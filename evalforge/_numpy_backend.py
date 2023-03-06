"""Pure NumPy implementation of the resampling kernels.

Mirrors the signatures and semantics of the C++ ``_core`` module. The random
streams differ (NumPy ``Generator`` vs. the native splitmix64 streams), so the
two backends agree statistically, not bit-for-bit. Within one backend the same
seed always yields the same resamples, which is what paired analyses rely on.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

name = "numpy"

# Cap on the number of float64 cells materialised per chunk (~32 MB).
_CHUNK_CELLS = 4_000_000


def _chunks(n_rep: int, n: int):
    step = max(1, _CHUNK_CELLS // max(n, 1))
    for lo in range(0, n_rep, step):
        yield lo, min(n_rep, lo + step)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(int(seed)))


def _counts(idx: np.ndarray, n: int) -> np.ndarray:
    """Resample multiplicities, shape (c, n), from index matrix (c, n)."""
    c = idx.shape[0]
    flat = (idx + (np.arange(c)[:, None] * n)).ravel()
    return np.bincount(flat, minlength=c * n).reshape(c, n)


def boot_mean(values, n_boot, seed, n_threads=0):
    v = np.ascontiguousarray(values, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    n = v.size
    rng = _rng(seed)
    out = np.empty(n_boot)
    for lo, hi in _chunks(n_boot, n):
        idx = rng.integers(0, n, size=(hi - lo, n))
        out[lo:hi] = v[idx].mean(axis=1)
    return out


def _macro_f1_from_counts(tp, pc, tc):
    denom = pc + tc
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2.0 * tp / np.where(denom > 0, denom, 1), 0.0)
    return f1.mean(axis=-1)


def boot_macro_f1(y, pred, n_classes, n_boot, seed, n_threads=0):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    n = y.size
    if y.shape != pred.shape or y.ndim != 1 or n == 0:
        raise ValueError("y/pred must be equal-length non-empty 1-D arrays")
    if y.min() < 0 or pred.min() < 0 or y.max() >= n_classes or pred.max() >= n_classes:
        raise ValueError("labels outside [0, n_classes)")
    eye = np.eye(n_classes)
    m_tp = eye[y] * (y == pred)[:, None]
    m_pc, m_tc = eye[pred], eye[y]
    rng = _rng(seed)
    out = np.empty(n_boot)
    for lo, hi in _chunks(n_boot, n):
        w = _counts(rng.integers(0, n, size=(hi - lo, n)), n).astype(float)
        out[lo:hi] = _macro_f1_from_counts(w @ m_tp, w @ m_pc, w @ m_tc)
    return out


def boot_auc(y, score, n_boot, seed, n_threads=0):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(score, dtype=float)
    n = y.size
    if y.shape != s.shape or y.ndim != 1 or n == 0:
        raise ValueError("y/score must be equal-length non-empty 1-D arrays")
    if y.min() < 0 or y.max() > 1:
        raise ValueError("y must be binary (0/1)")
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    pos_of = np.empty(n, dtype=np.int64)
    pos_of[order] = np.arange(n)
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    is_pos = (y_sorted == 1).astype(float)
    rng = _rng(seed)
    out = np.empty(n_boot)
    for lo, hi in _chunks(n_boot, n):
        idx = rng.integers(0, n, size=(hi - lo, n))
        w = _counts(pos_of[idx], n).astype(float)
        gp = np.add.reduceat(w * is_pos, starts, axis=1)
        gn = np.add.reduceat(w * (1.0 - is_pos), starts, axis=1)
        cum_neg_before = np.cumsum(gn, axis=1) - gn
        num = (gp * (cum_neg_before + 0.5 * gn)).sum(axis=1)
        tp, tn = gp.sum(axis=1), gn.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            res = num / (tp * tn)
        res[(tp == 0) | (tn == 0)] = np.nan
        out[lo:hi] = res
    return out


def boot_ece(bins, gap, n_bins, n_boot, seed, n_threads=0):
    bins = np.asarray(bins, dtype=np.int64)
    gap = np.asarray(gap, dtype=float)
    n = bins.size
    if bins.shape != gap.shape or bins.ndim != 1 or n == 0:
        raise ValueError("bins/gap must be equal-length non-empty 1-D arrays")
    if bins.min() < 0 or bins.max() >= n_bins:
        raise ValueError("bins outside [0, n_bins)")
    m = np.eye(n_bins)[bins] * gap[:, None]
    rng = _rng(seed)
    out = np.empty(n_boot)
    for lo, hi in _chunks(n_boot, n):
        w = _counts(rng.integers(0, n, size=(hi - lo, n)), n).astype(float)
        out[lo:hi] = np.abs(w @ m).sum(axis=1) / n
    return out


def _swap_masks(rng, c, n):
    return rng.random((c, n)) < 0.5


def perm_mean(a, b, n_perm, seed, n_threads=0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    if a.shape != b.shape or a.ndim != 1 or n == 0:
        raise ValueError("a/b must be equal-length non-empty 1-D arrays")
    d = a - b
    rng = _rng(seed)
    out = np.empty(n_perm)
    for lo, hi in _chunks(n_perm, n):
        sign = np.where(_swap_masks(rng, hi - lo, n), -1.0, 1.0)
        out[lo:hi] = (sign * d).mean(axis=1)
    return out


def _batch_macro_f1(y, pred, K):
    """Macro-F1 for a (c, n) matrix of predictions against a fixed y."""
    f1 = np.zeros((pred.shape[0], K))
    hit = pred == y[None, :]
    for k in range(K):
        tp = (hit & (y == k)[None, :]).sum(axis=1)
        pc = (pred == k).sum(axis=1)
        tc = int((y == k).sum())
        denom = pc + tc
        f1[:, k] = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean(axis=1)


def perm_macro_f1(y, pred_a, pred_b, n_classes, n_perm, seed, n_threads=0):
    y = np.asarray(y, dtype=np.int64)
    A = np.asarray(pred_a, dtype=np.int64)
    B = np.asarray(pred_b, dtype=np.int64)
    n = y.size
    if not (y.shape == A.shape == B.shape) or y.ndim != 1 or n == 0:
        raise ValueError("y/pred_a/pred_b must be equal-length non-empty 1-D arrays")
    rng = _rng(seed)
    out = np.empty(n_perm)
    for lo, hi in _chunks(n_perm, n * max(n_classes, 1)):
        sw = _swap_masks(rng, hi - lo, n)
        out[lo:hi] = _batch_macro_f1(y, np.where(sw, B, A), n_classes) - _batch_macro_f1(
            y, np.where(sw, A, B), n_classes
        )
    return out


def perm_auc(y, score_a, score_b, n_perm, seed, n_threads=0):
    y = np.asarray(y, dtype=np.int64)
    A = np.asarray(score_a, dtype=float)
    B = np.asarray(score_b, dtype=float)
    n = y.size
    if not (y.shape == A.shape == B.shape) or y.ndim != 1 or n == 0:
        raise ValueError("y/score_a/score_b must be equal-length non-empty 1-D arrays")
    is_pos = y == 1
    p, q = int(is_pos.sum()), int((~is_pos).sum())
    rng = _rng(seed)
    out = np.empty(n_perm)
    if p == 0 or q == 0:
        out[:] = np.nan
        return out

    def auc(s):
        r = rankdata(s, axis=1)
        return (r[:, is_pos].sum(axis=1) - p * (p + 1) / 2.0) / (p * q)

    for lo, hi in _chunks(n_perm, n * 4):
        sw = _swap_masks(rng, hi - lo, n)
        out[lo:hi] = auc(np.where(sw, B, A)) - auc(np.where(sw, A, B))
    return out
