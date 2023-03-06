"""Automatic failure-slice discovery.

Candidate slices are conjunctions (up to ``max_depth`` predicates, on distinct
features) of simple feature predicates: quantile bins for numeric features and
level equality for categorical / low-cardinality ones. For each slice we test
H1: "error rate inside the slice > error rate outside" with a one-sided Fisher
exact test (hypergeometric tail), correct *all* tested slices together for
multiplicity, and keep only the significant, non-redundant ones.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .multitest import adjust_pvalues


@dataclass
class Slice:
    description: str
    n: int
    n_errors: int
    error_rate: float
    complement_error_rate: float
    gap: float          # error_rate - complement_error_rate
    gap_lo: float       # 95% Newcombe-Wilson CI of the gap
    gap_hi: float
    p_value: float
    p_adjusted: float
    depth: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SliceReport:
    slices: list[Slice]          # significant, de-duplicated, sorted by adjusted p then gap
    n_tested: int
    method: str
    alpha: float
    overall_error_rate: float

    def to_dict(self) -> dict:
        return {"slices": [s.to_dict() for s in self.slices], "n_tested": self.n_tested,
                "method": self.method, "alpha": self.alpha, "overall_error_rate": self.overall_error_rate}


@dataclass
class _Pred:
    feature: str
    text: str
    mask: np.ndarray


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def build_predicates(X: pd.DataFrame, n_bins: int = 4, max_levels: int = 6, min_support: int = 20) -> list[_Pred]:
    preds: list[_Pred] = []
    seen: set[bytes] = set()

    def add(feature, text, mask):
        mask = np.asarray(mask, dtype=bool)
        s = int(mask.sum())
        if s < min_support or s > len(mask) - min_support:
            return
        key = np.packbits(mask).tobytes()
        if key in seen:      # identical sample set already covered by another predicate
            return
        seen.add(key)
        preds.append(_Pred(feature, text, mask))

    for col in X.columns:
        s = X[col]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        valid = s.notna().to_numpy()
        if not numeric or s.nunique() <= max_levels:
            for level in s.dropna().unique():
                add(col, f"{col} = {level}", (s == level).to_numpy())
            continue
        vals = s.to_numpy(dtype=float)
        edges = np.unique(np.nanquantile(vals, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:   # heavily tied (e.g. mostly zeros): split at the minimum
            add(col, f"{col} = {_fmt(edges[0])}", valid & (vals == edges[0]))
            add(col, f"{col} > {_fmt(edges[0])}", valid & (vals > edges[0]))
            continue
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if i == 0:
                add(col, f"{col} <= {_fmt(hi)}", valid & (vals <= hi))
            elif i == len(edges) - 2:
                add(col, f"{col} > {_fmt(lo)}", valid & (vals > lo))
            else:
                add(col, f"{col} in ({_fmt(lo)}, {_fmt(hi)}]", valid & (vals > lo) & (vals <= hi))
    return preds


def _wilson(k, n, z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _newcombe_gap(k1, n1, k2, n2, z=1.959964):
    """Newcombe hybrid-score CI for p1 - p2 (inputs may be vectors)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = _wilson(k1, n1, z)
    l2, u2 = _wilson(k2, n2, z)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def discover_slices(X: pd.DataFrame, errors, max_depth: int = 2, n_bins: int = 4, min_support: int = 20,
                    alpha: float = 0.05, method: str = "bh", min_gap: float = 0.0,
                    redundancy_jaccard: float = 0.8, max_slices: int = 10) -> SliceReport:
    """Find feature-defined subgroups with a significantly higher error rate.

    ``errors`` is a 0/1 vector (1 = model wrong). ``min_gap`` is an extra
    effect-size filter applied *after* correction, so it never affects the
    family of tests that the correction is computed over.
    """
    err = np.asarray(errors, dtype=float)
    X = X.reset_index(drop=True)
    n = len(err)
    if len(X) != n:
        raise ValueError("X and errors must have the same length")
    if max_depth not in (1, 2):
        raise ValueError("max_depth must be 1 or 2")
    N, K = n, int(err.sum())
    preds = build_predicates(X, n_bins=n_bins, min_support=min_support)
    if not preds or K == 0 or K == N:
        return SliceReport([], 0, method, alpha, K / N)

    M = np.stack([p.mask for p in preds]).astype(np.float32)          # (P, n)
    feats = np.array([p.feature for p in preds])
    cand: list[tuple[tuple[int, ...], int, int]] = []                  # (members, size, errors)
    sizes, errs = M.sum(1), M @ err.astype(np.float32)
    for i in range(len(preds)):
        cand.append(((i,), int(sizes[i]), int(round(errs[i]))))
    if max_depth == 2:
        pair_n = M @ M.T
        pair_e = (M * err.astype(np.float32)) @ M.T
        ii, jj = np.triu_indices(len(preds), k=1)
        keep = (feats[ii] != feats[jj]) & (pair_n[ii, jj] >= min_support) & (pair_n[ii, jj] <= N - min_support)
        for i, j in zip(ii[keep], jj[keep]):
            cand.append(((int(i), int(j)), int(round(pair_n[i, j])), int(round(pair_e[i, j]))))

    n_s = np.array([c[1] for c in cand])
    k_s = np.array([c[2] for c in cand])
    # one-sided Fisher exact: P(X >= k) for X ~ Hypergeom(N, K, n_s)
    pvals = hypergeom.sf(k_s - 1, N, K, n_s)
    padj = adjust_pvalues(pvals, method)

    n_c, k_c = N - n_s, K - k_s
    rate, rate_c = k_s / n_s, k_c / n_c
    lo, hi = _newcombe_gap(k_s, n_s, k_c, n_c)
    gap = rate - rate_c

    order = [i for i in np.argsort(padj + 1e-12 * -gap, kind="stable") if padj[i] <= alpha and gap[i] > min_gap]
    chosen: list[int] = []
    masks: list[np.ndarray] = []
    for i in order:
        mask = np.logical_and.reduce([preds[m].mask for m in cand[i][0]])
        redundant = any((mask & o).sum() / max((mask | o).sum(), 1) >= redundancy_jaccard for o in masks)
        if redundant:
            continue
        chosen.append(i)
        masks.append(mask)
        if len(chosen) >= max_slices:
            break
    out = [Slice(" AND ".join(preds[m].text for m in cand[i][0]), int(n_s[i]), int(k_s[i]), float(rate[i]),
                 float(rate_c[i]), float(gap[i]), float(lo[i]), float(hi[i]), float(pvals[i]), float(padj[i]),
                 len(cand[i][0])) for i in chosen]
    return SliceReport(out, len(cand), method, alpha, K / N)
