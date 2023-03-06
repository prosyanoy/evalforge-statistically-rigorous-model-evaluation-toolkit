"""Paired model comparison: bootstrap CI of the difference + permutation test."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .kernels import get_backend
from .metrics import (DEFAULT_BINS, auc_score, bootstrap_replicates, macro_f1_score,
                      percentile_interval, point_metrics, predictions, prepare_inputs)

COMPARED = ("accuracy", "macro_f1", "auc", "brier")


@dataclass
class Comparison:
    metric: str
    candidate: float
    baseline: float
    delta: float
    lo: float
    hi: float
    p_value: float
    n_perm: int

    def to_dict(self) -> dict:
        return asdict(self)


def permutation_p_value(observed: float, null: np.ndarray) -> float:
    """Two-sided p-value with the +1 correction (never exactly zero)."""
    null = null[np.isfinite(null)]
    extreme = np.sum(np.abs(null) >= abs(observed) - 1e-12)
    return float((1 + extreme) / (1 + null.size))


def permutation_null(metric, y, proba_a, proba_b, n_perm, seed, backend=None, n_threads=0) -> np.ndarray:
    """Null distribution of metric(A) - metric(B) under per-sample exchangeability."""
    k = backend or get_backend()
    K = proba_a.shape[1]
    pa, pb = predictions(proba_a), predictions(proba_b)
    if metric == "accuracy":
        return k.perm_mean((pa == y).astype(float), (pb == y).astype(float), n_perm, seed, n_threads)
    if metric == "brier":
        eye = np.eye(K)[y]
        return k.perm_mean(((proba_a - eye) ** 2).sum(1), ((proba_b - eye) ** 2).sum(1), n_perm, seed, n_threads)
    if metric == "macro_f1":
        return k.perm_macro_f1(y, pa, pb, K, n_perm, seed, n_threads)
    if metric == "auc":
        if K == 2:
            return k.perm_auc(y, proba_a[:, 1], proba_b[:, 1], n_perm, seed, n_threads)
        # same seed => identical swap masks for every class
        return np.mean([k.perm_auc((y == c).astype(np.int32), proba_a[:, c], proba_b[:, c], n_perm, seed, n_threads)
                        for c in range(K)], axis=0)
    raise ValueError(f"unsupported metric {metric!r}")


def compare_models(y_true, proba_candidate, proba_baseline, classes=None, metrics=COMPARED,
                   n_boot=2000, n_perm=10000, confidence=0.95, seed=0, backend=None,
                   n_bins=DEFAULT_BINS, n_threads=0) -> dict[str, Comparison]:
    """Compare ``candidate - baseline`` on the same samples.

    The CI comes from a paired bootstrap (both models scored on identical
    resamples); the p-value from a paired permutation test that randomly swaps
    the two models' outputs per sample. Delta > 0 means the candidate is better
    (for Brier, which is lower-is-better, delta < 0 means the candidate is better).
    """
    y, pa, _ = prepare_inputs(y_true, proba_candidate, classes)
    _, pb, _ = prepare_inputs(y_true, proba_baseline, classes)
    return compare_encoded(y, pa, pb, metrics, n_boot, n_perm, confidence, seed, backend, n_bins, n_threads)


def compare_encoded(y, pa, pb, metrics=COMPARED, n_boot=2000, n_perm=10000, confidence=0.95, seed=0,
                    backend=None, n_bins=DEFAULT_BINS, n_threads=0, reps_a=None, reps_b=None):
    ma, mb = point_metrics(y, pa, n_bins), point_metrics(y, pb, n_bins)
    reps_a = reps_a or bootstrap_replicates(y, pa, n_boot, seed, backend, n_bins, n_threads)
    reps_b = reps_b or bootstrap_replicates(y, pb, n_boot, seed, backend, n_bins, n_threads)
    out = {}
    for m in metrics:
        if not (np.isfinite(ma[m]) and np.isfinite(mb[m])):
            continue
        delta = ma[m] - mb[m]
        lo, hi = percentile_interval(reps_a[m] - reps_b[m], confidence)
        null = permutation_null(m, y, pa, pb, n_perm, seed + 1, backend, n_threads)
        out[m] = Comparison(m, ma[m], mb[m], delta, lo, hi, permutation_p_value(delta, null), n_perm)
    return out
