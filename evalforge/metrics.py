"""Point estimates and bootstrap confidence intervals for classification metrics.

Every metric is computed from the same bootstrap resamples (shared seed), so
intervals for different metrics, or for different models, are directly
comparable and paired.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from .kernels import get_backend

METRIC_NAMES = ("accuracy", "macro_f1", "auc", "brier", "ece")
DEFAULT_BINS = 15


@dataclass
class MetricResult:
    name: str
    point: float
    lo: float
    hi: float
    confidence: float = 0.95
    n_boot: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def width(self) -> float:
        return self.hi - self.lo


def prepare_inputs(y_true, proba, classes=None):
    """Validate and encode inputs.

    Returns ``(y, proba, classes)`` with ``y`` as int labels in ``[0, K)`` and
    ``proba`` an ``(n, K)`` row-stochastic matrix. A 1-D ``proba`` is treated as
    P(class == classes[1]) for a binary problem.
    """
    y_raw = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    if p.ndim != 2 or p.shape[0] != y_raw.shape[0] or y_raw.ndim != 1:
        raise ValueError("y_true must be 1-D and proba (n,) or (n, K) with matching n")
    if y_raw.shape[0] == 0:
        raise ValueError("empty evaluation set")
    if classes is None:
        classes = np.arange(p.shape[1]) if np.issubdtype(y_raw.dtype, np.integer) else np.unique(y_raw)
    classes = np.asarray(classes)
    if len(classes) != p.shape[1]:
        raise ValueError(f"proba has {p.shape[1]} columns but {len(classes)} classes")
    lookup = {c: i for i, c in enumerate(classes.tolist())}
    try:
        y = np.array([lookup[v] for v in y_raw.tolist()], dtype=np.int32)
    except KeyError as e:
        raise ValueError(f"label {e.args[0]!r} not in classes") from None
    if not np.isfinite(p).all() or p.min() < -1e-9 or p.max() > 1 + 1e-9:
        raise ValueError("proba must be finite and within [0, 1]")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("each row of proba must sum to 1")
    return y, np.clip(p, 0.0, 1.0), classes


def predictions(proba: np.ndarray) -> np.ndarray:
    return np.argmax(proba, axis=1).astype(np.int32)


def macro_f1_score(y, pred, n_classes) -> float:
    """Macro-F1 over all ``n_classes`` (absent classes score 0, like sklearn zero_division=0)."""
    f1 = []
    for k in range(n_classes):
        tp = np.sum((y == k) & (pred == k))
        d = np.sum(y == k) + np.sum(pred == k)
        f1.append(2.0 * tp / d if d > 0 else 0.0)
    return float(np.mean(f1))


def auc_score(y, proba) -> float:
    """Binary AUC, or macro one-vs-rest AUC for K>2. NaN when undefined."""
    K = proba.shape[1]
    try:
        if K == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        vals = [roc_auc_score((y == k).astype(int), proba[:, k]) for k in range(K)]
        return float(np.mean(vals))
    except ValueError:
        return float("nan")


def _bin_index(conf: np.ndarray, n_bins: int) -> np.ndarray:
    return np.minimum((conf * n_bins).astype(np.int32), n_bins - 1)


def ece_score(y, proba, n_bins: int = DEFAULT_BINS) -> float:
    """Expected calibration error of the top-label confidence (equal-width bins)."""
    conf = proba.max(axis=1)
    correct = (predictions(proba) == y).astype(float)
    bins = _bin_index(conf, n_bins)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += abs(np.sum(correct[m] - conf[m]))
    return float(total / len(y))


def brier_per_sample(y, proba) -> np.ndarray:
    onehot = np.eye(proba.shape[1])[y]
    return ((proba - onehot) ** 2).sum(axis=1)


def point_metrics(y, proba, n_bins: int = DEFAULT_BINS) -> dict[str, float]:
    pred = predictions(proba)
    out = {
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": macro_f1_score(y, pred, proba.shape[1]),
        "auc": auc_score(y, proba),
        "brier": float(brier_per_sample(y, proba).mean()),
        "ece": ece_score(y, proba, n_bins),
    }
    return out


def bootstrap_replicates(y, proba, n_boot=2000, seed=0, backend=None, n_bins=DEFAULT_BINS,
                         n_threads=0) -> dict[str, np.ndarray]:
    """Bootstrap distributions for every metric. Replicate b uses the same
    resample for all metrics, and for any other model bootstrapped with the same
    ``seed`` and sample size (needed for paired deltas)."""
    k = backend or get_backend()
    K = proba.shape[1]
    pred = predictions(proba)
    reps = {
        "accuracy": k.boot_mean((pred == y).astype(float), n_boot, seed, n_threads),
        "macro_f1": k.boot_macro_f1(y, pred, K, n_boot, seed, n_threads),
        "brier": k.boot_mean(brier_per_sample(y, proba), n_boot, seed, n_threads),
    }
    conf = proba.max(axis=1)
    reps["ece"] = k.boot_ece(_bin_index(conf, n_bins), (pred == y) - conf, n_bins, n_boot, seed, n_threads)
    if K == 2:
        auc = k.boot_auc(y, proba[:, 1], n_boot, seed, n_threads)
    else:
        auc = np.mean([k.boot_auc((y == c).astype(np.int32), proba[:, c], n_boot, seed, n_threads)
                       for c in range(K)], axis=0)
    reps["auc"] = auc
    return reps


def percentile_interval(reps: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    r = reps[np.isfinite(reps)]
    if r.size == 0:
        return float("nan"), float("nan")
    a = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(r, [a, 1.0 - a])
    return float(lo), float(hi)


def bootstrap_metrics(y_true, proba, classes=None, n_boot=2000, confidence=0.95, seed=0,
                      backend=None, n_bins=DEFAULT_BINS, n_threads=0) -> dict[str, MetricResult]:
    """Point estimates plus percentile bootstrap CIs for the whole metric suite.
    Metrics that are undefined on the data (e.g. AUC with one class) are omitted."""
    y, p, _ = prepare_inputs(y_true, proba, classes)
    return metrics_from_encoded(y, p, n_boot, confidence, seed, backend, n_bins, n_threads)[0]


def metrics_from_encoded(y, p, n_boot, confidence, seed, backend=None, n_bins=DEFAULT_BINS, n_threads=0):
    pts = point_metrics(y, p, n_bins)
    reps = bootstrap_replicates(y, p, n_boot, seed, backend, n_bins, n_threads)
    res = {}
    for name in METRIC_NAMES:
        if not np.isfinite(pts[name]):
            continue
        lo, hi = percentile_interval(reps[name], confidence)
        res[name] = MetricResult(name, pts[name], lo, hi, confidence, n_boot)
    return res, reps


def reliability_curve(y, proba, n_bins: int = 10):
    """Per-bin (mean confidence, accuracy, count) for a reliability diagram."""
    conf = proba.max(axis=1)
    correct = (predictions(proba) == y).astype(float)
    bins = _bin_index(conf, n_bins)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows
