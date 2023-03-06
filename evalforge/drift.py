"""Feature drift between a reference sample (e.g. training data) and the evaluation data."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .multitest import adjust_pvalues

PSI_WARN = 0.10
PSI_ALERT = 0.25   # conventional industry thresholds


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_stat: float | None
    p_value: float | None
    p_adjusted: float | None
    drifted: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    features: list[FeatureDrift]
    max_psi: float
    n_drifted: int
    alpha: float

    def to_dict(self) -> dict:
        return {"features": [f.to_dict() for f in self.features], "max_psi": self.max_psi,
                "n_drifted": self.n_drifted, "alpha": self.alpha}


def psi(reference, current, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index. Numeric features use reference-quantile bins;
    others use category frequencies (unseen categories get their own bin)."""
    ref, cur = pd.Series(reference).dropna(), pd.Series(current).dropna()
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > n_bins:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pr = np.histogram(ref, edges)[0] / len(ref)
        pc = np.histogram(cur, edges)[0] / len(cur)
    else:
        cats = pd.Index(ref.unique()).union(pd.Index(cur.unique()))
        pr = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        pc = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    pr, pc = np.maximum(pr, eps), np.maximum(pc, eps)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05,
                 psi_threshold: float = PSI_ALERT, method: str = "bh") -> DriftReport:
    """Per-feature PSI and two-sample KS test (numeric), BH-corrected across features.
    A feature is flagged when it is statistically significant *and* PSI exceeds
    ``psi_threshold`` (or, for non-numeric features, on PSI alone)."""
    cols = [c for c in current.columns if c in reference.columns]
    if not cols:
        raise ValueError("reference and current share no columns")
    rows, pvals = [], []
    for c in cols:
        numeric = pd.api.types.is_numeric_dtype(current[c]) and pd.api.types.is_numeric_dtype(reference[c])
        stat = p = None
        if numeric:
            r = ks_2samp(reference[c].dropna(), current[c].dropna())
            stat, p = float(r.statistic), float(r.pvalue)
        rows.append([c, psi(reference[c], current[c], n_bins=max(3, min(10, min(len(reference), len(current)) // 15))), stat, p])
        pvals.append(p if p is not None else np.nan)
    pv = np.array(pvals, dtype=float)
    adj = np.full(len(cols), np.nan)
    ok = np.isfinite(pv)
    if ok.any():
        adj[ok] = adjust_pvalues(pv[ok], method)
    feats = []
    for (c, ps, stat, p), a in zip(rows, adj):
        drifted = bool(ps >= psi_threshold and (np.isnan(a) or a <= alpha))
        feats.append(FeatureDrift(c, ps, stat, p, None if np.isnan(a) else float(a), drifted))
    feats.sort(key=lambda f: -f.psi)
    return DriftReport(feats, max(f.psi for f in feats), sum(f.drifted for f in feats), alpha)
