"""Multiple-testing corrections returning adjusted p-values."""
from __future__ import annotations

import numpy as np


def adjust_pvalues(p, method: str = "bh") -> np.ndarray:
    """Adjusted p-values: 'bh' (Benjamini-Hochberg FDR), 'holm' or 'bonferroni' (FWER)."""
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    method = method.lower()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    order = np.argsort(p)
    ps = p[order]
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * ps)
    elif method in ("bh", "fdr_bh"):
        adj = np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1]
    else:
        raise ValueError(f"unknown correction {method!r}; expected bh, holm or bonferroni")
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out
