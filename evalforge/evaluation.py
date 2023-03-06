"""End-to-end evaluation: metrics + comparison + slices + drift + decision."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import kernels
from .compare import COMPARED, Comparison, compare_encoded
from .drift import DriftReport, detect_drift
from .metrics import (DEFAULT_BINS, MetricResult, bootstrap_replicates, metrics_from_encoded,
                      prepare_inputs, predictions, reliability_curve)
from .policy import Decision, Policy, decide, load_policy
from .slices import SliceReport, discover_slices


@dataclass
class EvalConfig:
    n_boot: int = 2000
    n_perm: int = 10000
    confidence: float = 0.95
    seed: int = 0
    n_bins: int = DEFAULT_BINS
    alpha: float = 0.05
    slice_depth: int = 2
    slice_min_support: int = 20
    slice_correction: str = "bh"
    backend: str | None = None
    n_threads: int = 0


@dataclass
class EvaluationResult:
    name: str
    n_eval: int
    classes: list
    config: EvalConfig
    backend: str
    metrics: dict[str, MetricResult]
    reliability: list
    comparison: dict[str, Comparison] | None
    baseline_metrics: dict[str, MetricResult] | None
    slices: SliceReport | None
    drift: DriftReport | None
    decision: Decision
    boot_replicates: dict = field(default_factory=dict, repr=False)

    def policy_values(self) -> dict[str, dict[str, float]]:
        return build_policy_values(self)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "n_eval": self.n_eval, "classes": [str(c) for c in self.classes],
            "backend": self.backend,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "baseline_metrics": {k: v.to_dict() for k, v in (self.baseline_metrics or {}).items()},
            "comparison": {k: v.to_dict() for k, v in (self.comparison or {}).items()},
            "slices": self.slices.to_dict() if self.slices else None,
            "drift": self.drift.to_dict() if self.drift else None,
            "decision": self.decision.to_dict(),
        }


def build_policy_values(r: EvaluationResult) -> dict[str, dict[str, float]]:
    v: dict[str, dict[str, float]] = {"n_eval": {"point": float(r.n_eval)}}
    for k, m in r.metrics.items():
        v[k] = {"point": m.point, "ci_lower": m.lo, "ci_upper": m.hi}
    for k, c in (r.comparison or {}).items():
        v[f"{k}_delta"] = {"point": c.delta, "ci_lower": c.lo, "ci_upper": c.hi, "p_value": c.p_value}
    if r.slices is not None:
        sl = r.slices.slices
        worst = max(sl, key=lambda s: s.gap, default=None)
        v["significant_slices"] = {"point": float(len(sl))}
        v["worst_slice_gap"] = {"point": worst.gap if worst else 0.0,
                                "ci_lower": worst.gap_lo if worst else 0.0,
                                "ci_upper": worst.gap_hi if worst else 0.0}
    if r.drift is not None:
        v["max_psi"] = {"point": r.drift.max_psi}
        v["drifted_features"] = {"point": float(r.drift.n_drifted)}
    return v


def evaluate(y_true, proba, X: pd.DataFrame | None = None, *, classes=None, baseline_proba=None,
             X_reference: pd.DataFrame | None = None, policy: Policy | str | None = None,
             config: EvalConfig | None = None, name: str = "model") -> EvaluationResult:
    """Evaluate a classifier's predicted probabilities on held-out data.

    ``X`` enables slice discovery, ``baseline_proba`` the paired comparison and
    ``X_reference`` (e.g. training features) the drift check. Anything not
    supplied is skipped, and the policy reports the dependent rules as skipped.
    """
    cfg = config or EvalConfig()
    backend = kernels.get_backend(cfg.backend)
    y, p, classes = prepare_inputs(y_true, proba, classes)
    metrics, reps = metrics_from_encoded(y, p, cfg.n_boot, cfg.confidence, cfg.seed, backend, cfg.n_bins,
                                         cfg.n_threads)
    comparison = base_metrics = None
    if baseline_proba is not None:
        _, pb, _ = prepare_inputs(y_true, baseline_proba, classes)
        base_metrics, reps_b = metrics_from_encoded(y, pb, cfg.n_boot, cfg.confidence, cfg.seed, backend,
                                                    cfg.n_bins, cfg.n_threads)
        comparison = compare_encoded(y, p, pb, COMPARED, cfg.n_boot, cfg.n_perm, cfg.confidence, cfg.seed,
                                     backend, cfg.n_bins, cfg.n_threads, reps_a=reps, reps_b=reps_b)
    slices = None
    if X is not None:
        errors = (predictions(p) != y).astype(float)
        slices = discover_slices(X, errors, max_depth=cfg.slice_depth, min_support=cfg.slice_min_support,
                                 alpha=cfg.alpha, method=cfg.slice_correction)
    drift = detect_drift(X_reference, X, alpha=cfg.alpha) if (X is not None and X_reference is not None) else None
    result = EvaluationResult(name, len(y), classes.tolist(), cfg, backend.name, metrics,
                              reliability_curve(y, p), comparison, base_metrics, slices, drift,
                              Decision("SHIP"), reps)
    pol = policy if isinstance(policy, Policy) else load_policy(policy)
    result.decision = decide(pol, build_policy_values(result))
    result.boot_replicates["_y"], result.boot_replicates["_p"] = y, p
    return result
