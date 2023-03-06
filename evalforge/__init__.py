"""EvalForge - statistically rigorous model evaluation."""
from .compare import Comparison, compare_models
from .drift import detect_drift, psi
from .evaluation import EvalConfig, EvaluationResult, evaluate
from .kernels import cpp_available, get_backend
from .metrics import MetricResult, bootstrap_metrics, ece_score
from .multitest import adjust_pvalues
from .policy import Decision, Policy, PolicyError, decide, load_policy
from .slices import discover_slices

__version__ = "0.1.0"
__all__ = [
    "Comparison", "compare_models", "detect_drift", "psi", "EvalConfig", "EvaluationResult", "evaluate",
    "cpp_available", "get_backend", "MetricResult", "bootstrap_metrics", "ece_score", "adjust_pvalues",
    "Decision", "Policy", "PolicyError", "decide", "load_policy", "discover_slices",
]
