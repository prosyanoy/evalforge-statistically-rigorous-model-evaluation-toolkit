"""Ship / iterate / stop decision rule driven by a YAML policy.

A policy is a list of rules. Each rule compares one statistic of one metric to
a threshold; if the comparison fails, the rule's ``on_fail`` verdict
(``stop`` or ``iterate``) is triggered. The final decision is the most severe
triggered verdict: STOP > ITERATE > SHIP. Rules whose metric is unavailable
(e.g. no baseline supplied) are reported as skipped, never silently passed.
"""
from __future__ import annotations

import operator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}
STATS = ("point", "ci_lower", "ci_upper", "p_value")
VERDICTS = ("stop", "iterate")
SCALAR_METRICS = ("significant_slices", "worst_slice_gap", "max_psi", "drifted_features", "n_eval")
CI_METRICS = ("accuracy", "macro_f1", "auc", "brier", "ece")
DELTA_METRICS = tuple(f"{m}_delta" for m in ("accuracy", "macro_f1", "auc", "brier"))
KNOWN_METRICS = CI_METRICS + DELTA_METRICS + SCALAR_METRICS
DEFAULT_POLICY_PATH = Path(__file__).parent / "policies" / "default.yaml"


class PolicyError(ValueError):
    pass


@dataclass
class Rule:
    id: str
    metric: str
    op: str
    threshold: float
    stat: str = "point"
    on_fail: str = "iterate"
    description: str = ""


@dataclass
class Policy:
    name: str
    rules: list[Rule]
    description: str = ""


@dataclass
class RuleOutcome:
    rule: Rule
    status: str            # "pass" | "fail" | "skipped"
    observed: float | None
    message: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Decision:
    verdict: str           # "SHIP" | "ITERATE" | "STOP"
    outcomes: list[RuleOutcome] = field(default_factory=list)
    policy_name: str = ""

    @property
    def failed(self) -> list[RuleOutcome]:
        return [o for o in self.outcomes if o.status == "fail"]

    @property
    def skipped(self) -> list[RuleOutcome]:
        return [o for o in self.outcomes if o.status == "skipped"]

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "policy": self.policy_name,
                "outcomes": [o.to_dict() for o in self.outcomes]}


def parse_policy(data: dict) -> Policy:
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list) or not data["rules"]:
        raise PolicyError("policy must be a mapping with a non-empty 'rules' list")
    rules, ids = [], set()
    for i, raw in enumerate(data["rules"]):
        if not isinstance(raw, dict):
            raise PolicyError(f"rule #{i + 1} must be a mapping")
        missing = {"metric", "op", "threshold"} - raw.keys()
        if missing:
            raise PolicyError(f"rule #{i + 1} is missing {sorted(missing)}")
        r = Rule(id=str(raw.get("id", f"rule{i + 1}")), metric=str(raw["metric"]), op=str(raw["op"]),
                 threshold=raw["threshold"], stat=str(raw.get("stat", "point")),
                 on_fail=str(raw.get("on_fail", "iterate")), description=str(raw.get("description", "")))
        if r.id in ids:
            raise PolicyError(f"duplicate rule id {r.id!r}")
        ids.add(r.id)
        if r.metric not in KNOWN_METRICS:
            raise PolicyError(f"rule {r.id!r}: unknown metric {r.metric!r}; known: {', '.join(KNOWN_METRICS)}")
        if r.op not in OPS:
            raise PolicyError(f"rule {r.id!r}: op must be one of {list(OPS)}")
        if r.stat not in STATS:
            raise PolicyError(f"rule {r.id!r}: stat must be one of {list(STATS)}")
        if r.stat == "p_value" and r.metric not in DELTA_METRICS:
            raise PolicyError(f"rule {r.id!r}: p_value is only available for *_delta metrics")
        if r.stat in ("ci_lower", "ci_upper") and r.metric in ("n_eval", "significant_slices", "max_psi",
                                                              "drifted_features"):
            raise PolicyError(f"rule {r.id!r}: metric {r.metric!r} has no confidence interval")
        if r.on_fail not in VERDICTS:
            raise PolicyError(f"rule {r.id!r}: on_fail must be one of {list(VERDICTS)}")
        if isinstance(r.threshold, bool) or not isinstance(r.threshold, (int, float)):
            raise PolicyError(f"rule {r.id!r}: threshold must be a number")
        r.threshold = float(r.threshold)
        rules.append(r)
    return Policy(str(data.get("name", "unnamed")), rules, str(data.get("description", "")))


def load_policy(path: str | Path | None = None) -> Policy:
    p = Path(path) if path else DEFAULT_POLICY_PATH
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise PolicyError(f"invalid YAML in {p}: {e}") from e
    return parse_policy(data)


def decide(policy: Policy, values: dict[str, dict[str, float]]) -> Decision:
    """``values[metric][stat]`` -> float. Missing / NaN entries skip the rule."""
    outcomes = []
    for r in policy.rules:
        v = values.get(r.metric, {}).get(r.stat)
        if v is None or v != v:
            outcomes.append(RuleOutcome(r, "skipped", None, f"{r.metric}.{r.stat} not available"))
            continue
        ok = OPS[r.op](v, r.threshold)
        msg = f"{r.metric}.{r.stat} = {v:.4g} (required {r.op} {r.threshold:g})"
        outcomes.append(RuleOutcome(r, "pass" if ok else "fail", float(v), msg))
    failed = [o for o in outcomes if o.status == "fail"]
    if any(o.rule.on_fail == "stop" for o in failed):
        verdict = "STOP"
    elif failed:
        verdict = "ITERATE"
    else:
        verdict = "SHIP"
    return Decision(verdict, outcomes, policy.name)
