import numpy as np
import pandas as pd
import pytest
import yaml

from evalforge import (EvalConfig, PolicyError, adjust_pvalues, decide, detect_drift, discover_slices, evaluate,
                       load_policy, psi)
from evalforge.policy import DEFAULT_POLICY_PATH, parse_policy


def make_frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"age": rng.uniform(18, 80, n), "income": rng.normal(50, 10, n),
                         "region": rng.choice(["n", "s", "e", "w"], n), "noise": rng.normal(size=n)})


def test_multiple_testing_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert adjust_pvalues(p, "bonferroni") == pytest.approx([0.04, 0.16, 0.12, 0.02])
    assert adjust_pvalues(p, "holm") == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert adjust_pvalues(p, "bh") == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert adjust_pvalues([], "bh").size == 0
    with pytest.raises(ValueError):
        adjust_pvalues(p, "nope")


def test_slice_discovery_finds_planted_slice():
    X = make_frame()
    rng = np.random.default_rng(1)
    in_slice = ((X.age > 60) & (X.region == "s")).to_numpy()
    err = np.where(in_slice, rng.random(len(X)) < 0.6, rng.random(len(X)) < 0.08).astype(float)
    rep = discover_slices(X, err)
    assert rep.slices, "planted slice not found"
    top = rep.slices[0]
    assert "region = s" in top.description or "age >" in top.description
    assert top.gap > 0.3 and top.p_adjusted < 0.05 and top.gap_lo > 0
    assert rep.n_tested > 20


def test_slice_discovery_controls_false_positives():
    """Pure-noise errors: with BH at alpha=.05 we expect (almost) no discoveries."""
    X = make_frame(1000)
    false_runs = 0
    for seed in range(15):
        err = (np.random.default_rng(100 + seed).random(len(X)) < 0.15).astype(float)
        false_runs += bool(discover_slices(X, err).slices)
    assert false_runs <= 3  # FDR ~5% per run; allow slack for 15 trials


def test_uncorrected_testing_would_have_false_positives():
    """Guards the point of the correction: the raw p-values on noise do cross 0.05."""
    X = make_frame(1000)
    err = (np.random.default_rng(7).random(len(X)) < 0.15).astype(float)
    rep = discover_slices(X, err, method="bonferroni", alpha=1.0, max_slices=10_000, redundancy_jaccard=1.01)
    assert (np.array([s.p_value for s in rep.slices]) < 0.05).any()


def test_slice_edge_cases():
    X = make_frame(200)
    assert discover_slices(X, np.zeros(200)).slices == []
    assert discover_slices(X, np.ones(200)).slices == []
    with pytest.raises(ValueError):
        discover_slices(X, np.zeros(10))
    with pytest.raises(ValueError):
        discover_slices(X, np.zeros(200), max_depth=3)


def test_drift_detects_shift_and_ignores_iid():
    ref = make_frame(2000, 0)
    same = make_frame(1000, 1)
    assert detect_drift(ref, same).n_drifted == 0
    shifted = same.assign(income=same.income + 15)
    rep = detect_drift(ref, shifted)
    assert rep.features[0].feature == "income" and rep.features[0].drifted and rep.n_drifted == 1
    assert psi(ref.income, shifted.income) > 0.25 > psi(ref.income, same.income)
    cat = same.assign(region="n")
    assert psi(ref.region, cat.region) > 0.25


VALUES = {"n_eval": {"point": 500}, "accuracy": {"point": 0.93, "ci_lower": 0.91, "ci_upper": 0.95},
          "ece": {"point": 0.02}, "significant_slices": {"point": 0}}


def pol(*rules):
    return parse_policy({"name": "t", "rules": list(rules)})


def test_policy_verdict_precedence():
    ship = pol({"metric": "accuracy", "stat": "ci_lower", "op": ">=", "threshold": 0.9})
    assert decide(ship, VALUES).verdict == "SHIP"
    iterate = pol({"metric": "accuracy", "stat": "ci_lower", "op": ">=", "threshold": 0.95})
    assert decide(iterate, VALUES).verdict == "ITERATE"
    both = pol({"id": "a", "metric": "accuracy", "stat": "ci_lower", "op": ">=", "threshold": 0.95},
               {"id": "b", "metric": "ece", "op": "<=", "threshold": 0.01, "on_fail": "stop"})
    d = decide(both, VALUES)
    assert d.verdict == "STOP" and len(d.failed) == 2


def test_policy_skips_unavailable_metrics():
    p = pol({"metric": "accuracy_delta", "stat": "p_value", "op": "<=", "threshold": 0.05})
    d = decide(p, VALUES)
    assert d.verdict == "SHIP" and len(d.skipped) == 1


@pytest.mark.parametrize("bad", [
    {"metric": "bogus", "op": ">=", "threshold": 1},
    {"metric": "accuracy", "op": "~", "threshold": 1},
    {"metric": "accuracy", "op": ">=", "threshold": "high"},
    {"metric": "accuracy", "op": ">=", "threshold": 1, "on_fail": "panic"},
    {"metric": "accuracy", "op": ">=", "threshold": 1, "stat": "p_value"},
    {"metric": "n_eval", "op": ">=", "threshold": 1, "stat": "ci_lower"},
    {"metric": "accuracy", "op": ">="},
])
def test_policy_validation(bad):
    with pytest.raises(PolicyError):
        pol(bad)


def test_policy_files_load(tmp_path):
    assert len(load_policy().rules) >= 5
    strict = DEFAULT_POLICY_PATH.parent / "strict.yaml"
    assert load_policy(strict).name == "strict"
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump({"name": "x", "rules": [{"metric": "ece", "op": "<=", "threshold": 0.1}]}))
    assert load_policy(f).rules[0].metric == "ece"
    f.write_text("rules: [")
    with pytest.raises(PolicyError):
        load_policy(f)
    with pytest.raises(PolicyError):
        parse_policy({"rules": []})


def test_end_to_end_evaluate_decision_flow(backend):
    rng = np.random.default_rng(0)
    X = make_frame(800)
    y = rng.integers(0, 2, 800)
    conf = rng.uniform(0.7, 0.99, 800)
    pred = np.where(rng.random(800) < 0.97, y, 1 - y)
    p1 = np.where(pred == 1, conf, 1 - conf)
    proba = np.column_stack([1 - p1, p1])
    cfg = EvalConfig(n_boot=300, n_perm=300, backend=backend.name)
    good = evaluate(y, proba, X, X_reference=make_frame(800, 5), config=cfg)
    assert good.backend == backend.name and good.decision.verdict in {"SHIP", "ITERATE"}
    assert good.drift.n_drifted == 0
    bad = evaluate(y, 1 - proba, X, config=cfg)  # inverted predictions
    assert bad.decision.verdict == "STOP"
    assert {o.status for o in bad.decision.outcomes} >= {"fail", "skipped"}
    d = bad.to_dict()
    assert d["decision"]["verdict"] == "STOP" and "accuracy" in d["metrics"]
