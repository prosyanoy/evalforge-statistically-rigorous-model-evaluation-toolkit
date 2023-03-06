import numpy as np
import pytest

from evalforge import bootstrap_metrics, compare_models, ece_score
from evalforge.compare import permutation_p_value
from evalforge.metrics import prepare_inputs


def synth(n, acc, seed):
    """Probabilistic predictions that are correct with probability ~acc."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    right = rng.random(n) < acc
    pred = np.where(right, y, 1 - y)
    conf = rng.uniform(0.6, 0.95, n)
    p1 = np.where(pred == 1, conf, 1 - conf)
    return y, np.column_stack([1 - p1, p1])


def test_ci_contains_point_and_narrows_with_n(backend):
    y, p = synth(200, 0.85, 0)
    small = bootstrap_metrics(y, p, n_boot=1500, backend=backend)["accuracy"]
    y, p = synth(2000, 0.85, 0)
    big = bootstrap_metrics(y, p, n_boot=1500, backend=backend)["accuracy"]
    assert small.lo <= small.point <= small.hi and big.lo <= big.point <= big.hi
    assert big.width < small.width / 2


def test_bootstrap_ci_coverage(backend):
    """Empirical coverage of the 95% accuracy CI should be close to nominal."""
    hits, trials, true_acc = 0, 150, 0.8
    for seed in range(trials):
        y, p = synth(150, true_acc, seed)
        m = bootstrap_metrics(y, p, n_boot=400, seed=seed, backend=backend)["accuracy"]
        hits += m.lo <= true_acc <= m.hi
    assert 0.88 <= hits / trials <= 1.0


def test_multiclass_and_string_labels(backend):
    rng = np.random.default_rng(3)
    labels = np.array(["a", "b", "c"])
    y = rng.choice(labels, 120)
    p = rng.dirichlet(np.ones(3), 120)
    m = bootstrap_metrics(y, p, classes=labels, n_boot=300, backend=backend)
    assert set(m) == {"accuracy", "macro_f1", "auc", "brier", "ece"}
    assert all(r.lo <= r.hi for r in m.values())


def test_ece_perfectly_calibrated_vs_overconfident():
    rng = np.random.default_rng(0)
    n = 20000
    conf = rng.uniform(0.5, 1, n)
    p = np.column_stack([1 - conf, conf])             # always predicts class 1 with confidence `conf`
    y_cal = np.where(rng.random(n) < conf, 1, 0)
    assert ece_score(y_cal, p) < 0.02
    p_over = np.column_stack([np.full(n, 0.01), np.full(n, 0.99)])
    y_half = rng.integers(0, 2, n)
    assert ece_score(y_half, p_over) == pytest.approx(0.49, abs=0.02)


def test_ece_bootstrap_interval_covers_point(backend):
    y, p = synth(500, 0.7, 4)
    e = bootstrap_metrics(y, p, n_boot=800, backend=backend)["ece"]
    assert e.lo <= e.point <= e.hi


def test_input_validation():
    with pytest.raises(ValueError):
        prepare_inputs([0, 1], [[0.5, 0.5]])
    with pytest.raises(ValueError):
        prepare_inputs([0, 1], [[0.9, 0.9], [0.5, 0.5]])
    with pytest.raises(ValueError):
        prepare_inputs([0, 2], [[0.5, 0.5], [0.5, 0.5]], classes=[0, 1])


def test_permutation_p_value_math():
    assert permutation_p_value(1.0, np.zeros(99)) == pytest.approx(1 / 100)
    assert permutation_p_value(0.0, np.zeros(99)) == 1.0


def test_comparison_detects_real_difference(backend):
    y, good = synth(600, 0.9, 1)
    rng = np.random.default_rng(9)
    right = rng.random(600) < 0.7
    pred = np.where(right, y, 1 - y)
    conf = rng.uniform(0.6, 0.95, 600)
    p1 = np.where(pred == 1, conf, 1 - conf)
    bad = np.column_stack([1 - p1, p1])
    res = compare_models(y, good, bad, n_boot=600, n_perm=2000, backend=backend)
    assert res["accuracy"].delta > 0.1
    assert res["accuracy"].p_value < 0.01 and res["accuracy"].lo > 0


def test_comparison_null_is_not_significant(backend):
    """Two equally good models should rarely be flagged; p-values must not pile up near 0."""
    ps = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, 200)
        outs = []
        for _ in range(2):
            pred = np.where(rng.random(200) < 0.8, y, 1 - y)
            c = rng.uniform(0.6, 0.9, 200)
            p1 = np.where(pred == 1, c, 1 - c)
            outs.append(np.column_stack([1 - p1, p1]))
        ps.append(compare_models(y, outs[0], outs[1], metrics=("accuracy",), n_boot=100, n_perm=500,
                                 seed=seed, backend=backend)["accuracy"].p_value)
    ps = np.array(ps)
    assert (ps < 0.05).mean() <= 0.15
    assert 0.3 < ps.mean() < 0.7


def test_identical_models_have_zero_delta_and_p_one(backend):
    y, p = synth(150, 0.8, 5)
    res = compare_models(y, p, p.copy(), n_boot=200, n_perm=200, backend=backend)
    for c in res.values():
        assert c.delta == 0 and c.p_value == 1.0 and c.lo == 0 == c.hi


def test_multiclass_comparison(backend):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 150)
    a = rng.dirichlet(np.ones(3), 150)
    a[np.arange(150), y] += 1.0
    a /= a.sum(1, keepdims=True)
    b = rng.dirichlet(np.ones(3), 150)
    res = compare_models(y, a, b, n_boot=200, n_perm=500, backend=backend)
    assert res["auc"].delta > 0 and res["auc"].p_value < 0.05
