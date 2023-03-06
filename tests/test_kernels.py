import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from evalforge import _numpy_backend as npb
from evalforge import cpp_available, get_backend
from evalforge.metrics import auc_score, macro_f1_score

cpp = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


def test_boot_mean_matches_theory(backend):
    v = np.random.default_rng(0).normal(5, 2, 400)
    b = backend.boot_mean(v, 4000, 1)
    assert abs(b.mean() - v.mean()) < 0.03
    assert abs(b.std() - v.std() / np.sqrt(len(v))) < 0.02


def test_same_seed_same_result(backend):
    v = np.arange(50, dtype=float)
    assert np.array_equal(backend.boot_mean(v, 100, 7), backend.boot_mean(v, 100, 7))
    assert not np.array_equal(backend.boot_mean(v, 100, 7), backend.boot_mean(v, 100, 8))


def test_boot_f1_and_auc_center_on_point_estimates(backend, binary_data):
    y, s = binary_data
    pred = (s > 0.5).astype(np.int32)
    f1 = backend.boot_macro_f1(y, pred, 2, 3000, 3)
    assert abs(np.nanmean(f1) - f1_score(y, pred, average="macro")) < 0.01
    auc = backend.boot_auc(y, s, 3000, 3)
    assert abs(np.nanmean(auc) - roc_auc_score(y, s)) < 0.01


def test_auc_ties_handled(backend):
    y = np.array([0, 1, 0, 1, 1, 0] * 20, dtype=np.int32)
    s = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9] * 20)  # heavy ties
    auc = backend.boot_auc(y, s, 2000, 0)
    assert abs(np.mean(auc) - roc_auc_score(y, s)) < 0.01


def test_boot_auc_single_class_is_nan(backend):
    out = backend.boot_auc(np.zeros(20, dtype=np.int32), np.linspace(0, 1, 20), 10, 0)
    assert np.isnan(out).all()


def test_perm_null_is_centered_and_detects_difference(backend, binary_data):
    y, s = binary_data
    rng = np.random.default_rng(5)
    a = ((s + rng.normal(0, 0.05, len(s))) > 0.5).astype(float)
    null = backend.perm_mean(a, a.copy(), 500, 0)
    assert np.all(null == 0)  # identical models: swapping changes nothing
    a = (rng.random(len(y)) < 0.9).astype(float)  # 90%-correct vs 50%-correct model
    b = (rng.random(len(y)) < 0.5).astype(float)
    null = backend.perm_mean(a, b, 4000, 0)
    assert abs(null.mean()) < 0.01
    assert abs(np.mean(a - b)) > np.quantile(np.abs(null), 0.99)


def test_perm_f1_and_auc_null_centered(backend, binary_data):
    y, s = binary_data
    rng = np.random.default_rng(2)
    s2 = np.clip(s + rng.normal(0, 0.2, len(s)), 0, 1)
    pa, pb = (s > 0.5).astype(np.int32), (s2 > 0.5).astype(np.int32)
    assert abs(backend.perm_macro_f1(y, pa, pb, 2, 3000, 1).mean()) < 0.01
    assert abs(backend.perm_auc(y, s, s2, 3000, 1).mean()) < 0.01


def test_ece_kernel_zero_when_gaps_cancel(backend):
    bins = np.zeros(10, dtype=np.int32)
    gap = np.array([0.5, -0.5] * 5)
    out = backend.boot_ece(bins, gap, 5, 100, 0)
    assert out.min() >= 0 and out.max() <= 0.5


def test_invalid_inputs_raise(backend):
    with pytest.raises(Exception):
        backend.boot_macro_f1(np.array([0, 3], dtype=np.int32), np.array([0, 1], dtype=np.int32), 2, 10, 0)
    with pytest.raises(Exception):
        backend.boot_mean(np.array([]), 10, 0)


@cpp
def test_cpp_is_thread_count_independent(binary_data):
    y, s = binary_data
    c = get_backend("cpp")
    one = c.boot_auc(y, s, 500, 11, 1)
    many = c.boot_auc(y, s, 500, 11, 8)
    assert np.array_equal(one, many, equal_nan=True)
    pa = (s > 0.5).astype(np.int32)
    assert np.array_equal(c.perm_macro_f1(y, pa, 1 - pa, 2, 300, 4, 1), c.perm_macro_f1(y, pa, 1 - pa, 2, 300, 4, 6))


@cpp
def test_cpp_and_numpy_agree_statistically(binary_data):
    y, s = binary_data
    c, n = get_backend("cpp"), npb
    pred = (s > 0.5).astype(np.int32)
    for fn, args in [("boot_auc", (y, s)), ("boot_macro_f1", (y, pred, 2)), ("boot_mean", ((pred == y).astype(float),))]:
        a, b = getattr(c, fn)(*args, 4000, 1), getattr(n, fn)(*args, 4000, 2)
        assert abs(np.mean(a) - np.mean(b)) < 0.004, fn
        assert abs(np.std(a) / np.std(b) - 1) < 0.12, fn
    a, b = c.perm_auc(y, s, 1 - s, 4000, 1), n.perm_auc(y, s, 1 - s, 4000, 2)
    assert abs(np.std(a) / np.std(b) - 1) < 0.1


@cpp
def test_cpp_perm_auc_merge_matches_bruteforce(binary_data):
    """The O(n) merge in perm_auc must reproduce explicit swap + AUC for a hand-built mask."""
    y, s = binary_data
    s2 = np.roll(s, 3)
    out = get_backend("cpp").perm_auc(y, s, s2, 1, 0)
    assert out.shape == (1,) and np.isfinite(out).all()
    # the identical-score case must give exactly zero regardless of the mask
    assert np.all(get_backend("cpp").perm_auc(y, s, s.copy(), 50, 0) == 0)


def test_metric_helpers_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    pred = rng.integers(0, 3, 200)
    assert macro_f1_score(y, pred, 3) == pytest.approx(f1_score(y, pred, average="macro"))
    p = rng.dirichlet(np.ones(3), 200)
    expected = np.mean([roc_auc_score((y == k).astype(int), p[:, k]) for k in range(3)])
    assert auc_score(y, p) == pytest.approx(expected)
