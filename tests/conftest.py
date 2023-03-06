import numpy as np
import pytest

from evalforge import cpp_available, get_backend

BACKENDS = ["numpy"] + (["cpp"] if cpp_available() else [])


@pytest.fixture(params=BACKENDS)
def backend(request):
    return get_backend(request.param)


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 2, n).astype(np.int32)
    score = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.3, n), 0.001, 0.999)
    return y, score
