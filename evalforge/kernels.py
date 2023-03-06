"""Backend dispatch for the resampling kernels.

``get_backend()`` returns the C++ module when it was built, otherwise the NumPy
fallback. Set ``EVALFORGE_BACKEND=numpy`` (or ``cpp``) to force a choice.
"""
from __future__ import annotations

import os

from . import _numpy_backend

try:  # pragma: no cover - exercised implicitly depending on the build
    from . import _core as _cpp
    _cpp.name = "cpp"
except ImportError:  # extension not compiled
    _cpp = None


def cpp_available() -> bool:
    return _cpp is not None


def get_backend(name: str | None = None):
    """Return a kernel module. ``name`` is 'auto' (default), 'cpp' or 'numpy'."""
    name = (name or os.environ.get("EVALFORGE_BACKEND", "auto")).lower()
    if name == "numpy":
        return _numpy_backend
    if name == "cpp":
        if _cpp is None:
            raise RuntimeError("C++ backend requested but evalforge._core is not built (pip install -e .)")
        return _cpp
    if name == "auto":
        return _cpp if _cpp is not None else _numpy_backend
    raise ValueError(f"unknown backend {name!r}; expected auto, cpp or numpy")
