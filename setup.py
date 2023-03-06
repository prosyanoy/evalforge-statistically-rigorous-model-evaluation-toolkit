"""Builds the optional C++17 kernel. If compilation fails the pure NumPy
fallback in ``evalforge/_numpy_backend.py`` is used transparently."""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "evalforge._core",
    ["cpp/core.cpp"],
    cxx_std=17,
    extra_compile_args=["-O3"],
)
ext.optional = True  # do not abort installation when no compiler is present

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
