"""Benchmark: C++ multithreaded kernels vs. the pure NumPy fallback.

    python benchmarks/bench.py [--n 5000] [--reps 2000] [--repeat 3]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from evalforge import cpp_available, get_backend


def best_of(fn, repeat):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="evaluation-set size")
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap / permutation replicates")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    n, B = args.n, args.reps
    y = rng.integers(0, 2, n).astype(np.int32)
    s_a = np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.3, n), 0, 1)
    s_b = np.clip(0.5 + (y - 0.5) * 0.3 + rng.normal(0, 0.3, n), 0, 1)
    pa, pb = (s_a > 0.5).astype(np.int32), (s_b > 0.5).astype(np.int32)
    correct = (pa == y).astype(float)

    jobs = {
        "bootstrap accuracy": lambda k: k.boot_mean(correct, B, 0),
        "bootstrap macro-F1": lambda k: k.boot_macro_f1(y, pa, 2, B, 0),
        "bootstrap AUC": lambda k: k.boot_auc(y, s_a, B, 0),
        "permutation accuracy": lambda k: k.perm_mean(correct, (pb == y).astype(float), B, 0),
        "permutation macro-F1": lambda k: k.perm_macro_f1(y, pa, pb, 2, B, 0),
        "permutation AUC": lambda k: k.perm_auc(y, s_a, s_b, B, 0),
    }
    numpy_backend = get_backend("numpy")
    cpp = get_backend("cpp") if cpp_available() else None
    print(f"n={n} samples, {B} replicates, best of {args.repeat}")
    print(f"{'task':24s} {'numpy (s)':>10s} {'C++ (s)':>10s} {'speedup':>8s}")
    for name, job in jobs.items():
        t_np = best_of(lambda: job(numpy_backend), args.repeat)
        if cpp is None:
            print(f"{name:24s} {t_np:10.3f} {'n/a':>10s} {'n/a':>8s}")
            continue
        t_cpp = best_of(lambda: job(cpp), args.repeat)
        print(f"{name:24s} {t_np:10.3f} {t_cpp:10.3f} {t_np / t_cpp:7.1f}x")


if __name__ == "__main__":
    main()
