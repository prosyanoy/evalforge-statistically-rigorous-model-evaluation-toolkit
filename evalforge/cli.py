"""Command line interface: ``evalforge report`` and ``evalforge info``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, cpp_available
from .datasets import DATASETS, load_demo
from .evaluation import EvalConfig, evaluate
from .policy import PolicyError, load_policy
from .report import render_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalforge", description="Statistically rigorous model evaluation.")
    p.add_argument("--version", action="version", version=f"evalforge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="evaluate bundled demo datasets and write HTML reports")
    r.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    r.add_argument("--out", default="reports", help="output directory (default: reports)")
    r.add_argument("--policy", default=None, help="YAML policy file (default: bundled default.yaml)")
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--n-perm", type=int, default=10000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--backend", choices=["auto", "cpp", "numpy"], default=None)
    r.add_argument("--threads", type=int, default=0, help="C++ threads (0 = all cores)")
    sub.add_parser("info", help="show which kernel backend is active")
    return p


def run_report(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.policy)
    cfg = EvalConfig(n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed, backend=args.backend,
                     n_threads=args.threads)
    names = DATASETS if args.dataset == "all" else (args.dataset,)
    index = []
    for name in names:
        d = load_demo(name)
        res = evaluate(d.y_test, d.proba_candidate, d.X_test, classes=d.classes, baseline_proba=d.proba_baseline,
                       X_reference=d.X_train, policy=policy, config=cfg, name=name)
        html = render_report(res, title=f"{name} - ship readiness", description=d.description,
                             candidate_name=d.candidate_name, baseline_name=d.baseline_name)
        (out / f"{name}.html").write_text(html, encoding="utf-8")
        (out / f"{name}.json").write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
        acc = res.metrics["accuracy"]
        print(f"{name:14s} {res.decision.verdict:8s} accuracy {acc.point:.3f} [{acc.lo:.3f}, {acc.hi:.3f}] "
              f"backend={res.backend} -> {out / (name + '.html')}")
        index.append((name, res.decision.verdict, d.description))
    rows = "".join(f'<li><a href="{n}.html">{n}</a> - <b>{v}</b> - {desc}</li>' for n, v, desc in index)
    (out / "index.html").write_text(f"<!doctype html><meta charset=utf-8><title>EvalForge reports</title>"
                                    f"<h1>EvalForge reports</h1><ul>{rows}</ul>", encoding="utf-8")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "info":
            from .kernels import get_backend
            print(f"evalforge {__version__}; C++ extension built: {cpp_available()}; active backend: {get_backend().name}")
            return 0
        return run_report(args)
    except (PolicyError, ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
