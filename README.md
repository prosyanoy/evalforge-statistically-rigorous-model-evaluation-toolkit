# EvalForge: Statistically Rigorous Model Evaluation Toolkit

> Tells you whether a model is truly ready to ship, using confidence intervals, automatic failure-slice
> discovery, calibration and drift checks instead of a single accuracy number.

## The 30-second example

**Model A looks 2% better, but EvalForge shows the gain is not significant and the model fails on slice X.**

```text
$ python examples/ab_comparison.py          # logistic regression (A) vs random forest (B), breast cancer
Plain accuracy:  A 96.5%  vs  B 94.3%   -> A looks +2.2% better
EvalForge:       A accuracy 95% CI [93.9%, 98.7%];  delta CI [-1.3%, +5.7%], permutation p = 0.30  -> not significant
Failure slice:   mean area in (561.7, 791.1] AND worst concave points in (0.1017, 0.1679]
                 n=20, error 25% vs 1% elsewhere (BH-adjusted p = 0.036, 1389 slices tested)
Decision:        ITERATE  (no-significant-failure-slices)
```

The headline accuracy gap is well inside the noise (its CI spans zero), while a hidden 20-sample subgroup has a
~25x higher error rate that survives correction over 1,389 candidate slices. A single accuracy number shows neither.
The output above is real, seeded output of [`examples/ab_comparison.py`](examples/ab_comparison.py).

![60-second terminal demo](docs/demo.gif)

*Terminal demo (about 60 s): `evalforge info`, the A/B example, the three-dataset report run and the benchmark.
Regenerate with `python docs/make_demo_gif.py`; it executes the commands and replays their captured output.*

**Sample report:** [`docs/sample_report.html`](docs/sample_report.html) is the full self-contained HTML report for the
example above (download and open it in a browser; GitHub shows HTML files as source).
Run `evalforge report` to produce reports for all three bundled datasets.

## Benchmark: C++ vs NumPy

2,000 replicates on 5,000 evaluation samples, best of 3, 2 CPU cores (`python benchmarks/bench.py`):

| Task | NumPy (s) | C++ (s) | Speedup |
|---|---:|---:|---:|
| Bootstrap accuracy | 0.114 | 0.009 | 12.2x |
| Bootstrap macro-F1 | 0.228 | 0.036 | 6.4x |
| Bootstrap AUC | 0.606 | 0.025 | 24.7x |
| Permutation test, accuracy | 0.139 | 0.030 | 4.6x |
| Permutation test, macro-F1 | 0.265 | 0.074 | 3.6x |
| Permutation test, AUC | 2.068 | 0.237 | 8.7x |

Geometric-mean speedup is about **8x** (3.6x to 24.7x). Timings vary a few tens of percent between runs and scale with
core count (the kernel is multithreaded; this machine only had 2), so rerun the script to get numbers for your hardware.
The NumPy fallback is already vectorised, so this is a comparison against a reasonable baseline, not a naive loop.

## Resume line

> Built an evaluation framework with C++-accelerated bootstrap testing and automated failure-slice discovery; cut
> evaluation time ~8-fold (geometric mean over six resampling tasks vs. vectorised NumPy) and surfaced hidden subgroup
> regressions on 3 public datasets.

On the bundled datasets the default policy flags a significant failure slice on `breast_cancer` and `digits`
(none on `wine`), and gives three different verdicts (ITERATE / SHIP / STOP).

## Features

- **Metric suite with bootstrap CIs**: accuracy, macro-F1, AUC (binary or one-vs-rest), Brier score and
  expected calibration error (ECE), all with percentile bootstrap intervals computed on shared resamples.
- **Paired model comparison**: candidate vs. baseline with a paired-bootstrap CI of the difference and a
  paired permutation test (outputs of the two models are swapped per sample under H0).
- **C++17 / pybind11 kernels**: multithreaded bootstrap and permutation core with a pure NumPy fallback that has
  the same API, plus `benchmarks/bench.py` comparing the two.
- **Automatic slice discovery**: searches feature-defined subgroups (single predicates and pairs) where the
  model is significantly worse, with Benjamini-Hochberg (default), Holm or Bonferroni correction over *all*
  tested slices.
- **Drift check**: per-feature PSI and KS test between training and evaluation data, BH-corrected.
- **Ship / iterate / stop policy**: configurable YAML rules (`policies/default.yaml`, `strict.yaml`).
- **One-command HTML report** with plots and a plain-language summary, run on three bundled datasets
  (breast cancer, wine, digits; all shipped with scikit-learn, no downloads).

## Architecture

```
cpp/core.cpp               C++17 + pybind11 kernels (bootstrap / permutation), std::thread, GIL released
evalforge/
  _numpy_backend.py        pure-NumPy implementation of the same kernel API
  kernels.py               backend dispatch (C++ if built, EVALFORGE_BACKEND override)
  metrics.py               point metrics, bootstrap CIs, ECE, reliability curve
  compare.py               paired bootstrap + permutation test
  slices.py  multitest.py  failure-slice discovery, p-value adjustment
  drift.py                 PSI + KS drift
  policy.py  policies/     YAML policy parsing, validation and decision rule
  evaluation.py            evaluate(): ties everything together -> EvaluationResult
  datasets.py              bundled demo datasets + candidate/baseline models
  report.py  templates/    Jinja2 + matplotlib HTML report
  cli.py                   `evalforge report|info`
tests/  benchmarks/  examples/  .github/workflows/ci.yml
docs/                      demo.gif, sample_report.html, make_demo_gif.py
```

Data flow: `y_true, proba (, baseline_proba, X, X_reference)` -> metrics/CIs -> comparison -> slices -> drift
-> policy -> `EvaluationResult` -> HTML/JSON.

## Setup

Requires Python >= 3.10 (developed on 3.14, CI on 3.11 and 3.12) and, optionally, a C++17 compiler for the fast kernels.

```bash
make setup            # or: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/evalforge info   # shows whether the C++ extension was built
```

If no compiler is available the build step is skipped and everything runs on the NumPy fallback.
No secrets or services are needed; `.env.example` documents the one optional variable (`EVALFORGE_BACKEND`).

## Run

```bash
.venv/bin/evalforge report --out reports     # all 3 datasets -> reports/*.html + *.json (about 10 s)
.venv/bin/evalforge report --dataset digits --policy evalforge/policies/strict.yaml --backend numpy
.venv/bin/python benchmarks/bench.py         # C++ vs NumPy
.venv/bin/python examples/ab_comparison.py --report docs/sample_report.html   # the example above
```

Bundled demo outcomes with the default policy (seeded, so reproducible): `breast_cancer` -> **ITERATE**
(logistic regression is excellent but 7 significant failure slices are found), `wine` -> **SHIP**, `digits` -> **STOP**
(a deliberately weak shallow forest: low CI lower bound, miscalibrated, large failure slices).

Use it on your own model:

```python
from evalforge import evaluate, EvalConfig
from evalforge.report import render_report

res = evaluate(y_test, model.predict_proba(X_test), X_test, baseline_proba=old.predict_proba(X_test),
               X_reference=X_train, policy="my_policy.yaml", config=EvalConfig(n_boot=5000))
print(res.decision.verdict)
open("report.html", "w").write(render_report(res))
```

Policy rule format (`stat` is `point`, `ci_lower`, `ci_upper` or `p_value`; `on_fail` is `stop` or `iterate`):

```yaml
rules:
  - {id: accuracy-lower-bound, metric: accuracy, stat: ci_lower, op: ">=", threshold: 0.90, on_fail: iterate}
  - {id: no-regression, metric: accuracy_delta, stat: ci_upper, op: ">=", threshold: 0.0, on_fail: stop}
```

Metrics: `accuracy macro_f1 auc brier ece`, `<metric>_delta` (vs baseline), `significant_slices`,
`worst_slice_gap`, `max_psi`, `drifted_features`, `n_eval`. Verdict = most severe failed rule
(STOP > ITERATE > SHIP); rules whose inputs are missing are shown as *skipped*, never silently passed.

## Test

```bash
make test                                  # 65 tests
EVALFORGE_BACKEND=numpy .venv/bin/python -m pytest -q   # force the fallback
```

Tests cover kernel correctness against scikit-learn, thread-count determinism, C++/NumPy statistical agreement,
bootstrap CI coverage, permutation-test calibration under H0, ECE, planted-slice recovery and false-positive
control, multiple-testing formulas, drift detection, policy validation and the report/CLI.

## Key design decisions

- **Percentile bootstrap, not BCa.** Simple, shared across all metrics, and fast in the kernel. For tiny samples it can be
  slightly optimistic; the empirical coverage test guards against gross errors.
- **Deterministic, thread-independent C++ RNG.** Replicate *b* draws only from `(seed, b)` (splitmix64), so results do not
  depend on the thread count, and two models bootstrapped with the same seed see identical resamples, which makes the
  paired delta CI valid. Index streams depend only on `(seed, b, n)`; multi-class AUC reuses the seed per class.
  C++ and NumPy use different RNGs, so they agree statistically, not bit-for-bit.
- **O(n) per-replicate AUC.** Scores are sorted once; each bootstrap replicate only needs resample counts, and each
  permutation merges two pre-sorted orders instead of re-sorting.
- **Paired permutation test** swaps the two models' outputs per sample (exchangeability under H0), valid for non-decomposable
  metrics like F1 and AUC; two-sided p-values use the `(1 + extreme)/(1 + B)` correction so they are never zero.
- **Slice discovery tests a fixed, pre-declared family** (quantile bins / levels, depth <= 2, min support 20), with a one-sided
  Fisher exact test vectorised through the hypergeometric tail, and corrects over every candidate tested. The effect-size
  filter and Jaccard de-duplication are applied after correction, so they do not shrink the family. Slices are
  exploratory leads, not causal claims.
- **Decisions use CI bounds**, so more evaluation data can turn ITERATE into SHIP; that is the intended incentive.
- **Macro-F1 convention**: classes absent from both truth and prediction score 0 (sklearn `zero_division=0`).
- **Demo data is offline**: only datasets bundled in scikit-learn, so no network or OpenML access is needed. Splits are
  random, so the demo drift check is expected to find nothing; its detection ability is covered by unit tests.
- **Python version**: the spec names 3.11; the code is compatible with >= 3.10 and CI runs 3.11/3.12.
- Generated `reports/` and compiled extensions are git-ignored.
