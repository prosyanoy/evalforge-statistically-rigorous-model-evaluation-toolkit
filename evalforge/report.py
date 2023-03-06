"""Self-contained HTML report (plots embedded as base64 PNG) with a plain-language summary."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from .evaluation import EvaluationResult  # noqa: E402

BLUE, ORANGE, GREY, RED = "#2a6fbb", "#d9822b", "#8a8f98", "#c0392b"
VERDICT_TEXT = {
    "SHIP": "The model clears every rule in the policy.",
    "ITERATE": "The model is promising but at least one quality rule failed; fix it before shipping.",
    "STOP": "At least one blocking rule failed; do not ship this model as is.",
}
LABELS = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "auc": "AUC (OvR)", "brier": "Brier score", "ece": "ECE"}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e3e6ea", lw=0.8)
    ax.set_axisbelow(True)


def plot_metrics(r: EvaluationResult) -> str:
    names = [m for m in ("accuracy", "macro_f1", "auc") if m in r.metrics]
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.8 * len(names)))
    for i, m in enumerate(names):
        c = r.metrics[m]
        off = -0.15 if r.baseline_metrics else 0
        ax.errorbar(c.point, i + off, xerr=[[c.point - c.lo], [c.hi - c.point]], fmt="o", color=BLUE, capsize=3,
                    label="candidate" if i == 0 else None)
        if r.baseline_metrics and m in r.baseline_metrics:
            b = r.baseline_metrics[m]
            ax.errorbar(b.point, i + 0.15, xerr=[[b.point - b.lo], [b.hi - b.point]], fmt="s", color=ORANGE,
                        capsize=3, label="baseline" if i == 0 else None)
    ax.set_yticks(range(len(names)), [LABELS[m] for m in names])
    ax.invert_yaxis()
    ax.set_xlabel(f"{int(r.config.confidence * 100)}% bootstrap CI")
    _style(ax)
    if r.baseline_metrics:
        ax.legend(frameon=False, loc="lower left")
    return _png(fig)


def plot_reliability(r: EvaluationResult) -> str:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    conf, acc, cnt = zip(*r.reliability) if r.reliability else ([], [], [])
    lo = min([0.5, *conf]) if conf else 0.0
    ax.plot([lo, 1], [lo, 1], color=GREY, ls="--", lw=1, label="perfect")
    ax.plot(conf, acc, "o-", color=BLUE, label="model")
    ax.set_ylabel("accuracy in bin")
    ax.legend(frameon=False)
    ax2.bar(conf, cnt, width=0.03, color=BLUE)
    ax2.set_ylabel("samples")
    ax2.set_xlabel("predicted confidence")
    for a in (ax, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    return _png(fig)


def plot_deltas(r: EvaluationResult) -> str:
    items = [(m, c) for m, c in (r.comparison or {}).items() if m != "brier"]
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.7 * len(items)))
    for i, (m, c) in enumerate(items):
        col = BLUE if c.p_value <= r.config.alpha else GREY
        ax.errorbar(c.delta, i, xerr=[[c.delta - c.lo], [c.hi - c.delta]], fmt="o", color=col, capsize=3)
        ax.annotate(f"p={c.p_value:.3g}", (c.hi, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    ax.axvline(0, color=RED, lw=1)
    ax.set_yticks(range(len(items)), [LABELS[m] for m, _ in items])
    ax.invert_yaxis()
    ax.set_xlabel("candidate - baseline (paired bootstrap CI; blue = permutation-significant)")
    _style(ax)
    return _png(fig)


def plot_slices(r: EvaluationResult) -> str:
    sl = r.slices.slices
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.5 * len(sl)))
    for i, s in enumerate(sl):
        ax.errorbar(s.gap, i, xerr=[[s.gap - s.gap_lo], [s.gap_hi - s.gap]], fmt="o", color=ORANGE, capsize=3)
    ax.axvline(0, color=GREY, lw=1)
    labels = [(s.description if len(s.description) < 58 else s.description[:55] + "...") for s in sl]
    ax.set_yticks(range(len(sl)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("error-rate gap vs. rest of data (95% Newcombe CI)")
    _style(ax)
    return _png(fig)


def plot_drift(r: EvaluationResult, top: int = 10) -> str:
    feats = r.drift.features[:top]
    fig, ax = plt.subplots(figsize=(6.4, 0.7 + 0.35 * len(feats)))
    ax.barh([f.feature for f in feats], [f.psi for f in feats], color=[RED if f.drifted else BLUE for f in feats])
    ax.axvline(0.10, color=GREY, ls=":", lw=1)
    ax.axvline(0.25, color=GREY, ls="--", lw=1)
    ax.invert_yaxis()
    ax.set_xlabel("PSI (dotted 0.10 = moderate, dashed 0.25 = major)")
    _style(ax)
    return _png(fig)


def summarize(r: EvaluationResult, candidate: str = "The model") -> list[str]:
    """Plain-language bullet points that justify the verdict."""
    pct = int(r.config.confidence * 100)
    out = []
    a = r.metrics.get("accuracy")
    if a:
        out.append(f"{candidate} is {a.point:.1%} accurate on {r.n_eval} held-out samples; with {pct}% confidence "
                   f"the true accuracy lies between {a.lo:.1%} and {a.hi:.1%}. "
                   f"{'The interval is wide - more evaluation data would sharpen it.' if a.width > 0.08 else ''}".strip())
    f1 = r.metrics.get("macro_f1")
    if f1:
        out.append(f"Macro-F1 is {f1.point:.3f} ({f1.lo:.3f}-{f1.hi:.3f}); AUC is "
                   f"{r.metrics['auc'].point:.3f}." if "auc" in r.metrics else f"Macro-F1 is {f1.point:.3f}.")
    e = r.metrics.get("ece")
    if e:
        kind = "well calibrated" if e.point <= 0.05 else ("moderately miscalibrated" if e.point <= 0.10 else "poorly calibrated")
        out.append(f"Expected calibration error is {e.point:.3f}, so the confidence scores are {kind}.")
    if r.comparison and "accuracy" in r.comparison:
        c = r.comparison["accuracy"]
        if c.p_value <= r.config.alpha:
            word = "better" if c.delta > 0 else "worse"
            out.append(f"Versus the baseline, accuracy is significantly {word} ({c.delta:+.1%}, permutation p={c.p_value:.3g}).")
        else:
            out.append(f"Versus the baseline, the accuracy difference ({c.delta:+.1%}) is not statistically "
                       f"significant (permutation p={c.p_value:.3g}).")
    if r.slices is not None:
        n = len(r.slices.slices)
        if n:
            s = r.slices.slices[0]
            out.append(f"{n} failure slice(s) are significantly worse than the rest after {r.slices.method.upper()} "
                       f"correction over {r.slices.n_tested} candidate slices. Worst: \"{s.description}\" "
                       f"({s.n} samples, error {s.error_rate:.0%} vs {s.complement_error_rate:.0%} elsewhere).")
        else:
            out.append(f"No subgroup is significantly worse than the rest (tested {r.slices.n_tested} candidate slices, "
                       f"{r.slices.method.upper()} correction at alpha={r.slices.alpha}).")
    if r.drift is not None:
        out.append(f"{r.drift.n_drifted} feature(s) show significant drift between the reference and evaluation data "
                   f"(max PSI {r.drift.max_psi:.2f}).")
    return out


def render_report(r: EvaluationResult, title: str | None = None, description: str = "",
                  candidate_name: str = "candidate", baseline_name: str | None = None) -> str:
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                      autoescape=select_autoescape(["html", "j2"]))
    plots = {"metrics": plot_metrics(r), "reliability": plot_reliability(r)}
    if r.comparison:
        plots["deltas"] = plot_deltas(r)
    if r.slices and r.slices.slices:
        plots["slices"] = plot_slices(r)
    if r.drift:
        plots["drift"] = plot_drift(r)
    return env.get_template("report.html.j2").render(
        r=r, title=title or r.name, description=description, plots=plots, labels=LABELS,
        summary=summarize(r), verdict_text=VERDICT_TEXT[r.decision.verdict], candidate_name=candidate_name,
        baseline_name=baseline_name)
