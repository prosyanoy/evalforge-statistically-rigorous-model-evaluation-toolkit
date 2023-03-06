import json

import pytest

from evalforge import EvalConfig, evaluate
from evalforge.cli import main
from evalforge.datasets import load_demo
from evalforge.report import render_report


@pytest.fixture(scope="module")
def wine_result():
    d = load_demo("wine")
    cfg = EvalConfig(n_boot=300, n_perm=300)
    r = evaluate(d.y_test, d.proba_candidate, d.X_test, classes=d.classes, baseline_proba=d.proba_baseline,
                 X_reference=d.X_train, config=cfg, name="wine")
    return d, r


def test_report_html_contents(wine_result):
    d, r = wine_result
    html = render_report(r, title="wine <demo>", description=d.description, candidate_name=d.candidate_name,
                         baseline_name=d.baseline_name)
    assert r.decision.verdict in html
    assert html.count("data:image/png;base64,") >= 3
    assert "Policy rules" in html and "Paired comparison" in html and "Feature drift" in html
    assert "wine &lt;demo&gt;" in html  # autoescaped


def test_demo_datasets_are_offline_and_sane():
    for name in ("breast_cancer", "wine", "digits"):
        d = load_demo(name)
        assert d.proba_candidate.shape == (len(d.y_test), len(d.classes))
        assert (d.proba_candidate.argmax(1) == d.y_test).mean() > 0.6
    with pytest.raises(ValueError):
        load_demo("nope")


def test_cli_report_writes_files(tmp_path, capsys):
    rc = main(["report", "--dataset", "wine", "--out", str(tmp_path), "--n-boot", "200", "--n-perm", "200"])
    assert rc == 0
    assert (tmp_path / "wine.html").read_text().startswith("<!doctype html>")
    data = json.loads((tmp_path / "wine.json").read_text())
    assert data["decision"]["verdict"] in {"SHIP", "ITERATE", "STOP"}
    assert "wine" in capsys.readouterr().out


def test_cli_errors(tmp_path, capsys):
    assert main(["report", "--policy", str(tmp_path / "missing.yaml"), "--out", str(tmp_path)]) == 2
    assert "error" in capsys.readouterr().err
    assert main(["info"]) == 0


def test_backend_env_override():
    import os
    import subprocess
    import sys

    env = {**os.environ, "EVALFORGE_BACKEND": "numpy"}
    out = subprocess.run([sys.executable, "-m", "evalforge", "info"], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and "active backend: numpy" in out.stdout
    bad = subprocess.run([sys.executable, "-m", "evalforge", "info"], env={**os.environ, "EVALFORGE_BACKEND": "x"},
                         capture_output=True, text=True)
    assert bad.returncode != 0


def test_ab_example_runs(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "examples" / "ab_comparison.py"
    out = subprocess.run([sys.executable, str(script), "--report", str(tmp_path / "r.html")],
                         capture_output=True, text=True, check=True).stdout
    assert "Plain accuracy:" in out and "Failure slice:" in out and "Decision:" in out
    assert (tmp_path / "r.html").read_text().startswith("<!doctype html>")
