import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mulehunt import pipeline
from mulehunt.features import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS
from mulehunt.report import write_markdown_report

REPO = Path(__file__).resolve().parents[2]
REPORTS = ("a_01_simulator.md", "a_05_classifiers.md", "a_06_clustering.md")


def linked_figures(md_path: Path) -> list[Path]:
    text = md_path.read_text(encoding="utf-8")
    return [md_path.parent / link for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)]


def test_markdown_report_renders_table_and_figure(tmp_path):
    fig = tmp_path / "figs" / "f.png"
    fig.parent.mkdir()
    fig.write_bytes(b"png")
    out = tmp_path / "r.md"
    write_markdown_report(out, "Title", [
        ("Intro", "Some prose."),
        ("Table", pd.DataFrame({"a": [1, 2], "b": [0.5, 0.25]})),
        ("Figure", fig),
        ("Mixed", ["text first", pd.DataFrame({"x": ["p"]})]),
    ])
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Title\n")
    assert "## Intro\n\nSome prose." in text
    assert "| a | b |\n|---|---|\n| 1 | 0.5 |" in text
    assert "![f](figs/f.png)" in text
    assert linked_figures(out) == [fig]
    assert "text first" in text and "| x |" in text


def test_markdown_report_rejects_missing_figure(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_markdown_report(tmp_path / "r.md", "T", [("F", tmp_path / "missing.png")])


def test_every_feature_has_a_description():
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_COLUMNS)
    assert all(len(d) > 10 for d in FEATURE_DESCRIPTIONS.values())


def test_pipeline_stage_subset(monkeypatch):
    calls = []
    for name in pipeline.STAGE_ORDER:
        monkeypatch.setitem(pipeline.STAGES, name, lambda cfg, small, n=name: calls.append((n, small)))
    times = pipeline.run_pipeline(stages=["validate", "generate"], small=True)
    assert calls == [("generate", True), ("validate", True)]       # always in pipeline order
    assert list(times) == ["generate", "validate"]
    assert all(t >= 0 for t in times.values())


def test_pipeline_unknown_stage():
    with pytest.raises(ValueError):
        pipeline.run_pipeline(stages=["generate", "deploy"])


@pytest.mark.slow
def test_pipeline_small_end_to_end(tmp_path, monkeypatch):
    # Inputs the stages read, copied into an empty folder; every output lands there too.
    for rel in ("simulator/config.yaml", "data/raw"):
        src, dst = REPO / rel, tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy)(src, dst)
    shutil.copytree(REPO / "data/generated/toy", tmp_path / "data/generated/toy")
    monkeypatch.chdir(tmp_path)
    times = pipeline.run_pipeline(small=True)
    assert list(times) == list(pipeline.STAGE_ORDER)
    for name in REPORTS:
        report = tmp_path / "reports" / name
        assert report.exists(), name
        for fig in linked_figures(report):
            assert fig.exists(), f"{name} links a missing figure: {fig}"
    assert (tmp_path / "models/risk_scores.csv").exists()
    assert list((tmp_path / "models").glob("*.joblib"))
