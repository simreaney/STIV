"""Tests for report.pdf: written on request, complete, and cleaned up like
everything else a run writes."""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import pytest
import yaml

from stiv import Config, cli, run
from stiv.pdf_report import write_pdf_report
from stiv.report import MANIFEST, add_pdf_report, write_outputs
from stiv.synthetic import SyntheticRiver, make_config, render


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("pdf")
    river = SyntheticRiver(width_px=480, height_px=300, duration_s=3.0,
                           bank_top_px=70, bank_bottom_px=230, v_max_ms=1.0)
    video = render(river, d / "river", progress=False)
    raw = make_config(river, video, d / "out", n_stations=7)
    return raw, run(Config.from_dict(raw), progress=False)


def _cfg(raw, out_dir, **output):
    raw = copy.deepcopy(raw)
    raw["output"] = {"out_dir": str(out_dir), **output}
    return Config.from_dict(raw)


def _pages(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


def _listed(out: Path) -> list[str]:
    return json.loads((out / MANIFEST).read_text())["files"]


def test_off_unless_asked_for(base, tmp_path):
    raw, result = base
    assert Config().output.pdf_report is False
    write_outputs(result, _cfg(raw, tmp_path / "out"))
    assert not (tmp_path / "out" / "report.pdf").exists()


def test_written_with_the_run_and_cleared_by_the_next(base, tmp_path):
    raw, result = base
    out = tmp_path / "out"
    write_outputs(result, _cfg(raw, out, pdf_report=True))
    pdf = out / "report.pdf"
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF")
    assert b"STIV report - synthetic" in data            # the document title
    # Summary, 4 figures, 3 station details, the table and the config.
    assert _pages(pdf) >= 10
    assert "report.pdf" in _listed(out)

    write_outputs(result, _cfg(raw, out))                  # a re-run without it
    assert not pdf.exists()


def test_with_discharge_and_without_any_velocity(base, tmp_path):
    raw, result = base
    res = copy.deepcopy(result)
    for st in res.stations:
        st.velocity_ms = math.nan
        st.flags = ["low_coherency"]
    path = write_pdf_report(res, result.config, tmp_path / "r.pdf")
    assert path.read_bytes().startswith(b"%PDF")


def test_a_long_report_runs_onto_more_pages(base, tmp_path):
    raw, result = base
    res = copy.deepcopy(result)
    res.warnings = [f"warning number {i}: " + "words " * 60 for i in range(40)]
    short = write_pdf_report(result, result.config, tmp_path / "short.pdf")
    long = write_pdf_report(res, result.config, tmp_path / "long.pdf")
    assert _pages(long) > _pages(short)


def test_added_after_the_run_joins_the_manifest(base, tmp_path):
    raw, result = base
    cfg = _cfg(raw, tmp_path / "out")
    write_outputs(result, cfg)
    before = set(_listed(tmp_path / "out"))
    path = add_pdf_report(result, cfg)
    assert path == tmp_path / "out" / "report.pdf" and path.is_file()
    assert set(_listed(tmp_path / "out")) == before | {"report.pdf"}


def test_cli_run_pdf(base, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "agg")
    raw, _ = base
    survey = tmp_path / "survey.yaml"
    survey.write_text(yaml.safe_dump(_cfg(raw, tmp_path / "out").to_dict()), encoding="utf-8")
    assert cli.main(["run", str(survey), "-q", "--pdf"]) == 0
    assert (tmp_path / "out" / "report.pdf").is_file()
    assert yaml.safe_load((tmp_path / "out" / "config_used.yaml").read_text())["output"]["pdf_report"]


def test_wizard_writes_it_by_default(tmp_path, base):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("ipyfilechooser")
    import ipywidgets as W

    from stiv import notebook_ui as nb

    raw, _ = base
    wz = nb.Wizard(out_dir=str(tmp_path / "out"))
    wz._load_video(raw["video"]["path"], is_demo=False, status=W.HTML())
    s = wz.state
    s.m_per_px = 0.02
    s.section_pts = [tuple(raw["section"]["start_px"]), tuple(raw["section"]["end_px"])]
    s.downstream_pts = [tuple(raw["flow"]["downstream_px"])]
    assert wz._build_config().output.pdf_report is True

    def find(widget, text):
        if text in str(getattr(widget, "description", "")):
            return widget
        for child in getattr(widget, "children", ()):
            hit = find(child, text)
            if hit is not None:
                return hit

    box = find(wz._build_step3(), "PDF report")
    box.value = False
    assert wz._build_config().output.pdf_report is False
