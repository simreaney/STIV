"""Tests for the on-disk report: tables, summary.json and the figures.

One small synthetic run is analysed once and then doctored per test, so the
cases that are awkward to provoke from footage -- a station that failed and was
*not* interpolated, a profile with no usable velocity at all -- can be checked
directly.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stiv import Config, plotting, run
from stiv.report import MANIFEST, clear_outputs, write_outputs
from stiv.synthetic import SyntheticRiver, make_config, render

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("report")
    river = SyntheticRiver(width_px=480, height_px=300, duration_s=3.0,
                           bank_top_px=70, bank_bottom_px=230, v_max_ms=1.0)
    video = render(river, d / "river", progress=False)
    raw = make_config(river, video, d / "out", n_stations=7)
    result = run(Config.from_dict(raw), progress=False)
    return raw, result


def _cfg(raw, out_dir, **overrides):
    raw = copy.deepcopy(raw)
    raw["output"]["out_dir"] = str(out_dir)
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)
    return Config.from_dict(raw)


def _clean(result):
    """A copy of ``result`` in which every station passed quality control.

    The bank stations of the base run fail (their search lines straddle the
    bank), which would muddle tests about one particular failure.
    """
    res = copy.deepcopy(result)
    for st in res.stations:
        st.flags = []
        st.velocity_raw_ms = st.velocity_ms
    return res


def _fail(result, i, interpolated, raw_value=0.5):
    """Make station ``i`` a QC failure, as the analysis reports one."""
    st = result.stations[i]
    st.flags = ["low_coherency"] + (["interpolated"] if interpolated else [])
    st.velocity_raw_ms = raw_value
    if not interpolated:
        st.velocity_ms = math.nan
    return st


def _strict_json(path):
    def _reject(token):
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(Path(path).read_text(), parse_constant=_reject)


# --------------------------------------------------------------------------- #
class TestRerunClearsOldOutputs:
    def test_fewer_stations_leave_no_stale_stis(self, base, tmp_path):
        raw, result = base
        out = tmp_path / "out"
        write_outputs(result, _cfg(raw, out, output={"save_fields": True}))
        assert len(list((out / "sti").glob("*.png"))) == 7
        assert (out / "orientation_fields.npz").is_file()

        five = copy.deepcopy(result)
        five.stations, five.stis = five.stations[:5], five.stis[:5]
        five.fields, five.estimates = five.fields[:5], five.estimates[:5]
        write_outputs(five, _cfg(raw, out))
        assert len(list((out / "sti").glob("*.png"))) == 5
        assert not (out / "orientation_fields.npz").exists()

    def test_a_folder_without_a_manifest_loses_nothing(self, base, tmp_path):
        """``stiv run --no-figures --out-dir <survey folder>`` used to delete the
        survey's own overview.png and discharge.png, which it never rewrites."""
        raw, result = base
        out = tmp_path / "survey"
        out.mkdir()
        mine = {name: out / name for name in ("overview.png", "discharge.png", "notes.txt")}
        for f in mine.values():
            f.write_text("mine")
        (out / "sti").mkdir()
        (out / "sti" / "sti_000_0000.00m.png").write_text("mine")

        write_outputs(result, _cfg(raw, out, output={"save_diagnostics": False,
                                                     "save_sti_images": False}))
        assert all(f.read_text() == "mine" for f in mine.values())
        assert (out / "sti" / "sti_000_0000.00m.png").read_text() == "mine"
        assert (out / "velocity.csv").is_file() and (out / MANIFEST).is_file()

    def test_only_what_the_last_run_listed_is_removed(self, base, tmp_path):
        raw, result = base
        out = tmp_path / "out"
        write_outputs(result, _cfg(raw, out))          # figures, STIs, manifest
        listed = json.loads((out / MANIFEST).read_text())["files"]
        assert "overview.png" in listed and any(f.startswith("sti/") for f in listed)

        # Somebody's own files, added afterwards, beside and among the outputs.
        keep = [out / "flight.mp4", out / "sti" / "my_sketch.png", out / "velocity_old.csv"]
        for f in keep:
            f.write_text("mine")
        removed = clear_outputs(out)
        assert {p.relative_to(out).as_posix() for p in removed} == set(listed)
        assert all(f.read_text() == "mine" for f in keep)
        assert (out / "sti").is_dir()                  # still holds my_sketch.png
        assert not (out / "sti_detail").exists()       # emptied, so removed
        assert not (out / MANIFEST).exists()

    def test_a_tampered_manifest_cannot_reach_other_files(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        outside = tmp_path / "precious.csv"
        outside.write_text("mine")
        (out / "notes.txt").write_text("mine")
        (out / MANIFEST).write_text(json.dumps(
            {"files": ["../precious.csv", str(outside), "notes.txt", "sti/../notes.txt"]}))
        assert clear_outputs(out) == []
        assert outside.read_text() == "mine" and (out / "notes.txt").read_text() == "mine"


class TestSummaryJson:
    def test_statistics_are_over_the_reported_profile(self, base, tmp_path):
        raw, result = base
        res = _clean(result)
        # A failed, non-interpolated station: its raw value must not leak into
        # mean or max.
        _fail(res, 3, interpolated=False, raw_value=99.0)
        out = tmp_path / "out"
        write_outputs(res, _cfg(raw, out, output={"save_diagnostics": False,
                                                  "save_sti_images": False}))
        s = _strict_json(out / "summary.json")
        v = res.velocities
        assert math.isnan(v[3])
        assert s["max_surface_velocity_ms"] == pytest.approx(np.nanmax(v))
        assert s["max_surface_velocity_ms"] < 99.0
        assert s["mean_surface_velocity_ms"] == pytest.approx(np.nanmean(v))
        assert s["n_failed"] == 1 and s["n_interpolated"] == 0

    def test_no_valid_velocity_writes_null_not_nan(self, base, tmp_path):
        raw, result = base
        res = copy.deepcopy(result)
        for i in range(len(res.stations)):
            _fail(res, i, interpolated=False, raw_value=math.nan)
        res.geometry.flow_bearing_deg = math.nan
        out = tmp_path / "out"
        write_outputs(res, _cfg(raw, out, output={"save_diagnostics": False,
                                                  "save_sti_images": False}))
        s = _strict_json(out / "summary.json")      # bare NaN would raise here
        assert s["mean_surface_velocity_ms"] is None
        assert s["max_surface_velocity_ms"] is None
        assert s["flow_bearing_deg"] is None


# --------------------------------------------------------------------------- #
class TestFigures:
    @pytest.fixture
    def figures(self, monkeypatch):
        made = []
        original = plotting._figure

        def _spy(**kw):
            made.append(original(**kw))
            return made[-1]

        monkeypatch.setattr(plotting, "_figure", _spy)
        return made

    @staticmethod
    def _legend(fig):
        return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]

    def test_interpolated_station_is_labelled_interpolated(self, base, tmp_path, figures):
        res = _clean(base[1])
        _fail(res, 2, interpolated=True)
        plotting.plot_velocity_profile(res, tmp_path / "p.png", dpi=50)
        labels = self._legend(figures[0])
        assert "failed QC (interpolated)" in labels
        assert not any("excluded" in t for t in labels)

    def test_excluded_station_is_not_called_interpolated(self, base, tmp_path, figures):
        """``quality.interpolate_flagged: false``: the label used to say
        "interpolated" regardless."""
        res = _clean(base[1])
        _fail(res, 2, interpolated=False, raw_value=0.4)
        _fail(res, 4, interpolated=False, raw_value=math.nan)
        plotting.plot_velocity_profile(res, tmp_path / "p.png", dpi=50)
        labels = self._legend(figures[0])
        assert "failed QC (excluded; raw value)" in labels
        assert "failed QC (excluded; no value)" in labels
        assert not any("interpolated" in t for t in labels)
        # The profile line has a gap at the excluded stations.
        line = figures[0].axes[0].get_lines()[0]
        assert np.isnan(np.asarray(line.get_ydata(), dtype=float)[[2, 4]]).all()

    def test_full_figure_set_with_an_excluded_station(self, base, tmp_path):
        res = copy.deepcopy(base[1])
        _fail(res, 0, interpolated=False, raw_value=math.nan)
        written = plotting.save_all(res, tmp_path / "figs", dpi=50)
        assert all(p.stat().st_size > 1000 for p in written)

    @staticmethod
    def _arrows(fig):
        """(base points, arrow vectors) of every quiver on the figure's map axes."""
        from matplotlib.quiver import Quiver

        qs = [c for c in fig.axes[0].collections if isinstance(c, Quiver)]
        return [(np.asarray(q.XY), np.column_stack([q.U, q.V])) for q in qs]

    def test_velocity_map_arrow_length_is_proportional_to_speed(self, base, tmp_path, figures):
        res = _clean(base[1])
        plotting.plot_velocity_map(res, tmp_path / "m.png", dpi=50)
        ((xy, uv),) = self._arrows(figures[0])
        v = res.velocities
        assert len(uv) == len(v)
        lengths = np.linalg.norm(uv, axis=1)
        # A uniform-scale view: pixels per (m/s) is the same for every arrow.
        np.testing.assert_allclose(lengths / np.abs(v), (lengths / np.abs(v))[0], rtol=1e-6)
        # ... and every arrow points downstream: the synthetic river runs left
        # to right across the frame.
        assert (uv[:, 0] > 0).all() and np.allclose(uv[:, 1], 0.0, atol=1e-6 * lengths.max())

    def test_velocity_map_marks_failed_stations(self, base, tmp_path, figures):
        res = _clean(base[1])
        _fail(res, 1, interpolated=True)
        _fail(res, 3, interpolated=False, raw_value=math.nan)
        res.stations[5].velocity_ms = -0.2          # flowing upstream at one station
        plotting.plot_velocity_map(res, tmp_path / "m.png", dpi=50)
        quivers = self._arrows(figures[0])
        assert sorted(len(uv) for _, uv in quivers) == [1, 5]   # interpolated drawn apart
        labels = [t.get_text() for t in figures[0].axes[0].get_legend().get_texts()]
        assert "failed QC (interpolated)" in labels and "no velocity (excluded)" in labels
        assert (tmp_path / "m.png").stat().st_size > 1000

    def test_velocity_map_with_no_velocity_at_all(self, base, tmp_path):
        res = _clean(base[1])
        for st in res.stations:
            st.velocity_ms = math.nan
            st.flags = ["low_coherency"]
        plotting.plot_velocity_map(res, tmp_path / "m.png", dpi=50)
        assert (tmp_path / "m.png").stat().st_size > 1000

    def test_velocity_map_is_part_of_the_figure_set(self, base, tmp_path):
        written = plotting.save_all(_clean(base[1]), tmp_path / "figs", dpi=50)
        assert tmp_path / "figs" / "velocity_map.png" in written

    @pytest.mark.parametrize("w,h", [(336, 256), (480, 300), (1920, 1080), (1080, 1920)])
    def test_overview_layout_holds_for_small_frames(self, base, tmp_path, w, h):
        """At 110 px to the inch a thermal-camera or proxy frame made a figure
        narrower than its own legend, and constrained layout gave up."""
        import warnings

        res = copy.copy(base[1])
        res.reference_frame_bgr = np.zeros((h, w, 3), np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plotting.plot_overview(res, tmp_path / "overview.png", dpi=50)

    def test_importing_plotting_leaves_the_backend_alone(self):
        """A notebook's inline/ipympl backend must survive a run's report."""
        code = (
            "import sys, matplotlib\n"
            "matplotlib.use('svg')\n"
            "import stiv.plotting, stiv.report\n"
            "assert matplotlib.get_backend() == 'svg', matplotlib.get_backend()\n"
            "assert 'matplotlib.pyplot' not in sys.modules\n"
        )
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
