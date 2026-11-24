"""The other velocity methods: FFT-STIV, LS-PIV and particle tracking.

Unit tests hold each estimator to a field it can recover exactly -- streaks
of known slope, a texture translated by a known amount -- and the end-to-end
tests hold every method to the synthetic river's known profile, alone and
compared in one pass, as ``test_end_to_end`` does for STIV.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.ndimage import gaussian_filter

from stiv import Config, cli, orientation
from stiv.analysis import ADVISORY_FLAGS, FAILURE_FLAGS, run, run_survey
from stiv.config import (
    ConfigError, LspivConfig, PtvConfig, StiConfig, VelocimetryConfig, survey_dict,
)
from stiv.geometry import StationBox
from stiv.lspiv import PatchCollector, PatchLayout, estimate as lspiv_estimate
from stiv.ptv import TrackCollector, TrackSet, _at_centre, estimate as ptv_estimate
from stiv.report import write_outputs
from stiv.sti import preprocess
from stiv.synthetic import SyntheticRiver, make_config, render
from stiv.vectors import subpixel_peak

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
BASE = {
    "video": {"path": "clip.mp4"},
    "rectify": {"m_per_px": 0.02},
    "section": {"start_px": [100, 50], "end_px": [100, 300]},
}


def _cfg(**blocks) -> Config:
    raw = {k: dict(v) for k, v in BASE.items()}
    raw.update(blocks)
    return Config.from_dict(raw)


class TestConfig:
    def test_defaults_are_stiv_alone(self):
        cfg = _cfg()
        assert cfg.velocimetry.method == "stiv"
        assert cfg.velocimetry.methods == ["stiv"]
        assert cfg.lspiv.background == "mean" and cfg.lspiv.ia_size_m is None

    def test_methods_lists_the_reported_one_first(self):
        cfg = _cfg(velocimetry={"method": "lspiv", "compare": ["stiv", "ptv"]})
        assert cfg.velocimetry.methods == ["lspiv", "stiv", "ptv"]

    @pytest.mark.parametrize("block, match", [
        ({"velocimetry": {"method": "piv"}}, "velocimetry.method"),
        ({"velocimetry": {"compare": ["optical"]}}, "velocimetry.compare"),
        ({"velocimetry": {"method": "lspiv", "compare": ["lspiv"]}}, "already velocimetry.method"),
        ({"velocimetry": {"compare": ["ptv", "ptv"]}}, "more than once"),
        ({"lspiv": {"ia_size_m": 0}}, "lspiv.ia_size_m"),
        ({"lspiv": {"pair_step": 0}}, "lspiv.pair_step"),
        ({"lspiv": {"background": "ema"}}, "lspiv.background"),
        ({"ptv": {"track_frames": 0}}, "ptv.track_frames"),
        ({"ptv": {"max_angle_deg": 120}}, "ptv.max_angle_deg"),
        ({"quality": {"min_peak_ratio": 0.5}}, "quality.min_peak_ratio"),
        ({"stiv": {"estimator": "wavelet"}}, "'fft'"),
    ])
    def test_bad_values_are_refused(self, block, match):
        with pytest.raises(ConfigError, match=match):
            _cfg(**block)

    def test_round_trips_through_a_dict(self):
        cfg = _cfg(velocimetry={"method": "ptv", "compare": ["lspiv"]}, lspiv={"ia_size_m": 0.4})
        again = Config.from_dict(cfg.to_dict())
        assert again.to_dict() == cfg.to_dict()
        assert again.velocimetry.compare == ["lspiv"] and again.lspiv.ia_size_m == 0.4

    def test_survey_dict_writes_the_methods(self):
        raw = survey_dict("x.mp4", method="lspiv", compare=["stiv"], estimator="fft",
                          ia_size_m=0.5)
        assert raw["velocimetry"] == {"method": "lspiv", "compare": ["stiv"]}
        assert raw["stiv"]["estimator"] == "fft"
        assert raw["lspiv"] == {"ia_size_m": 0.5}
        # Only a survey that uses LS-PIV carries its block.
        assert "lspiv" not in survey_dict("x.mp4")

    def test_a_config_from_before_the_methods_is_stiv(self):
        # Such as the survey.yaml the iPhone app exports.
        assert _cfg().velocimetry == VelocimetryConfig()


# --------------------------------------------------------------------------- #
# FFT-STIV
# --------------------------------------------------------------------------- #
def _streaks(slope: float, n_t: int = 180, n_x: int = 120, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal(8000), 3)
    x = np.arange(n_x)
    sti = np.array([np.interp(x - slope * t + 4000, np.arange(8000), f) for t in range(n_t)])
    return sti + 0.3 * f.std() * rng.standard_normal(sti.shape)


class TestSpectralEstimator:
    @pytest.mark.parametrize("slope", [-0.6, 0.3, 1.0, 2.0, 3.5])
    @pytest.mark.parametrize("shape", [(180, 120), (600, 150), (100, 200)])
    def test_recovers_the_streak_slope(self, slope, shape):
        cond = preprocess(_streaks(slope, *shape), StiConfig())
        est = orientation.spectral_estimate(orientation.spectral_profile(cond), -1.55, 1.55)
        # Square Welch tiles: within 2 % from a slow to a fast streak, either
        # way, whatever the STI's shape -- but a short STI (100 rows, a few
        # seconds) has coarser angular resolution: 0.6 degrees at tan phi 3.5
        # is 4 % of the velocity.
        assert est.tan_phi == pytest.approx(slope, rel=0.05 if shape == (100, 200) else 0.02)
        assert est.mean_coherency > 0.5 and est.valid_fraction == 1.0
        assert 0 < est.phi_std < math.radians(15)

    def test_respects_the_search_range(self):
        prof = orientation.spectral_profile(preprocess(_streaks(1.0), StiConfig()))
        assert orientation.spectral_estimate(prof, 0.0, 1.5).tan_phi == pytest.approx(1.0, rel=0.02)
        # A range that excludes the streaks finds nothing standing above the
        # spectrum's background, as Eq. (14) finds no window in it.
        est = orientation.spectral_estimate(prof, -1.5, -0.2)
        assert np.isnan(est.tan_phi) and est.valid_fraction == 0.0

    def test_refuses_an_sti_too_small_for_a_spectrum(self):
        with pytest.raises(ValueError, match="too small"):
            orientation.spectral_profile(np.zeros((10, 100)))


# --------------------------------------------------------------------------- #
# sub-pixel peak
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cx, cy, rho", [(5.3, 4.8, 0.0), (5.3, 4.8, 0.7),
                                         (4.6, 5.4, -0.6), (5.45, 5.45, 0.8)])
def test_subpixel_peak_finds_a_tilted_gaussian(cx, cy, rho):
    # A shear flow tilts the correlation peak; fitting each axis alone would
    # put a spurious cross-flow velocity into every such station.
    yy, xx = np.mgrid[0:11, 0:11]
    X, Y = (xx - cx) / 2.5, (yy - cy) / 1.5
    p = np.exp(-(X ** 2 - 2 * rho * X * Y + Y ** 2) / (2 * (1 - rho ** 2)))
    iy, ix = np.unravel_index(np.argmax(p), p.shape)
    dy, dx = subpixel_peak(p, iy, ix)
    assert (ix + dx, iy + dy) == pytest.approx((cx, cy), abs=1e-6)


# --------------------------------------------------------------------------- #
# LS-PIV on a texture of known motion
# --------------------------------------------------------------------------- #
H, W, PAD = 240, 600, 400


def _texture(sigma: float, seed: int) -> np.ndarray:
    f = gaussian_filter(np.random.default_rng(seed).standard_normal((H, W + 2 * PAD)), sigma)
    return f / f.std()


def _layout(ia: int = 25, m: int = 9) -> PatchLayout:
    n = ia + 2 * m
    c = np.array([300.0, 120.0])
    offs = np.arange(n) - (n - 1) / 2
    gx, gy = np.meshgrid(offs, offs)
    pts = np.column_stack([c[0] + gx.ravel(), c[1] + gy.ravel()])
    # World = image, one sample a pixel: velocity comes out in pixels a frame.
    return PatchLayout(0, c, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, ia, m, pts, pts)


def _translate(tex, u, n_frames=150, static=None, cfg=LspivConfig()):
    col = PatchCollector([_layout()], [1], 1.0, cfg)
    base_x = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :] + PAD, (H, W))
    grid_y = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, axis=1)
    for k in range(n_frames):
        mx = np.ascontiguousarray(np.mod(base_x - u[0] * k, tex.shape[1]), dtype=np.float32)
        my = np.ascontiguousarray(np.mod(grid_y - u[1] * k, H), dtype=np.float32)
        f = cv2.remap(tex.astype(np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        col.add_frame(f if static is None else f + static, None)
    return col.finish()[0]


class TestLspiv:
    @pytest.mark.parametrize("sigma", [2.0, 6.0])
    def test_recovers_a_translation_without_bias(self, sigma):
        # sigma 6 is the synthetic river's texture: coarse against the
        # interrogation area, which is what exposed a window-mean bias.
        got = [lspiv_estimate(_translate(_texture(sigma, s), (2.0, 0.0))).velocity_world
               for s in range(3)]
        assert np.mean(got, axis=0) == pytest.approx([2.0, 0.0], abs=0.01)

    def test_recovers_a_diagonal_sub_pixel_motion(self):
        e = lspiv_estimate(_translate(_texture(3.0, 9), (1.3, 0.4)))
        assert e.velocity_world == pytest.approx([1.3, 0.4], abs=0.05)
        assert e.peak_ratio > 1.5 and not e.flags

    def test_mean_background_removes_static_texture_exactly(self):
        tex = _texture(3.0, 9)
        static = gaussian_filter(np.random.default_rng(4).standard_normal((H, W)), 3) * 2.0
        clean = lspiv_estimate(_translate(tex, (2.0, 0.0))).velocity_world
        glare = lspiv_estimate(_translate(tex, (2.0, 0.0), static=static)).velocity_world
        raw = lspiv_estimate(_translate(tex, (2.0, 0.0), static=static,
                                        cfg=LspivConfig(background="none"))).velocity_world
        # The bilinear expansion subtracts the clip's mean patch as if it had
        # been known from the first frame: the static texture leaves no trace
        # beyond float32 rounding in the correlation.
        np.testing.assert_allclose(glare, clean, atol=1e-6)
        assert abs(raw[0] - 2.0) > abs(glare[0] - 2.0)

    def test_sub_ensembles_agree_on_a_steady_motion(self):
        r = _translate(_texture(3.0, 9), (2.0, 0.0))
        assert len(r.block_disp) >= 2
        assert np.std(r.block_disp[:, 0]) < 0.05

    def test_a_featureless_patch_gives_no_peak(self):
        col = PatchCollector([_layout()], [1], 1.0, LspivConfig())
        for _ in range(20):
            col.add_frame(np.full((H, W), 100, np.uint8), None)
        e = lspiv_estimate(col.finish()[0])
        assert "no_correlation_peak" in e.flags and not e.ok


# --------------------------------------------------------------------------- #
# PTV
# --------------------------------------------------------------------------- #
class TestPtv:
    def test_fit_across_the_box_reads_the_centre(self):
        # Tracks crowded on the fast side of a gradient pull the median up;
        # the straight-line fit is read at the box's centre instead.
        rng = np.random.default_rng(0)
        offset = np.concatenate([rng.uniform(-0.5, 0.0, 20), rng.uniform(0.0, 0.5, 80)])
        values = 1.0 + 0.4 * offset + 0.005 * rng.standard_normal(100)
        assert np.median(values) > 1.04
        assert _at_centre(values, offset, 0.5) == pytest.approx(1.0, abs=0.01)
        # Too few tracks to fit: the median.
        assert _at_centre(values[:5], offset[:5], 0.5) == pytest.approx(np.median(values[:5]))

    def test_tracks_a_translated_texture(self):
        from stiv.geometry import ScaleRectifier

        tex = (np.clip(0.5 + 0.2 * _texture(2.0, 3), 0, 1) * 255).astype(np.float32)
        rect = ScaleRectifier(1.0)
        centre = rect.image_to_world([[300.0, 120.0]])[0]
        along = rect.image_to_world([[301.0, 120.0]])[0] - centre
        box = StationBox(0, centre, along, np.array([-along[1], along[0]]), 80.0, 60.0)
        col = TrackCollector([box], rect, 1.0, PtvConfig(track_frames=4), (H, W))
        base_x = np.broadcast_to(np.arange(W, dtype=np.float32)[None, :] + PAD, (H, W))
        grid_y = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, axis=1)
        for k in range(40):
            mx = np.ascontiguousarray(base_x - 1.5 * k, dtype=np.float32)
            col.add_frame(cv2.remap(tex, mx, grid_y, cv2.INTER_CUBIC).astype(np.uint8), None)
        tracks = col.finish()
        assert isinstance(tracks, TrackSet) and len(tracks) > 50
        cfg = _cfg(stiv={"velocity_range_ms": [-0.5, 5.0]})
        e = ptv_estimate(cfg, tracks, box)
        assert e.velocity_world @ box.along_world == pytest.approx(1.5, abs=0.03)
        assert abs(e.velocity_world @ box.across_world) < 0.03


# --------------------------------------------------------------------------- #
# end to end, on the synthetic river
# --------------------------------------------------------------------------- #
N = 15


@pytest.fixture(scope="module")
def river_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("velocimetry")
    river = SyntheticRiver(width_px=720, height_px=420, duration_s=6.0,
                           bank_top_px=90, bank_bottom_px=330, v_max_ms=1.2)
    return river, render(river, d / "river", progress=False), d


def _run(clip, name, multi=False, **blocks):
    river, video, d = clip
    raw = make_config(river, video, d / name, n_stations=N)
    for k, v in blocks.items():
        raw.setdefault(k, {}).update(v)
    if multi:
        x = river.width_px // 2
        raw["section"] = {"n_stations": N}
        raw["sections"] = [{"name": "a", "start_px": [x, 90], "end_px": [x, 330]},
                           {"name": "b", "start_px": [x - 150, 90], "end_px": [x - 150, 330]}]
        return run_survey(Config.from_dict(raw), progress=False)
    return run(Config.from_dict(raw), progress=False)


def _errors(stations, river):
    _, truth = river.truth_profile(N)
    v = np.array([s.velocity_ms for s in stations])
    m = (truth > 0.2) & np.isfinite(v)
    err = v[m] - truth[m]
    return float(np.sqrt(np.mean(err ** 2))), float(err.mean())


class TestEndToEnd:
    @pytest.mark.parametrize("blocks, rmse_max", [
        ({"stiv": {"estimator": "fft"}}, 0.03),
        ({"velocimetry": {"method": "lspiv"}}, 0.04),
        ({"velocimetry": {"method": "ptv"}}, 0.04),
    ])
    def test_each_method_recovers_the_profile(self, river_clip, blocks, rmse_max):
        river = river_clip[0]
        r = _run(river_clip, "alone-" + "-".join(map(str, blocks.values())), **blocks)
        rmse, bias = _errors(r.stations, river)
        assert rmse < rmse_max and abs(bias) < 0.025
        assert sum(1 for s in r.stations if s.failed) <= 3
        if r.method != "stiv":
            # The water runs straight down the channel.
            cross = np.array([s.velocity_cross_ms for s in r.stations if not s.failed])
            assert np.median(np.abs(cross)) < 0.03
            assert all(s.method == r.method for s in r.stations)
            assert all(np.isnan(s.tan_phi) for s in r.stations)

    def test_compare_runs_every_method_in_one_pass(self, river_clip):
        river = river_clip[0]
        r = _run(river_clip, "compare", velocimetry={"compare": ["lspiv", "ptv"]})
        assert r.method == "stiv" and list(r.comparisons) == ["lspiv", "ptv"]
        assert r.stis and r.correlations and r.tracks is not None and len(r.boxes) == N
        for m in r.methods:
            rmse, _ = _errors(r.stations_for(m), river)
            assert rmse < 0.04, m
        df = r.to_frame()
        assert {"velocity_lspiv_ms", "velocity_ptv_ms", "method", "velocity_cross_ms"} <= set(df)
        assert "LS-PIV" in r.summary() and "compared with" in r.summary()

    def test_stiv_alone_writes_the_columns_it_always_has(self, river_clip):
        r = _run(river_clip, "stiv-alone")
        assert list(r.to_frame().columns) == [
            "index", "station_m", "x_world_m", "y_world_m", "u_px", "v_px", "phi_deg",
            "tan_phi", "velocity_ms", "velocity_raw_ms", "velocity_std_ms", "coherency",
            "valid_fraction", "n_windows", "sx_m_per_px", "st_s_per_px", "flags",
        ]
        assert "method" not in r.summary()

    def test_reported_lspiv_compared_with_stiv_and_discharge(self, river_clip):
        depths = [0.2 + 0.1 * min(i, N - 1 - i) for i in range(N)]
        r = _run(river_clip, "lspiv-q", velocimetry={"method": "lspiv", "compare": ["stiv"]},
                 discharge={"enabled": True, "depths_m": depths})
        assert r.discharge is not None and r.comparison_discharge["stiv"] is not None
        q, q_stiv = r.discharge.discharge_m3s, r.comparison_discharge["stiv"].discharge_m3s
        assert q == pytest.approx(q_stiv, rel=0.05)
        assert "Q (m3/s)" in r.comparison_table()

    @pytest.mark.parametrize("method", ["lspiv", "ptv"])
    def test_reversed_direction_is_flipped(self, river_clip, method):
        r = _run(river_clip, f"rev-{method}", velocimetry={"method": method},
                 flow={"reverse": True})
        assert np.nanmedian(r.velocities) > 0.5
        assert any("opposite to the assumed downstream" in w for w in r.warnings)
        # The direction reported follows the velocities: downstream is +x again.
        assert r.geometry.flow_dir_world[0] > 0.99

    def test_reversed_direction_without_auto_flip_is_only_warned(self, river_clip):
        r = _run(river_clip, "rev-noflip", velocimetry={"method": "lspiv"},
                 flow={"reverse": True, "auto_flip": False})
        assert np.nanmedian([s.velocity_raw_ms for s in r.stations]) < -0.5
        assert any("flow.reverse: false" in w for w in r.warnings)

    def test_methods_that_disagree_on_direction_are_warned_about(self, river_clip):
        r = _run(river_clip, "rev-mixed", velocimetry={"method": "lspiv", "compare": ["stiv"]},
                 flow={"reverse": True})
        # Both find the water running the other way, so they agree.
        assert np.nanmedian([s.velocity_ms for s in r.comparisons["stiv"]]) > 0.5
        assert not any("disagree" in w for w in r.warnings)

    def test_survey_of_two_sections(self, river_clip):
        survey = _run(river_clip, "multi", multi=True,
                      velocimetry={"method": "ptv", "compare": ["lspiv"]})
        assert len(survey) == 2
        for r in survey:
            assert r.method == "ptv" and "lspiv" in r.comparisons
            assert r.tracks is not None and len(r.tracks) > 0
        assert {"method", "discharge_lspiv_m3s"} <= set(survey.to_frame().columns)

    def test_outputs(self, river_clip, tmp_path):
        r = _run(river_clip, "outputs", velocimetry={"method": "lspiv", "compare": ["stiv", "ptv"]})
        r.config.output.out_dir = str(tmp_path / "out")
        out = write_outputs(r, r.config)
        for name in ("velocity.csv", "velocity_stiv.csv", "velocity_ptv.csv",
                     "correlation_panel.png", "tracks.png", "sti_panel.png"):
            assert (out / name).is_file(), name
        summary = yaml.safe_load((out / "summary.json").read_text())
        assert summary["method"] == "lspiv" and set(summary["comparisons"]) == {"stiv", "ptv"}
        assert pd.read_csv(out / "velocity_ptv.csv")["method"].eq("ptv").all()
        # The next run into the folder clears them.
        r2 = _run(river_clip, "outputs2")
        r2.config.output.out_dir = str(out)
        write_outputs(r2, r2.config)
        assert not (out / "velocity_ptv.csv").exists() and not (out / "tracks.png").exists()


def test_the_comparison_example_loads():
    from pathlib import Path

    cfg = Config.from_yaml(Path(__file__).resolve().parents[1] / "examples" / "compare_methods.yaml")
    assert cfg.velocimetry.methods == ["stiv", "lspiv", "ptv"]


def test_new_flags_are_classified():
    for flag in ("no_correlation_peak", "low_peak_ratio", "displacement_at_search_edge",
                 "too_few_tracks"):
        assert flag in FAILURE_FLAGS
    for flag in ("displacement_low", "cross_flow_high"):
        assert flag in ADVISORY_FLAGS


# --------------------------------------------------------------------------- #
# notebook wizard
# --------------------------------------------------------------------------- #
def _find(widget, description):
    found = []

    def walk(w):
        if getattr(w, "description", None) == description:
            found.append(w)
        for child in getattr(w, "children", ()):
            walk(child)

    walk(widget)
    assert len(found) == 1, f"{len(found)} widgets labelled {description!r}"
    return found[0]


class TestWizard:
    @pytest.fixture
    def nb(self):
        pytest.importorskip("ipywidgets")
        pytest.importorskip("ipyfilechooser")
        from stiv import notebook_ui

        return notebook_ui

    @pytest.fixture
    def wizard(self, nb, river_clip):
        import ipywidgets as W

        _, video, _ = river_clip
        wz = nb.Wizard()
        wz._load_video(str(video), is_demo=False, status=W.HTML())
        s = wz.state
        s.m_per_px = 0.02
        s.section_pts, s.downstream_pts = [(360.0, 90.0), (360.0, 330.0)], [(719.0, 210.0)]
        return wz

    def test_the_method_choice_reaches_the_config(self, wizard):
        step = wizard._build_step3()
        _find(step, "Report").value = "lspiv"
        lspiv_box = _find(step, "LS-PIV")
        assert lspiv_box.disabled                  # the reported method is not also compared
        _find(step, "STIV").value = True
        _find(step, "PTV").value = True
        _find(step, "STIV angle from").value = "fft"
        _find(step, "LS-PIV area (m)").value = "0.4"
        cfg = wizard._build_config()
        assert cfg.velocimetry.method == "lspiv" and cfg.velocimetry.compare == ["stiv", "ptv"]
        assert cfg.stiv.estimator == "fft" and cfg.lspiv.ia_size_m == 0.4
        assert "LS-PIV" in wizard._recap_html(cfg) and "compared with" in wizard._recap_html(cfg)

        # Reporting a method that was being compared takes it off the list.
        _find(step, "Report").value = "ptv"
        assert wizard.state.method == "ptv" and wizard.state.compare == ["stiv"]

    def test_reopening_a_survey_keeps_its_methods(self, nb, wizard, tmp_path):
        import ipywidgets as W

        wizard.state.method, wizard.state.compare = "ptv", ["lspiv"]
        wizard.state.ia_size_m = 0.35
        path = tmp_path / "survey.yaml"
        wizard._build_config().dump_yaml(path)

        wz = nb.Wizard()
        wz._reopen_survey(str(path), W.HTML())
        assert (wz.state.method, wz.state.compare, wz.state.ia_size_m) == ("ptv", ["lspiv"], 0.35)
        cfg = wz._build_config()
        assert cfg.velocimetry.method == "ptv" and cfg.lspiv.ia_size_m == 0.35

    def test_results_show_the_comparison(self, wizard, river_clip, tmp_path):
        r = _run(river_clip, "wizard-results", velocimetry={"method": "lspiv", "compare": ["stiv"]})
        r.config.output.out_dir = str(tmp_path / "out")
        out = write_outputs(r, r.config)
        view = wizard._result_view(r, out)
        texts = []

        def walk(w):
            if isinstance(getattr(w, "value", None), str):
                texts.append(w.value)
            for child in getattr(w, "children", ()):
                walk(child)

        walk(view)
        html = "".join(texts)
        assert "Methods compared" in html and "LS-PIV</b> (reported)" in html
        tabs = [c for c in view.children if hasattr(c, "titles") or hasattr(c, "get_title")][0]
        titles = (list(tabs.titles) if hasattr(tabs, "titles")
                  else [tabs.get_title(i) for i in range(len(tabs.children))])
        assert "LS-PIV correlation" in titles and "Space-time" in titles


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
class TestCli:
    def test_run_overrides_the_methods(self, river_clip, tmp_path, capsys):
        river, video, _ = river_clip
        p = tmp_path / "survey.yaml"
        p.write_text(yaml.safe_dump(make_config(river, video, tmp_path / "out", n_stations=N)))
        assert cli.main(["-q", "run", str(p), "--method", "lspiv", "--compare", "stiv",
                         "--estimator", "fft", "--no-figures"]) == 0
        assert "LS-PIV, compared with STIV" in capsys.readouterr().out
        used = Config.from_yaml(tmp_path / "out" / "config_used.yaml")
        assert used.velocimetry.method == "lspiv" and used.velocimetry.compare == ["stiv"]
        assert used.stiv.estimator == "fft"

    def test_compare_none_clears_it(self):
        args = cli.build_parser().parse_args(["run", "x.yaml", "--compare", "none"])
        assert args.compare == []

    def test_unknown_compare_method_is_refused(self, capsys):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["run", "x.yaml", "--compare", "lspiv,radar"])
        assert "radar" in capsys.readouterr().err

    def test_init_writes_the_methods(self, river_clip, tmp_path, capsys):
        _, video, _ = river_clip
        out = tmp_path / "survey.yaml"
        assert cli.main(["init", str(video), "-o", str(out), "--m-per-px", "0.02",
                         "--method", "ptv", "--compare", "lspiv"]) == 0
        raw = yaml.safe_load(out.read_text())
        assert raw["velocimetry"] == {"method": "ptv", "compare": ["lspiv"]}
        assert "lspiv" in raw
