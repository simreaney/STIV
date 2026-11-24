"""Tests for the command line's survey aids, matching the notebook wizard's.

Drone field-of-view presets (``--camera``, ``stiv cameras``), the altitude from
a DJI flight log (``--takeoff-height``), the automatic water outline offered by
``stiv pick``, and ``stabilise.water_mask_px: auto`` for ``stiv run``.  No test
here opens a window: every interactive call is stood in for.
"""

from __future__ import annotations

import json
import logging
import shutil

import numpy as np
import pytest
import yaml

from stiv import analysis, cli, picker
from stiv.analysis import run
from stiv.cameras import camera
from stiv.config import Config, ConfigError, RectifyConfig, StabiliseConfig, survey_dict
from stiv.report import write_outputs
from stiv.stabilise import build_mask
from stiv.synthetic import SyntheticRiver, make_config, render
from stiv.watermask import WaterDetection

RIVER = dict(width_px=320, height_px=200, duration_s=3.0, bank_top_px=50, bank_bottom_px=150,
             v_max_ms=0.8, drone_drift_px=20, drone_yaw_deg=1.0)
SECTION = [(160.0, 50.0), (160.0, 150.0)]
RIGHT = (310.0, 100.0)                  # downstream: the river runs left to right


def _srt(alts=(41.5, 41.7, 41.6), focal=24.0, with_alt=True, plain=False) -> str:
    """A flight log: ``rel_alt`` as the Mini and Air 3 write it, a plain
    ``altitude`` as the Air 2S and Mavic 2 do (``plain``), or no height."""
    blocks = []
    for i, alt in enumerate(alts):
        if not with_alt:
            height = ""
        elif plain:
            height = f" [altitude: {alt:.6f}]"
        else:
            height = f" [rel_alt: {alt:.3f} abs_alt: 99.0]"
        fields = f"[focal_len: {focal:.2f}]" + height
        blocks.append(f"{i + 1}\n00:00:0{i},000 --> 00:00:0{i},033\n{fields}\n")
    return "\n".join(blocks)


@pytest.fixture(scope="module")
def drifting(tmp_path_factory):
    river = SyntheticRiver(**RIVER)
    return river, render(river, tmp_path_factory.mktemp("drift") / "river", progress=False)


@pytest.fixture
def flight(drifting, tmp_path):
    """The clip with a DJI flight log beside it."""
    video = tmp_path / "flight.mp4"
    shutil.copy(drifting[1], video)
    video.with_suffix(".SRT").write_text(_srt(), encoding="utf-8")
    return video


def _init(video, out, *extra):
    return cli.main(["init", str(video), "-o", str(out), *extra])


def _header(path):
    return [ln for ln in path.read_text().splitlines() if ln.startswith("#")]


# --------------------------------------------------------------------------- #
class TestCameraPresets:
    def test_cameras_lists_every_preset(self, capsys):
        from stiv.cameras import DJI_CAMERAS

        assert cli.main(["cameras"]) == 0
        out = capsys.readouterr().out
        assert all(p.key in out for p in DJI_CAMERAS)
        assert "DIAGONAL" in out and "69.7 deg" in out and "82.1 deg" in out

    def test_camera_fills_in_the_horizontal_fov(self, drifting, tmp_path, capsys):
        out = tmp_path / "s.yaml"
        assert _init(drifting[1], out, "--camera", "mini-4-pro", "--altitude-m", "40") == 0
        raw = yaml.safe_load(out.read_text())
        assert raw["rectify"] == {"method": "scale", "altitude_m": 40.0, "hfov_deg": 69.7}
        assert any("from --camera mini-4-pro" in ln for ln in _header(out))
        assert not any("m_per_px" in ln and "fill" in ln.lower() for ln in _header(out))
        assert "the section coordinates" in capsys.readouterr().out
        raw["section"].update(start_px=[160, 50], end_px=[160, 150])
        out.write_text(yaml.safe_dump(raw))
        got = Config.from_yaml(out).rectify.resolve_m_per_px(320)
        want = RectifyConfig(altitude_m=40.0, hfov_deg=camera("mini-4-pro").hfov_deg).resolve_m_per_px(320)
        assert got == pytest.approx(want, rel=2e-3)

    def test_info_turns_a_camera_into_a_scale(self, drifting, capsys):
        assert cli.main(["info", str(drifting[1]), "--camera", "air-2s", "--altitude-m", "30"]) == 0
        out = capsys.readouterr().out
        assert "rectify.m_per_px" in out and "77.6 deg HFOV" in out

    def test_a_camera_alone_leaves_the_altitude_to_fill_in(self, drifting, tmp_path):
        out = tmp_path / "s.yaml"
        assert _init(drifting[1], out, "--camera", "mini-4-pro") == 0
        raw = yaml.safe_load(out.read_text())
        assert raw["rectify"]["altitude_m"] is None and raw["rectify"]["hfov_deg"] == 69.7
        assert any("rectify.altitude_m" in ln for ln in _header(out))
        raw["section"].update(start_px=[160, 50], end_px=[160, 150])
        out.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError):
            Config.from_yaml(out)

    def test_m_per_px_directly(self, drifting, tmp_path):
        out = tmp_path / "s.yaml"
        assert _init(drifting[1], out, "--m-per-px", "0.02") == 0
        assert yaml.safe_load(out.read_text())["rectify"] == {"method": "scale", "m_per_px": 0.02}

    @pytest.mark.parametrize("extra,match", [
        (["--camera", "mini-9-pro"], "mini-4-pro"),                 # lists the choices
        (["--camera", "mini-4-pro", "--hfov-deg", "70"], "not both"),
        (["--m-per-px", "0.02", "--camera", "mini-4-pro"], "--m-per-px"),
        (["--hfov-deg", "190"], "180"),
        (["--altitude-m", "-5"], "greater than zero"),
    ])
    def test_contradictions_are_refused_before_writing(self, drifting, tmp_path, capsys, extra, match):
        out = tmp_path / "s.yaml"
        assert _init(drifting[1], out, *extra) == 2
        assert match in capsys.readouterr().err
        assert not out.exists()


# --------------------------------------------------------------------------- #
class TestCameraFromVideo:
    """The drone the video names in its metadata stands in for --camera.

    Reading the name is tested in test_videometa.py; here the reader is stood
    in for, since the clip must also be one OpenCV can decode."""

    @pytest.fixture
    def names(self, monkeypatch):
        from stiv import videometa
        from stiv.cameras import preset_for_name

        calls = []

        def install(name, source=videometa.TRACK):
            def fake(path):
                calls.append(path)
                return videometa.CameraId(name, source, preset_for_name(name)) if name else None
            monkeypatch.setattr(videometa, "identify_camera", fake)
            return calls
        return install

    def test_init_uses_the_drone_the_video_names(self, drifting, tmp_path, capsys, names):
        names("DJI Mini4 Pro")
        out = tmp_path / "s.yaml"
        assert _init(drifting[1], out, "--altitude-m", "40") == 0
        assert yaml.safe_load(out.read_text())["rectify"] == {
            "method": "scale", "altitude_m": 40.0, "hfov_deg": 69.7}
        note = 'hfov_deg 69.7 for the DJI Mini 4 Pro, as the video\'s DJI metadata track says "DJI Mini4 Pro"'
        assert any(note in ln for ln in _header(out))
        assert note in capsys.readouterr().out

    def test_info_asks_only_for_the_altitude(self, drifting, capsys, names):
        names("FC3411", "model tag")
        assert cli.main(["info", str(drifting[1])]) == 0
        out = capsys.readouterr().out
        assert "for the DJI Air 2S" in out
        assert "pass --altitude-m (or --takeoff-height, with a flight log)\n" in out
        assert "and --camera (or --hfov-deg)" not in out

    def test_a_different_camera_is_warned_about(self, drifting, tmp_path, capsys, names):
        names("DJI Mini4 Pro")
        out = tmp_path / "s.yaml"
        assert _init(drifting[1], out, "--camera", "air-2s", "--altitude-m", "40") == 0
        assert yaml.safe_load(out.read_text())["rectify"]["hfov_deg"] == 77.6     # as asked
        err = capsys.readouterr().err
        assert "--camera air-2s is the DJI Air 2S" in err and "--camera mini-4-pro would give 69.7" in err

    def test_the_same_camera_is_not(self, drifting, tmp_path, capsys, names):
        names("DJI Mini4 Pro")
        assert _init(drifting[1], tmp_path / "s.yaml", "--camera", "mini-4-pro") == 0
        assert "warning" not in capsys.readouterr().err

    @pytest.mark.parametrize("extra", [["--hfov-deg", "70"], ["--m-per-px", "0.02"]])
    def test_a_scale_given_outright_is_not_second_guessed(self, drifting, tmp_path, names, extra):
        calls = names("DJI Mini4 Pro")
        assert _init(drifting[1], tmp_path / "s.yaml", *extra) == 0
        assert calls == []

    def test_not_read_for_ground_control(self, drifting, names):
        import argparse

        calls = names("DJI Mini4 Pro")
        sc = cli._resolve_scale(argparse.Namespace(gcps=True), drifting[1])
        assert calls == [] and sc.hfov_deg is None

    def test_a_drone_with_no_preset_is_reported(self, drifting, tmp_path, capsys, names):
        names("DJI Avata2")
        out = tmp_path / "s.yaml"
        assert _init(drifting[1], out) == 0
        assert yaml.safe_load(out.read_text())["rectify"]["m_per_px"] is None
        assert 'says "DJI Avata2", a camera not in the drone list' in capsys.readouterr().out

    def test_nothing_named_leaves_the_scale_open(self, drifting, tmp_path, names):
        names(None)
        out = tmp_path / "s.yaml"
        assert _init(drifting[1], out) == 0
        assert yaml.safe_load(out.read_text())["rectify"]["m_per_px"] is None

    def test_not_used_for_vertical_video(self, tmp_path, capsys, names):
        import cv2

        names("DJI Mini4 Pro")
        video = tmp_path / "tall.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 160))
        for _ in range(5):
            writer.write(np.zeros((160, 96, 3), np.uint8))
        writer.release()
        out = tmp_path / "s.yaml"
        assert _init(video, out) == 0
        assert yaml.safe_load(out.read_text())["rectify"]["m_per_px"] is None
        assert "vertical mode" in capsys.readouterr().out

    def test_zoom_is_still_caught(self, flight, tmp_path, capsys, names):
        names("DJI Mini4 Pro")
        flight.with_suffix(".SRT").write_text(_srt(focal=48.0), encoding="utf-8")
        assert _init(flight, tmp_path / "s.yaml", "--takeoff-height", "0") == 0
        narrowed = camera("mini-4-pro").zoomed_hfov_deg(48.0)
        assert f"--hfov-deg {narrowed:.1f} instead of the preset" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
class TestFlightLog:
    def test_info_reports_it_without_calling_it_the_altitude(self, flight, capsys):
        assert cli.main(["info", str(flight), "--camera", "mini-4-pro"]) == 0
        out = capsys.readouterr().out
        assert "Flight log (flight.SRT)" in out
        assert "41.6 m" in out and "41.5-41.7 m" in out and "24 mm" in out
        assert "NOT the height above the water" in out
        assert "rectify.m_per_px" not in out            # no altitude was given

    def test_takeoff_height_adds_to_the_logged_altitude(self, flight, tmp_path, capsys):
        out = tmp_path / "s.yaml"
        assert _init(flight, out, "--camera", "air-2s", "--takeoff-height", "1.5") == 0
        assert yaml.safe_load(out.read_text())["rectify"]["altitude_m"] == pytest.approx(43.1)
        assert any("above take-off" in ln and "flight.SRT" in ln for ln in _header(out))
        assert "43.1" in capsys.readouterr().out

    def test_info_with_a_takeoff_height(self, flight, capsys):
        assert cli.main(["info", str(flight), "--camera", "mini-4-pro",
                         "--takeoff-height", "2"]) == 0
        out = capsys.readouterr().out
        assert "rectify.m_per_px" in out and "43.6 m altitude" in out

    def test_takeoff_height_without_a_log(self, drifting, tmp_path, capsys):
        assert _init(drifting[1], tmp_path / "s.yaml", "--takeoff-height", "1") == 2
        assert "flight log" in capsys.readouterr().err

    def test_takeoff_height_with_a_log_that_has_no_altitude(self, flight, tmp_path, capsys):
        flight.with_suffix(".SRT").write_text(_srt(with_alt=False), encoding="utf-8")
        assert _init(flight, tmp_path / "s.yaml", "--takeoff-height", "1") == 2
        assert "records no height above take-off" in capsys.readouterr().err

    def test_an_air_2s_log_is_used_and_flagged_for_checking(self, flight, tmp_path, capsys):
        flight.with_suffix(".SRT").write_text(_srt(focal=224, plain=True), encoding="utf-8")
        out = tmp_path / "s.yaml"
        assert _init(flight, out, "--camera", "air-2s", "--takeoff-height", "1.5") == 0
        assert yaml.safe_load(out.read_text())["rectify"]["altitude_m"] == pytest.approx(43.1)
        printed = capsys.readouterr()
        assert "sea level" in printed.out + printed.err

    def test_a_log_reading_above_sea_level_is_not_used(self, flight, tmp_path, capsys):
        flight.with_suffix(".SRT").write_text(_srt(alts=(812.0, 812.4, 812.2), plain=True),
                                              encoding="utf-8")
        assert _init(flight, tmp_path / "s.yaml", "--takeoff-height", "1") == 2
        assert "records no height above take-off" in capsys.readouterr().err

    def test_altitude_and_takeoff_height_together(self, flight, tmp_path, capsys):
        assert _init(flight, tmp_path / "s.yaml", "--altitude-m", "40",
                     "--takeoff-height", "1") == 2
        assert "not both" in capsys.readouterr().err

    def test_digital_zoom_is_warned_about(self, flight, tmp_path, capsys):
        flight.with_suffix(".SRT").write_text(_srt(focal=48.0), encoding="utf-8")
        assert _init(flight, tmp_path / "s.yaml", "--camera", "mini-4-pro",
                     "--takeoff-height", "0") == 0
        err = capsys.readouterr().err
        narrowed = camera("mini-4-pro").zoomed_hfov_deg(48.0)
        assert "2.0x digital zoom" in err and f"--hfov-deg {narrowed:.1f}" in err


# --------------------------------------------------------------------------- #
class TestPickWater:
    @pytest.fixture
    def clicks(self, monkeypatch):
        """Stand in for the click windows; returns the list of prompts shown."""
        shown = []

        def install(*answers):
            it = iter(answers)

            def fake(video, n, prompt, t_s=0.0):
                shown.append(prompt)
                return next(it)

            monkeypatch.setattr(picker, "pick_points", fake)
            return shown

        return install

    def test_an_accepted_outline_replaces_the_clicking(self, drifting, tmp_path, clicks,
                                                       monkeypatch, capsys):
        shown = clicks(SECTION, [RIGHT])          # no third answer: outline not clicked
        offered = []
        monkeypatch.setattr(picker, "confirm_water_outline",
                            lambda video, poly, section, t_s=0.0, note="": offered.append(poly) or True)
        out = tmp_path / "s.yaml"
        assert cli.main(["pick", str(drifting[1]), "-o", str(out), "--m-per-px", "0.02"]) == 0
        raw = yaml.safe_load(out.read_text())
        (poly,) = raw["stabilise"]["water_mask_px"]
        assert len(poly) >= 3 and len(offered) == 1 and len(shown) == 2
        assert raw["rectify"]["m_per_px"] == 0.02
        assert "Next: `stiv run" in capsys.readouterr().out
        Config.from_yaml(out)                     # complete: runs as written

    def test_a_refused_outline_falls_back_to_clicking(self, drifting, tmp_path, clicks, monkeypatch):
        water = [(0.0, 60.0), (319.0, 60.0), (319.0, 140.0)]
        shown = clicks(SECTION, [RIGHT], water)
        monkeypatch.setattr(picker, "confirm_water_outline", lambda *a, **k: False)
        out = tmp_path / "s.yaml"
        assert cli.main(["pick", str(drifting[1]), "-o", str(out)]) == 0
        assert len(shown) == 3
        assert yaml.safe_load(out.read_text())["stabilise"]["water_mask_px"] == [[list(p) for p in water]]

    def test_no_auto_water(self, drifting, tmp_path, clicks, monkeypatch):
        clicks(SECTION, [RIGHT], [])
        monkeypatch.setattr("stiv.watermask.detect_water", lambda *a, **k: pytest.fail("detected"))
        out = tmp_path / "s.yaml"
        assert cli.main(["pick", str(drifting[1]), "-o", str(out), "--no-auto-water"]) == 0
        assert "water_mask_px" not in yaml.safe_load(out.read_text())["stabilise"]

    def test_a_failed_detection_says_why(self, drifting, tmp_path, clicks, monkeypatch, caplog):
        clicks(SECTION, [RIGHT], [])
        monkeypatch.setattr("stiv.watermask.detect_water",
                            lambda *a, **k: WaterDetection([], "failed", "the banks moved too"))
        monkeypatch.setattr(picker, "confirm_water_outline", lambda *a, **k: pytest.fail("offered"))
        with caplog.at_level(logging.INFO, logger="stiv.picker"):
            assert cli.main(["pick", str(drifting[1]), "-o", str(tmp_path / "s.yaml")]) == 0
        assert "could not outline the water automatically: the banks moved too" in caplog.text

    def test_water_flowing_the_other_way_is_reported(self, drifting, tmp_path, clicks,
                                                     monkeypatch, caplog):
        clicks(SECTION, [(5.0, 100.0)], [])       # downstream on the wrong side
        monkeypatch.setattr(picker, "confirm_water_outline", lambda *a, **k: False)
        with caplog.at_level(logging.WARNING, logger="stiv.picker"):
            assert cli.main(["pick", str(drifting[1]), "-o", str(tmp_path / "s.yaml")]) == 0
        assert "other side of the cross-section" in caplog.text

    def test_gcps_and_a_scale_are_refused_before_clicking(self, drifting, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(picker, "pick_points", lambda *a, **k: pytest.fail("clicked"))
        assert cli.main(["pick", str(drifting[1]), "-o", str(tmp_path / "s.yaml"), "--gcps",
                         "--camera", "mini-4-pro"]) == 2
        assert "--gcps" in capsys.readouterr().err

    def test_takeoff_height_errors_come_before_clicking(self, drifting, tmp_path, monkeypatch):
        monkeypatch.setattr(picker, "pick_points", lambda *a, **k: pytest.fail("clicked"))
        assert cli.main(["pick", str(drifting[1]), "-o", str(tmp_path / "s.yaml"),
                         "--takeoff-height", "1"]) == 2


# --------------------------------------------------------------------------- #
def _raw(video, out_dir, water="auto", **stab):
    raw = survey_dict(str(video), m_per_px=0.02, start_px=SECTION[0], end_px=SECTION[1],
                      downstream_px=RIGHT, n_stations=7, out_dir=str(out_dir))
    raw["stabilise"].update(stab)
    if water is not None:
        raw["stabilise"]["water_mask_px"] = water
    return raw


class TestAutoWaterMaskConfig:
    @pytest.mark.parametrize("word", ["auto", "AUTO", " Auto "])
    def test_auto_in_any_case(self, word, tmp_path):
        assert Config.from_dict(_raw("x.mp4", tmp_path, word)).stabilise.water_mask_px == "auto"

    def test_another_word_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="or `auto`"):
            Config.from_dict(_raw("x.mp4", tmp_path, "automatic"))

    def test_polygons_load_as_before(self, tmp_path):
        poly = [[0, 60], [319, 60], [319, 140]]
        cfg = Config.from_dict(_raw("x.mp4", tmp_path, [poly]))
        assert cfg.stabilise.water_mask_px == [[[0.0, 60.0], [319.0, 60.0], [319.0, 140.0]]]

    def test_survey_dict(self):
        assert survey_dict("x.mp4", water_mask_px="auto")["stabilise"]["water_mask_px"] == "auto"
        assert "water_mask_px" not in survey_dict("x.mp4")["stabilise"]       # still opt-in

    def test_yaml_round_trip(self, tmp_path):
        p = tmp_path / "s.yaml"
        Config.from_dict(_raw("x.mp4", tmp_path)).dump_yaml(p)
        assert "water_mask_px: auto" in p.read_text()
        assert Config.from_yaml(p).stabilise.water_mask_px == "auto"

    def test_an_unresolved_auto_never_reaches_the_tracker(self):
        with pytest.raises(ValueError, match="resolved"):
            build_mask(StabiliseConfig(water_mask_px="auto"), (100, 100))

    def test_static_mask_wins(self, tmp_path, monkeypatch):
        monkeypatch.setattr("stiv.watermask.detect_water", lambda *a, **k: pytest.fail("detected"))
        static = [[[0, 0], [319, 0], [319, 40]]]
        cfg = Config.from_dict(_raw("x.mp4", tmp_path, static_mask_px=static))
        stab, note = analysis._automatic_water_mask(cfg, [])
        assert stab.water_mask_px is None and "static_mask_px" in note

    def test_not_needed_without_stabilisation(self, tmp_path, monkeypatch):
        monkeypatch.setattr("stiv.watermask.detect_water", lambda *a, **k: pytest.fail("detected"))
        cfg = Config.from_dict(_raw("x.mp4", tmp_path, enabled=False))
        assert "not needed" in analysis._automatic_water_mask(cfg, [])[1]


class TestAutoWaterMaskRun:
    @pytest.fixture
    def masks(self, monkeypatch):
        """The water_mask_px each tracking mask was actually built with."""
        seen = []
        original = analysis.build_mask

        def spy(cfg, *a, **k):
            seen.append(cfg.water_mask_px)
            return original(cfg, *a, **k)

        monkeypatch.setattr(analysis, "build_mask", spy)
        return seen

    def test_run_outlines_the_water_and_keeps_auto(self, drifting, tmp_path, masks):
        river, video = drifting
        raw = make_config(river, video, tmp_path / "out", n_stations=7)
        raw["stabilise"]["water_mask_px"] = "auto"
        cfg = Config.from_dict(raw)
        res = run(cfg, progress=False)
        assert res.water_mask_note.startswith("auto: water outlined")
        assert "water mask     auto: water outlined" in res.summary()
        (used,) = masks
        assert len(used) == 1 and len(used[0]) >= 3            # one detected polygon
        assert not any("water outline" in w for w in res.warnings)

        out = write_outputs(res, cfg)
        assert yaml.safe_load((out / "config_used.yaml").read_text())["stabilise"]["water_mask_px"] == "auto"
        json.loads((out / "summary.json").read_text())
        assert Config.from_yaml(out / "config_used.yaml").stabilise.water_mask_px == "auto"

    def test_still_water_runs_on_without_a_mask(self, tmp_path, masks):
        river = SyntheticRiver(**{**RIVER, "v_max_ms": 0.0})
        video = render(river, tmp_path / "still", progress=False)
        raw = make_config(river, video, tmp_path / "out", n_stations=7)
        raw["stabilise"]["water_mask_px"] = "auto"
        res = run(Config.from_dict(raw), progress=False)
        assert masks == [None]
        assert res.water_mask_note.startswith("auto: not used")
        assert any("automatic water outline failed" in w for w in res.warnings)


# --------------------------------------------------------------------------- #
def test_wizard_reopens_an_auto_survey(drifting, tmp_path):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("ipyfilechooser")
    import ipywidgets as W

    from stiv.notebook_ui import Wizard

    river, video = drifting
    path = tmp_path / "config_used.yaml"
    Config.from_dict(_raw(video, tmp_path / "results")).dump_yaml(path)
    wz = Wizard()
    status = W.HTML()
    wz._reopen_survey(str(path), status)
    assert "Reopened" in status.value and wz.step == 4
    assert wz.state.water_pts == [] and wz._water_at_run()
    assert "found when the analysis runs" in wz._summary_html()
    assert wz._build_config().stabilise.water_mask_px == "auto"     # not silently dropped

    # An outline of the wizard's own replaces it.
    wz.state.water_pts = [(0.0, 60.0), (319.0, 60.0), (319.0, 140.0)]
    wz.state.water_source = "manual"
    assert not wz._water_at_run()
    assert len(wz._build_config().stabilise.water_mask_px[0]) == 3


# --------------------------------------------------------------------------- #
def _widgets(w):
    yield w
    for child in getattr(w, "children", ()):
        yield from _widgets(child)


def test_wizard_chooses_the_drone_the_video_names(drifting, tmp_path, monkeypatch):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("ipyfilechooser")
    import ipywidgets as W

    from stiv import videometa
    from stiv.cameras import preset_for_name
    from stiv.notebook_ui import Wizard

    named = {"name": "DJI Mini4 Pro"}
    monkeypatch.setattr(videometa, "identify_camera", lambda path: videometa.CameraId(
        named["name"], videometa.TRACK, preset_for_name(named["name"])))

    wz = Wizard()
    status = W.HTML()
    wz._load_video(str(drifting[1]), is_demo=False, status=status)
    s = wz.state
    assert (s.scale_mode, s.camera_key, s.hfov_deg) == ("flight", "mini-4-pro", 69.7)
    assert "recorded by a <b>DJI Mini 4 Pro</b>" in status.value

    step = wz._build_step1()
    html = lambda: " ".join(w.value for w in _widgets(step) if isinstance(w, W.HTML))
    assert "Chosen from the video itself" in html()
    dropdown = next(w for w in _widgets(step) if isinstance(w, W.Dropdown)
                    and w.description == "Drone / camera")
    dropdown.value = "air-2s"
    assert "not the DJI Air 2S" in html() and "Chosen from the video" not in html()

    # Reopening a survey: the Mini 3 Pro and Mini 4 Pro share a field of view,
    # and the video says which it was.
    raw = _raw(drifting[1], tmp_path / "results")
    raw["rectify"] = {"method": "scale", "altitude_m": 40.0, "hfov_deg": 69.7}
    path = tmp_path / "config_used.yaml"
    Config.from_dict(raw).dump_yaml(path)
    for name, key in (("DJI Mini 3 Pro", "mini-3-pro"), ("DJI Mini4 Pro", "mini-4-pro")):
        named["name"] = name
        wz._reopen_survey(str(path), W.HTML())
        assert wz.state.camera_key == key


def test_wizard_names_a_drone_it_has_no_preset_for(drifting, monkeypatch):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("ipyfilechooser")
    import ipywidgets as W

    from stiv import videometa
    from stiv.notebook_ui import Wizard

    monkeypatch.setattr(videometa, "identify_camera",
                        lambda path: videometa.CameraId("DJI Avata2", videometa.TRACK, None))
    wz = Wizard()
    status = W.HTML()
    wz._load_video(str(drifting[1]), is_demo=False, status=status)
    assert wz.state.camera_key is None and wz.state.scale_mode == "manual"
    assert "&ldquo;DJI Avata2&rdquo;" in status.value
    step = wz._build_step1()
    assert any("not in the list" in w.value for w in _widgets(step) if isinstance(w, W.HTML))


# --------------------------------------------------------------------------- #
class TestFocalLengthHint:
    """A flight log logging a lens wider than 24 mm points to the Air 2S."""

    HINT = "only the DJI Air 2S's lens is that wide"

    @pytest.fixture
    def air_2s(self, flight):
        flight.with_suffix(".SRT").write_text(_srt(focal=22.4, plain=True), encoding="utf-8")
        return flight

    def test_info_points_to_the_air_2s(self, air_2s, capsys):
        assert cli.main(["info", str(air_2s)]) == 0
        out = capsys.readouterr().out
        assert self.HINT in out and "--camera air-2s gives 77.6 deg" in out

    def test_init_suggests_it_but_does_not_choose_it(self, air_2s, tmp_path, capsys):
        out = tmp_path / "s.yaml"
        assert _init(air_2s, out) == 0
        assert yaml.safe_load(out.read_text())["rectify"]["m_per_px"] is None
        assert any(self.HINT in ln for ln in _header(out))
        assert self.HINT in capsys.readouterr().out

    def test_a_drone_whose_lens_is_too_long_is_warned_about(self, air_2s, tmp_path, capsys):
        assert _init(air_2s, tmp_path / "s.yaml", "--camera", "mini-4-pro") == 0
        err = capsys.readouterr().err
        assert "wider than the DJI Mini 4 Pro's 24 mm lens can give" in err
        assert "--camera air-2s, 77.6 deg" in err

    def test_the_air_2s_itself_is_quiet(self, air_2s, tmp_path, capsys):
        assert _init(air_2s, tmp_path / "s.yaml", "--camera", "air-2s") == 0
        cap = capsys.readouterr()
        assert "warning" not in cap.err and self.HINT not in cap.out

    def test_metadata_that_disagrees_with_the_log(self, air_2s, tmp_path, capsys, monkeypatch):
        from stiv import videometa

        monkeypatch.setattr(videometa, "identify_camera", lambda path: videometa.CameraId(
            "DJI Mini4 Pro", videometa.TRACK, camera("mini-4-pro")))
        assert _init(air_2s, tmp_path / "s.yaml") == 0
        assert "wider than the DJI Mini 4 Pro's 24 mm lens" in capsys.readouterr().err

    @pytest.mark.parametrize("focal,extra", [
        (24.0, []),                          # fits nearly every drone listed
        (44.8, []),                          # a zoomed Air 2S, or a zoomed anything
        (22.4, ["--hfov-deg", "77.6"]),      # the field of view is settled already
        (22.4, ["--m-per-px", "0.02"]),
    ])
    def test_no_hint(self, flight, tmp_path, capsys, focal, extra):
        flight.with_suffix(".SRT").write_text(_srt(focal=focal), encoding="utf-8")
        assert _init(flight, tmp_path / "s.yaml", *extra) == 0
        cap = capsys.readouterr()
        assert "lens is that wide" not in cap.out + cap.err


def test_wizard_points_to_the_air_2s_from_the_flight_log(flight):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("ipyfilechooser")
    import ipywidgets as W

    from stiv.notebook_ui import Wizard

    flight.with_suffix(".SRT").write_text(_srt(focal=22.4, plain=True), encoding="utf-8")
    wz = Wizard()
    status = W.HTML()
    wz._load_video(str(flight), is_demo=False, status=status)
    assert "flight log points to a <b>DJI Air 2S</b>" in status.value
    assert wz.state.camera_key is None and wz.state.hfov_deg is None      # a hint, not a choice

    step = wz._build_step1()
    html = lambda: " ".join(w.value for w in _widgets(step) if isinstance(w, W.HTML))
    dropdown = next(w for w in _widgets(step) if isinstance(w, W.Dropdown)
                    and w.description == "Drone / camera")
    assert "only the <b>DJI Air 2S</b>'s lens is that wide" in html()
    dropdown.value = "mini-4-pro"
    assert "wider than the DJI Mini 4 Pro's 24 mm lens can give" in html()
    dropdown.value = "air-2s"
    assert "lens is that wide" not in html() and "lens can give" not in html()
