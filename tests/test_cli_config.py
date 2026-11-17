"""Tests for the config schema, the command line and the notebook wizard.

These cover the ways a survey reaches the analysis: a hand-written or
generated YAML file, ``stiv init`` / ``stiv pick``, and the Jupyter wizard.  The
common thread is that a mistake should be refused up front with a message that
says what to fix -- not surface as a traceback, or as a silently wrong number,
after the whole video has been read.
"""

from __future__ import annotations

import copy
import datetime
import math
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from stiv import cli
from stiv.config import (
    Config,
    ConfigError,
    SectionConfig,
    StiConfig,
    path_for_config,
    suggest_frame_step,
    survey_dict,
)
from stiv.synthetic import SyntheticRiver, make_config, render


def tiny_river(**kw) -> SyntheticRiver:
    base = dict(width_px=320, height_px=200, duration_s=2.0, bank_top_px=50,
                bank_bottom_px=150, v_max_ms=0.8)
    base.update(kw)
    return SyntheticRiver(**base)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("clip")
    return render(tiny_river(), d / "river", progress=False)


BASE = {
    "video": {"path": "dummy.mp4"},
    "rectify": {"method": "scale", "m_per_px": 0.02},
    "section": {"start_px": [480, 120], "end_px": [480, 420], "n_stations": 11},
}


def config_with(**overrides):
    raw = copy.deepcopy(BASE)
    for k, v in overrides.items():
        raw.setdefault(k, {}).update(v)
    return Config.from_dict(raw)


def write_yaml(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
class TestTypes:
    def test_yaml_exponent_floats_are_numbers(self, tmp_path):
        """PyYAML reads ``15e-1`` as the *string* '15e-1' (no decimal point)."""
        p = write_yaml(tmp_path / "s.yaml", """\
            video: {path: clip.mp4}
            rectify: {method: scale, m_per_px: 2e-2}
            section: {start_px: [10, 10], end_px: [200, 10], n_stations: 6e1}
            stiv: {smooth_sigma: 15e-1}
            """)
        cfg = Config.from_yaml(p)
        assert cfg.stiv.smooth_sigma == 1.5
        assert cfg.rectify.m_per_px == pytest.approx(0.02)
        assert cfg.section.n_stations == 60 and isinstance(cfg.section.n_stations, int)

    def test_integral_float_count_is_accepted(self):
        assert config_with(section={"n_stations": 30.0}).section.n_stations == 30

    def test_numpy_values_are_accepted(self):
        """Code building a config often holds points as arrays."""
        cfg = config_with(section={"start_px": np.array([480.0, 120.0]),
                                   "end_px": np.array([480, 420]),
                                   "n_stations": np.int64(11)},
                          stiv={"velocity_range_ms": np.array([-0.5, 3.0])},
                          flow={"auto_flip": np.True_})
        assert cfg.section.start_px == [480.0, 120.0]
        assert cfg.stiv.velocity_range_ms == [-0.5, 3.0]
        assert cfg.flow.auto_flip is True
        # ...and when set afterwards, as a script tweaking a loaded config would.
        cfg.section.end_px = np.array([480.0, 400.0])
        cfg.stabilise.water_mask_px = [np.array([[0, 0], [9, 0], [9, 9]])]
        cfg.validate()
        assert cfg.section.end_px == [480.0, 400.0]

    def test_yaml_names_are_kept_as_typed(self, tmp_path):
        """YAML 1.1 reads ``2024-05-01`` as a date and ``12:30`` as 750."""
        for typed in ("2024-05-01", "12:30", "2024-05-01 10:00:00"):
            p = write_yaml(tmp_path / "s.yaml", f"""\
                video: {{path: clip.mp4}}
                rectify: {{method: scale, m_per_px: 0.02}}
                section: {{start_px: [10, 10], end_px: [200, 10], name: {typed}}}
                """)
            assert Config.from_yaml(p).section.name == typed

    def test_base60_is_not_a_number(self, tmp_path):
        p = write_yaml(tmp_path / "s.yaml", """\
            video: {path: clip.mp4, start_s: 1:30}
            rectify: {method: scale, m_per_px: 0.02}
            section: {start_px: [10, 10], end_px: [200, 10]}
            """)
        with pytest.raises(ConfigError, match="video.start_s must be a number, got '1:30'"):
            Config.from_yaml(p)

    def test_a_date_object_becomes_its_iso_text(self):
        cfg = config_with(section={"name": datetime.date(2024, 5, 1)})
        assert cfg.section.name == "2024-05-01"

    @pytest.mark.parametrize("section,key,value", [
        ("section", "n_stations", 30.5),
        ("stiv", "window_px", 31.5),
        ("stiv", "window_px", "wide"),
        ("stiv", "smooth_sigma", "lots"),
        ("stabilise", "enabled", "yes please"),
        ("section", "n_stations", True),
    ])
    def test_wrong_types_are_refused(self, section, key, value):
        with pytest.raises(ConfigError, match=f"{section}.{key}"):
            config_with(**{section: {key: value}})

    def test_null_in_a_required_field_says_what_to_do(self):
        with pytest.raises(ConfigError, match="n_stations is empty.*default"):
            config_with(section={"n_stations": None})

    @pytest.mark.parametrize("section,key,value", [
        ("stiv", "velocity_range_ms", [-0.5, 2.0, 5.0]),
        ("quality", "tan_phi_range", [0.1]),
        ("section", "start_px", [1, 2, 3]),
        ("flow", "downstream_px", [5]),
        ("video", "channel_weights", [0.5, 0.5]),
    ])
    def test_sequence_lengths_are_checked(self, section, key, value):
        with pytest.raises(ConfigError, match=f"{section}.{key}"):
            config_with(**{section: {key: value}})

    def test_malformed_yaml_names_the_line(self, tmp_path):
        p = write_yaml(tmp_path / "bad.yaml", """\
            video:
              path: clip.mp4
            section: {start_px: [1, 2
            """)
        with pytest.raises(ConfigError, match=r"bad\.yaml at line \d+, column \d+"):
            Config.from_yaml(p)

    def test_a_yaml_list_is_not_a_config(self, tmp_path):
        p = write_yaml(tmp_path / "list.yaml", "- video\n- section\n")
        with pytest.raises(ConfigError, match="mapping"):
            Config.from_yaml(p)

    def test_cli_reports_a_config_error_instead_of_a_traceback(self, tmp_path, capsys):
        p = write_yaml(tmp_path / "bad.yaml", "stiv: {velocity_range_ms: [0, 1, 2]}\nvideo: [\n")
        assert cli.main(["run", str(p)]) == 2
        assert "error:" in capsys.readouterr().err

    def test_config_used_round_trips(self, tmp_path):
        cfg = config_with(stiv={"velocity_range_ms": [0, 3]},
                          video={"path": str(tmp_path / "clip.mp4")},
                          output={"out_dir": str(tmp_path / "out")})
        cfg.dump_yaml(tmp_path / "used.yaml")
        again = Config.from_yaml(tmp_path / "used.yaml")
        assert again.to_dict() == cfg.to_dict()


# --------------------------------------------------------------------------- #
class TestValidation:
    @pytest.mark.parametrize("section,values,match", [
        ("video", {"max_frames": 0}, "max_frames"),
        ("video", {"fps_override": 0}, "fps_override"),
        ("quality", {"median_filter_stations": -1}, "median_filter_stations"),
        ("quality", {"tan_phi_range": [8.0, 0.05]}, "tan_phi_range"),
        ("rectify", {"m_per_px": None, "altitude_m": 60.0}, "hfov_deg"),
    ])
    def test_rejected(self, section, values, match):
        with pytest.raises(ConfigError, match=match):
            config_with(**{section: values})

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
    def test_any_median_window_is_accepted(self, n):
        # An even window is widened to the next odd one by the filter, which
        # logs it; configs (and iOS exports) with 2 or 4 must still load.
        assert config_with(quality={"median_filter_stations": n}).quality.median_filter_stations == n

    def test_tilde_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        p = write_yaml(tmp_path / "s.yaml", """\
            video: {path: ~/clip.mp4}
            rectify: {method: scale, m_per_px: 0.02}
            section: {start_px: [10, 10], end_px: [200, 10]}
            output: {out_dir: ~/out}
            """)
        cfg = Config.from_yaml(p)
        assert cfg.video.path == str(tmp_path / "clip.mp4")
        assert cfg.output.out_dir == str(tmp_path / "out")


class TestDischargeUpFront:
    """Discharge inputs used to be read only after the whole video."""

    def test_missing_bathymetry_file(self, tmp_path):
        p = write_yaml(tmp_path / "s.yaml", """\
            video: {path: clip.mp4}
            rectify: {method: scale, m_per_px: 0.02}
            section: {start_px: [10, 10], end_px: [200, 10]}
            discharge: {enabled: true, bathymetry_csv: nowhere.csv}
            """)
        with pytest.raises(ConfigError, match="bathymetry_csv not found"):
            Config.from_yaml(p)

    def test_bathymetry_is_found_next_to_the_config(self, tmp_path, monkeypatch):
        (tmp_path / "bed.csv").write_text("distance_m,depth_m\n0,0\n5,1\n10,0\n")
        p = write_yaml(tmp_path / "s.yaml", """\
            video: {path: clip.mp4}
            rectify: {method: scale, m_per_px: 0.02}
            section: {start_px: [10, 10], end_px: [200, 10]}
            discharge: {enabled: true, bathymetry_csv: bed.csv}
            """)
        monkeypatch.chdir(tmp_path.parent)
        assert Config.from_yaml(p).discharge.bathymetry_csv == str(tmp_path / "bed.csv")

    def test_unreadable_bathymetry_columns(self, tmp_path):
        (tmp_path / "bed.csv").write_text("x,y\n0,0\n1,1\n")
        with pytest.raises(ConfigError, match="distance"):
            config_with(discharge={"enabled": True, "bathymetry_csv": str(tmp_path / "bed.csv")})

    def test_depth_count_must_match_the_stations(self):
        with pytest.raises(ConfigError, match="depths_m has 3 entries.*n_stations is 11"):
            config_with(discharge={"enabled": True, "depths_m": [1.0, 2.0, 1.0]})
        config_with(discharge={"enabled": True, "depths_m": [1.0] * 11})

    def test_station_override_is_rechecked(self, tmp_path, capsys):
        cfg = config_with(discharge={"enabled": True, "depths_m": [1.0] * 11})
        cfg.section.n_stations = 9
        with pytest.raises(ConfigError, match="depths_m"):
            cfg.validate()


# --------------------------------------------------------------------------- #
class TestSharedBuilder:
    def test_defaults_come_from_the_schema(self):
        raw = survey_dict("clip.mp4")
        assert raw["stiv"]["velocity_range_ms"] == list(StiConfig.velocity_range_ms)
        assert raw["section"]["edge_trim"] == SectionConfig.edge_trim
        assert raw["section"]["n_stations"] == SectionConfig.n_stations
        assert raw["stiv"]["search_line_length_m"] == StiConfig.search_line_length_m

    def test_unknown_scale_and_section_are_null_and_refused(self):
        raw = survey_dict("clip.mp4")
        assert raw["rectify"]["m_per_px"] is None
        assert raw["section"]["start_px"] is None
        with pytest.raises(ConfigError, match="m_per_px is not set: fill in"):
            Config.from_dict(raw)
        raw["rectify"]["m_per_px"] = 0.02
        with pytest.raises(ConfigError, match="start_px and section.end_px are not set: fill in"):
            Config.from_dict(raw)

    def test_gcp_world_placeholders_are_refused(self):
        raw = survey_dict("clip.mp4", gcp_px=[[0, 0], [9, 0], [9, 9], [0, 9]],
                          start_px=[1, 1], end_px=[8, 1])
        assert raw["rectify"]["gcp_world"] == [[None, None]] * 4
        with pytest.raises(ConfigError, match=r"gcp_world\[0\]\[0\] is empty"):
            Config.from_dict(raw)

    def test_frame_step_suggestion(self):
        # tan(phi) = 1 at frame_step = m_per_px * fps / U.
        assert suggest_frame_step(0.02, 30.0, 0.5) == 1
        assert suggest_frame_step(0.05, 60.0, 1.0) == 3

    def test_path_for_config(self, tmp_path):
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "clip.mp4").touch()
        (tmp_path / "footage").mkdir()
        assert path_for_config(tmp_path / "configs" / "clip.mp4",
                               tmp_path / "configs" / "s.yaml") == "clip.mp4"
        assert path_for_config(tmp_path / "footage" / "clip.mp4",
                               tmp_path / "configs" / "s.yaml") == str(tmp_path / "footage" / "clip.mp4")


# --------------------------------------------------------------------------- #
class TestCli:
    @pytest.mark.parametrize("argv,quiet,verbose", [
        (["-q", "run", "x.yaml"], True, False),
        (["run", "-q", "x.yaml"], True, False),
        (["-v", "run", "x.yaml"], False, True),
        (["run", "x.yaml"], False, False),
    ])
    def test_verbosity_either_side_of_the_subcommand(self, argv, quiet, verbose):
        args = cli.build_parser().parse_args(argv)
        assert (args.quiet, args.verbose) == (quiet, verbose)

    def test_init_writes_placeholders_that_refuse_to_run(self, clip, tmp_path, monkeypatch, capsys):
        # Typed relative to the shell, written next to a config in another folder.
        monkeypatch.chdir(clip.parent)
        out = tmp_path / "configs" / "survey.yaml"
        out.parent.mkdir()
        assert cli.main(["init", clip.name, "-o", str(out)]) == 0
        raw = yaml.safe_load(out.read_text())
        assert raw["rectify"]["m_per_px"] is None
        assert raw["section"]["start_px"] is None
        assert raw["stiv"]["velocity_range_ms"] == list(StiConfig.velocity_range_ms)
        assert not any(k.startswith("#") for k in raw["rectify"])
        assert "altitude_m: 60.0" in out.read_text()          # a real YAML comment

        with pytest.raises(ConfigError, match="fill in"):
            Config.from_yaml(out)
        raw["rectify"]["m_per_px"] = 0.02
        raw["section"].update(start_px=[160, 50], end_px=[160, 150])
        out.write_text(yaml.safe_dump(raw))
        assert Config.from_yaml(out).video.path == str(clip.resolve())

    def test_init_refuses_to_overwrite_without_force(self, clip, tmp_path, capsys):
        out = tmp_path / "survey.yaml"
        out.write_text("# careful hand edits\n")
        assert cli.main(["init", str(clip), "-o", str(out)]) == 2
        assert "--force" in capsys.readouterr().err
        assert out.read_text() == "# careful hand edits\n"
        assert cli.main(["init", str(clip), "-o", str(out), "--force"]) == 0
        assert "careful" not in out.read_text()

    def test_pick_writes_a_path_the_config_can_find(self, clip, tmp_path, monkeypatch):
        from stiv import picker

        clicks = iter([[(10.0, 50.0), (10.0, 150.0)], [(300.0, 100.0)], []])
        frames_at = []

        def fake_pick(video, n, prompt, t_s=0.0):
            frames_at.append(t_s)
            return next(clicks)

        monkeypatch.setattr(picker, "pick_points", fake_pick)
        monkeypatch.chdir(clip.parent)
        out = tmp_path / "configs" / "survey.yaml"
        out.parent.mkdir()
        assert cli.main(["pick", clip.name, "-o", str(out), "-t", "0.5"]) == 0
        raw = yaml.safe_load(out.read_text())

        assert frames_at == [0.5, 0.5, 0.5]
        assert raw["video"]["start_s"] == 0.5          # clicked on the first analysed frame
        assert raw["section"]["start_px"] == [10.0, 50.0]
        assert raw["rectify"]["m_per_px"] is None
        with pytest.raises(ConfigError, match="m_per_px is not set"):
            Config.from_yaml(out)
        raw["rectify"]["m_per_px"] = 0.02
        out.write_text(yaml.safe_dump(raw))
        assert Config.from_yaml(out).video.path == str(clip.resolve())

    def test_pick_refuses_to_overwrite_before_any_clicking(self, clip, tmp_path, monkeypatch):
        from stiv import picker

        monkeypatch.setattr(picker, "pick_points", lambda *a, **k: pytest.fail("clicked"))
        out = tmp_path / "survey.yaml"
        out.write_text("keep me\n")
        assert cli.main(["pick", str(clip), "-o", str(out)]) == 2
        assert out.read_text() == "keep me\n"

    @pytest.mark.parametrize("fps,expect_steps", [(0.0, False), (30.0, True)])
    def test_info_with_an_unknown_frame_rate(self, monkeypatch, capsys, fps, expect_steps):
        """A file with no recorded fps must not get frame_step suggestions,
        which would all collapse to 1; it gets told to set fps_override."""
        from stiv import video
        from stiv.video import VideoInfo

        monkeypatch.setattr(video, "probe",
                            lambda path: VideoInfo(str(path), fps, 0, 3840, 2160, 0.0))
        assert cli.main(["info", "clip.h264", "--altitude-m", "60", "--hfov-deg", "84"]) == 0
        out = capsys.readouterr().out
        assert "rectify.m_per_px" in out
        assert ("video.frame_step:" in out) is expect_steps
        assert ("fps_override" in out) is not expect_steps

    def test_demo_validation_with_no_moving_station(self):
        """``stiv demo --v-max 0.1``: nothing exceeds 0.15 m/s."""
        truth = np.linspace(0.0, 0.1, 9)
        lines, rmse = cli.demo_validation(truth + 0.01, truth)
        assert rmse == pytest.approx(0.01)
        assert not any("relative" in ln and "%" in ln for ln in lines)

    def test_demo_validation_skips_unreported_stations(self):
        truth = np.array([0.5, 1.0, 0.5])
        lines, rmse = cli.demo_validation(np.array([0.5, np.nan, 0.6]), truth)
        assert math.isfinite(rmse)
        assert any("without a value : 1" in ln for ln in lines)
        assert cli.demo_validation(np.full(3, np.nan), truth)[1] is None


# --------------------------------------------------------------------------- #
class TestWizard:
    @pytest.fixture
    def nb(self):
        pytest.importorskip("ipywidgets")
        pytest.importorskip("ipyfilechooser")
        from stiv import notebook_ui

        return notebook_ui

    def test_loading_a_new_video_forgets_the_old_survey(self, nb, clip):
        import ipywidgets as W

        wz = nb.Wizard()
        wz._load_video(str(clip), is_demo=False, status=W.HTML())
        s = wz.state
        s.section_pts = [(1.0, 2.0), (3.0, 4.0)]
        s.downstream_pts = [(5.0, 6.0)]
        s.water_pts = [(0, 0), (1, 0), (1, 1)]
        s.n_stations, s.velocity_range, s.m_per_px, s.start_s = 77, (0.0, 2.0), 0.5, 1.0
        wz._pick_stage = len(wz._PICK_STAGES)

        wz._load_video(str(clip), is_demo=False, status=W.HTML())
        assert wz.state is s                     # callbacks hold this object
        assert (s.section_pts, s.downstream_pts, s.water_pts) == ([], [], [])
        assert wz._pick_stage == 0
        assert s.n_stations == SectionConfig.n_stations
        assert s.velocity_range == tuple(StiConfig.velocity_range_ms)
        assert s.m_per_px is None and s.start_s == 0.0
        assert wz._step_ready(2)[0] is False
        with pytest.raises(ConfigError):
            wz._build_config()

    def test_scale_is_not_prefilled(self, nb, clip):
        import ipywidgets as W

        wz = nb.Wizard()
        wz._load_video(str(clip), is_demo=False, status=W.HTML())
        assert wz._step_ready(1)[0] is False
        wz.state.scale_mode = "flight"
        assert "altitude" in wz._step_ready(1)[1]
        wz.state.altitude_m = 60.0
        assert "field of view" in wz._step_ready(1)[1]
        wz.state.hfov_deg = 84.0
        assert wz._step_ready(1)[0] is True

    def test_unknown_frame_rate_must_be_entered(self, nb, clip, monkeypatch):
        import ipywidgets as W

        from stiv.video import VideoInfo

        monkeypatch.setattr(nb, "probe", lambda path: VideoInfo(str(path), 0.0, 60, 320, 200, 0.0))
        wz = nb.Wizard()
        wz._load_video(str(clip), is_demo=False, status=W.HTML())
        wz.state.m_per_px = 0.05
        ok, msg = wz._step_ready(1)
        assert not ok and "frame rate" in msg
        # No suggestions rather than a column of 1s.
        assert "<td>1</td>" not in wz._frame_step_table(0.05)

        # Typed into the box the scale step shows for exactly this case.
        step1 = wz._build_step1()
        (fps_box,) = [w for w in step1.children if getattr(w, "description", "") == "Frame rate (fps)"]
        fps_box.value = "30"
        assert wz.state.fps_override == 30.0
        assert wz._step_ready(1)[0] is True
        assert "<td>3</td>" in wz._frame_step_table(0.05)      # 0.05 * 30 / 0.5 m/s
        wz.state.section_pts = [(10.0, 50.0), (10.0, 150.0)]
        wz.state.downstream_pts = [(300.0, 100.0)]
        assert wz._build_config().video.fps_override == 30.0

    def test_picker_shows_the_frame_at_the_start_time(self, nb, clip):
        import ipywidgets as W

        wz = nb.Wizard()
        wz._load_video(str(clip), is_demo=False, status=W.HTML())
        s = wz.state
        assert s.frame_t_s == 0.0
        s.section_pts = [(10.0, 50.0), (10.0, 150.0)]
        s.downstream_pts = [(300.0, 100.0)]
        s.picks_t_s = 0.0
        s.start_s = 1.0
        assert wz._picker_frame() is True        # points were clicked on another frame
        assert s.frame_t_s == 1.0
        s.picks_t_s = 1.0
        assert wz._picker_frame() is False

    def test_each_video_gets_its_own_results_folder(self, nb, clip, tmp_path):
        """One shared default folder let each survey wipe the last one's
        results: writing them first clears what the previous run there wrote."""
        import ipywidgets as W

        other = tmp_path / f"other{clip.suffix}"
        shutil.copy(clip, other)

        def out_dir(wz, video):
            wz._load_video(str(video), is_demo=False, status=W.HTML())
            s = wz.state
            s.m_per_px = 0.05
            s.section_pts, s.downstream_pts = [(10.0, 50.0), (10.0, 150.0)], [(300.0, 100.0)]
            return wz._build_config().output.out_dir

        wz = nb.Wizard()
        assert out_dir(wz, clip) != out_dir(wz, other)

        # A folder typed into step 4, or given to Wizard(), still wins.
        (box,) = [w for w in wz._build_step3().children
                  if getattr(w, "description", "") == "Save results to"]
        box.value = "mine"
        assert out_dir(wz, clip) == "mine"
        assert out_dir(nb.Wizard(out_dir="given"), other) == "given"

    def test_picker_is_shown_only_inside_the_wizard(self, nb, clip):
        """ipympl displays every figure made in interactive mode after the
        cell runs; the picker's canvas must not turn up a second time there."""
        pytest.importorskip("ipympl")
        code = textwrap.dedent("""
            import sys, matplotlib
            matplotlib.use("module://ipympl.backend_nbagg")
            import matplotlib.pyplot as plt, ipywidgets as W
            from ipympl.backend_nbagg import _Backend_ipympl
            from stiv.notebook_ui import Wizard

            wz = Wizard()
            wz._ensure_interactive_backend = plt.ion   # what %matplotlib widget leaves
            wz._load_video(sys.argv[1], is_demo=False, status=W.HTML())
            box = wz._build_step2()
            assert wz._fig.canvas in box.children
            assert wz._fig not in _Backend_ipympl._to_show, "queued for a second display"
            assert plt.isinteractive()
        """)
        subprocess.run([sys.executable, "-c", code, str(clip)],
                       cwd=Path(__file__).resolve().parents[1], check=True)

    def test_demo_runs_the_same_analysis_as_stiv_demo(self, nb, tmp_path, monkeypatch):
        from stiv.video import FrameSource

        river = tiny_river()
        monkeypatch.setattr(cli, "make_demo_river", lambda: river)
        monkeypatch.chdir(tmp_path)
        wz = nb.Wizard(out_dir=str(tmp_path / "out"))
        wz._generate_demo_video()
        got = wz._build_config()

        want = Config.from_dict(make_config(river, wz.state.video_path, tmp_path / "out",
                                            n_stations=cli.DEMO_N_STATIONS))
        a, b = got.to_dict(), want.to_dict()
        for d in (a, b):
            d["video"].pop("start_s"), d["video"].pop("duration_s")
            d["output"].pop("out_dir")
        assert a == b
        # duration_s is the whole clip in one and null in the other: same frames.
        assert FrameSource(got.video).n_frames == FrameSource(want.video).n_frames
