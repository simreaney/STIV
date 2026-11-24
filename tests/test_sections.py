"""Several cross-sections of one clip, analysed together (``sections:``).

The synthetic river flows left to right with the same profile everywhere, so
cross-sections at different points along it must each recover that profile --
and, with stabilisation off, give exactly the numbers each gives on its own.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from stiv import Config, SurveyResult, cli, run, run_survey
from stiv.config import ConfigError, section_folder, survey_dict, yaml_text
from stiv.report import MANIFEST, add_pdf_report, write_outputs, write_survey_outputs
from stiv.synthetic import SyntheticRiver, make_config, render

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = [0.0, 0.3, 0.5, 0.6, 0.5, 0.3, 0.0]


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("sections")
    river = SyntheticRiver(width_px=480, height_px=300, duration_s=3.0,
                           bank_top_px=70, bank_bottom_px=230, v_max_ms=1.0)
    return river, render(river, d / "river", progress=False)


def survey_raw(river, video, out_dir, **sections_extra):
    """Two sections, a third of the way in from each end of the frame."""
    raw = make_config(river, video, out_dir, n_stations=7)
    raw.pop("section")
    raw["section"] = {"n_stations": 7}
    raw["output"].update(save_diagnostics=False, save_sti_images=False)
    raw["sections"] = [
        {"name": "upstream", "start_px": [160, 70], "end_px": [160, 230]},
        {"name": "downstream", "start_px": [320, 70], "end_px": [320, 230]},
    ]
    for i, extra in sections_extra.items():
        raw["sections"][int(i[1:])].update(extra)
    return raw


@pytest.fixture(scope="module")
def survey(clip, tmp_path_factory):
    river, video = clip
    raw = survey_raw(river, video, tmp_path_factory.mktemp("survey_out"),
                     s1={"depths_m": DEPTHS})
    raw["stabilise"] = {"enabled": False}
    cfg = Config.from_dict(raw)
    return cfg, run_survey(cfg, progress=False)


# --------------------------------------------------------------------------- #
class TestConfig:
    def test_entries_inherit_what_they_leave_out(self, tmp_path):
        raw = survey_dict("clip.mp4", m_per_px=0.02, n_stations=12, edge_trim=0.1,
                          downstream_px=[400, 100], out_dir=str(tmp_path / "out"), sections=[
                              {"name": "a", "start_px": [10, 20], "end_px": [10, 200]},
                              {"start_px": [90, 20], "end_px": [90, 200], "downstream_px": [5, 5]},
                          ])
        raw["sections"][0]["n_stations"] = 20
        cfg = Config.from_dict(raw)
        a, b = cfg.section_configs()
        assert (a.section.name, b.section.name) == ("a", "section_2")
        assert (a.section.n_stations, b.section.n_stations) == (20, 12)
        assert a.section.edge_trim == b.section.edge_trim == 0.1
        assert a.flow.downstream_px == [400.0, 100.0]
        assert b.flow.downstream_px == [5.0, 5.0]
        assert a.output.out_dir == str(tmp_path / "out" / "a")
        assert b.output.out_dir == str(tmp_path / "out" / "section_2")
        # Each is an ordinary one-section config that validates on its own.
        for c in (a, b):
            assert c.sections is None
            Config.from_dict(c.to_dict())

    def test_depths_given_to_a_section_turn_its_discharge_on(self, tmp_path):
        csv = tmp_path / "bridge.csv"
        csv.write_text("distance_m,depth_m\n0,0\n2,1\n4,0\n")
        raw = survey_dict("clip.mp4", m_per_px=0.02, sections=[
            {"name": "a", "start_px": [10, 20], "end_px": [10, 200]},
            {"name": "b", "start_px": [90, 20], "end_px": [90, 200], "bathymetry_csv": "bridge.csv"},
        ])
        p = tmp_path / "survey.yaml"
        p.write_text(yaml_text(raw))
        a, b = Config.from_yaml(p).section_configs()
        assert not a.discharge.enabled
        assert b.discharge.enabled
        assert b.discharge.bathymetry_csv == str(csv.resolve())   # beside the config file

    def test_discharge_everywhere_needs_depths_everywhere(self):
        raw = survey_dict("clip.mp4", m_per_px=0.02, sections=[
            {"name": "a", "start_px": [10, 20], "end_px": [10, 200]},
            {"name": "b", "start_px": [90, 20], "end_px": [90, 200]},
        ])
        raw["sections"][1]["depths_m"] = [0.5] * 30
        raw["discharge"]["enabled"] = True
        with pytest.raises(ConfigError, match=r"cross-section 1 \('a'\).*bathymetry_csv or depths_m"):
            Config.from_dict(raw)

    @pytest.mark.parametrize("change,match", [
        (lambda r: r["section"].update(start_px=[1, 2], end_px=[3, 4]),
         "section.start_px and section.end_px must be left out"),
        (lambda r: r.update(sections=[]), "sections is empty"),
        (lambda r: r.update(sections={"name": "a"}), "must be a list of cross-sections"),
        (lambda r: r["sections"].append("c"), r"sections\[2\] must be a mapping"),
        (lambda r: r["sections"][1].update(name="UP stream"), "would share the results folder"),
        (lambda r: r["sections"][1].update(name="up_stream"), "both called 'up_stream'"),
        (lambda r: r["sections"][0].pop("end_px"), r"sections\[0\] \('up_stream'\) needs start_px"),
        (lambda r: r["sections"][0].update(colour="red"), r"unknown key\(s\) in config section 'sections\[0\]'"),
        (lambda r: r["sections"][1].update(n_stations="lots"), r"sections\[1\].n_stations must be a whole"),
        (lambda r: r["sections"][1].update(depths_m=[1, 2]),
         r"cross-section 2 \('down'\): discharge.depths_m has 2 entries"),
    ])
    def test_mistakes_are_refused_by_name(self, change, match):
        raw = survey_dict("clip.mp4", m_per_px=0.02, sections=[
            {"name": "up_stream", "start_px": [10, 20], "end_px": [10, 200]},
            {"name": "down", "start_px": [90, 20], "end_px": [90, 200]},
        ])
        change(raw)
        with pytest.raises(ConfigError, match=match):
            Config.from_dict(raw)

    def test_round_trips_through_yaml(self, tmp_path):
        raw = survey_dict("clip.mp4", m_per_px=0.02, sections=[
            {"name": "a", "start_px": [10, 20], "end_px": [10, 200]},
            {"name": "b", "start_px": [90, 20], "end_px": [90, 200], "downstream_px": [5, 5]},
        ])
        cfg = Config.from_dict(raw, base=tmp_path)
        p = tmp_path / "used.yaml"
        cfg.dump_yaml(p)
        text = p.read_text()
        assert "null" not in text.split("sections:")[1].split("flow:")[0]   # unset = inherited
        again = Config.from_yaml(p)
        assert again.to_dict() == cfg.to_dict()
        assert [c.to_dict() for c in again.section_configs()] == \
               [c.to_dict() for c in cfg.section_configs()]

    def test_a_single_section_reads_as_it_always_has(self):
        one = survey_dict("clip.mp4", m_per_px=0.02, downstream_px=[1, 1], sections=[
            {"name": "only", "start_px": [10, 20], "end_px": [10, 200], "downstream_px": [7, 7]}])
        assert "sections" not in one
        assert one["section"]["name"] == "only"
        assert one["flow"]["downstream_px"] == [7.0, 7.0]
        cfg = Config.from_dict(one)
        assert "sections" not in cfg.to_dict()
        assert cfg.section_configs() == [cfg]

    def test_section_folders_are_safe_names(self):
        assert section_folder("Bridge pool #2") == "Bridge_pool_2"
        assert section_folder("../up") == "up"
        assert section_folder("...") == "section"
        assert section_folder("sti") == "section_sti"      # the STI images' own folder


# --------------------------------------------------------------------------- #
class TestAnalysis:
    def test_every_section_recovers_the_profile(self, survey, clip):
        river, _ = clip
        cfg, sv = survey
        assert isinstance(sv, SurveyResult)
        assert sv.names == ["upstream", "downstream"]
        _, truth = river.truth_profile(7)
        moving = truth > 0.2
        for r in sv:
            err = r.velocities[moving] - truth[moving]
            assert np.sqrt(np.mean(err ** 2)) < 0.06
            assert r.n_frames == sv[0].n_frames      # one pass over the video

    def test_matches_each_section_run_alone(self, survey):
        cfg, sv = survey
        for c, r in zip(cfg.section_configs(), sv):
            alone = run(c, progress=False)
            assert np.array_equal(alone.velocities, r.velocities, equal_nan=True)
            assert alone.warnings == r.warnings

    def test_discharge_only_where_there_are_depths(self, survey):
        _, sv = survey
        assert sv[0].discharge is None
        assert sv[1].discharge is not None and sv[1].discharge.discharge_m3s > 0
        table = sv.to_frame()
        assert list(table["section"]) == ["upstream", "downstream"]
        assert np.isnan(table["discharge_m3s"][0]) and table["discharge_m3s"][1] > 0

    def test_a_section_pointed_upstream_is_corrected_alone(self, clip, tmp_path):
        river, video = clip
        # The second section's own downstream point is on the upstream side.
        raw = survey_raw(river, video, tmp_path, s1={"downstream_px": [0, 150]})
        raw["stabilise"] = {"enabled": False}
        sv = run_survey(Config.from_dict(raw), progress=False)
        flipped = [any("opposite to the assumed downstream" in w for w in r.warnings) for r in sv]
        assert flipped == [False, True]
        assert all(np.nanmedian(r.velocities) > 0 for r in sv)
        assert sv.shared_warnings == [w for w in sv[0].warnings if w in sv.shared_warnings]
        assert "opposite to the assumed downstream" in sv.summary()

    def test_run_wants_run_survey_for_several(self, clip, tmp_path):
        river, video = clip
        cfg = Config.from_dict(survey_raw(river, video, tmp_path))
        with pytest.raises(ConfigError, match="run_survey"):
            run(cfg, progress=False)

    def test_a_one_section_config_is_a_survey_of_one(self, clip, tmp_path):
        river, video = clip
        raw = make_config(river, video, tmp_path, n_stations=7)
        raw["stabilise"] = {"enabled": False}
        cfg = Config.from_dict(raw)
        sv = run_survey(cfg, progress=False)
        assert len(sv) == 1 and sv[0].config is cfg
        assert sv.summary() == sv[0].summary()
        assert np.array_equal(sv[0].velocities, run(cfg, progress=False).velocities,
                              equal_nan=True)

    def test_stabilised_survey_stays_close_to_each_alone(self, clip, tmp_path):
        """The stabiliser keeps off every section's lines, so it tracks a few
        different bank features: close, but not to the last bit."""
        river, video = clip
        cfg = Config.from_dict(survey_raw(river, video, tmp_path))
        cfg.stabilise.enabled = True
        sv = run_survey(cfg, progress=False)
        for c, r in zip(cfg.section_configs(), sv):
            alone = run(c, progress=False)
            assert np.nanmax(np.abs(alone.velocities - r.velocities)) < 0.02


# --------------------------------------------------------------------------- #
class TestOutputs:
    def test_layout(self, survey, tmp_path):
        cfg, sv = survey
        cfg = copy.deepcopy(cfg)
        cfg.output.out_dir = str(tmp_path / "out")
        out = write_survey_outputs(sv, cfg)
        top = {p.name for p in out.iterdir()}
        assert {"sections.csv", "report.txt", "summary.json", "config_used.yaml",
                "upstream", "downstream", MANIFEST} <= top
        for name in ("upstream", "downstream"):
            sub = out / name
            assert (sub / "velocity.csv").is_file() and (sub / "config_used.yaml").is_file()
            # A section's own config re-runs that section alone.
            assert Config.from_yaml(sub / "config_used.yaml").section.name == name
        assert (out / "downstream" / "discharge_panels.csv").is_file()
        assert not (out / "upstream" / "discharge_panels.csv").exists()
        assert Config.from_yaml(out / "config_used.yaml").sections is not None

        summary = json.loads((out / "summary.json").read_text())
        assert summary["n_sections"] == 2
        assert [s["section"] for s in summary["sections"]] == ["upstream", "downstream"]
        assert [s["folder"] for s in summary["sections"]] == ["upstream", "downstream"]
        manifest = json.loads((out / MANIFEST).read_text())
        assert manifest["sections"] == ["upstream", "downstream"]
        assert "upstream" in (out / "report.txt").read_text()

    def test_figures_and_one_pdf_for_the_survey(self, survey, tmp_path):
        cfg, sv = survey
        cfg = copy.deepcopy(cfg)
        cfg.output.out_dir = str(tmp_path / "out")
        cfg.output.save_diagnostics = True
        cfg.output.pdf_report = True
        out = write_survey_outputs(sv, cfg)
        assert (out / "overview.png").is_file() and (out / "velocity_map.png").is_file()
        assert (out / "report.pdf").is_file()
        assert (out / "upstream" / "velocity_map.png").is_file()
        assert not (out / "upstream" / "report.pdf").exists()      # the survey's covers it
        assert "/Title (STIV report - 2 cross-sections)" in (out / "report.pdf").read_bytes().decode("latin-1")

    def test_pdf_added_later_keeps_the_section_folders_listed(self, survey, tmp_path):
        cfg, sv = survey
        cfg = copy.deepcopy(cfg)
        cfg.output.out_dir = str(tmp_path / "out")
        out = write_survey_outputs(sv, cfg)
        add_pdf_report(sv, cfg)
        manifest = json.loads((out / MANIFEST).read_text())
        assert "report.pdf" in manifest["files"]
        assert manifest["sections"] == ["upstream", "downstream"]

    def test_the_next_run_clears_section_folders_and_nothing_else(self, survey, tmp_path):
        cfg, sv = survey
        cfg = copy.deepcopy(cfg)
        cfg.output.out_dir = str(tmp_path / "out")
        out = write_survey_outputs(sv, cfg)
        (out / "upstream" / "field notes.txt").write_text("mine")

        # The same folder, now a one-section run: the old section folders go,
        # except for what STIV did not write.
        single = sv[0].config
        single = copy.deepcopy(single)
        single.output.out_dir = str(out)
        write_outputs(sv[0], single)
        assert not (out / "downstream").exists()
        assert (out / "upstream" / "field notes.txt").read_text() == "mine"
        assert not (out / "upstream" / "velocity.csv").exists()
        assert not (out / "sections.csv").exists()
        assert (out / "velocity.csv").is_file()


# --------------------------------------------------------------------------- #
class TestCli:
    def test_run_writes_every_section(self, clip, tmp_path, capsys):
        river, video = clip
        raw = survey_raw(river, video, tmp_path / "out")
        p = tmp_path / "survey.yaml"
        p.write_text(yaml_text(raw))
        assert cli.main(["-q", "run", str(p), "-n", "5", "--no-stabilise"]) == 0
        printed = capsys.readouterr().out
        assert "2 cross-sections, analysed together" in printed
        for name in ("upstream", "downstream"):
            used = Config.from_yaml(tmp_path / "out" / name / "config_used.yaml")
            assert used.section.n_stations == 5          # -n reaches every section

    def test_init_lays_out_placeholders(self, clip, tmp_path, capsys):
        _, video = clip
        out = tmp_path / "survey.yaml"
        assert cli.main(["init", str(video), "-o", str(out), "--sections", "3"]) == 0
        raw = yaml.safe_load(out.read_text())
        assert [e["name"] for e in raw["sections"]] == ["section_1", "section_2", "section_3"]
        assert all(e["start_px"] is None for e in raw["sections"])
        assert "--sections 3" in out.read_text()
        with pytest.raises(ConfigError):
            Config.from_yaml(out)
        assert "3 cross-sections" in capsys.readouterr().out

    def test_pick_clicks_each_section_then_one_downstream_point(self, clip, tmp_path, monkeypatch):
        from stiv import picker

        _, video = clip
        clicks = iter([[(10.0, 50.0), (10.0, 150.0)], [(200.0, 50.0), (200.0, 150.0)],
                       [(400.0, 100.0)], []])
        calls = []

        def fake_pick(video, n, prompt, t_s=0.0, shown=()):
            calls.append((prompt, [list(map(list, line)) for line in shown]))
            return next(clicks)

        monkeypatch.setattr(picker, "pick_points", fake_pick)
        monkeypatch.setattr(picker, "confirm_water_outline", lambda *a, **k: False)
        out = tmp_path / "survey.yaml"
        assert cli.main(["pick", str(video), "-o", str(out), "--sections", "2",
                         "--m-per-px", "0.02"]) == 0
        raw = yaml.safe_load(out.read_text())
        assert [(e["name"], e["start_px"]) for e in raw["sections"]] == \
               [("section_1", [10.0, 50.0]), ("section_2", [200.0, 50.0])]
        assert raw["flow"]["downstream_px"] == [400.0, 100.0]
        assert "cross-section 2 of 2" in calls[1][0]
        assert calls[1][1] == [[[10.0, 50.0], [10.0, 150.0]]]      # the first one is shown
        assert "ALL the cross-sections" in calls[2][0]
        Config.from_yaml(out)                                       # ready to run

    def test_zero_sections_is_refused_before_clicking(self, clip, tmp_path, monkeypatch):
        from stiv import picker

        _, video = clip
        monkeypatch.setattr(picker, "pick_points", lambda *a, **k: pytest.fail("clicked"))
        assert cli.main(["pick", str(video), "-o", str(tmp_path / "s.yaml"), "--sections", "0"]) == 2


# --------------------------------------------------------------------------- #
def _find(widget, text, kind=None):
    """The first widget under ``widget`` whose description (or label) holds ``text``."""
    if (text in str(getattr(widget, "description", ""))
            and (kind is None or isinstance(widget, kind))):
        return widget
    for child in getattr(widget, "children", ()):
        hit = _find(child, text, kind)
        if hit is not None:
            return hit
    return None


class TestWizard:
    @pytest.fixture
    def nb(self):
        pytest.importorskip("ipywidgets")
        pytest.importorskip("ipyfilechooser")
        from stiv import notebook_ui

        return notebook_ui

    @staticmethod
    def _two_sections(nb, video, out_dir):
        import ipywidgets as W

        wz = nb.Wizard(out_dir=str(out_dir))
        wz._load_video(str(video), is_demo=False, status=W.HTML())
        s = wz.state
        s.m_per_px = 0.02
        s.section_pts = [(160.0, 70.0), (160.0, 230.0)]
        s.downstream_pts = [(470.0, 150.0)]
        s.more_sections = [{"name": None, "pts": [(320.0, 70.0), (320.0, 230.0)],
                            "bathymetry_csv": None, "extra": {}}]
        s.n_stations, s.stabilise, s.pdf_report = 7, False, False
        s.search_line_length_m, s.velocity_range = 3.0, (-0.5, 4.0)
        wz._pick_stage = len(wz._PICK_STAGES)
        return wz

    def test_adding_sections_on_the_picker(self, nb, clip):
        """Driven through the real ipympl canvas, in a process of its own."""
        pytest.importorskip("ipympl")
        _, video = clip
        code = textwrap.dedent("""
            import sys, matplotlib
            matplotlib.use("module://ipympl.backend_nbagg")
            import matplotlib.pyplot as plt, ipywidgets as W
            from matplotlib.backend_bases import MouseEvent
            from stiv.notebook_ui import Wizard

            def find(w, text):
                if text in str(getattr(w, "description", "")):
                    return w
                for c in getattr(w, "children", ()):
                    hit = find(c, text)
                    if hit is not None:
                        return hit

            wz = Wizard()
            wz._ensure_interactive_backend = plt.ion
            wz._load_video(sys.argv[1], is_demo=False, status=W.HTML())
            s = wz.state
            s.m_per_px = 0.02
            s.section_pts = [(160.0, 70.0), (160.0, 230.0)]
            s.downstream_pts = [(470.0, 150.0)]
            wz._pick_stage = len(wz._PICK_STAGES)
            box = wz._build_step2()
            ax, canvas = wz._ax, wz._fig.canvas

            def click(x, y):
                X, Y = ax.transData.transform((x, y))
                canvas.callbacks.process("button_press_event",
                                         MouseEvent("button_press_event", canvas, X, Y, button=1))

            add, confirm, cancel = find(box, "Add another"), find(box, "Confirm"), find(box, "Cancel")
            remove = find(box, "Remove the last")
            assert add.layout.display != "none" and remove.layout.display == "none"
            add.click()
            assert add.layout.display == "none" and cancel.layout.display != "none"
            click(320, 70)
            assert confirm.disabled
            ok, msg = wz._step_ready(2)
            assert not ok and "Finish the cross-section" in msg
            click(320, 230)
            click(400, 100)                     # a third click is ignored
            confirm.click()
            assert [sec["pts"] for sec in s.more_sections] == [[(320.0, 70.0), (320.0, 230.0)]]
            assert wz._step_ready(2)[0] and remove.layout.display != "none"

            add.click(); click(10, 10); cancel.click()
            assert len(s.more_sections) == 1 and wz._adding_pts is None
            cfg = wz._build_config()
            assert [c.section.name for c in cfg.section_configs()] == ["section_1", "section_2"]

            remove.click()
            assert s.more_sections == [] and wz._build_config().sections is None
        """)
        subprocess.run([sys.executable, "-c", code, str(video)], cwd=ROOT, check=True)

    def test_settings_name_the_sections_and_give_each_its_depths(self, nb, clip, tmp_path):
        import ipywidgets as W

        _, video = clip
        wz = self._two_sections(nb, video, tmp_path / "out")
        step = wz._build_step3()
        _find(step, "Cross-section 1", W.Text).value = "Bridge"
        _find(step, "Cross-section 2", W.Text).value = "Ford"
        assert "Bridge" in str(_find(step, "Depths for", W.Dropdown).options)

        csv = tmp_path / "ford.csv"
        csv.write_text("distance_m,depth_m\n0,0\n1.6,0.6\n3.2,0\n")
        _find(step, "Also compute discharge", W.Checkbox).value = True
        assert "at least one cross-section" in wz._step_ready(3)[1]
        wz.state.more_sections[0]["bathymetry_csv"] = str(csv)
        assert wz._step_ready(3)[0]
        cfg = wz._build_config()
        bridge, ford = cfg.section_configs()
        assert (bridge.section.name, ford.section.name) == ("Bridge", "Ford")
        assert not bridge.discharge.enabled and ford.discharge.bathymetry_csv == str(csv)
        assert ford.output.out_dir.endswith("Ford")

        _find(step, "Cross-section 2", W.Text).value = "bridge"
        assert "same name" in wz._step_ready(3)[1]

    def test_runs_and_shows_every_section(self, nb, clip, tmp_path):
        import ipywidgets as W

        from stiv.analysis import SurveyResult

        _, video = clip
        wz = self._two_sections(nb, video, tmp_path / "out")
        wz._run_analysis()
        assert wz.step == 5, wz._error.value
        assert isinstance(wz.state.result, SurveyResult)
        assert (tmp_path / "out" / "section_2" / "velocity.csv").is_file()
        view = wz._build_step5()
        picker = _find(view, "", W.ToggleButtons)
        assert list(picker.options) == ["All cross-sections", "section_1", "section_2"]
        assert "sections.csv" in str([c.value for c in view.children if isinstance(c, W.VBox)
                                      for c in c.children if isinstance(c, W.HTML)])
        picker.value = "section_2"
        holder = view.children[3]
        assert "section_2" in holder.children[0].children[0].value

    def test_reopens_a_survey_with_its_sections(self, nb, clip, tmp_path):
        import ipywidgets as W

        river, video = clip
        raw = survey_raw(river, video, tmp_path / "results",
                         s1={"n_stations": 9, "downstream_px": [470, 150]})
        path = tmp_path / "survey.yaml"
        path.write_text(yaml_text(raw))
        wz = nb.Wizard()
        status = W.HTML()
        wz._reopen_survey(str(path), status)
        assert "Reopened" in status.value and wz.step == 4
        s = wz.state
        assert s.section_name == "upstream" and s.section_pts == [(160.0, 70.0), (160.0, 230.0)]
        assert [sec["name"] for sec in s.more_sections] == ["downstream"]
        assert "2 marked" in wz._summary_html()

        # What the wizard has no control for comes through unchanged.
        up, down = wz._build_config().section_configs()
        assert (up.section.n_stations, down.section.n_stations) == (7, 9)
        assert down.flow.downstream_px == [470.0, 150.0]

        # Down to one section: a valid one-section survey again.
        s.more_sections.clear()
        cfg = wz._build_config()
        assert cfg.sections is None and cfg.section.start_px == [160.0, 70.0]

    def test_a_one_section_survey_can_grow(self, nb, clip, tmp_path):
        import ipywidgets as W

        river, video = clip
        raw = make_config(river, video, tmp_path / "results", n_stations=7)
        path = tmp_path / "survey.yaml"
        Config.from_dict(raw).dump_yaml(path)
        wz = nb.Wizard()
        wz._reopen_survey(str(path), W.HTML())
        wz.state.more_sections.append({"name": "second", "pts": [(320.0, 70.0), (320.0, 230.0)],
                                       "bathymetry_csv": None, "extra": {}})
        cfg = wz._build_config()
        assert [c.section.name for c in cfg.section_configs()] == ["synthetic", "second"]
        assert cfg.section.start_px is None


def test_the_example_survey_is_valid():
    cfg = Config.from_yaml(ROOT / "examples" / "multi_section.yaml")
    sections = cfg.section_configs()
    assert [c.section.name for c in sections] == ["gauge", "riffle", "pool_tail"]
    assert [c.section.n_stations for c in sections] == [40, 30, 40]
    assert [c.discharge.enabled for c in sections] == [True, False, False]
    assert sections[0].discharge.bathymetry_csv.endswith("bathymetry_example.csv")
