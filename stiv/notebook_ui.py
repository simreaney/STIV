"""A guided, step-by-step Jupyter interface for building and running a STIV survey.

Aimed at students setting up their first survey: one decision per screen, the
running choices are always visible, sensible defaults are pre-filled -- except
the scale, which has no sensible default and must be typed in -- and nothing is
analysed until "Run analysis" is pressed. It is a thin widget layer
over the same public API used by the CLI (:mod:`stiv.config`, :mod:`stiv.analysis`,
:mod:`stiv.report`) -- it does not duplicate any pipeline logic.

Needs the ``notebook`` extra::

    pip install -e '.[notebook]'

Usage, in a notebook cell::

    from stiv.notebook_ui import launch
    wizard = launch()

The interactive cross-section picker (step 3) needs the ``ipympl`` backend;
this module switches to it automatically the first time that step is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import (
    FRAME_STEP_SPEEDS_MS,
    Config,
    ConfigError,
    DischargeConfig,
    OutputConfig,
    RectifyConfig,
    SectionConfig,
    StabiliseConfig,
    StiConfig,
    VideoConfig,
    default_duration_s,
    suggest_frame_step,
    survey_dict,
)
from .video import VideoInfo, probe, read_frame_at

try:
    import ipywidgets as W
    from IPython.display import Image, clear_output, display
    from ipyfilechooser import FileChooser
except ImportError as exc:  # pragma: no cover - exercised by missing-deps installs
    raise ImportError(
        "the interactive notebook UI needs ipywidgets, ipympl and ipyfilechooser. "
        "Install them with `pip install -e '.[notebook]'`, then restart the kernel."
    ) from exc

# Palette matches stiv.plotting, so the wizard and the report figures agree.
ORANGE = "#eb6834"
BLUE = "#2a78d6"
GREEN = "#1baf7a"
CRITICAL = "#d03b3b"
WARNING = "#fab219"
MUTED = "#898781"


@dataclass
class _State:
    """Everything the wizard knows about the survey being built.

    Defaults come from the config schema, so a survey built here analyses the
    same way as one started with ``stiv init`` or ``stiv pick``.
    """

    #: The results folder the user chose, or None for one named after the
    #: video (see :meth:`Wizard._out_dir`).
    out_dir: Optional[str] = None

    video_path: Optional[str] = None
    is_demo: bool = False
    info: Optional[VideoInfo] = None
    frame_bgr: Optional[np.ndarray] = None
    #: Time in the video of ``frame_bgr``, s.  The picker must show the frame
    #: the analysis starts on: the stabiliser measures drift from there.
    frame_t_s: float = 0.0

    start_s: float = VideoConfig.start_s
    duration_s: float = 20.0
    #: The recording frame rate, when the file does not say (or says wrongly).
    fps_override: Optional[float] = None

    scale_mode: str = "manual"  # "manual" | "flight"
    # No pre-filled scale: every velocity is proportional to it, and a
    # plausible-looking default that nobody checked is worse than a blank.
    m_per_px: Optional[float] = None
    altitude_m: Optional[float] = None
    hfov_deg: Optional[float] = None

    section_pts: list = field(default_factory=list)
    downstream_pts: list = field(default_factory=list)
    water_pts: list = field(default_factory=list)
    #: Time of the frame the points above were clicked or confirmed on, s.
    picks_t_s: Optional[float] = None

    n_stations: int = SectionConfig.n_stations
    velocity_range: tuple = tuple(StiConfig.velocity_range_ms)
    stabilise: bool = StabiliseConfig.enabled
    frame_step: Optional[int] = None
    window_px: int = StiConfig.window_px
    edge_trim: float = SectionConfig.edge_trim
    search_line_length_m: float = StiConfig.search_line_length_m
    section_name: Optional[str] = None

    discharge_enabled: bool = False
    bathymetry_csv: Optional[str] = None
    alpha: float = DischargeConfig.alpha
    bank_coefficient: float = DischargeConfig.bank_coefficient

    result: object = None
    out_path: Optional[Path] = None

    def clear(self) -> None:
        """Back to a blank survey, keeping only a results folder the user chose.

        Done in place rather than by replacing the object, because widget
        callbacks built earlier hold a reference to it.
        """
        self.__dict__.update(_State(out_dir=self.out_dir).__dict__)


def _parse_positive(text: str) -> Optional[float]:
    """A strictly positive number typed into a text box, or None."""
    try:
        x = float(str(text).strip())
    except ValueError:
        return None
    return x if np.isfinite(x) and x > 0 else None


def _fmt_opt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:g}"


# --------------------------------------------------------------------------- #
class Wizard:
    """A linear, six-step survey builder rendered as one ipywidgets panel."""

    _TITLES = ("Video", "Scale", "Cross-section", "Settings", "Run", "Results")
    _NEXT_LABELS = (
        "Next →", "Next →", "Next →", "Next →",
        "Run analysis ▶", "Start a new survey ↻",
    )
    _PICK_STAGES = (
        {"attr": "section_pts", "min": 2, "max": 2,
         "prompt": "Click the two ends of the measurement cross-section: left bank, then right bank."},
        {"attr": "downstream_pts", "min": 1, "max": 1,
         "prompt": "Click one point further DOWNSTREAM of the section. This tells STIV which way the flow runs."},
        {"attr": "water_pts", "min": 0, "max": None,
         "prompt": "Optional: click around the edge of the water, then Confirm. This stops the camera-shake "
                   "correction from locking onto the moving river. Press Confirm with no points to skip."},
    )
    _PICK_COLOURS = {"section_pts": ORANGE, "downstream_pts": BLUE, "water_pts": GREEN}

    def __init__(self, out_dir: Optional[str] = None):
        self.state = _State(out_dir=out_dir)
        self.step = 0
        self._pick_stage = 0
        self._step2_box = None
        self._step2_key = None
        self._fig = None
        self._ax = None
        self._preview_box = W.Output()

        self._crumbs = W.HTML()
        self._error = W.HTML()
        self._body = W.VBox([])
        self._back_btn = W.Button(description="← Back", layout=W.Layout(width="100px"))
        self._next_btn = W.Button(description=self._NEXT_LABELS[0], button_style="primary",
                                   layout=W.Layout(width="180px"))
        self._back_btn.on_click(self._on_back)
        self._next_btn.on_click(self._on_next)
        nav = W.HBox([self._back_btn, self._next_btn],
                      layout=W.Layout(justify_content="space-between", margin="14px 0 0 0"))
        self.root = W.VBox(
            [self._crumbs, self._error, self._body, nav],
            layout=W.Layout(border=f"1px solid #ddd", padding="16px", max_width="920px"),
        )
        self._refresh()

    def display(self) -> "Wizard":
        display(self.root)
        return self

    def reset(self) -> None:
        self._forget_survey()
        self.step = 0
        self._refresh()

    def _forget_survey(self) -> None:
        """Drop everything tied to the current video: picks, scale, settings."""
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
        self.state.clear()
        self._pick_stage = 0
        self._step2_box = None
        self._step2_key = None
        self._fig = self._ax = None

    # ------------------------------------------------------------------ #
    # navigation
    # ------------------------------------------------------------------ #
    def _on_next(self, _btn) -> None:
        if self.step == 4:
            self._run_analysis()
            return
        if self.step == 5:
            self.reset()
            return
        ok, msg = self._step_ready(self.step)
        if not ok:
            self._show_error(msg)
            return
        self._show_error("")
        self.step += 1
        self._refresh()

    def _on_back(self, _btn) -> None:
        if self.step > 0:
            self._show_error("")
            self.step -= 1
            self._refresh()

    def _show_error(self, msg: str) -> None:
        self._error.value = (
            f"<div style='background:#fdecec;border:1px solid {CRITICAL};color:{CRITICAL};"
            f"padding:6px 10px;border-radius:4px;margin-bottom:8px'>{msg}</div>"
            if msg else ""
        )

    def _refresh(self) -> None:
        self._crumbs.value = self._crumbs_html()
        try:
            body = self._current_body()
        except Exception as exc:  # keeps a bad widget build from taking down the kernel
            body = W.HTML(f"<div style='color:{CRITICAL}'>Could not build this step: {exc}</div>")
        self._body.children = [body]
        self._back_btn.disabled = self.step == 0
        self._next_btn.description = self._NEXT_LABELS[self.step]

    def _crumbs_html(self) -> str:
        parts = []
        for i, title in enumerate(self._TITLES):
            if i == self.step:
                parts.append(f"<b style='color:{ORANGE}'>{i + 1}. {title}</b>")
            elif i < self.step:
                parts.append(f"<span style='color:{GREEN}'>{i + 1}. {title} ✓</span>")
            else:
                parts.append(f"<span style='color:#aaa'>{i + 1}. {title}</span>")
        return "<div style='margin-bottom:10px'>" + " &nbsp;→&nbsp; ".join(parts) + "</div>"

    def _current_body(self):
        return (
            self._build_step0, self._build_step1, self._build_step2,
            self._build_step3, self._build_step4, self._build_step5,
        )[self.step]()

    def _step_ready(self, step: int) -> tuple[bool, str]:
        s = self.state
        if step == 0:
            if not s.video_path or s.info is None:
                return False, "Load a video, or generate the demo river, before continuing."
            return True, ""
        if step == 1:
            m = self._resolved_m_per_px()
            if not m or m <= 0:
                if s.scale_mode == "manual":
                    return False, "Enter the ground resolution (metres per pixel) before continuing."
                if s.altitude_m is None:
                    return False, "Enter the flight altitude above the water surface before continuing."
                return False, "Enter your camera's horizontal field of view (between 0 and 180 degrees) before continuing."
            if self._fps() is None:
                return False, ("This video does not record its frame rate. Enter the rate it was recorded "
                               "at (frames per second) below the time window before continuing.")
            if s.duration_s <= 0:
                return False, "Duration must be greater than zero."
            return True, ""
        if step == 2:
            if len(s.section_pts) != 2:
                return False, "Click both ends of the cross-section, then press Confirm."
            if len(s.downstream_pts) != 1:
                return False, "Click a point downstream of the section, then press Confirm."
            if self._pick_stage < len(self._PICK_STAGES):
                return False, "Confirm each picking stage (or skip the water outline), then press Next."
            return True, ""
        if step == 3:
            if s.discharge_enabled and not s.bathymetry_csv:
                return False, "Choose a bathymetry CSV file, or turn discharge off."
            return True, ""
        return True, ""

    def _resolved_m_per_px(self) -> Optional[float]:
        s = self.state
        if s.info is None:
            return None
        if s.scale_mode == "manual":
            return s.m_per_px
        if s.altitude_m is None or s.hfov_deg is None or not s.hfov_deg < 180.0:
            return None
        r = RectifyConfig(method="scale", altitude_m=s.altitude_m, hfov_deg=s.hfov_deg)
        return r.resolve_m_per_px(s.info.width)

    def _out_dir(self) -> str:
        """The results folder: the one chosen, else one named after the video.

        Named as ``stiv init`` names it after the survey file.  A folder shared
        by every survey would let each one wipe the last one's results, since
        writing them first clears what the previous run there wrote.
        """
        s = self.state
        if s.out_dir:
            return s.out_dir
        if not s.video_path:
            return OutputConfig.out_dir
        from .cli import _default_out_dir

        return _default_out_dir(Path(s.video_path))

    def _fps(self) -> Optional[float]:
        """The frame rate the analysis will use: the user's, else the file's."""
        s = self.state
        if s.fps_override:
            return s.fps_override
        if s.info is not None and s.info.fps > 0:
            return s.info.fps
        return None

    def _frame_step_table(self, m_per_px: float) -> str:
        fps = self._fps()
        if fps is None:
            # Every suggestion would collapse to 1 without one.
            return ("<p style='color:#888;font-size:0.9em'>Enter the frame rate below to get "
                    "<code>frame_step</code> suggestions.</p>")
        rows = []
        for v in FRAME_STEP_SPEEDS_MS:
            step = suggest_frame_step(m_per_px, fps, v)
            rows.append(f"<tr><td>{v:.1f} m/s</td><td>{step}</td></tr>")
        return (
            "<table style='margin-top:6px;font-size:0.9em'>"
            "<tr><th style='text-align:left'>expected speed</th>"
            "<th style='text-align:left'>suggested frame_step</th></tr>"
            + "".join(rows) + "</table>"
        )

    # ------------------------------------------------------------------ #
    # step 0 -- video
    # ------------------------------------------------------------------ #
    def _build_step0(self):
        s = self.state
        mode = W.ToggleButtons(
            options=[("Use my own video", "own"), ("Try the demo river", "demo")],
            value="demo" if s.is_demo else "own",
        )

        fc = FileChooser(str(Path.cwd()))
        fc.filter_pattern = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.mkv", "*.m4v"]
        fc.title = "Choose a drone video file"
        own_status = W.HTML()

        def _on_file_selected(chooser):
            self._load_video(chooser.selected, is_demo=False, status=own_status)

        fc.register_callback(_on_file_selected)
        own_box = W.VBox([W.HTML("<p>Browse to a video file on disk, then wait for it to load.</p>"),
                           fc, own_status])

        demo_status = W.HTML()
        demo_btn = W.Button(description="Generate demo river", button_style="primary")

        def _on_demo(_btn):
            demo_btn.disabled = True
            demo_status.value = "<i>Rendering a synthetic river with a known velocity profile…</i>"
            try:
                self._generate_demo_video()
                demo_status.value = f"<span style='color:{GREEN}'>Demo video ready — see the preview below.</span>"
            except Exception as exc:
                demo_status.value = f"<span style='color:{CRITICAL}'>Could not render the demo: {exc}</span>"
            demo_btn.disabled = False

        demo_btn.on_click(_on_demo)
        demo_box = W.VBox([
            W.HTML(
                "<p>Renders a short synthetic river with a <b>known</b> velocity profile and mild "
                "simulated drone drift &mdash; a safe way to learn the workflow before using real "
                "footage. The scale and cross-section are filled in for you in the next steps, but "
                "you can still edit them.</p>"
            ),
            demo_btn, demo_status,
        ])

        stack = W.Stack(children=[own_box, demo_box], selected_index=1 if s.is_demo else 0)
        W.jslink((mode, "index"), (stack, "selected_index"))

        def _on_mode(change):
            s.is_demo = change["new"] == "demo"

        mode.observe(_on_mode, names="value")

        self._refresh_preview()

        return W.VBox([W.HTML("<h4>1. Choose a video</h4>"), mode, stack, self._preview_box])

    def _refresh_preview(self) -> None:
        with self._preview_box:
            clear_output(wait=True)
            s = self.state
            if s.info is None:
                print("No video loaded yet.")
                return
            print(str(s.info))
            if s.frame_bgr is not None:
                ok, buf = cv2.imencode(".png", s.frame_bgr)
                if ok:
                    display(Image(data=buf.tobytes(), width=560))

    def _load_video(self, path: str, is_demo: bool, status) -> None:
        try:
            info = probe(path)
            frame = read_frame_at(path, 0.0)
        except ConfigError as exc:
            status.value = f"<span style='color:{CRITICAL}'>{exc}</span>"
            return
        # A new video is a new survey: the previous video's picks would be
        # off-frame or on the wrong features, and its scale, station count and
        # velocity range were chosen for a different river.
        self._forget_survey()
        s = self.state
        s.video_path = str(path)
        s.is_demo = is_demo
        s.info = info
        s.frame_bgr = frame
        s.frame_t_s = 0.0
        s.duration_s = default_duration_s(info.duration_s)
        status.value = f"<span style='color:{GREEN}'>Loaded: {info}</span>" + (
            "" if info.fps > 0 else
            f" <span style='color:{WARNING}'>The file does not record its frame rate; "
            f"you will be asked for it in the next step.</span>"
        )
        self._refresh_preview()

    def _generate_demo_video(self) -> None:
        # The same river and the same config as `stiv demo`, so the wizard's
        # demo reproduces the numbers the README quotes for it.
        from .cli import DEMO_N_STATIONS, make_demo_river
        from .synthetic import make_config, render

        river = make_demo_river()
        out_dir = Path.cwd() / "stiv_demo_data"
        video_path = render(river, out_dir / "synthetic_river.mp4", progress=False)
        self._load_video(str(video_path), is_demo=True, status=W.HTML())

        s = self.state
        raw = make_config(river, video_path, out_dir / "analysis", n_stations=DEMO_N_STATIONS)
        s.scale_mode = "manual"
        s.m_per_px = raw["rectify"]["m_per_px"]
        s.section_pts = [tuple(raw["section"]["start_px"]), tuple(raw["section"]["end_px"])]
        s.downstream_pts = [tuple(raw["flow"]["downstream_px"])]
        s.water_pts = []
        s.picks_t_s = s.frame_t_s
        s.section_name = raw["section"]["name"]
        s.n_stations = raw["section"]["n_stations"]
        s.frame_step = raw["video"].get("frame_step")
        s.stabilise = raw["stabilise"]["enabled"]
        s.search_line_length_m = raw["stiv"]["search_line_length_m"]
        s.window_px = raw["stiv"]["window_px"]
        s.velocity_range = tuple(raw["stiv"]["velocity_range_ms"])
        self._pick_stage = len(self._PICK_STAGES)

    # ------------------------------------------------------------------ #
    # step 1 -- scale
    # ------------------------------------------------------------------ #
    def _build_step1(self):
        s = self.state
        info = s.info

        mode = W.ToggleButtons(
            options=[("I know the ground resolution", "manual"), ("Calculate from flight altitude", "flight")],
            value=s.scale_mode,
        )
        # Text boxes rather than number boxes, so they can start out blank:
        # the scale has no safe default and must be entered, not accepted.
        m_per_px_box = W.Text(value=_fmt_opt(s.m_per_px), placeholder="e.g. 0.02",
                              description="m / pixel")
        altitude_box = W.Text(value=_fmt_opt(s.altitude_m), placeholder="e.g. 60",
                              description="Altitude (m)")
        hfov_box = W.Text(value=_fmt_opt(s.hfov_deg), placeholder="e.g. 84",
                          description="Camera HFOV (deg)", style={"description_width": "initial"})
        resolved_label = W.HTML()

        manual_box = W.VBox([m_per_px_box])
        flight_box = W.VBox([altitude_box, hfov_box,
                              W.HTML("<span style='color:#888;font-size:0.9em'>Altitude is height above the "
                                     "<b>water surface</b>, not take-off point. DJI wide lenses are typically "
                                     "82-84&deg; HFOV.</span>")])
        stack = W.Stack(children=[manual_box, flight_box], selected_index=0 if s.scale_mode == "manual" else 1)
        W.jslink((mode, "index"), (stack, "selected_index"))

        def _update_resolved(*_):
            s.scale_mode = mode.value
            s.m_per_px = _parse_positive(m_per_px_box.value)
            s.altitude_m = _parse_positive(altitude_box.value)
            s.hfov_deg = _parse_positive(hfov_box.value)
            m = self._resolved_m_per_px()
            if m:
                gw = m * info.width
                resolved_label.value = (
                    f"<p>Ground resolution: <b>{m:.4f} m/pixel</b> ({gw:.1f} m across the frame)</p>"
                    + self._frame_step_table(m)
                )
            else:
                resolved_label.value = f"<p style='color:{CRITICAL}'>Enter a resolution, or an altitude and field of view.</p>"

        mode.observe(_update_resolved, names="value")
        m_per_px_box.observe(_update_resolved, names="value")
        altitude_box.observe(_update_resolved, names="value")
        hfov_box.observe(_update_resolved, names="value")
        _update_resolved()

        max_dur = info.duration_s if info.duration_s and info.duration_s > 0 else 3600.0
        start_s_box = W.BoundedFloatText(value=min(s.start_s, max_dur), min=0.0, max=max_dur, step=0.5,
                                          description="Start (s)")
        duration_box = W.BoundedFloatText(value=min(s.duration_s, max_dur), min=1.0, max=max_dur, step=1.0,
                                           description="Duration (s)")

        def _update_time(*_):
            s.start_s = start_s_box.value
            s.duration_s = duration_box.value

        start_s_box.observe(_update_time, names="value")
        duration_box.observe(_update_time, names="value")

        # Every velocity is proportional to the frame rate.  A file that does
        # not record one (a raw .h264 stream, say) cannot be analysed without
        # it, and one whose container reports the wrong rate needs correcting.
        known = info.fps > 0
        fps_box = W.Text(value=_fmt_opt(s.fps_override),
                         placeholder=f"{info.fps:.3f} (from the file)" if known else "e.g. 30",
                         description="Frame rate (fps)", style={"description_width": "initial"})
        fps_note = W.HTML(
            "<p style='color:#888;font-size:0.9em'>Leave blank to use the frame rate the file reports; "
            "set it only if the run warns that the rate is wrong.</p>" if known else
            f"<p style='color:{CRITICAL}'><b>This video does not record its frame rate.</b> Enter the "
            f"rate it was recorded at -- every velocity is proportional to it.</p>"
        )

        def _update_fps(*_):
            s.fps_override = _parse_positive(fps_box.value)
            _update_resolved()            # the frame_step suggestions depend on it

        fps_box.observe(_update_fps, names="value")

        return W.VBox([
            W.HTML("<h4>2. Set the scale</h4>"),
            mode, stack, resolved_label,
            W.HTML("<hr><p>Which part of the video to analyse (20-40 s of steady hover is usually plenty). "
                   "The cross-section is marked on the frame at the start time in the next step.</p>"),
            start_s_box, duration_box,
            fps_note, fps_box,
        ])

    # ------------------------------------------------------------------ #
    # step 2 -- cross-section picker
    # ------------------------------------------------------------------ #
    def _ensure_interactive_backend(self) -> None:
        try:
            from IPython import get_ipython

            ip = get_ipython()
            if ip is None:
                raise RuntimeError("not running inside an IPython/Jupyter kernel")
            ip.run_line_magic("matplotlib", "widget")
        except Exception as exc:
            raise RuntimeError(
                "the interactive picker needs the ipympl backend and a running Jupyter kernel. "
                "Install it with `pip install -e '.[notebook]'`, then restart the kernel and re-run "
                "this cell."
            ) from exc

    def _picker_frame(self) -> bool:
        """Make ``frame_bgr`` the frame at the analysis start time.

        The stabiliser takes the first *analysed* frame as its reference, so
        points clicked on any other frame are off by however far the aircraft
        drifted in between.  Returns True when points already clicked on a
        different frame need checking again.
        """
        s = self.state
        if s.frame_bgr is None or s.frame_t_s != s.start_s:
            s.frame_bgr = read_frame_at(s.video_path, s.start_s)
            s.frame_t_s = s.start_s
        has_points = bool(s.section_pts or s.downstream_pts or s.water_pts)
        return has_points and s.picks_t_s is not None and s.picks_t_s != s.frame_t_s

    def _build_step2(self):
        s = self.state
        key = (s.video_path, s.start_s)
        if self._step2_box is not None and self._step2_key == key:
            return self._step2_box

        stale_from = s.picks_t_s
        stale = self._picker_frame()
        if stale:
            # Keep the points, but make every stage be looked at and confirmed
            # again on the new frame before the wizard will move on.
            self._pick_stage = 0

        self._ensure_interactive_backend()
        import matplotlib.pyplot as plt

        if self._fig is not None:
            plt.close(self._fig)

        frame = s.frame_bgr
        h, w = frame.shape[:2]
        # Made with interactive mode off: ipympl would otherwise queue the new
        # figure and display its canvas again under whichever cell runs next,
        # a second picker outside the wizard.  The canvas is shown in the
        # wizard's own panel below.
        with plt.ioff():
            fig, ax = plt.subplots(figsize=(min(9.5, w / 130), min(6.2, h / 130)))
        fig.canvas.header_visible = False
        ax.imshow(frame[:, :, ::-1])
        ax.set_axis_off()
        self._fig, self._ax = fig, ax

        self._pick_stage = min(self._pick_stage, len(self._PICK_STAGES))
        stage_label = W.HTML()
        confirm_btn = W.Button(description="Confirm", button_style="success")
        skip_btn = W.Button(description="Skip", layout=W.Layout(display="none"))
        undo_btn = W.Button(description="Undo last point")
        clear_btn = W.Button(description="Clear this stage")
        restart_btn = W.Button(description="Restart all picking")

        lines = {
            stage["attr"]: ax.plot([], [], "o-", color=self._PICK_COLOURS[stage["attr"]], markersize=7,
                                    markeredgecolor="white", markeredgewidth=1.3, lw=2)[0]
            for stage in self._PICK_STAGES
        }

        def _current_stage():
            return self._PICK_STAGES[self._pick_stage] if self._pick_stage < len(self._PICK_STAGES) else None

        def _points(stage):
            return getattr(s, stage["attr"])

        def _redraw():
            for stage in self._PICK_STAGES:
                pts = _points(stage)
                arr = np.array(pts) if pts else np.empty((0, 2))
                xs = list(arr[:, 0]) if len(arr) else []
                ys = list(arr[:, 1]) if len(arr) else []
                if stage["attr"] == "water_pts" and len(pts) >= 3:
                    xs = xs + [xs[0]]
                    ys = ys + [ys[0]]
                lines[stage["attr"]].set_data(xs, ys)
            fig.canvas.draw_idle()

        stale_note = (
            f"<div style='background:#fff4e5;border:1px solid {WARNING};padding:6px 10px;"
            f"border-radius:4px;margin-bottom:6px'>The analysis now starts at {s.frame_t_s:.1f} s, so "
            f"this is the frame at {s.frame_t_s:.1f} s; the points were placed on the frame at "
            f"{stale_from:.1f} s. Any drone drift in between offsets them. Check each stage and press "
            f"Confirm, or use Clear / Restart to re-click.</div>"
            if stale else ""
        )

        def _update_controls():
            stage = _current_stage()
            if stage is None:
                s.picks_t_s = s.frame_t_s
                stage_label.value = "<b>All set.</b> Press Next below to continue, or Restart to redo the picking."
                confirm_btn.disabled = True
                skip_btn.layout.display = "none"
                return
            n = len(_points(stage))
            need = stage["min"]
            extra = f", need at least {need}" if n < need else ""
            stage_label.value = stale_note + (
                f"<b>Stage {self._pick_stage + 1} of {len(self._PICK_STAGES)}.</b> {stage['prompt']} "
                f"({n} point{'s' if n != 1 else ''} clicked{extra})"
            )
            confirm_btn.disabled = n < need
            skip_btn.layout.display = "inline-block" if need == 0 else "none"

        def _on_click(event):
            if event.inaxes is not ax or event.xdata is None:
                return
            stage = _current_stage()
            if stage is None:
                return
            if event.button == 3:
                _on_undo(None)
                return
            pts = _points(stage)
            if stage["max"] is not None and len(pts) >= stage["max"]:
                return
            pts.append((float(event.xdata), float(event.ydata)))
            s.picks_t_s = s.frame_t_s   # the frame these points belong to
            _redraw()
            _update_controls()

        def _on_undo(_btn):
            stage = _current_stage()
            if stage is None:
                return
            pts = _points(stage)
            if pts:
                pts.pop()
                _redraw()
                _update_controls()

        def _on_clear(_btn):
            stage = _current_stage()
            if stage is None:
                return
            _points(stage).clear()
            _redraw()
            _update_controls()

        def _on_confirm(_btn):
            s.picks_t_s = s.frame_t_s
            self._pick_stage += 1
            _redraw()
            _update_controls()

        def _on_restart(_btn):
            for stage in self._PICK_STAGES:
                _points(stage).clear()
            s.picks_t_s = None
            self._pick_stage = 0
            _redraw()
            _update_controls()

        fig.canvas.mpl_connect("button_press_event", _on_click)
        confirm_btn.on_click(_on_confirm)
        skip_btn.on_click(_on_confirm)
        undo_btn.on_click(_on_undo)
        clear_btn.on_click(_on_clear)
        restart_btn.on_click(_on_restart)

        _redraw()
        _update_controls()

        box = W.VBox([
            W.HTML("<h4>3. Mark the cross-section on the frame</h4>"),
            stage_label,
            fig.canvas,
            W.HBox([confirm_btn, skip_btn, undo_btn, clear_btn, restart_btn]),
            W.HTML(
                "<p style='color:#888;font-size:0.9em'>Left click = add a point &middot; right click = undo &middot; "
                "use the toolbar under the image to zoom in first if you need precision.</p>"
            ),
        ])
        self._step2_box = box
        self._step2_key = key
        return box

    # ------------------------------------------------------------------ #
    # step 3 -- settings
    # ------------------------------------------------------------------ #
    def _build_step3(self):
        s = self.state
        n_stations = W.IntSlider(value=s.n_stations, min=5, max=100, step=1, description="Stations")
        vrange = W.FloatRangeSlider(value=list(s.velocity_range), min=-2.0, max=10.0, step=0.1,
                                     description="Velocity range (m/s)",
                                     style={"description_width": "160px"}, layout=W.Layout(width="480px"))
        stabilise = W.Checkbox(value=s.stabilise,
                                description="Compensate for drone motion (leave on unless the camera is on a fixed mount)")

        frame_step_box = W.Text(value=str(s.frame_step) if s.frame_step else "", placeholder="auto (1)",
                                 description="Frame step")
        window_px = W.IntSlider(value=s.window_px, min=11, max=81, step=2, description="Window size (px)")
        edge_trim = W.FloatSlider(value=s.edge_trim, min=0.0, max=0.4, step=0.01, description="Edge trim")
        advanced = W.Accordion(children=[W.VBox([frame_step_box, window_px, edge_trim])])
        advanced.set_title(0, "Advanced (optional)")
        advanced.selected_index = None

        discharge_chk = W.Checkbox(value=s.discharge_enabled,
                                    description="Also compute discharge (needs a depth/bathymetry file)")
        bathy_fc = FileChooser(str(Path.cwd()))
        bathy_fc.filter_pattern = ["*.csv"]
        bathy_fc.title = "Bathymetry CSV (columns: distance_m, depth_m)"
        alpha = W.FloatSlider(value=s.alpha, min=0.5, max=1.2, step=0.01, description="alpha (velocity index)")
        bank_coeff = W.FloatSlider(value=s.bank_coefficient, min=0.0, max=1.0, step=0.05, description="Bank coefficient")
        bathy_note = W.HTML(f"<i>Currently: {s.bathymetry_csv}</i>" if s.bathymetry_csv else "")
        discharge_box = W.VBox([bathy_fc, bathy_note, alpha, bank_coeff])
        discharge_box.layout.display = "block" if s.discharge_enabled else "none"

        def _on_discharge_toggle(change):
            s.discharge_enabled = change["new"]
            discharge_box.layout.display = "block" if change["new"] else "none"

        discharge_chk.observe(_on_discharge_toggle, names="value")

        def _on_bathy(chooser):
            s.bathymetry_csv = chooser.selected
            bathy_note.value = f"<i>Currently: {s.bathymetry_csv}</i>"

        bathy_fc.register_callback(_on_bathy)

        out_dir_box = W.Text(value=self._out_dir(), description="Save results to")

        def _on_out_dir(change):
            # Only an edit here is a choice that outlives this survey; the name
            # shown until then follows the video, and an emptied box returns
            # to it.
            s.out_dir = change["new"].strip() or None

        out_dir_box.observe(_on_out_dir, names="value")

        def _sync(*_):
            s.n_stations = n_stations.value
            s.velocity_range = tuple(vrange.value)
            s.stabilise = stabilise.value
            s.frame_step = int(frame_step_box.value) if frame_step_box.value.strip().isdigit() else None
            s.window_px = window_px.value
            s.edge_trim = edge_trim.value
            s.alpha = alpha.value
            s.bank_coefficient = bank_coeff.value

        for w_ in (n_stations, vrange, stabilise, frame_step_box, window_px, edge_trim, alpha, bank_coeff):
            w_.observe(_sync, names="value")

        return W.VBox([
            W.HTML("<h4>4. Analysis settings</h4>"),
            n_stations, vrange, stabilise,
            advanced,
            W.HTML("<hr>"),
            discharge_chk, discharge_box,
            W.HTML("<hr>"),
            out_dir_box,
        ])

    # ------------------------------------------------------------------ #
    # step 4 -- run
    # ------------------------------------------------------------------ #
    def _build_config(self) -> Config:
        s = self.state
        if not s.video_path:
            raise ConfigError("no video has been loaded (step 1)")
        manual = s.scale_mode == "manual"
        raw = survey_dict(
            s.video_path,
            start_s=s.start_s,
            duration_s=s.duration_s,
            frame_step=s.frame_step or VideoConfig.frame_step,
            fps_override=s.fps_override,
            m_per_px=s.m_per_px if manual else None,
            altitude_m=None if manual else s.altitude_m,
            hfov_deg=None if manual else s.hfov_deg,
            start_px=s.section_pts[0] if len(s.section_pts) == 2 else None,
            end_px=s.section_pts[1] if len(s.section_pts) == 2 else None,
            n_stations=s.n_stations,
            edge_trim=s.edge_trim,
            name=s.section_name or Path(s.video_path).stem,
            downstream_px=s.downstream_pts[0] if s.downstream_pts else None,
            stabilise=s.stabilise,
            water_mask_px=[s.water_pts] if len(s.water_pts) >= 3 else None,
            search_line_length_m=s.search_line_length_m,
            window_px=s.window_px,
            velocity_range_ms=s.velocity_range,
            bathymetry_csv=s.bathymetry_csv if s.discharge_enabled else None,
            alpha=s.alpha,
            bank_coefficient=s.bank_coefficient,
            out_dir=self._out_dir(),
        )
        return Config.from_dict(raw)

    def _recap_html(self, cfg: Config) -> str:
        s = self.state
        width_px = float(np.hypot(*(np.array(s.section_pts[0]) - np.array(s.section_pts[1]))))
        rows = [
            ("Video", Path(s.video_path).name),
            ("Ground resolution", f"{self._resolved_m_per_px():.4f} m/pixel"),
            ("Frame rate", (f"{self._fps():.3f} fps" if self._fps() else "unknown")
             + (" (entered)" if s.fps_override else "")),
            ("Analysis window", f"{s.start_s:.1f}s → {s.start_s + s.duration_s:.1f}s"),
            ("Cross-section", f"{width_px:.0f} px wide"),
            ("Stations", str(s.n_stations)),
            ("Expected velocity range", f"{s.velocity_range[0]:.1f} to {s.velocity_range[1]:.1f} m/s"),
            ("Drone-motion compensation", "on" if s.stabilise else "off"),
            ("Discharge", "on" if s.discharge_enabled else "off"),
            ("Results folder", self._out_dir()),
        ]
        body = "".join(
            f"<tr><td style='color:#888;padding-right:10px'>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows
        )
        return f"<table>{body}</table>"

    def _build_step4(self):
        try:
            cfg = self._build_config()
            recap = self._recap_html(cfg)
        except ConfigError as exc:
            recap = f"<div style='color:{CRITICAL}'>Configuration problem: {exc}</div>"
        return W.VBox([
            W.HTML("<h4>5. Review and run</h4>"),
            W.HTML(recap),
            W.HTML("<p>Press <b>Run analysis</b> below. A full-length clip can take a minute or two &mdash; "
                   "progress will appear here.</p>"),
        ])

    def _run_analysis(self) -> None:
        self._show_error("")
        try:
            cfg = self._build_config()
        except ConfigError as exc:
            self._show_error(f"Configuration problem: {exc}")
            return

        self._next_btn.disabled = True
        self._back_btn.disabled = True
        status = W.HTML("<b>Running the analysis…</b> this can take a little while for a full-length video.")
        progress_out = W.Output(layout=W.Layout(border="1px solid #eee", padding="4px",
                                                 max_height="180px", overflow="auto"))
        self._body.children = [W.VBox([W.HTML("<h4>5. Review and run</h4>"), status, progress_out])]

        try:
            from .analysis import run as run_analysis
            from .report import write_outputs

            with progress_out:
                result = run_analysis(cfg, progress=True)
                out_path = write_outputs(result, cfg)
        except ConfigError as exc:
            self._next_btn.disabled = False
            self._back_btn.disabled = False
            self._show_error(f"Could not run: {exc}")
            self._refresh()
            return
        except Exception as exc:  # a boundary between untrusted footage/settings and the pipeline
            self._next_btn.disabled = False
            self._back_btn.disabled = False
            self._show_error(f"Unexpected error: {exc}")
            self._refresh()
            return

        self.state.result = result
        self.state.out_path = out_path
        self._next_btn.disabled = False
        self._back_btn.disabled = False
        self.step = 5
        self._refresh()

    # ------------------------------------------------------------------ #
    # step 5 -- results
    # ------------------------------------------------------------------ #
    def _build_step5(self):
        s = self.state
        if s.result is None:
            return W.HTML("<p>No results yet.</p>")
        result = s.result
        out = s.out_path

        warn_html = ""
        if result.warnings:
            items = "".join(f"<li>{w}</li>" for w in result.warnings)
            warn_html = (
                f"<div style='background:#fff4e5;border:1px solid {WARNING};padding:8px;"
                f"border-radius:4px;margin:8px 0'><b>Warnings</b><ul>{items}</ul></div>"
            )
        summary_html = f"<pre style='background:#f7f7f5;padding:10px;border-radius:4px'>{result.summary()}</pre>"

        images = []
        names = ["overview.png", "velocity_profile.png"]
        if result.discharge is not None:
            names.append("discharge.png")
        for name in names:
            p = out / name
            if p.is_file():
                images.append(W.Image(value=p.read_bytes(), format="png", layout=W.Layout(max_width="860px")))

        table_out = W.Output()
        with table_out:
            df = result.to_frame()
            cols = ["station_m", "velocity_ms", "velocity_std_ms", "coherency", "flags"]
            display(df[cols].round(3))

        footer = W.HTML(f"<p>Full tables, config and figures were written to <code>{out.resolve()}</code>.</p>")

        return W.VBox([
            W.HTML("<h4>6. Results</h4>"),
            W.HTML(summary_html), W.HTML(warn_html),
            *images,
            W.HTML("<h5>Velocity by station</h5>"), table_out,
            footer,
        ])


def launch(out_dir: Optional[str] = None) -> Wizard:
    """Build and display the interactive survey wizard.

    Results go to ``out_dir`` if given, else to ``<video name>_output`` in the
    working directory, one folder per video.  Returns the :class:`Wizard` so
    its state (e.g. ``wizard.state.result``) can be inspected from later cells.
    """
    return Wizard(out_dir=out_dir).display()
