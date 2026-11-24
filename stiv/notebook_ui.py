"""A guided, step-by-step Jupyter interface for building and running a STIV survey.

Aimed at students setting up their first survey: one decision per screen, the
running choices always visible in a side panel, sensible defaults pre-filled --
except the scale, which has no sensible default and must be typed in or taken
from a drone preset -- and nothing is analysed until "Run analysis" is pressed.
It is a thin widget layer over the same public API used by the CLI
(:mod:`stiv.config`, :mod:`stiv.analysis`, :mod:`stiv.report`) -- it does not
duplicate any pipeline logic.

Needs the ``widgets`` extra.  Straight from GitHub, in a notebook cell::

    %pip install "stiv[widgets] @ git+https://github.com/simreaney/STIV.git"

or, in a clone of the repository, ``pip install -e '.[notebook]'``.

Usage, in a notebook cell::

    from stiv.notebook_ui import launch
    wizard = launch()

It runs in JupyterLab, VS Code and Google Colab (Colab needs
``google.colab.output.enable_custom_widget_manager()`` first; the bundled
notebook's setup cell does that).  The interactive cross-section picker
(step 3) needs the ``ipympl`` backend; this module switches to it automatically
the first time that step is built.
"""

from __future__ import annotations

import copy
import html
import shutil
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .cameras import DJI_CAMERAS, camera, preset_for_focal_length
from .config import (
    FRAME_STEP_SPEEDS_MS,
    METHOD_LABELS,
    METHODS,
    Config,
    ConfigError,
    DischargeConfig,
    LspivConfig,
    OutputConfig,
    RectifyConfig,
    SectionConfig,
    StabiliseConfig,
    StiConfig,
    WATER_MASK_AUTO,
    VideoConfig,
    default_duration_s,
    is_auto_water_mask,
    section_folder,
    suggest_frame_step,
    survey_dict,
)
from .srt import shortest_focal_len
from .video import VideoInfo, probe, read_frame_at

#: How to install what this module needs, from GitHub.
INSTALL_COMMAND = 'pip install "stiv[widgets] @ git+https://github.com/simreaney/STIV.git"'

try:
    import ipywidgets as W
    from IPython.display import display
    from ipyfilechooser import FileChooser
except ImportError as exc:  # pragma: no cover - exercised by missing-deps installs
    raise ImportError(
        "the interactive notebook UI needs ipywidgets, ipympl and ipyfilechooser. "
        f"Install them with `{INSTALL_COMMAND}` (or `pip install -e '.[notebook]'` in a "
        "clone of the repository), then restart the kernel."
    ) from exc

# Palette matches stiv.plotting, so the wizard and the report figures agree.
ORANGE = "#eb6834"
BLUE = "#2a78d6"
GREEN = "#1baf7a"
CRITICAL = "#d03b3b"
WARNING = "#fab219"
MUTED = "#898781"

_VIDEO_PATTERNS = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv", "*.m4v"]
#: The same, as bare suffixes: "mp4", "mov", ...
_VIDEO_SUFFIXES = {p.split(".")[-1].lower() for p in _VIDEO_PATTERNS}
#: What the upload button lets the browser offer: the videos and their flight
#: logs.  ``video/*`` as well as the suffixes, since some browsers filter by
#: type rather than by name.
_UPLOAD_ACCEPT = ",".join(["video/*"] + sorted("." + x for x in _VIDEO_SUFFIXES) + [".srt"])
#: Label width shared by every form field, so the boxes line up.
_DESC = {"description_width": "190px"}


def _field(width: str = "440px") -> W.Layout:
    """A fresh layout for one form field.  Never shared: a Layout is a single
    model, and hiding one widget through it would hide every widget using it."""
    return W.Layout(width=width)


# Two colours come from the host -- the text and the background, from
# JupyterLab's theme or VS Code's, else plain light ones for Colab -- and every
# other shade is mixed from that pair.  So the panel follows a light or dark
# theme, and its quieter text stays readable in both: JupyterLab's own
# secondary text colours are 54% and 38% black, too faint for instructions, and
# a tile or panel shaded independently of the text can end up light-on-light.
_CSS = """<style>
.stiv-wiz {
  --stiv-bg: var(--jp-layout-color1, var(--vscode-editor-background, #ffffff));
  --stiv-fg: var(--jp-ui-font-color1, var(--vscode-editor-foreground, #1d1d1b));
  --stiv-fg2: color-mix(in srgb, var(--stiv-fg) 88%, var(--stiv-bg));
  --stiv-fg3: color-mix(in srgb, var(--stiv-fg) 74%, var(--stiv-bg));
  --stiv-bg2: color-mix(in srgb, var(--stiv-fg) 5%, var(--stiv-bg));
  --stiv-line: color-mix(in srgb, var(--stiv-fg) 18%, var(--stiv-bg));
  /* The palette's green, as text: darkened on a light page, lightened on a dark. */
  --stiv-green-text: color-mix(in srgb, #1baf7a 62%, var(--stiv-fg));
  background: var(--stiv-bg); color: var(--stiv-fg);
  border: 1px solid var(--stiv-line); border-radius: 12px; overflow: hidden;
  max-width: 1180px;
}
.stiv-wiz .stiv-header { padding: 16px 22px 14px;
  background: linear-gradient(100deg, #0d366b 0%, #2a78d6 100%); }
.stiv-wiz .stiv-header h2 { margin: 0; font-size: 1.3em; font-weight: 650; color: #fff; }
.stiv-wiz .stiv-header p { margin: 3px 0 0; font-size: .92em; color: rgba(255,255,255,.85); }
.stiv-wiz .stiv-stepper { padding: 10px 16px 8px; gap: 2px; flex-wrap: wrap;
  border-bottom: 1px solid var(--stiv-line); }
.stiv-wiz button.stiv-step { background: transparent !important; box-shadow: none !important;
  color: var(--stiv-fg3) !important; border-radius: 999px !important; padding: 0 14px;
  height: 30px; font-weight: 500; }
.stiv-wiz button.stiv-step.stiv-current { background: #eb6834 !important; color: #fff !important;
  font-weight: 650; opacity: 1 !important; }
.stiv-wiz button.stiv-step.stiv-done { color: var(--stiv-green-text) !important; font-weight: 600; }
.stiv-wiz button.stiv-step.stiv-done:hover { background: var(--stiv-bg2) !important; }
.stiv-wiz button.stiv-step:disabled { cursor: default; opacity: 1; }
.stiv-wiz .stiv-main { gap: 18px; padding: 16px 20px 18px; flex-flow: row wrap;
  align-items: flex-start; }
.stiv-wiz .stiv-content { flex: 1 1 560px; min-width: 0; }
.stiv-wiz .stiv-aside { flex: 0 1 270px; background: var(--stiv-bg2); border-radius: 10px;
  border: 1px solid var(--stiv-line); padding: 12px 14px; }
.stiv-wiz h3.stiv-title { margin: 0 0 2px; font-size: 1.15em; font-weight: 650; color: var(--stiv-fg); }
.stiv-wiz p.stiv-lead { margin: 0 0 12px; color: var(--stiv-fg2); }
.stiv-wiz .stiv-card { border: 1px solid var(--stiv-line); border-radius: 10px;
  padding: 10px 14px 12px; margin: 0 0 12px; }
.stiv-wiz .stiv-card-title { font-weight: 650; margin: 0 0 4px; color: var(--stiv-fg); }
.stiv-wiz .stiv-help { color: var(--stiv-fg2); font-size: .92em; margin: 4px 0; line-height: 1.45; }
.stiv-wiz .stiv-callout { border-radius: 8px; padding: 8px 12px; margin: 4px 0 10px;
  line-height: 1.45; border: 1px solid; border-left-width: 4px; color: var(--stiv-fg); }
.stiv-wiz .stiv-info { background: rgba(42,120,214,.08); border-color: rgba(42,120,214,.5); }
.stiv-wiz .stiv-ok { background: rgba(27,175,122,.09); border-color: rgba(27,175,122,.55); }
.stiv-wiz .stiv-warn { background: rgba(250,178,25,.13); border-color: rgba(214,150,0,.7); }
.stiv-wiz .stiv-error { background: rgba(208,59,59,.09); border-color: rgba(208,59,59,.65); }
.stiv-wiz .stiv-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px; margin: 8px 0 10px; }
.stiv-wiz .stiv-tile { background: var(--stiv-bg2); border: 1px solid var(--stiv-line);
  border-radius: 8px; padding: 8px 10px; }
.stiv-wiz .stiv-tile .k { font-size: .75em; color: var(--stiv-fg2); text-transform: uppercase;
  letter-spacing: .05em; }
.stiv-wiz .stiv-tile .v { font-size: 1.35em; font-weight: 650; color: var(--stiv-fg); margin-top: 2px; }
.stiv-wiz .stiv-tile .s { font-size: .8em; color: var(--stiv-fg3); }
.stiv-wiz .stiv-summary h4 { margin: 0 0 6px; font-size: .95em; color: var(--stiv-fg); }
.stiv-wiz .stiv-summary .row { display: flex; gap: 8px; padding: 5px 0;
  border-top: 1px solid var(--stiv-line); font-size: .87em; line-height: 1.35; }
.stiv-wiz .stiv-summary .mark { width: 14px; flex: none; text-align: center; color: var(--stiv-fg3); }
.stiv-wiz .stiv-summary .mark.ok { color: var(--stiv-green-text); font-weight: 700; }
.stiv-wiz .stiv-summary .lbl { color: var(--stiv-fg2); }
.stiv-wiz .stiv-summary .val { color: var(--stiv-fg); font-weight: 600; word-break: break-word; }
.stiv-wiz .stiv-summary .sub { color: var(--stiv-fg3); font-size: .92em; }
.stiv-wiz .stiv-nav { border-top: 1px solid var(--stiv-line); padding-top: 12px; margin-top: 4px;
  align-items: center; }
.stiv-wiz .stiv-counter { color: var(--stiv-fg3); font-size: .9em; }
.stiv-wiz .stiv-chips span { display: inline-block; background: var(--stiv-bg2);
  border: 1px solid var(--stiv-line); border-radius: 999px; padding: 1px 10px;
  margin: 0 4px 4px 0; font-size: .85em; color: var(--stiv-fg2); }
.stiv-wiz .stiv-chips span.now { border-color: #eb6834; color: var(--stiv-fg); font-weight: 600; }
.stiv-wiz .stiv-chips span.done { color: var(--stiv-green-text); }
.stiv-wiz table.stiv-table { border-collapse: collapse; font-size: .92em; margin: 4px 0; }
.stiv-wiz table.stiv-table td, .stiv-wiz table.stiv-table th { padding: 3px 14px 3px 0;
  text-align: left; vertical-align: top; }
.stiv-wiz table.stiv-table th { color: var(--stiv-fg2); font-weight: 600; }
.stiv-wiz table.stiv-table td.k { color: var(--stiv-fg2); }
.stiv-wiz pre.stiv-report { background: var(--stiv-bg2); padding: 10px 12px; border-radius: 8px;
  font-size: .84em; white-space: pre-wrap; color: var(--stiv-fg); }
.stiv-wiz .stiv-frame img, .stiv-wiz img.stiv-frame { border-radius: 8px; }
/* ipywidgets sets HTML text to the 28px height of an input box; prose and
   tables read better at a normal line height. */
.stiv-wiz .widget-html-content { line-height: 1.45; }
</style>"""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _callout(kind: str, body: str) -> str:
    """A coloured note; ``kind`` is info, ok, warn or error.  ``body`` is HTML."""
    return f"<div class='stiv-callout stiv-{kind}'>{body}</div>"


def _tiles(items) -> str:
    """A row of headline numbers: ``(label, value, sub-label or None)``."""
    out = []
    for k, v, sub in items:
        sub_html = f"<div class='s'>{sub}</div>" if sub else ""
        out.append(f"<div class='stiv-tile'><div class='k'>{k}</div><div class='v'>{v}</div>{sub_html}</div>")
    return "<div class='stiv-tiles'>" + "".join(out) + "</div>"


def _heading(title: str, lead: str) -> W.HTML:
    return W.HTML(f"<h3 class='stiv-title'>{title}</h3><p class='stiv-lead'>{lead}</p>")


def _help(text: str) -> W.HTML:
    return W.HTML(f"<div class='stiv-help'>{text}</div>")


def _card(title: Optional[str], *children) -> W.VBox:
    head = [W.HTML(f"<div class='stiv-card-title'>{title}</div>")] if title else []
    box = W.VBox(head + list(children))
    box.add_class("stiv-card")
    return box


def _show(widget, visible: bool) -> None:
    widget.layout.display = None if visible else "none"


def _parse_positive(text: str) -> Optional[float]:
    """A strictly positive number typed into a text box, or None."""
    try:
        x = float(str(text).strip())
    except ValueError:
        return None
    return x if np.isfinite(x) and x > 0 else None


def _parse_float(text: str) -> Optional[float]:
    """Any finite number typed into a text box, or None."""
    try:
        x = float(str(text).strip())
    except ValueError:
        return None
    return x if np.isfinite(x) else None


def _fmt_opt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:g}"


def _jpeg(frame_bgr: np.ndarray, max_width: int = 960) -> bytes:
    """A reduced JPEG of a frame, small enough to keep in the notebook."""
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        frame_bgr = cv2.resize(frame_bgr, (max_width, max(1, round(h * max_width / w))),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def _in_colab() -> bool:
    return "google.colab" in sys.modules


def _drive_dir() -> Optional[Path]:
    """Google Drive, when this is Colab and the setup cell mounted it."""
    p = Path("/content/drive/MyDrive")
    return p if _in_colab() and p.is_dir() else None


def _browse_root() -> str:
    d = _drive_dir()
    return str(d) if d else str(Path.cwd())


def _refresh_matplotlib_backends() -> None:
    """Let matplotlib see the ipympl backend when it was installed after start-up.

    Matplotlib 3.9 and later read the list of installed backends once per
    session.  In Colab the notebook's setup cell installs ipympl after that --
    Colab starts matplotlib before any cell runs -- and from then on ipympl is
    refused ("not a valid value for backend"), even as it is imported, until
    the session restarts.  Clearing the list makes matplotlib read it afresh.
    ``_clear`` is matplotlib's own reset, private; if it ever goes, the picker
    says to restart the session instead.
    """
    try:
        from matplotlib.backends import backend_registry
    except ImportError:              # matplotlib < 3.9 keeps no such list
        return
    try:
        if backend_registry.is_valid_backend("module://ipympl.backend_nbagg"):
            return
        reset = getattr(backend_registry, "_clear", None)
        if callable(reset):
            reset()
    except Exception:                # a refresh is a courtesy; never the failure itself
        pass


def _read_survey(path: str | Path) -> Config:
    """A survey YAML, validated if it is complete, else read as far as it goes.

    ``stiv init`` writes surveys with ``null`` placeholders that
    :meth:`Config.validate` refuses; the wizard can fill those in, so they are
    read without the validation rather than turned away.
    """
    path = Path(path).expanduser()
    try:
        return Config.from_yaml(path)
    except ConfigError as strict:
        import yaml

        from .config import _build, _build_sections, _ConfigLoader

        try:
            raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader) or {}
        except Exception:
            raise strict from None
        if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
            raise strict from None
        cfg = Config()
        for f in fields(Config):
            section = raw.get(f.name)
            if f.name == "sections":
                try:
                    cfg.sections = _build_sections(section)
                except ConfigError:
                    raise strict from None
            elif isinstance(section, dict):
                setattr(cfg, f.name, _build(type(getattr(cfg, f.name)), section, f.name))
        cfg.normalise_types()
        cfg.resolve_paths(path.parent)
        return cfg


def _merge_survey(base: dict, new: dict) -> dict:
    """``new`` (what the wizard manages) laid over a reopened survey ``base``.

    Settings the wizard has no control for -- ``static_mask_px``, the quality
    thresholds, the output options -- come through from the file unchanged.
    """
    out = copy.deepcopy(base)
    for section, values in new.items():
        if section == "rectify" or not isinstance(values, dict):
            # The scale is one choice: altitude + FOV, or m/px, never a mixture.
            # And `sections` arrives complete, with what the file gave each
            # section and the wizard does not manage already folded in.
            out[section] = copy.deepcopy(values)
        else:
            out.setdefault(section, {}).update(copy.deepcopy(values))
    # One cross-section or several: whichever the wizard has now, not the file.
    if "sections" in new:
        out["section"] = {k: v for k, v in out.get("section", {}).items()
                          if k not in ("start_px", "end_px", "name")}
    else:
        out.pop("sections", None)
    # survey_dict leaves these out when unset; the file's must not fill in.
    out["video"]["fps_override"] = new["video"].get("fps_override")
    water = new["stabilise"].get("water_mask_px")
    if water is None and is_auto_water_mask(base.get("stabilise", {}).get("water_mask_px")):
        # The file asked for the automatic outline at run time, and the wizard
        # has no outline of its own to replace it with.
        water = WATER_MASK_AUTO
    out["stabilise"]["water_mask_px"] = water
    return out


@dataclass
class _State:
    """Everything the wizard knows about the survey being built.

    Defaults come from the config schema, so a survey built here analyses the
    same way as one started with ``stiv init`` or ``stiv pick``.
    """

    #: The results folder the user chose, or None for one named after the
    #: video (see :meth:`Wizard._out_dir`).
    out_dir: Optional[str] = None
    #: The results folder of a reopened survey; forgotten along with it.
    loaded_out_dir: Optional[str] = None

    video_path: Optional[str] = None
    is_demo: bool = False
    info: Optional[VideoInfo] = None
    frame_bgr: Optional[np.ndarray] = None
    #: Time in the video of ``frame_bgr``, s.  The picker must show the frame
    #: the analysis starts on: the stabiliser measures drift from there.
    frame_t_s: float = 0.0
    #: The DJI flight log (.SRT) beside the video, and what it recorded.
    srt_path: Optional[str] = None
    srt_records: list = field(default_factory=list)

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
    #: A key of :data:`stiv.cameras.DJI_CAMERAS`, "custom", or None.
    camera_key: Optional[str] = None
    #: The camera the video names in its own metadata (a
    #: :class:`stiv.videometa.CameraId`), or None.
    camera_id: Optional[object] = None
    #: Height of the take-off point above the water, for the flight-log altitude.
    takeoff_height_m: Optional[float] = None

    section_pts: list = field(default_factory=list)
    #: Further cross-sections, analysed with the first in one pass over the
    #: video: each a dict of "pts" (its two ends), "name" and "bathymetry_csv",
    #: and "extra" -- keys of a reopened survey's entry the wizard leaves alone.
    more_sections: list = field(default_factory=list)
    downstream_pts: list = field(default_factory=list)
    water_pts: list = field(default_factory=list)
    #: Further water polygons from a reopened survey, kept as they were.
    extra_water_polys: list = field(default_factory=list)
    #: Where ``water_pts`` came from: "auto", "manual", "loaded" or None.
    water_source: Optional[str] = None
    #: Time of the frame the points above were clicked or confirmed on, s.
    picks_t_s: Optional[float] = None

    n_stations: int = SectionConfig.n_stations
    velocity_range: tuple = tuple(StiConfig.velocity_range_ms)
    stabilise: bool = StabiliseConfig.enabled
    frame_step: Optional[int] = None
    window_px: int = StiConfig.window_px
    edge_trim: float = SectionConfig.edge_trim
    search_line_length_m: float = StiConfig.search_line_length_m
    #: The velocity method reported, those run beside it, STIV's angle
    #: estimator and LS-PIV's interrogation area (None: automatic).
    method: str = "stiv"
    compare: list = field(default_factory=list)
    estimator: str = StiConfig.estimator
    ia_size_m: Optional[float] = LspivConfig.ia_size_m
    section_name: Optional[str] = None
    #: The first section's entry keys the wizard leaves alone, from a reopened
    #: multi-section survey (see ``more_sections``).
    section_extra: dict = field(default_factory=dict)

    discharge_enabled: bool = False
    #: Write report.pdf with the results: on by default here, where it is a
    #: visible choice -- a single document is what a student hands in.
    pdf_report: bool = True
    bathymetry_csv: Optional[str] = None
    alpha: float = DischargeConfig.alpha
    bank_coefficient: float = DischargeConfig.bank_coefficient

    #: A reopened survey, as a raw config: what the wizard does not manage
    #: is taken from here.
    base_raw: Optional[dict] = None
    loaded_from: Optional[str] = None

    result: object = None
    out_path: Optional[Path] = None

    def clear(self) -> None:
        """Back to a blank survey, keeping only a results folder the user chose.

        Done in place rather than by replacing the object, because widget
        callbacks built earlier hold a reference to it.
        """
        self.__dict__.update(_State(out_dir=self.out_dir).__dict__)


# --------------------------------------------------------------------------- #
class Wizard:
    """A linear, six-step survey builder rendered as one ipywidgets panel."""

    _TITLES = ("Video", "Scale", "Cross-section", "Settings", "Run", "Results")
    _NEXT_LABELS = ("Next", "Next", "Next", "Next", "Run analysis", "Start a new survey")
    _NEXT_ICONS = ("arrow-right",) * 4 + ("play", "refresh")
    _PICK_STAGES = (
        {"attr": "section_pts", "min": 2, "max": 2, "label": "Cross-section",
         "prompt": "Click the two ends of the measurement cross-section: left bank, then right bank."},
        {"attr": "downstream_pts", "min": 1, "max": 1, "label": "Downstream point",
         "prompt": "Click one point further DOWNSTREAM of the section. This tells STIV which way the flow runs."},
        {"attr": "water_pts", "min": 0, "max": None, "label": "Water outline (optional)",
         "prompt": "Optional: click around the edge of the water, then Confirm. This stops the camera-shake "
                   "correction from locking onto the moving river. Press Skip to go without one."},
    )
    _PICK_COLOURS = {"section_pts": ORANGE, "downstream_pts": BLUE, "water_pts": GREEN}

    def __init__(self, out_dir: Optional[str] = None):
        self.state = _State(out_dir=out_dir)
        self.step = 0
        #: Furthest step reached with everything before it complete.
        self._furthest = 0
        self._pick_stage = 0
        #: Ends clicked so far of a cross-section being added, else None.
        self._adding_pts: Optional[list] = None
        self._step2_box = None
        self._step2_key = None
        self._step2_sync = None
        self._fig = None
        self._ax = None
        self._video_mode = "own"
        #: Inputs of the last automatic water detection, and what it found.
        self._water_key = None
        self._water_detection = None
        self._preview = W.VBox([])

        header = W.HTML(
            "<div class='stiv-header'><h2>STIV &middot; river surface velocity</h2>"
            "<p>Measure the surface speed of a river from a drone video, one step at a time.</p></div>"
        )
        self._step_btns = []
        for i, title in enumerate(self._TITLES):
            btn = W.Button(description=f"{i + 1}  {title}", layout=W.Layout(width="auto"))
            btn.add_class("stiv-step")
            btn.on_click(lambda _b, i=i: self._go_to(i))
            self._step_btns.append(btn)
        stepper = W.HBox(self._step_btns)
        stepper.add_class("stiv-stepper")

        self._error = W.HTML()
        self._body = W.VBox([])
        self._back_btn = W.Button(description="Back", icon="arrow-left", layout=W.Layout(width="110px"))
        self._next_btn = W.Button(description=self._NEXT_LABELS[0], icon=self._NEXT_ICONS[0],
                                  button_style="primary", layout=W.Layout(width="200px"))
        self._back_btn.on_click(self._on_back)
        self._next_btn.on_click(self._on_next)
        self._counter = W.HTML()
        nav = W.HBox([self._back_btn, self._counter, self._next_btn],
                     layout=W.Layout(justify_content="space-between"))
        nav.add_class("stiv-nav")
        content = W.VBox([self._error, self._body, nav])
        content.add_class("stiv-content")

        self._summary = W.HTML()
        aside = W.VBox([self._summary])
        aside.add_class("stiv-aside")
        main = W.HBox([content, aside])
        main.add_class("stiv-main")

        self.root = W.VBox([W.HTML(_CSS), header, stepper, main])
        self.root.add_class("stiv-wiz")
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
        self._adding_pts = None
        self._furthest = 0
        self._step2_box = None
        self._step2_key = None
        self._step2_sync = None
        self._fig = self._ax = None
        self._water_key = None
        self._water_detection = None

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
        self._furthest = max(self._furthest, self.step)
        self._refresh()

    def _on_back(self, _btn) -> None:
        if self.step > 0:
            self._show_error("")
            self.step -= 1
            self._refresh()

    def _go_to(self, target: int) -> None:
        """Jump to a step from the stepper; forward only over completed steps."""
        if target == self.step or target > self._furthest:
            return
        if target == 5 and self.state.result is None:
            return
        for j in range(self.step, min(target, 4)):
            ok, msg = self._step_ready(j)
            if not ok:
                self.step = j
                self._show_error(msg)
                self._refresh()
                return
        self._show_error("")
        self.step = target
        self._refresh()

    def _show_error(self, msg: str) -> None:
        self._error.value = _callout("error", msg) if msg else ""

    def _refresh(self) -> None:
        try:
            body = self._current_body()
        except Exception as exc:  # keeps a bad widget build from taking down the kernel
            body = W.HTML(_callout("error", f"Could not build this step: {_esc(exc)}"))
        self._body.children = [body]
        self._back_btn.disabled = self.step == 0
        self._next_btn.description = self._NEXT_LABELS[self.step]
        self._next_btn.icon = self._NEXT_ICONS[self.step]
        self._counter.value = f"<span class='stiv-counter'>Step {self.step + 1} of {len(self._TITLES)}</span>"
        self._refresh_stepper()
        self._refresh_summary()

    def _refresh_stepper(self) -> None:
        for i, (btn, title) in enumerate(zip(self._step_btns, self._TITLES)):
            for cls in ("stiv-current", "stiv-done"):
                btn.remove_class(cls)
            reachable = i <= self._furthest and (i != 5 or self.state.result is not None)
            if i == self.step:
                btn.add_class("stiv-current")
                btn.description = f"{i + 1}  {title}"
            elif reachable:
                btn.add_class("stiv-done")
                btn.description = f"✓ {title}" if i < self.step or i < self._furthest else f"{i + 1}  {title}"
            else:
                btn.description = f"{i + 1}  {title}"
            btn.disabled = not reachable or i == self.step

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
                return False, ("Choose your drone, or enter your camera's horizontal field of view "
                               "(between 0 and 180 degrees), before continuing.")
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
            if self._adding_pts is not None:
                return False, ("Finish the cross-section you are adding (click both ends, then "
                               "Confirm), or press Cancel.")
            return True, ""
        if step == 3:
            if s.discharge_enabled and not any(sec["bathymetry_csv"] for sec in self._sections()):
                return False, ("Choose a bathymetry CSV file, or turn discharge off."
                               if not s.more_sections else
                               "Choose a bathymetry CSV file for at least one cross-section, "
                               "or turn discharge off.")
            names = [sec["name"] for sec in self._sections()]
            folders = [section_folder(n).lower() for n in names]
            if len(set(folders)) < len(folders):
                return False, ("Two cross-sections have the same name (or names that make the "
                               "same results folder); give each its own name.")
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

    def _sections(self) -> list[dict]:
        """Every cross-section marked so far: "name", "pts", "bathymetry_csv".

        The first is the one the picking stages mark.  A lone section is
        named as it always was -- the name given, else the video's -- and
        several ``section_1``, ``section_2``, ... unless named.
        """
        s = self.state
        multi = bool(s.more_sections)
        first_name = s.section_name or (
            "section_1" if multi else Path(s.video_path).stem if s.video_path else "section")
        out = [{"name": first_name, "pts": s.section_pts,
                "bathymetry_csv": s.bathymetry_csv, "extra": s.section_extra}]
        for k, sec in enumerate(s.more_sections, start=2):
            out.append({"name": sec.get("name") or f"section_{k}", "pts": sec["pts"],
                        "bathymetry_csv": sec.get("bathymetry_csv"),
                        "extra": sec.get("extra", {})})
        return out

    def _out_dir(self) -> str:
        """The results folder: the one chosen, else one named after the video.

        Named as ``stiv init`` names it after the survey file.  A folder shared
        by every survey would let each one wipe the last one's results, since
        writing them first clears what the previous run there wrote.  In Colab
        it goes to Google Drive when that is mounted, since the Colab machine's
        own disk is wiped when the session ends.
        """
        s = self.state
        if s.out_dir:
            return s.out_dir
        if s.loaded_out_dir:
            return s.loaded_out_dir
        if not s.video_path:
            return OutputConfig.out_dir
        from .cli import _default_out_dir

        name = _default_out_dir(Path(s.video_path))
        drive = _drive_dir()
        return str(drive / "stiv_results" / name) if drive else name

    def _water_at_run(self) -> bool:
        """True when a reopened survey's ``water_mask_px: auto`` will stand.

        That is, the file asked for the automatic outline at run time and the
        wizard has no outline of its own to put in its place.
        """
        s = self.state
        return (bool(s.base_raw)
                and is_auto_water_mask((s.base_raw.get("stabilise") or {}).get("water_mask_px"))
                and len(s.water_pts) < 3 and not s.extra_water_polys)

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
            return ("<div class='stiv-help'>Enter the frame rate below to get "
                    "<code>frame_step</code> suggestions.</div>")
        rows = []
        for v in FRAME_STEP_SPEEDS_MS:
            step = suggest_frame_step(m_per_px, fps, v)
            rows.append(f"<tr><td>{v:.1f} m/s</td><td>{step}</td></tr>")
        return (
            "<table class='stiv-table'>"
            "<tr><th>expected speed</th><th>suggested frame_step</th></tr>"
            + "".join(rows) + "</table>"
        )

    # ------------------------------------------------------------------ #
    # the side panel
    # ------------------------------------------------------------------ #
    def _refresh_summary(self) -> None:
        self._summary.value = self._summary_html()

    def _summary_html(self) -> str:
        s = self.state
        rows = []

        def row(done: Optional[bool], label: str, value: str, sub: str = "") -> None:
            mark = "✓" if done else ("·" if done is None else "–")
            cls = "mark ok" if done else "mark"
            sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
            rows.append(f"<div class='row'><div class='{cls}'>{mark}</div><div>"
                        f"<div class='lbl'>{label}</div><div class='val'>{value}</div>{sub_html}</div></div>")

        info = s.info
        if info is not None and s.video_path:
            fps = f"{self._fps():.3g} fps" if self._fps() else "frame rate unknown"
            row(True, "Video", _esc(Path(s.video_path).name),
                f"{info.width}×{info.height} · {fps} · {info.duration_s:.1f} s"
                + (" · flight log" if s.srt_records else ""))
        else:
            row(False, "Video", "not chosen yet")

        m = self._resolved_m_per_px()
        if m:
            if s.scale_mode == "flight":
                preset = camera(s.camera_key or "")
                how = f"{s.altitude_m:g} m up · " + (_esc(preset.label) if preset else f"{s.hfov_deg:g}° HFOV")
            else:
                how = "entered directly"
            row(True, "Scale", f"{m * 100:.2f} cm / pixel", how)
        else:
            row(False, "Scale", "not set yet")

        if info is not None:
            row(None, "Analysis window", f"{s.start_s:g} – {s.start_s + s.duration_s:g} s")

        if len(s.section_pts) == 2 and s.more_sections:
            secs = self._sections()
            widths = [float(np.hypot(*np.subtract(sec["pts"][1], sec["pts"][0]))) for sec in secs]
            row(True, "Cross-sections", f"{len(secs)} marked",
                " · ".join(f"{w * m:.1f} m" if m else f"{w:.0f} px" for w in widths)
                + (" · flow direction set" if s.downstream_pts else " · flow direction not set"))
        elif len(s.section_pts) == 2:
            px = float(np.hypot(*np.subtract(s.section_pts[1], s.section_pts[0])))
            row(True, "Cross-section", f"{px * m:.1f} m wide" if m else f"{px:.0f} px wide",
                "flow direction set" if s.downstream_pts else "flow direction not set")
        else:
            row(False, "Cross-section", "not marked yet")
        water = {"auto": "detected automatically", "manual": "drawn by hand",
                 "loaded": "from the reopened survey"}.get(s.water_source or "")
        if len(s.water_pts) >= 3 and water:
            row(None, "Water outline", water)
        elif self._water_at_run():
            row(None, "Water outline", "found when the analysis runs (auto)")
        elif self._pick_stage >= len(self._PICK_STAGES):
            row(None, "Water outline", "none (section corridors only)" if s.more_sections
                else "none (section corridor only)")

        row(None, "Stations", f"{s.n_stations}",
            f"expected {s.velocity_range[0]:g} to {s.velocity_range[1]:g} m/s")
        row(None, "Velocity method", METHOD_LABELS[s.method],
            ("compared with " + ", ".join(METHOD_LABELS[m] for m in s.compare)) if s.compare else "")
        row(None, "Drone motion", "compensated" if s.stabilise else "not compensated")
        if s.more_sections and s.discharge_enabled:
            with_depths = sum(1 for sec in self._sections() if sec["bathymetry_csv"])
            row(None, "Discharge", "on", f"depths for {with_depths} of {len(self._sections())} sections")
        else:
            row(None, "Discharge", "on" if s.discharge_enabled else "off",
                _esc(Path(s.bathymetry_csv).name) if s.discharge_enabled and s.bathymetry_csv else "")
        if s.video_path:
            out = self._out_dir()
            row(None, "Results folder", _esc(Path(out).name), _esc(out) if out != Path(out).name else "")
        return "<div class='stiv-summary'><h4>Your survey</h4>" + "".join(rows) + "</div>"

    # ------------------------------------------------------------------ #
    # step 0 -- video
    # ------------------------------------------------------------------ #
    def _build_step0(self):
        s = self.state
        mode = W.ToggleButtons(
            options=[("Use my own video", "own"), ("Reopen a saved survey", "reopen"),
                     ("Try the demo river", "demo")],
            value=self._video_mode, style={"button_width": "auto"},
        )

        # -- own video: browse, or upload -------------------------------- #
        own_status = W.HTML()
        fc = FileChooser(_browse_root())
        fc.filter_pattern = _VIDEO_PATTERNS
        fc.title = "<b>Browse to a drone video</b>"
        fc.register_callback(lambda chooser: self._load_video(chooser.selected, is_demo=False,
                                                              status=own_status))
        upload = W.FileUpload(accept=_UPLOAD_ACCEPT,
                              multiple=True, description="Upload files", icon="upload")
        upload.observe(lambda _c: self._on_upload(upload, own_status), names="value")
        where = (
            "In Colab, the easiest place for a big video is Google Drive (the setup cell mounts it "
            "under <code>drive/MyDrive</code>)." if _in_colab() else
            "Video files are big: if the notebook runs on your own computer, browsing to the file "
            "is quicker than uploading it."
        )
        own_box = W.VBox([
            _card("Choose a video file", fc,
                  _help("If the drone saved a flight log next to the video (<code>.SRT</code>, same "
                        "name), it is picked up automatically and can fill in the altitude.")),
            _card("…or upload from this computer", upload,
                  _help("Upload the video, and its <code>.SRT</code> flight log if you have one. "
                        f"Uploads go to <code>stiv_uploads/</code>. {where}")),
            own_status,
        ])

        # -- reopen a saved survey --------------------------------------- #
        reopen_status = W.HTML()
        rc = FileChooser(_browse_root())
        rc.filter_pattern = ["*.yaml", "*.yml"]
        rc.title = "<b>Choose a survey file</b>"
        rc.register_callback(lambda chooser: self._reopen_survey(chooser.selected, reopen_status))
        reopen_box = W.VBox([
            _card("Reopen a saved survey", rc,
                  _help("Every run saves <code>config_used.yaml</code> in its results folder; "
                        "a survey from <code>stiv init</code> or <code>stiv pick</code> works too. "
                        "Everything is filled in from the file -- the scale, the clicked points and "
                        "the settings -- so you can change what you need and run again.")),
            reopen_status,
        ])

        # -- demo -------------------------------------------------------- #
        demo_status = W.HTML()
        demo_btn = W.Button(description="Generate demo river", icon="tint", button_style="primary",
                            layout=W.Layout(width="220px"))

        def _on_demo(_btn):
            demo_btn.disabled = True
            demo_status.value = _callout("info", "Rendering a synthetic river with a known velocity profile…")
            try:
                self._generate_demo_video()
                demo_status.value = _callout("ok", "Demo video ready. The scale and cross-section are "
                                                   "filled in for you; press <b>Next</b>.")
            except Exception as exc:
                demo_status.value = _callout("error", f"Could not render the demo: {_esc(exc)}")
            demo_btn.disabled = False

        demo_btn.on_click(_on_demo)
        demo_box = W.VBox([
            _card("Try the demo river",
                  _help("Renders a short synthetic river with a <b>known</b> velocity profile and mild "
                        "simulated drone drift &mdash; a safe way to learn the workflow, and to check "
                        "STIV is installed correctly, before using real footage. The scale and "
                        "cross-section are filled in for you in the next steps, but you can still "
                        "edit them."),
                  demo_btn),
            demo_status,
        ])

        panes = {"own": own_box, "reopen": reopen_box, "demo": demo_box}

        def _on_mode(change=None):
            self._video_mode = mode.value
            for key, pane in panes.items():
                _show(pane, key == mode.value)

        mode.observe(_on_mode, names="value")
        _on_mode()
        self._refresh_preview()

        return W.VBox([
            _heading("1. Choose a video",
                     "A steady, straight-down (nadir) drone hover over the river works best."),
            mode, W.HTML("<div style='height:8px'></div>"),
            own_box, reopen_box, demo_box, self._preview,
        ])

    def _refresh_preview(self) -> None:
        s = self.state
        if s.info is None:
            self._preview.children = []
            return
        info = s.info
        chips = [f"{info.width} × {info.height}",
                 f"{info.fps:.3f} fps" if info.fps > 0 else "frame rate not recorded",
                 f"{info.duration_s:.1f} s"]
        if info.n_frames_total > 0:
            chips.append(f"{info.n_frames_total} frames")
        if s.srt_records:
            chips.append(f"flight log: {_esc(Path(s.srt_path).name)}")
        children = [W.HTML(f"<div class='stiv-card-title'>{_esc(Path(s.video_path).name)}</div>"
                           "<div class='stiv-chips'>" + "".join(f"<span>{c}</span>" for c in chips)
                           + "</div>")]
        if s.frame_bgr is not None:
            img = W.Image(value=_jpeg(s.frame_bgr), format="jpeg",
                          layout=W.Layout(width="100%", max_width="720px"))
            img.add_class("stiv-frame")
            children.append(img)
        self._preview.children = [_card(None, *children)]

    def _on_upload(self, upload, status) -> None:
        files = upload.value
        if not files:
            return
        if isinstance(files, dict):          # ipywidgets 7 shape
            files = [{"name": k, "content": v["content"]} for k, v in files.items()]
        dest = Path.cwd() / "stiv_uploads"
        dest.mkdir(exist_ok=True)
        videos, saved = [], []
        for f in files:
            path = dest / Path(f["name"]).name
            path.write_bytes(bytes(f["content"]))
            saved.append(path.name)
            if path.suffix.lower().lstrip(".") in _VIDEO_SUFFIXES:
                videos.append(path)
        try:
            upload.value = ()               # let the browser copy go
        except Exception:
            pass
        if not videos:
            status.value = _callout("info", f"Uploaded {_esc(', '.join(saved))}. Now upload or choose "
                                            "the video itself.")
            return
        self._load_video(str(videos[0]), is_demo=False, status=status)

    def _load_video(self, path: str, is_demo: bool, status) -> None:
        try:
            info = probe(path)
            frame = read_frame_at(path, 0.0)
        except ConfigError as exc:
            status.value = _callout("error", _esc(exc))
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
        self._read_flight_log()
        self._identify_camera()
        named = s.camera_id
        # The drone the file names is chosen for the user, who can still change
        # it -- except in the vertical mode, which no preset describes.
        if named is not None and named.preset is not None and info.height <= info.width:
            s.scale_mode = "flight"
            s.camera_key = named.preset.key
            s.hfov_deg = round(named.preset.hfov_deg, 1)
        msg = f"Loaded <b>{_esc(Path(path).name)}</b>."
        if s.srt_records:
            msg += f" Found its flight log, <code>{_esc(Path(s.srt_path).name)}</code>."
        if named is not None:
            what = (f"a <b>{_esc(named.preset.label)}</b>" if named.preset
                    else f"&ldquo;{_esc(named.name)}&rdquo;")
            msg += f" The file says it was recorded by {what}."
        if named is None or named.preset is None:
            likely = preset_for_focal_length(shortest_focal_len(s.srt_records))
            if likely is not None:
                msg += (f" Its flight log points to a <b>{_esc(likely.label)}</b>, from the focal "
                        f"length it records (see the next step).")
        status.value = _callout("ok", msg) + (
            "" if info.fps > 0 else
            _callout("warn", "The file does not record its frame rate; you will be asked for it in the "
                             "next step.")
        )
        self._refresh_preview()
        self._refresh_summary()

    def _identify_camera(self) -> None:
        from .videometa import identify_camera

        s = self.state
        s.camera_id = (identify_camera(s.video_path)
                       if s.video_path and not s.is_demo else None)

    def _read_flight_log(self) -> None:
        from .srt import find_srt, parse_srt

        s = self.state
        srt = find_srt(s.video_path) if s.video_path else None
        s.srt_path = str(srt) if srt else None
        try:
            s.srt_records = parse_srt(srt) if srt else []
        except OSError:
            s.srt_records = []

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
        self._refresh_summary()

    def _reopen_survey(self, path: str, status) -> None:
        """Fill the wizard from a saved survey and go to the review step."""
        try:
            cfg = _read_survey(path)
        except ConfigError as exc:
            status.value = _callout("error", f"Could not read that survey: {_esc(exc)}")
            return
        if cfg.rectify.method == "gcp":
            status.value = _callout(
                "error", "That survey uses ground control points (an oblique view), which this wizard "
                         f"does not cover. Run it from a terminal instead: <code>stiv run {_esc(path)}</code>")
            return
        video = cfg.video.path
        if not video or not Path(video).is_file():
            status.value = _callout(
                "error", f"The video the survey refers to was not found: <code>{_esc(video)}</code>. "
                         "Put the video back there, or correct <code>video.path</code> in the file.")
            return
        start = float(cfg.video.start_s or 0.0)
        try:
            info = probe(video)
            frame = read_frame_at(video, start)
        except ConfigError as exc:
            status.value = _callout("error", _esc(exc))
            return

        self._forget_survey()
        s = self.state
        s.video_path, s.is_demo, s.info = video, False, info
        s.frame_bgr, s.frame_t_s, s.start_s = frame, start, start
        if cfg.video.duration_s:
            s.duration_s = float(cfg.video.duration_s)
        elif info.duration_s and info.duration_s > start:
            s.duration_s = round(info.duration_s - start, 3)        # null meant "to the end"
        else:
            s.duration_s = default_duration_s(info.duration_s, start)
        s.fps_override = cfg.video.fps_override
        s.frame_step = cfg.video.frame_step if cfg.video.frame_step > 1 else None
        self._read_flight_log()
        self._identify_camera()

        r = cfg.rectify
        if r.m_per_px is not None:
            s.scale_mode, s.m_per_px = "manual", float(r.m_per_px)
        elif r.altitude_m is not None and r.hfov_deg is not None:
            s.scale_mode, s.altitude_m, s.hfov_deg = "flight", float(r.altitude_m), float(r.hfov_deg)
            match = [p for p in DJI_CAMERAS if abs(round(p.hfov_deg, 1) - s.hfov_deg) < 0.051]
            # Drones can share a field of view (the Mini 3 Pro and Mini 4 Pro
            # do); the one the video names is the one it was.
            named = s.camera_id.preset if s.camera_id is not None else None
            if named in match:
                match = [named]
            s.camera_key = match[0].key if match else "custom"
        else:
            # Sensor width and focal length, which the wizard has no boxes for.
            s.scale_mode, s.m_per_px = "manual", r.resolve_m_per_px(info.width)

        sec = cfg.section
        if cfg.sections:
            self._reopen_sections(cfg)
        elif sec.start_px is not None and sec.end_px is not None:
            s.section_pts = [tuple(map(float, sec.start_px)), tuple(map(float, sec.end_px))]
        if cfg.flow.downstream_px is not None:
            s.downstream_pts = [tuple(map(float, cfg.flow.downstream_px))]
        # `auto` is outlined when the analysis runs; see _water_at_run.
        water_cfg = cfg.stabilise.water_mask_px
        polys = ([] if is_auto_water_mask(water_cfg)
                 else [list(map(tuple, p)) for p in (water_cfg or [])])
        if polys:
            s.water_pts, s.extra_water_polys, s.water_source = polys[0], polys[1:], "loaded"
        s.picks_t_s = start
        s.n_stations, s.edge_trim = sec.n_stations, sec.edge_trim
        s.velocity_range = tuple(cfg.stiv.velocity_range_ms)
        s.window_px, s.search_line_length_m = cfg.stiv.window_px, cfg.stiv.search_line_length_m
        s.method = cfg.velocimetry.method
        s.compare = [m for m in cfg.velocimetry.compare if m != s.method]
        s.estimator, s.ia_size_m = cfg.stiv.estimator, cfg.lspiv.ia_size_m
        s.stabilise = cfg.stabilise.enabled
        if cfg.sections:
            s.discharge_enabled = any(x["bathymetry_csv"] for x in self._sections())
        else:
            s.section_name = sec.name
            s.discharge_enabled = bool(cfg.discharge.enabled and cfg.discharge.bathymetry_csv)
            s.bathymetry_csv = cfg.discharge.bathymetry_csv
        s.alpha, s.bank_coefficient = cfg.discharge.alpha, cfg.discharge.bank_coefficient
        s.loaded_out_dir = cfg.output.out_dir
        s.pdf_report = cfg.output.pdf_report
        s.base_raw = cfg.to_dict()
        s.loaded_from = str(path)

        if len(s.section_pts) == 2 and s.downstream_pts:
            self._pick_stage = len(self._PICK_STAGES)
        else:
            self._pick_stage = 1 if len(s.section_pts) == 2 else 0
        self._water_key = self._water_inputs()      # keep the file's outline, detect nothing

        # Straight to the review, or to the first step that still needs input.
        target = 4
        for j in range(4):
            if not self._step_ready(j)[0]:
                target = j
                break
        self._furthest = max(target, 1)
        self.step = target
        self._show_error("")
        status.value = _callout("ok", f"Reopened <b>{_esc(Path(path).name)}</b>.")
        self._refresh()
        if target == 4:
            self._error.value = _callout(
                "ok", f"Survey <b>{_esc(Path(path).name)}</b> reopened. Review it and run, or use the "
                      "steps above to change anything first.")

    #: Keys of a ``sections`` entry the wizard shows and edits itself.
    _MANAGED_SECTION_KEYS = frozenset({"name", "start_px", "end_px", "bathymetry_csv"})

    def _reopen_sections(self, cfg: Config) -> None:
        """Fill the wizard's cross-sections from a survey's ``sections``.

        A section whose ends are not filled in yet is left out, bar the
        first, which the picking stages then ask for.  Shared depths with
        ``discharge.enabled`` become each section's own.
        """
        from dataclasses import asdict

        s = self.state
        shared = cfg.discharge.bathymetry_csv if cfg.discharge.enabled else None
        entries = []
        for k, e in enumerate(cfg.sections):
            have_ends = e.start_px is not None and e.end_px is not None
            if not have_ends and k > 0:
                continue
            entries.append({
                "name": e.name,
                "pts": ([tuple(map(float, e.start_px)), tuple(map(float, e.end_px))]
                        if have_ends else []),
                "bathymetry_csv": e.bathymetry_csv or (None if e.depths_m is not None else shared),
                "extra": {key: v for key, v in asdict(e).items()
                          if v is not None and key not in self._MANAGED_SECTION_KEYS},
            })
        first, rest = entries[0], entries[1:]
        s.section_pts = first["pts"]
        s.section_name = first["name"]
        s.bathymetry_csv = first["bathymetry_csv"]
        s.section_extra = first["extra"]
        s.more_sections = rest

    # ------------------------------------------------------------------ #
    # step 1 -- scale
    # ------------------------------------------------------------------ #
    def _build_step1(self):
        s = self.state
        info = s.info

        mode = W.ToggleButtons(
            options=[("I know the ground resolution", "manual"), ("Calculate from flight altitude", "flight")],
            value=s.scale_mode, style={"button_width": "auto"},
        )
        # Text boxes rather than number boxes, so they can start out blank:
        # the scale has no safe default and must be entered, not accepted.
        m_per_px_box = W.Text(value=_fmt_opt(s.m_per_px), placeholder="e.g. 0.02",
                              description="Ground resolution (m/pixel)", style=_DESC, layout=_field())
        altitude_box = W.Text(value=_fmt_opt(s.altitude_m), placeholder="e.g. 60",
                              description="Altitude above water (m)", style=_DESC, layout=_field())
        options = ([("Choose your drone…", "")] + [(p.label, p.key) for p in DJI_CAMERAS]
                   + [("Other camera: type its horizontal FOV", "custom")])
        cam_value = s.camera_key if s.camera_key in {v for _, v in options} else (
            "custom" if s.hfov_deg is not None else "")
        cam_dd = W.Dropdown(options=options, value=cam_value, description="Drone / camera",
                            style=_DESC, layout=W.Layout(width="560px"))
        hfov_box = W.Text(value=_fmt_opt(s.hfov_deg), placeholder="e.g. 71.5",
                          description="Horizontal FOV (degrees)", style=_DESC, layout=_field())
        cam_note = W.HTML()
        resolved = W.HTML()

        portrait = info.height > info.width
        fov_help = _help(
            "DJI spec sheets quote the <b>diagonal</b> field of view (e.g. 84&deg;); the scale needs "
            "the <b>horizontal</b> one, which is narrower, so the presets convert it for you. They "
            "assume the normal landscape video mode, which uses the full width of the sensor; digital "
            "zoom and cropped high-frame-rate modes narrow the view. If in doubt, check the scale "
            "against a distance you have measured on the ground."
        )
        portrait_note = W.HTML(_callout(
            "warn", "This video is taller than it is wide (vertical shooting mode). The drone presets "
                    "describe the normal landscape view and do not apply: enter the ground resolution "
                    "directly, or the horizontal field of view of this mode.") if portrait else "")

        # -- the flight log, when there is one ------------------------- #
        srt_widgets = []
        srt_note = W.HTML()
        srt_msg = W.HTML()
        takeoff_box = W.Text(value=_fmt_opt(s.takeoff_height_m), placeholder="e.g. 1.5 (0 if at water level)",
                             description="Take-off height above water (m)", style=_DESC, layout=_field())
        use_btn = W.Button(description="Use flight-log altitude", icon="download",
                           layout=W.Layout(width="220px"))
        if s.srt_records:
            srt_widgets = [srt_note, takeoff_box, use_btn, srt_msg]

        def _srt_stats():
            from .srt import altitude_stats

            return altitude_stats(s.srt_records, s.start_s, s.duration_s) if s.srt_records else None

        def _update_srt_note():
            st = _srt_stats()
            if st is None:
                srt_note.value = (_callout("info", f"Found the flight log <code>{_esc(Path(s.srt_path).name)}"
                                                   "</code>, but it records no height above take-off.")
                                  if s.srt_records else "")
                _show(takeoff_box, False)
                _show(use_btn, False)
                return
            spread = f" ({st.min_m:.1f}&ndash;{st.max_m:.1f} m)" if st.max_m - st.min_m >= 0.05 else ""
            assumed = (" The log calls this just &ldquo;altitude&rdquo;: on the Air 2S and Mavic 2 that "
                       "is the height above take-off, but check it is not the height above sea level."
                       if st.assumed else "")
            srt_note.value = _callout(
                "info", f"<b>Flight log found.</b> During the analysis window the drone was "
                        f"<b>{st.median_m:.1f} m above its take-off point</b>{spread}.{assumed} That is not "
                        f"its height above the water: enter how far the take-off point was above the water "
                        f"surface (0 if you took off at water level, negative if below it).")
            preset = camera(s.camera_key or "")
            narrowed = preset.zoomed_hfov_deg(st.focal_len_mm) if preset else None
            if narrowed is not None:
                zoom = st.focal_len_mm / preset.eq_focal_mm
                srt_note.value += _callout(
                    "warn", f"The flight log shows a {st.focal_len_mm:g} mm-equivalent focal length -- about "
                            f"{zoom:.1f}x digital zoom -- so the view is narrower than the preset's: about "
                            f"{narrowed:.1f}&deg; horizontally. Type that into the field of view box if the "
                            f"zoom was on for this clip.")

        def _use_srt(_btn):
            st = _srt_stats()
            off = _parse_float(takeoff_box.value)
            if st is None:
                return
            if off is None:
                srt_msg.value = _callout("warn", "Enter the take-off height above the water first "
                                                 "(0 if you took off at water level).")
                return
            altitude_box.value = f"{st.median_m + off:.2f}"
            srt_msg.value = _callout("ok", f"Altitude set to {st.median_m:.1f} + {off:g} = "
                                           f"<b>{st.median_m + off:.2f} m</b> above the water.")

        use_btn.on_click(_use_srt)

        def _on_takeoff(change):
            s.takeoff_height_m = _parse_float(change["new"])

        takeoff_box.observe(_on_takeoff, names="value")

        manual_box = W.VBox([m_per_px_box,
                             _help("The ground distance one pixel covers, e.g. from a measured "
                                   "distance between two marks visible in the frame.")])
        flight_box = W.VBox([portrait_note, *srt_widgets, altitude_box,
                             _help("Height above the <b>water surface</b>, not the take-off point."),
                             cam_dd, hfov_box, cam_note, fov_help])

        named = s.camera_id
        # The lens the flight log records: zoom only lengthens it, so no lens
        # logs a focal length shorter than its own.
        shortest = shortest_focal_len(s.srt_records)
        likely = preset_for_focal_length(shortest)

        def _cam_note():
            preset = camera(cam_dd.value)
            if preset:
                cam_note.value = f"<div class='stiv-help'><b>{_esc(preset.label)}:</b> {preset.describe()}</div>"
            elif cam_dd.value == "custom":
                cam_note.value = "<div class='stiv-help'>Enter your camera's horizontal field of view.</div>"
            else:
                cam_note.value = ""
            if named is not None:
                said = f"&ldquo;{_esc(named.name)}&rdquo;"
                if named.preset is None:
                    cam_note.value += (
                        f"<div class='stiv-help'>The video names its camera {said}, which is not in the "
                        f"list: choose <i>Other camera</i> and type its horizontal field of view.</div>")
                elif portrait:
                    pass                            # the portrait warning says why not
                elif cam_dd.value == named.preset.key:
                    cam_note.value += (f"<div class='stiv-help'>Chosen from the video itself, whose "
                                       f"metadata says {said}.</div>")
                elif preset is not None:
                    cam_note.value += _callout(
                        "warn", f"The video's metadata says it was recorded by the "
                                f"<b>{_esc(named.preset.label)}</b> ({said}), not the {_esc(preset.label)}. "
                                f"Check which drone you flew.")
            if shortest is None or portrait:
                return
            if preset is not None and not preset.can_log_focal_length(shortest):
                cam_note.value += _callout(
                    "warn", f"The flight log records a {shortest:g} mm-equivalent focal length, wider than "
                            f"the {_esc(preset.label)}'s {preset.eq_focal_mm:g} mm lens can give"
                            + (f": of the drones listed, only the <b>{_esc(likely.label)}</b>'s lens is "
                               f"that wide" if likely else "")
                            + ". Check which drone you flew.")
            elif (likely is not None and not cam_dd.value
                  and (named is None or named.preset is None)):
                cam_note.value += _callout(
                    "info", f"The flight log records a {shortest:g} mm-equivalent focal length, and of the "
                            f"drones listed only the <b>{_esc(likely.label)}</b>'s lens is that wide. If "
                            f"that is what you flew, choose it above.")

        def _on_cam(change):
            s.camera_key = change["new"] or None
            preset = camera(change["new"])
            if preset:
                hfov_box.value = f"{preset.hfov_deg:.1f}"
            _cam_note()
            if s.srt_records:
                _update_srt_note()

        def _on_hfov(change):
            preset = camera(cam_dd.value)
            typed = _parse_positive(change["new"])
            if preset and (typed is None or abs(typed - round(preset.hfov_deg, 1)) > 1e-6):
                cam_dd.value = "custom"

        def _update_resolved(*_):
            s.scale_mode = mode.value
            s.m_per_px = _parse_positive(m_per_px_box.value)
            s.altitude_m = _parse_positive(altitude_box.value)
            s.hfov_deg = _parse_positive(hfov_box.value)
            _show(manual_box, mode.value == "manual")
            _show(flight_box, mode.value == "flight")
            m = self._resolved_m_per_px()
            if m:
                gw = m * info.width
                resolved.value = _tiles([
                    ("Ground resolution", f"{m * 100:.2f} cm/px", f"{m:.4f} m per pixel"),
                    ("Frame width on the ground", f"{gw:.1f} m", f"{info.width} pixels"),
                ]) + self._frame_step_table(m)
            else:
                resolved.value = _callout(
                    "info", "Enter a ground resolution, or an altitude and a field of view, to see the "
                            "scale." if mode.value == "manual" or s.altitude_m is None else
                            "Choose your drone, or type its horizontal field of view, to see the scale.")
            self._refresh_summary()

        cam_dd.observe(_on_cam, names="value")
        hfov_box.observe(_on_hfov, names="value")
        mode.observe(_update_resolved, names="value")
        m_per_px_box.observe(_update_resolved, names="value")
        altitude_box.observe(_update_resolved, names="value")
        hfov_box.observe(_update_resolved, names="value")
        _cam_note()
        if s.srt_records:
            _update_srt_note()
        _update_resolved()

        max_dur = info.duration_s if info.duration_s and info.duration_s > 0 else 3600.0
        start_s_box = W.BoundedFloatText(value=min(s.start_s, max_dur), min=0.0, max=max_dur, step=0.5,
                                         description="Start (s)", style=_DESC, layout=_field())
        duration_box = W.BoundedFloatText(value=min(s.duration_s, max_dur), min=1.0, max=max_dur, step=1.0,
                                          description="Duration (s)", style=_DESC, layout=_field())

        def _update_time(*_):
            s.start_s = start_s_box.value
            s.duration_s = duration_box.value
            if s.srt_records:
                _update_srt_note()
            self._refresh_summary()

        start_s_box.observe(_update_time, names="value")
        duration_box.observe(_update_time, names="value")

        # Every velocity is proportional to the frame rate.  A file that does
        # not record one (a raw .h264 stream, say) cannot be analysed without
        # it, and one whose container reports the wrong rate needs correcting.
        known = info.fps > 0
        fps_box = W.Text(value=_fmt_opt(s.fps_override),
                         placeholder=f"{info.fps:.3f} (from the file)" if known else "e.g. 30",
                         description="Frame rate (fps)", style=_DESC, layout=_field())
        fps_note = W.HTML(
            _help("Leave blank to use the frame rate the file reports; set it only if the run warns "
                  "that the rate is wrong.").value if known else
            _callout("error", "<b>This video does not record its frame rate.</b> Enter the rate it was "
                              "recorded at -- every velocity is proportional to it.")
        )

        def _update_fps(*_):
            s.fps_override = _parse_positive(fps_box.value)
            _update_resolved()            # the frame_step suggestions depend on it

        fps_box.observe(_update_fps, names="value")

        return W.VBox([
            _heading("2. Set the scale",
                     "How many metres one pixel covers on the water. Every velocity is proportional to "
                     "it, so it is never filled in for you."),
            _card("Scale", mode, W.HTML("<div style='height:6px'></div>"), manual_box, flight_box, resolved),
            _card("Part of the video to analyse",
                  _help("20&ndash;40 s of steady hover is usually plenty. The cross-section is marked "
                        "on the frame at the start time in the next step."),
                  start_s_box, duration_box),
            _card("Frame rate", fps_note, fps_box),
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
            _refresh_matplotlib_backends()
            ip.run_line_magic("matplotlib", "widget")
        except Exception as exc:
            if _in_colab():
                fix = ("Run the notebook's setup cell first (it turns on the widget support ipympl "
                       "needs). If it has run, use Runtime > Restart session, then run the cells "
                       "again: the setup cell installed ipympl after matplotlib had started.")
            else:
                fix = (f"Install it with `{INSTALL_COMMAND}`, then restart the kernel and re-run "
                       f"this cell.")
            raise RuntimeError(
                f"the interactive picker needs the ipympl backend and a running Jupyter kernel "
                f"({exc}). {fix}"
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
        has_points = bool(s.section_pts or s.downstream_pts or s.water_pts or s.more_sections)
        return has_points and s.picks_t_s is not None and s.picks_t_s != s.frame_t_s

    def _water_inputs(self) -> tuple:
        """What an automatic water outline depends on."""
        s = self.state
        return (s.video_path, s.frame_t_s, tuple(map(tuple, s.section_pts)),
                tuple(map(tuple, s.downstream_pts)))

    def _detect_water(self) -> None:
        """Outline the water automatically and put it in ``water_pts``."""
        from .watermask import detect_water

        s = self.state
        self._water_key = self._water_inputs()
        try:
            det = detect_water(s.video_path, s.frame_t_s, s.section_pts,
                               s.downstream_pts[0] if s.downstream_pts else None,
                               fps=s.fps_override)
        except Exception as exc:   # footage the detector cannot read: fall back to clicking
            from .watermask import WaterDetection

            det = WaterDetection([], "failed", f"the detector stopped: {exc}")
        self._water_detection = det
        if det.ok:
            s.water_pts[:] = list(det.polygon_px)
            s.water_source = "auto"
            s.picks_t_s = s.frame_t_s

    def _water_note_html(self) -> str:
        s = self.state
        det = self._water_detection
        if s.water_source == "loaded" and len(s.water_pts) >= 3:
            return _callout("info", "The water outline (green) is the one saved with the survey.")
        if det is None:
            return ""
        out = ""
        if s.water_source == "auto" and det.ok:
            kind = "ok" if det.confidence == "good" else "warn"
            check = ("" if det.confidence == "good" else
                     f"<br><b>Check it carefully:</b> {_esc(det.message)}.")
            out = _callout(kind, "<b>The water was outlined automatically</b> (green). It should cover the "
                                 "moving water around the section but not the banks; it need not be exact, "
                                 "and too much water is better than too little. Press <b>Confirm</b> to "
                                 "use it, or <b>Clear stage</b> and click your own outline." + check)
        elif not det.ok:
            out = _callout("info", f"Could not outline the water automatically: {_esc(det.message)}. Click "
                                   "around the edge of the water and press <b>Confirm</b>, or press "
                                   "<b>Skip</b>.")
        if det.flow_agrees is False and s.downstream_pts:
            out += _callout("warn", "<b>The water appears to flow the other way</b>, towards the other side "
                                    "of the cross-section from your downstream point. Use <b>Move "
                                    "downstream point across</b> if it is on the wrong side.")
        return out

    def _stage_list_html(self) -> str:
        chips = []
        for i, stage in enumerate(self._PICK_STAGES):
            dot = f"<b style='color:{self._PICK_COLOURS[stage['attr']]}'>&#9679;</b>"
            if i < self._pick_stage:
                chips.append(f"<span class='done'>{dot} ✓ {stage['label']}</span>")
            elif i == self._pick_stage:
                chips.append(f"<span class='now'>{dot} {i + 1}. {stage['label']}</span>")
            else:
                chips.append(f"<span>{dot} {i + 1}. {stage['label']}</span>")
        return "<div class='stiv-chips'>" + "".join(chips) + "</div>"

    def _build_step2(self):
        s = self.state
        key = (s.video_path, s.start_s)
        if self._step2_box is not None and self._step2_key == key:
            if self._step2_sync is not None:
                self._step2_sync()
            return self._step2_box

        stale_from = s.picks_t_s
        stale = self._picker_frame()
        if stale:
            # Keep the points, but make every stage be looked at and confirmed
            # again on the new frame before the wizard will move on.
            self._pick_stage = 0

        self._ensure_interactive_backend()
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        if self._fig is not None:
            plt.close(self._fig)

        frame = s.frame_bgr
        h, w = frame.shape[:2]
        # Made with interactive mode off: ipympl would otherwise queue the new
        # figure and display its canvas again under whichever cell runs next,
        # a second picker outside the wizard.  The canvas is shown in the
        # wizard's own panel below.
        # Sized to the frame's own shape and to the wizard's content column, so
        # the image fills the canvas: no bands, and nothing cut off at the side.
        fig_w, fig_h = 7.0, 7.0 * h / w
        if fig_h > 6.0:
            fig_w, fig_h = 6.0 * w / h, 6.0
        with plt.ioff():
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.canvas.header_visible = False
        ax.imshow(frame[:, :, ::-1])
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_autoscale_on(False)
        ax.set_axis_off()
        self._fig, self._ax = fig, ax

        self._pick_stage = min(self._pick_stage, len(self._PICK_STAGES))
        stages_html = W.HTML()
        stage_label = W.HTML()
        water_note = W.HTML()
        confirm_btn = W.Button(description="Confirm", icon="check", button_style="success")
        skip_btn = W.Button(description="Skip", icon="forward", layout=W.Layout(display="none"))
        detect_btn = W.Button(description="Detect water", icon="magic", layout=W.Layout(display="none"))
        flip_btn = W.Button(description="Move downstream point across", icon="exchange",
                            button_style="warning", layout=W.Layout(display="none", width="auto"))
        undo_btn = W.Button(description="Undo point", icon="undo")
        clear_btn = W.Button(description="Clear stage", icon="eraser")
        restart_btn = W.Button(description="Restart", icon="refresh")
        # Further cross-sections, once the stages are done: all are analysed
        # together, in one pass over the video.
        add_sec_btn = W.Button(description="Add another cross-section", icon="plus",
                               layout=W.Layout(display="none", width="auto"))
        remove_sec_btn = W.Button(description="Remove the last one", icon="minus",
                                  layout=W.Layout(display="none", width="auto"))
        cancel_add_btn = W.Button(description="Cancel", icon="times", layout=W.Layout(display="none"))

        lines = {
            stage["attr"]: ax.plot([], [], "o-", color=self._PICK_COLOURS[stage["attr"]], markersize=7,
                                   markeredgecolor="white", markeredgewidth=1.3, lw=2)[0]
            for stage in self._PICK_STAGES
        }
        water_fill = Polygon(np.zeros((3, 2)), closed=True, facecolor=GREEN, edgecolor="none",
                             alpha=0.25, visible=False)
        ax.add_patch(water_fill)
        more_line = ax.plot([], [], "o-", color=ORANGE, markersize=7, markeredgecolor="white",
                            markeredgewidth=1.3, lw=2)[0]
        adding_line = ax.plot([], [], "o--", color=ORANGE, markersize=7, markeredgecolor="white",
                              markeredgewidth=1.3, lw=2)[0]
        numbers: list = []

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
            # An automatic outline has dozens of vertices; draw it as an area.
            water = s.water_pts
            lines["water_pts"].set_marker("o" if s.water_source != "auto" else "")
            if len(water) >= 3:
                water_fill.set_xy(np.array(water))
                water_fill.set_visible(True)
            else:
                water_fill.set_visible(False)
            # The further cross-sections, and the one being added; numbered
            # once there is more than one.
            xs, ys = [], []
            for sec in s.more_sections:
                (x0, y0), (x1, y1) = sec["pts"]
                xs += [x0, x1, np.nan]
                ys += [y0, y1, np.nan]
            more_line.set_data(xs, ys)
            adding = self._adding_pts or []
            adding_line.set_data([p[0] for p in adding], [p[1] for p in adding])
            for t in numbers:
                t.remove()
            numbers.clear()
            if s.more_sections or self._adding_pts:
                ends = [s.section_pts] + [sec["pts"] for sec in s.more_sections] + [adding]
                for k, pts in enumerate(ends, start=1):
                    if pts:
                        numbers.append(ax.annotate(
                            str(k), pts[0], xytext=(-9, 9), textcoords="offset points",
                            ha="right", va="bottom", color="white", fontsize=10, weight="bold",
                            bbox=dict(boxstyle="circle,pad=0.25", fc=ORANGE, ec="white", lw=1)))
            fig.canvas.draw_idle()

        stale_note = (
            _callout("warn", f"The analysis now starts at {s.frame_t_s:.1f} s, so this is the frame at "
                             f"{s.frame_t_s:.1f} s; the points were placed on the frame at {stale_from:.1f} s. "
                             f"Any drone drift in between offsets them. Check each stage and press Confirm, "
                             f"or use Clear / Restart to re-click.")
            if stale else ""
        )

        def _update_controls():
            stage = _current_stage()
            stages_html.value = self._stage_list_html()
            if self._adding_pts is not None:
                for b in (skip_btn, detect_btn, flip_btn, add_sec_btn, remove_sec_btn):
                    _show(b, False)
                _show(cancel_add_btn, True)
                water_note.value = ""
                n = len(self._adding_pts)
                stage_label.value = _callout(
                    "info", f"<b>Cross-section {len(s.more_sections) + 2}.</b> Click its two ends, bank "
                            f"to bank, then press <b>Confirm</b> ({n} of 2 clicked).")
                confirm_btn.disabled = n < 2
                self._refresh_summary()
                return
            _show(cancel_add_btn, False)
            on_water = stage is not None and stage["attr"] == "water_pts"
            if on_water and len(s.water_pts) < 3 and self._water_key != self._water_inputs():
                stage_label.value = _callout("info", "Looking for the water&hellip;")
                self._detect_water()
                _redraw()
            water_note.value = self._water_note_html() if (on_water or stage is None) else ""
            _show(detect_btn, on_water)
            det = self._water_detection
            _show(flip_btn, bool(det and det.flow_agrees is False and s.downstream_pts))
            self._refresh_summary()
            _show(add_sec_btn, stage is None)
            _show(remove_sec_btn, stage is None and bool(s.more_sections))
            if stage is None:
                s.picks_t_s = s.frame_t_s
                n_sec = len(s.more_sections) + 1
                if n_sec == 1:
                    msg = ("<b>All set.</b> Press <b>Next</b> to continue, or <b>Restart</b> to redo the "
                           "picking. To measure in more than one place, <b>Add another "
                           "cross-section</b>: they are all analysed together.")
                else:
                    msg = (f"<b>All set: {n_sec} cross-sections.</b> Press <b>Next</b> to continue, or add "
                           f"another. The one downstream point (blue) serves them all, so it should be "
                           f"downstream of every one.")
                stage_label.value = _callout("ok", msg)
                confirm_btn.disabled = True
                _show(skip_btn, False)
                return
            n = len(_points(stage))
            need = stage["min"]
            extra = f", need at least {need}" if n < need else ""
            if on_water and s.water_source == "auto":
                prompt = "Optional: check the water outline found for you, then Confirm."
            else:
                prompt = f"{stage['prompt']} ({n} point{'s' if n != 1 else ''} clicked{extra})"
            stage_label.value = stale_note + _callout(
                "info", f"<b>Stage {self._pick_stage + 1} of {len(self._PICK_STAGES)}.</b> {prompt}")
            confirm_btn.disabled = n < need
            _show(skip_btn, need == 0)

        def _on_click(event):
            if event.inaxes is not ax or event.xdata is None:
                return
            if self._adding_pts is not None:
                if event.button == 3:
                    _on_undo(None)
                    return
                if len(self._adding_pts) < 2:
                    self._adding_pts.append((float(event.xdata), float(event.ydata)))
                    s.picks_t_s = s.frame_t_s
                _redraw()
                _update_controls()
                return
            stage = _current_stage()
            if stage is None:
                return
            if event.button == 3:
                _on_undo(None)
                return
            pts = _points(stage)
            if stage["attr"] == "water_pts":
                if s.water_source in ("auto", "loaded"):
                    pts.clear()      # a click starts a hand-drawn outline afresh
                s.water_source = "manual"
            if stage["max"] is not None and len(pts) >= stage["max"]:
                return
            pts.append((float(event.xdata), float(event.ydata)))
            s.picks_t_s = s.frame_t_s   # the frame these points belong to
            _redraw()
            _update_controls()

        def _on_undo(_btn):
            if self._adding_pts is not None:
                if self._adding_pts:
                    self._adding_pts.pop()
                _redraw()
                _update_controls()
                return
            stage = _current_stage()
            if stage is None:
                return
            pts = _points(stage)
            if stage["attr"] == "water_pts" and s.water_source in ("auto", "loaded"):
                pts.clear()          # one undo removes the whole automatic outline
                s.water_source = None
            elif pts:
                pts.pop()
            _redraw()
            _update_controls()

        def _on_clear(_btn):
            if self._adding_pts is not None:
                self._adding_pts.clear()
                _redraw()
                _update_controls()
                return
            stage = _current_stage()
            if stage is None:
                return
            _points(stage).clear()
            if stage["attr"] == "water_pts":
                s.water_source = None
            _redraw()
            _update_controls()

        def _on_confirm(_btn):
            if self._adding_pts is not None:
                if len(self._adding_pts) == 2:
                    s.more_sections.append({"name": None, "pts": list(self._adding_pts),
                                            "bathymetry_csv": None, "extra": {}})
                    self._adding_pts = None
                _redraw()
                _update_controls()
                return
            s.picks_t_s = s.frame_t_s
            self._pick_stage += 1
            _redraw()
            _update_controls()

        def _on_restart(_btn):
            for stage in self._PICK_STAGES:
                _points(stage).clear()
            s.more_sections.clear()
            self._adding_pts = None
            s.extra_water_polys = []
            s.water_source = None
            s.picks_t_s = None
            self._pick_stage = 0
            self._water_key = None
            self._water_detection = None
            _redraw()
            _update_controls()

        def _on_detect(_btn):
            detect_btn.disabled = True
            stage_label.value = _callout("info", "Looking for the water&hellip;")
            s.water_pts.clear()
            s.water_source = None
            self._detect_water()
            detect_btn.disabled = False
            _redraw()
            _update_controls()

        def _on_flip(_btn):
            # Mirror the downstream point across the line of the section.
            a, b = (np.asarray(p, float) for p in s.section_pts)
            u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
            n = np.array([-u[1], u[0]])
            p = np.asarray(s.downstream_pts[0], float)
            q = p - 2.0 * float(np.dot(p - a, n)) * n
            q = np.clip(q, [0.0, 0.0], [w - 1.0, h - 1.0])
            s.downstream_pts[:] = [(float(q[0]), float(q[1]))]
            if self._water_detection is not None:
                self._water_detection.flow_agrees = True
            self._water_key = self._water_inputs()
            _redraw()
            _update_controls()

        def _on_add(_btn):
            self._adding_pts = []
            _redraw()
            _update_controls()

        def _on_remove(_btn):
            if s.more_sections:
                s.more_sections.pop()
            _redraw()
            _update_controls()

        def _on_cancel(_btn):
            self._adding_pts = None
            _redraw()
            _update_controls()

        add_sec_btn.on_click(_on_add)
        remove_sec_btn.on_click(_on_remove)
        cancel_add_btn.on_click(_on_cancel)
        fig.canvas.mpl_connect("button_press_event", _on_click)
        confirm_btn.on_click(_on_confirm)
        skip_btn.on_click(_on_confirm)
        undo_btn.on_click(_on_undo)
        clear_btn.on_click(_on_clear)
        restart_btn.on_click(_on_restart)
        detect_btn.on_click(_on_detect)
        flip_btn.on_click(_on_flip)

        def _sync():
            _redraw()
            _update_controls()

        _sync()

        box = W.VBox([
            _heading("3. Mark the cross-section",
                     "Click directly on the frame the analysis starts from."),
            stages_html,
            stage_label,
            water_note,
            fig.canvas,
            W.HBox([confirm_btn, cancel_add_btn, skip_btn, detect_btn, undo_btn, clear_btn, restart_btn],
                   layout=W.Layout(flex_flow="row wrap")),
            flip_btn,
            W.HBox([add_sec_btn, remove_sec_btn], layout=W.Layout(flex_flow="row wrap")),
            _help("Left click = add a point &middot; right click = undo &middot; use the toolbar beside "
                  "the image to zoom in first if you need precision."),
        ])
        self._step2_box = box
        self._step2_key = key
        self._step2_sync = _sync
        return box

    # ------------------------------------------------------------------ #
    # step 3 -- settings
    # ------------------------------------------------------------------ #
    def _build_step3(self):
        s = self.state
        n_stations = W.IntSlider(value=s.n_stations, min=5, max=100, step=1, description="Stations",
                                 style=_DESC, layout=_field("520px"))
        vrange = W.FloatRangeSlider(value=list(s.velocity_range), min=-2.0, max=10.0, step=0.1,
                                    description="Expected velocity (m/s)", style=_DESC, layout=_field("520px"))
        stabilise = W.Checkbox(value=s.stabilise, indent=False,
                               description="Compensate for drone motion (leave on unless the camera "
                                           "was on a fixed mount)",
                               layout=W.Layout(width="auto"))

        frame_step_box = W.Text(value=str(s.frame_step) if s.frame_step else "", placeholder="auto (1)",
                                description="Frame step", style=_DESC, layout=_field())
        window_px = W.IntSlider(value=s.window_px, min=11, max=81, step=2, description="Window size (px)",
                                style=_DESC, layout=_field("520px"))
        edge_trim = W.FloatSlider(value=s.edge_trim, min=0.0, max=0.4, step=0.01, description="Edge trim",
                                  style=_DESC, layout=_field("520px"))
        estimator_dd = W.Dropdown(options=[("Eq. 14 weighted mean (the paper)", "eq14"),
                                           ("Histogram peak", "peak"),
                                           ("Fourier spectrum (FFT)", "fft")],
                                  value=s.estimator, description="STIV angle from", style=_DESC,
                                  layout=_field("520px"))
        ia_box = W.Text(value=f"{s.ia_size_m:g}" if s.ia_size_m else "",
                        placeholder="auto (the station spacing)",
                        description="LS-PIV area (m)", style=_DESC, layout=_field())
        advanced = W.Accordion(children=[W.VBox([
            _help("Frame step: analyse every n-th frame (step 2 suggested values for your scale). "
                  "Window size: the patch the streak angle is measured over. Edge trim: the fraction of "
                  "the section skipped at each bank. STIV angle: how the streak angle is read from each "
                  "space-time image; the Fourier spectrum copes better with noisy images. LS-PIV area: "
                  "the side of the square of water each LS-PIV station correlates."),
            frame_step_box, window_px, edge_trim, estimator_dd, ia_box,
        ])])
        advanced.set_title(0, "Advanced (optional)")
        advanced.selected_index = None

        # The velocity method: one reported, the others optionally run beside
        # it.  The reported one cannot also be compared, so its box is off.
        method_dd = W.Dropdown(
            options=[("STIV: space-time images (default)", "stiv"),
                     ("LS-PIV: patch correlation", "lspiv"),
                     ("Particle tracking (PTV)", "ptv")],
            value=s.method, description="Report", style=_DESC, layout=_field("520px"))
        compare_boxes = {m: W.Checkbox(value=m in s.compare, indent=False,
                                       description=METHOD_LABELS[m], layout=W.Layout(width="auto"))
                         for m in METHODS}
        method_note = W.HTML()
        _method_notes = {
            "stiv": "Reads the slope of streaks in a space-time image along a line down the flow at "
                    "each station: streamwise velocity only. Robust to sparse or patchy texture.",
            "lspiv": "Correlates a small square of water at each station between frames: a 2-D "
                     "velocity, so the arrows show where the water went. Needs dense, lasting texture "
                     "(ripples, boils).",
            "ptv": "Follows corners on the water (foam, leaves, debris) from frame to frame: a 2-D "
                   "velocity. Best with distinct floating tracers; weak on smooth water.",
        }

        def _sync_methods(change=None):
            for m, box in compare_boxes.items():
                box.disabled = m == method_dd.value
                if box.disabled and box.value:
                    box.value = False
            s.method = method_dd.value
            s.compare = [m for m in METHODS if compare_boxes[m].value and m != s.method]
            method_note.value = f"<div class='stiv-help'>{_method_notes[s.method]}</div>"
            if change is not None:          # not while the step is still being built
                self._refresh_summary()

        _sync_methods()
        method_dd.observe(_sync_methods, names="value")
        for box in compare_boxes.values():
            box.observe(_sync_methods, names="value")
        method_card = _card(
            "Velocity method",
            method_dd, method_note,
            W.HBox([W.HTML("<span class='stiv-help'>Also run, to compare:&nbsp;</span>"),
                    *compare_boxes.values()], layout=W.Layout(flex_flow="row wrap",
                                                              align_items="center")),
            _help("Methods run to compare are measured in the same pass over the video and shown "
                  "beside the reported one, each with its own discharge; only the reported one is "
                  "in the headline numbers."))

        discharge_chk = W.Checkbox(value=s.discharge_enabled, indent=False,
                                   description="Also compute discharge (needs a depth / bathymetry file)",
                                   layout=W.Layout(width="auto"))
        bathy_fc = FileChooser(_browse_root())
        bathy_fc.filter_pattern = ["*.csv"]
        bathy_fc.title = "<b>Bathymetry CSV</b> (columns: distance_m, depth_m)"
        bathy_upload = W.FileUpload(accept=".csv", multiple=False, description="Upload CSV", icon="upload")
        alpha = W.FloatSlider(value=s.alpha, min=0.5, max=1.2, step=0.01, description="Velocity index (alpha)",
                              style=_DESC, layout=_field("520px"))
        bank_coeff = W.FloatSlider(value=s.bank_coefficient, min=0.0, max=1.0, step=0.05,
                                   description="Bank coefficient", style=_DESC, layout=_field("520px"))
        bathy_note = W.HTML()
        multi = bool(s.more_sections)

        # With several cross-sections each has its own depths, chosen one
        # section at a time; a section with none reports velocities only.
        def _target_options():
            return [(f"{k + 1}. {sec['name']}", k) for k, sec in enumerate(self._sections())]

        bathy_target = W.Dropdown(options=_target_options(), value=0, description="Depths for",
                                  style=_DESC, layout=_field())
        bathy_clear = W.Button(description="No depths for this section", icon="times",
                               layout=W.Layout(width="auto"))

        def _set_bathy(path: Optional[str]) -> None:
            k = bathy_target.value if multi else 0
            if k == 0:
                s.bathymetry_csv = path
            else:
                s.more_sections[k - 1]["bathymetry_csv"] = path

        def _bathy_note():
            if not multi:
                bathy_note.value = (f"<div class='stiv-help'>Using <code>{_esc(s.bathymetry_csv)}</code></div>"
                                    if s.bathymetry_csv else "")
                return
            rows = "".join(
                f"<tr><td>{k + 1}. {_esc(sec['name'])}</td><td>"
                + (f"<code>{_esc(Path(sec['bathymetry_csv']).name)}</code>" if sec["bathymetry_csv"]
                   else "<span class='stiv-help'>none: velocities only</span>")
                + "</td></tr>"
                for k, sec in enumerate(self._sections()))
            bathy_note.value = f"<table class='stiv-table'>{rows}</table>"

        _bathy_note()
        bathy_clear.on_click(lambda _b: (_set_bathy(None), _bathy_note(), self._refresh_summary()))
        discharge_box = W.VBox(
            ([_help("Each cross-section has its own depths. Pick the section, then choose or upload "
                    "its CSV; sections without depths report velocities only."),
              bathy_target] if multi else [])
            + [bathy_fc, bathy_upload]
            + ([bathy_clear] if multi else [])
            + [bathy_note, alpha, bank_coeff,
               _help("alpha converts surface to depth-averaged velocity (0.85 is the "
                     "usual value for natural channels).")])
        _show(discharge_box, s.discharge_enabled)

        def _on_discharge_toggle(change):
            s.discharge_enabled = change["new"]
            _show(discharge_box, change["new"])
            self._refresh_summary()

        discharge_chk.observe(_on_discharge_toggle, names="value")

        def _on_bathy(chooser):
            _set_bathy(chooser.selected)
            _bathy_note()
            self._refresh_summary()

        bathy_fc.register_callback(_on_bathy)

        def _on_bathy_upload(_change):
            files = bathy_upload.value
            if not files:
                return
            if isinstance(files, dict):
                files = [{"name": k, "content": v["content"]} for k, v in files.items()]
            dest = Path.cwd() / "stiv_uploads"
            dest.mkdir(exist_ok=True)
            path = dest / Path(files[0]["name"]).name
            path.write_bytes(bytes(files[0]["content"]))
            _set_bathy(str(path))
            _bathy_note()
            self._refresh_summary()

        bathy_upload.observe(_on_bathy_upload, names="value")

        out_dir_box = W.Text(value=self._out_dir(), description="Save results to", style=_DESC,
                             layout=W.Layout(width="620px"))

        def _on_out_dir(change):
            # Only an edit here is a choice that outlives this survey; the name
            # shown until then follows the video, and an emptied box returns
            # to it.
            s.out_dir = change["new"].strip() or None
            self._refresh_summary()

        out_dir_box.observe(_on_out_dir, names="value")

        pdf_chk = W.Checkbox(value=s.pdf_report, indent=False,
                             description="Also write a PDF report (report.pdf)",
                             layout=W.Layout(width="auto"))

        def _on_pdf(change):
            s.pdf_report = change["new"]

        pdf_chk.observe(_on_pdf, names="value")

        def _sync(*_):
            s.n_stations = n_stations.value
            s.velocity_range = tuple(vrange.value)
            s.stabilise = stabilise.value
            s.frame_step = int(frame_step_box.value) if frame_step_box.value.strip().isdigit() else None
            s.window_px = window_px.value
            s.edge_trim = edge_trim.value
            s.estimator = estimator_dd.value
            try:
                ia = float(ia_box.value.strip()) if ia_box.value.strip() else None
            except ValueError:
                ia = None
            s.ia_size_m = ia if ia is not None and ia > 0 else None
            s.alpha = alpha.value
            s.bank_coefficient = bank_coeff.value
            self._refresh_summary()

        for w_ in (n_stations, vrange, stabilise, frame_step_box, window_px, edge_trim, estimator_dd,
                   ia_box, alpha, bank_coeff):
            w_.observe(_sync, names="value")

        names_card = []
        if multi:
            name_boxes = []
            for k, sec in enumerate(self._sections()):
                box = W.Text(value=sec["name"], description=f"Cross-section {k + 1}", style=_DESC,
                             layout=_field())

                def _on_name(change, k=k):
                    name = change["new"].strip() or None
                    if k == 0:
                        s.section_name = name
                    else:
                        s.more_sections[k - 1]["name"] = name
                    keep = bathy_target.value
                    bathy_target.options = _target_options()
                    bathy_target.value = keep
                    _bathy_note()
                    self._refresh_summary()

                box.observe(_on_name, names="value")
                name_boxes.append(box)
            names_card = [_card("Cross-sections",
                                _help("All are analysed together, in one pass over the video. Each "
                                      "name also names the folder its results go in."),
                                *name_boxes)]

        return W.VBox([
            _heading("4. Analysis settings", "The defaults suit most surveys; adjust them if you know the river."),
            *names_card,
            _card("Measurement",
                  _help("Stations are the points across the section where velocity is measured"
                        + (" (the same number at every section)" if multi else "")
                        + ". The expected velocity range bounds the search: make it generous, but "
                          "not absurd."),
                  n_stations, vrange, advanced),
            method_card,
            _card("Camera motion", stabilise),
            _card("Discharge", discharge_chk, discharge_box),
            _card("Output", out_dir_box, pdf_chk,
                  _help("The PDF holds the summary, every figure, the station table and these "
                        "settings in one document, to keep or hand in. Running again into the same "
                        "folder replaces the results already there.")),
        ])

    # ------------------------------------------------------------------ #
    # step 4 -- run
    # ------------------------------------------------------------------ #
    def _build_config(self) -> Config:
        s = self.state
        if not s.video_path:
            raise ConfigError("no video has been loaded (step 1)")
        manual = s.scale_mode == "manual"
        water = ([s.water_pts] if len(s.water_pts) >= 3 else []) + list(s.extra_water_polys)
        secs = self._sections()
        sections = None
        if len(secs) > 1:
            sections = []
            for sec in secs:
                pts = sec["pts"]
                sections.append({
                    "name": sec["name"],
                    "start_px": pts[0] if len(pts) == 2 else None,
                    "end_px": pts[1] if len(pts) == 2 else None,
                    "bathymetry_csv": sec["bathymetry_csv"] if s.discharge_enabled else None,
                })
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
            water_mask_px=water or None,
            search_line_length_m=s.search_line_length_m,
            window_px=s.window_px,
            velocity_range_ms=s.velocity_range,
            bathymetry_csv=s.bathymetry_csv if s.discharge_enabled and sections is None else None,
            alpha=s.alpha,
            bank_coefficient=s.bank_coefficient,
            out_dir=self._out_dir(),
            sections=sections,
            method=s.method,
            compare=s.compare,
            estimator=s.estimator,
            ia_size_m=s.ia_size_m,
        )
        raw["output"]["pdf_report"] = bool(s.pdf_report)
        for entry, sec in zip(raw.get("sections", []), secs):
            # What a reopened survey gave each section that the wizard has no
            # control for: its own station count, downstream point, depths...
            for key, value in sec["extra"].items():
                if key == "depths_m" and (not s.discharge_enabled or sec["bathymetry_csv"]):
                    continue
                entry.setdefault(key, copy.deepcopy(value))
        if s.base_raw:
            raw = _merge_survey(s.base_raw, raw)
        return Config.from_dict(raw)

    def _recap_html(self, cfg: Config) -> str:
        s = self.state
        width_px = float(np.hypot(*(np.array(s.section_pts[0]) - np.array(s.section_pts[1]))))
        m = self._resolved_m_per_px()
        if s.scale_mode == "flight":
            preset = camera(s.camera_key or "")
            scale_how = (f"{s.altitude_m:g} m altitude, "
                         + (f"{_esc(preset.label)} ({s.hfov_deg:g}° HFOV)" if preset else f"{s.hfov_deg:g}° HFOV"))
        else:
            scale_how = "entered directly"
        water = {"auto": "detected automatically", "manual": "drawn by hand",
                 "loaded": "from the reopened survey"}.get(s.water_source or "", "none")
        secs = self._sections()
        if len(secs) > 1:
            parts = []
            for sec in secs:
                px = float(np.hypot(*np.subtract(sec["pts"][1], sec["pts"][0])))
                parts.append(f"{_esc(sec['name'])} {px * m:.1f} m")
            section_row = ("Cross-sections", f"{len(secs)}: " + ", ".join(parts)
                           + " <span class='stiv-help'>(analysed together)</span>")
        else:
            section_row = ("Cross-section", f"{width_px * m:.1f} m ({width_px:.0f} px)")
        if s.discharge_enabled and len(secs) > 1:
            n_q = sum(1 for sec in secs if sec["bathymetry_csv"])
            discharge_row = ("Discharge", f"on, at {n_q} of {len(secs)} cross-sections")
        else:
            discharge_row = ("Discharge", "on" if s.discharge_enabled else "off")
        rows = [
            ("Video", _esc(Path(s.video_path).name)),
            ("Ground resolution", f"{m:.4f} m/pixel <span class='stiv-help'>({scale_how})</span>"),
            ("Frame rate", (f"{self._fps():.3f} fps" if self._fps() else "unknown")
             + (" (entered)" if s.fps_override else "")),
            ("Analysis window", f"{s.start_s:.1f} s → {s.start_s + s.duration_s:.1f} s"),
            section_row,
            ("Stations", str(s.n_stations) + (" at each" if len(secs) > 1 else "")),
            ("Expected velocity range", f"{s.velocity_range[0]:.1f} to {s.velocity_range[1]:.1f} m/s"),
            ("Velocity method", METHOD_LABELS[s.method]
             + (" (FFT angle)" if s.method == "stiv" and s.estimator == "fft" else "")
             + (f" <span class='stiv-help'>compared with "
                f"{', '.join(METHOD_LABELS[m] for m in s.compare)}</span>" if s.compare else "")),
            ("Drone-motion compensation", "on" if s.stabilise else "off"),
            ("Water outline", water if len(s.water_pts) >= 3 or s.extra_water_polys
             else "found when the analysis runs (auto, from the survey file)"
             if self._water_at_run() else "none"),
            discharge_row,
            ("PDF report", "yes" if s.pdf_report else "no"),
            ("Results folder", f"<code>{_esc(self._out_dir())}</code>"),
        ]
        body = "".join(f"<tr><td class='k'>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows)
        return f"<table class='stiv-table'>{body}</table>"

    def _build_step4(self):
        notes = ""
        try:
            cfg = self._build_config()
            recap = self._recap_html(cfg)
            from .report import MANIFEST

            if (Path(cfg.output.out_dir) / MANIFEST).is_file():
                notes += _callout("warn", "That results folder already holds a previous run, which this "
                                          "run will replace. Change the folder in step 4 to keep both.")
        except ConfigError as exc:
            recap = _callout("error", f"Configuration problem: {_esc(exc)}")
        return W.VBox([
            _heading("5. Review and run", "Check the survey, then press <b>Run analysis</b>."),
            _card("Survey", W.HTML(recap)),
            W.HTML(notes + _callout("info", "A full-length clip can take a minute or two; progress appears "
                                            "here. Nothing is written until the analysis finishes.")),
        ])

    def _run_analysis(self) -> None:
        self._show_error("")
        try:
            cfg = self._build_config()
        except ConfigError as exc:
            self._show_error(f"Configuration problem: {_esc(exc)}")
            return

        self._next_btn.disabled = True
        self._back_btn.disabled = True
        for btn in self._step_btns:
            btn.disabled = True
        status = W.HTML(_callout("info", "<b>Running the analysis&hellip;</b> reading the video and "
                                         "building a space-time image at every station"
                                         + (f" of all {len(cfg.sections)} cross-sections"
                                            if cfg.sections else "") + "."))
        progress_out = W.Output(layout=W.Layout(padding="4px", max_height="200px", overflow="auto"))
        self._body.children = [W.VBox([
            _heading("5. Review and run", "Analysing&hellip;"), status, _card(None, progress_out),
        ])]

        try:
            from .analysis import run_survey
            from .report import write_survey_outputs

            with progress_out:
                survey = run_survey(cfg, progress=True)
                out_path = write_survey_outputs(survey, cfg)
            # One section's result is kept as it always was; several, as the survey.
            result = survey if cfg.sections is not None else survey.results[0]
        except ConfigError as exc:
            self._next_btn.disabled = False
            self._back_btn.disabled = False
            self._show_error(f"Could not run: {_esc(exc)}")
            self._refresh()
            return
        except Exception as exc:  # a boundary between untrusted footage/settings and the pipeline
            self._next_btn.disabled = False
            self._back_btn.disabled = False
            self._show_error(f"Unexpected error: {_esc(exc)}")
            self._refresh()
            return

        self.state.result = result
        self.state.out_path = out_path
        self._next_btn.disabled = False
        self._back_btn.disabled = False
        self.step = 5
        self._furthest = 5
        self._refresh()

    # ------------------------------------------------------------------ #
    # step 5 -- results
    # ------------------------------------------------------------------ #
    def _build_step5(self):
        from .analysis import SurveyResult

        s = self.state
        if s.result is None:
            return W.HTML("<p>No results yet.</p>")
        result = s.result
        out = s.out_path
        multi = isinstance(result, SurveyResult)

        if multi:
            survey = result
            first = survey.results[0]
            views = {"All cross-sections": lambda: self._survey_view(survey, Path(out))}
            for r in survey.results:
                folder = Path(out) / section_folder(r.config.section.name)
                views[r.config.section.name] = (
                    lambda r=r, folder=folder: self._result_view(r, folder, survey))
            picker = W.ToggleButtons(options=list(views), value="All cross-sections",
                                     style={"button_width": "auto"})
            built: dict = {}
            holder = W.VBox([])

            def _show_view(change=None):
                name = picker.value
                if name not in built:
                    built[name] = views[name]()
                holder.children = [built[name]]

            picker.observe(_show_view, names="value")
            _show_view()
            body = [_help("Choose a cross-section to see its own figures, table and report."),
                    picker, holder]
            lead = (f"{len(survey.results)} cross-sections &middot; "
                    f"{first.duration_s:.0f} s analysed together")
            cfg = survey.config
        else:
            body = [self._result_view(result, Path(out))]
            lead = (f"{_esc(result.config.section.name)} &middot; "
                    f"{result.geometry.width_m:.1f} m wide &middot; {result.duration_s:.0f} s analysed")
            cfg = result.config

        zip_btn = W.Button(description="Download results (.zip)" if _in_colab() else "Zip the results",
                           icon="download", layout=W.Layout(width="auto"))
        rerun_btn = W.Button(description="Change settings and re-run", icon="sliders",
                             layout=W.Layout(width="auto"))
        pdf_path = Path(out) / "report.pdf"
        pdf_btn = W.Button(icon="file-pdf-o", layout=W.Layout(width="auto"))
        files_note = W.HTML()

        def _pdf_label():
            if pdf_path.is_file():
                pdf_btn.description = "Download PDF report" if _in_colab() else "PDF report"
            else:
                pdf_btn.description = "Make a PDF report"

        def _on_pdf(_btn):
            pdf_btn.disabled = True
            try:
                if not pdf_path.is_file():
                    files_note.value = _callout("info", "Writing the PDF report&hellip;")
                    from .report import add_pdf_report

                    add_pdf_report(result, cfg)
                if _in_colab():
                    from google.colab import files

                    files.download(str(pdf_path))
                    files_note.value = _callout("ok", "Downloading <code>report.pdf</code>.")
                else:
                    files_note.value = _callout(
                        "ok", f"The PDF report is <code>{_esc(pdf_path.resolve())}</code>.")
            except Exception as exc:
                files_note.value = _callout("error", f"Could not write the PDF report: {_esc(exc)}")
            _pdf_label()
            pdf_btn.disabled = False

        _pdf_label()
        pdf_btn.on_click(_on_pdf)

        def _on_zip(_btn):
            zip_btn.disabled = True
            try:
                folder = Path(out).resolve()
                archive = shutil.make_archive(str(folder), "zip", root_dir=str(folder))
                if _in_colab():
                    from google.colab import files

                    files.download(archive)
                    files_note.value = _callout("ok", f"Downloading <code>{_esc(Path(archive).name)}</code>.")
                else:
                    files_note.value = _callout("ok", f"Saved <code>{_esc(archive)}</code>.")
            except Exception as exc:
                files_note.value = _callout("error", f"Could not zip the results: {_esc(exc)}")
            zip_btn.disabled = False

        def _on_rerun(_btn):
            self._show_error("")
            self.step = 3
            self._refresh()

        zip_btn.on_click(_on_zip)
        rerun_btn.on_click(_on_rerun)

        colab_note = (" Colab deletes its own disk when the session ends, so download them or keep them "
                      "on Google Drive." if _in_colab() and not str(Path(out).resolve()).startswith("/content/drive")
                      else "")
        where = (" Each cross-section's own tables, figures and report are in a sub-folder named after it; "
                 "<code>sections.csv</code> compares them." if multi else "")
        return W.VBox([
            _heading("6. Results", lead),
            *body,
            _card("Files",
                  _help(f"Tables, figures, the run report{', report.pdf' if pdf_path.is_file() else ''} "
                        f"and <code>config_used.yaml</code> were written to "
                        f"<code>{_esc(Path(out).resolve())}</code>.{where} Reopen that YAML in step 1 to "
                        f"come back to this survey later.{colab_note}"),
                  W.HBox([pdf_btn, zip_btn, rerun_btn], layout=W.Layout(flex_flow="row wrap")),
                  files_note),
        ])

    @staticmethod
    def _tabs(panes: list, titles: list) -> W.Tab:
        tabs = W.Tab(children=panes)
        if "titles" in tabs.trait_names():          # ipywidgets 8
            tabs.titles = tuple(titles)
        else:                                       # ipywidgets 7, which Colab may still ship
            for i, t in enumerate(titles):
                tabs.set_title(i, t)
        return tabs

    @staticmethod
    def _image(path: Path):
        if not path.is_file():
            return None
        return W.Image(value=path.read_bytes(), format="png",
                       layout=W.Layout(width="100%", max_width="900px"))

    @staticmethod
    def _warnings_html(warnings: list[str], title: str = "Warnings") -> str:
        if not warnings:
            return ""
        items = "".join(f"<li>{_esc(w)}</li>" for w in warnings)
        return _callout("warn", f"<b>{title}</b><ul style='margin:4px 0 0 18px;padding:0'>{items}</ul>")

    def _result_view(self, result, out: Path, survey=None) -> W.VBox:
        """Headline numbers, figures, table and report of one cross-section.

        ``out`` is the folder its figures were written to.  Within a
        ``survey``, the warnings it lists once for all its sections are left
        out here.
        """
        omit_warnings = survey.shared_warnings if survey is not None else []
        v = result.velocities
        finite = v[np.isfinite(v)]
        n = len(result.stations)
        n_ok = sum(1 for st in result.stations if not st.failed)
        stab = result.stabilisation
        tiles = []
        if finite.size:
            tiles += [("Mean surface velocity", f"{finite.mean():.2f} m/s", f"over {finite.size} stations"),
                      ("Max surface velocity", f"{finite.max():.2f} m/s", None)]
        if result.discharge is not None:
            tiles.append(("Discharge", f"{result.discharge.discharge_m3s:.3f} m³/s",
                          f"area {result.discharge.area_m2:.2f} m²"))
        tiles.append(("Stations passing QC", f"{n_ok} / {n}", None))
        if not stab.enabled:
            stab_txt = "off" if not stab.auto_disabled else "switched off"
        elif stab.warnings():
            stab_txt = "check warnings"
        else:
            stab_txt = "OK"
        tiles.append(("Camera motion", stab_txt, "drift correction"))

        omit = set(omit_warnings)
        warnings = [w for w in result.warnings if w not in omit]
        warn_html = self._warnings_html(
            warnings, "Warnings for this cross-section" if survey is not None else "Warnings")

        panes, titles = [], []
        for name, title in (("velocity_map.png", "Flow map"), ("velocity_profile.png", "Velocity profile"),
                            ("overview.png", "Section check"), ("sti_panel.png", "Space-time"),
                            ("correlation_panel.png", "LS-PIV correlation"), ("tracks.png", "Tracks"),
                            ("discharge.png", "Discharge")):
            img = self._image(out / name)
            if img is not None:
                panes.append(img)
                titles.append(title)
        table_out = W.Output()
        with table_out:
            df = result.to_frame()
            quality = {"stiv": ["coherency"], "lspiv": ["velocity_cross_ms", "peak_ratio"],
                       "ptv": ["velocity_cross_ms", "n_measurements"]}[result.method]
            cols = (["station_m", "velocity_ms", "velocity_std_ms"] + quality
                    + [f"velocity_{m}_ms" for m in result.comparisons] + ["flags"])
            display(df[cols].round(3))
        panes.append(table_out)
        titles.append("Table")
        panes.append(W.HTML(f"<pre class='stiv-report'>{_esc(result.summary(omit_warnings=omit_warnings))}</pre>"))
        titles.append("Run report")
        head = ([W.HTML(f"<div class='stiv-help'>{_esc(result.config.section.name)} &middot; "
                        f"{result.geometry.width_m:.1f} m wide</div>")] if survey is not None else [])
        compared = []
        if result.comparisons:
            # Each method side by side; the headline numbers above are the
            # reported method's alone.
            rows = []
            has_q = any(result.discharge_for(m) is not None for m in result.methods)
            for m in result.methods:
                st = result.stations_for(m)
                vm = np.array([x.velocity_ms for x in st], dtype=float)
                fin = vm[np.isfinite(vm)]
                q = result.discharge_for(m)
                rows.append(
                    f"<tr><td><b>{METHOD_LABELS[m]}</b>{' (reported)' if m == result.method else ''}</td>"
                    f"<td>{sum(1 for x in st if not x.failed)} / {len(st)}</td>"
                    f"<td>{f'{fin.mean():.2f}' if fin.size else '–'}</td>"
                    f"<td>{f'{fin.max():.2f}' if fin.size else '–'}</td>"
                    + (f"<td>{q.discharge_m3s:.3f}</td>" if q is not None else ("<td>–</td>" if has_q else ""))
                    + "</tr>")
            compared = [_card("Methods compared", W.HTML(
                "<table class='stiv-table'><tr><th>method</th><th>passing QC</th><th>mean U (m/s)</th>"
                "<th>max U (m/s)</th>" + ("<th>Q (m³/s)</th>" if has_q else "") + "</tr>"
                + "".join(rows) + "</table>"))]
        return W.VBox(head + [W.HTML(_tiles(tiles) + warn_html), *compared, self._tabs(panes, titles)])

    def _survey_view(self, survey, out: Path) -> W.VBox:
        """Every cross-section side by side: the table, and the figures with all of them."""
        results = survey.results
        n = sum(len(r.stations) for r in results)
        n_ok = sum(1 for r in results for st in r.stations if not st.failed)
        tiles = [("Cross-sections", str(len(results)), "analysed together"),
                 ("Stations passing QC", f"{n_ok} / {n}", "over every section")]
        q = [r.discharge.discharge_m3s for r in results if r.discharge is not None]
        if q:
            tiles.append(("Discharge", f"{len(q)} of {len(results)}", "sections with depths"))
        stab = results[0].stabilisation
        if not stab.enabled:
            stab_txt = "off" if not stab.auto_disabled else "switched off"
        else:
            stab_txt = "check warnings" if stab.warnings() else "OK"
        tiles.append(("Camera motion", stab_txt, "one correction for every section"))

        rows = []
        for r in results:
            v = r.velocities
            finite = v[np.isfinite(v)]
            ok = sum(1 for st in r.stations if not st.failed)
            rows.append(
                f"<tr><td><b>{_esc(r.config.section.name)}</b></td><td>{r.geometry.width_m:.1f} m</td>"
                f"<td>{ok} / {len(r.stations)}</td>"
                f"<td>{f'{finite.mean():.2f}' if finite.size else '–'}</td>"
                f"<td>{f'{finite.max():.2f}' if finite.size else '–'}</td>"
                + (f"<td>{r.discharge.discharge_m3s:.3f}</td>" if r.discharge is not None
                   else ("<td>–</td>" if q else "")) + "</tr>")
        table = ("<table class='stiv-table'><tr><th>section</th><th>width</th><th>passing QC</th>"
                 "<th>mean U (m/s)</th><th>max U (m/s)</th>" + ("<th>Q (m³/s)</th>" if q else "")
                 + "</tr>" + "".join(rows) + "</table>")

        panes, titles = [], []
        for name, title in (("velocity_map.png", "Flow map"), ("overview.png", "Section check")):
            img = self._image(out / name)
            if img is not None:
                panes.append(img)
                titles.append(title)
        panes.append(W.HTML(f"<pre class='stiv-report'>{_esc(survey.summary())}</pre>"))
        titles.append("Run report")
        return W.VBox([
            W.HTML(_tiles(tiles) + self._warnings_html(survey.shared_warnings,
                                                       "Warnings for the whole survey")),
            _card("Cross-sections", W.HTML(table)),
            self._tabs(panes, titles),
        ])


def launch(out_dir: Optional[str] = None) -> Wizard:
    """Build and display the interactive survey wizard.

    Results go to ``out_dir`` if given, else to ``<video name>_output`` in the
    working directory (on Google Drive, when running in Colab with Drive
    mounted), one folder per video.  Returns the :class:`Wizard` so its state
    (e.g. ``wizard.state.result``) can be inspected from later cells.
    """
    return Wizard(out_dir=out_dir).display()
