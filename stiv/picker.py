"""Interactive selection of the cross-section and control points on a frame.

Getting pixel coordinates out of a video by hand is the most tedious part of
setting up an image-velocimetry survey, so this opens a frame in a matplotlib
window and turns clicks into a ready-to-run config file.

The water outline -- the fiddliest set of clicks -- is found automatically once
the section and the downstream point are in (:mod:`stiv.watermask`), and
shown for the user to accept; clicking it by hand is the fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import (
    ConfigError,
    OutputConfig,
    SectionConfig,
    default_duration_s,
    path_for_config,
    survey_dict,
)
from .video import probe, read_frame_at

log = logging.getLogger(__name__)


class _Picker:
    """Collect an ordered list of clicked points from one frame."""

    def __init__(self, frame_bgr: np.ndarray, title: str, n_points: Optional[int],
                 shown: Sequence[Sequence[Sequence[float]]] = ()):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.points: list[tuple[float, float]] = []
        self.n_points = n_points

        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.ax.imshow(frame_bgr[:, :, ::-1])
        # Cross-sections clicked already, for reference.
        for k, line in enumerate(shown):
            p = np.asarray(line, float)
            self.ax.plot(p[:, 0], p[:, 1], "o-", color="#2a78d6", markersize=6,
                         markeredgecolor="white", markeredgewidth=1.2, lw=1.8, alpha=0.85)
            self.ax.annotate(f"section {k + 1}", p[0], xytext=(8, 8), textcoords="offset points",
                             color="#2a78d6", fontsize=10, weight="bold")
        self.ax.set_title(title, fontsize=11, loc="left")
        self.ax.set_axis_off()
        self.marks = self.ax.plot([], [], "o-", color="#eb6834", markersize=8,
                                  markeredgecolor="white", markeredgewidth=1.5, lw=2)[0]
        self.labels: list = []

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if event.button == 3:  # right click undoes
            self._undo()
            return
        self.points.append((float(event.xdata), float(event.ydata)))
        self._redraw()
        if self.n_points is not None and len(self.points) >= self.n_points:
            self.plt.close(self.fig)

    def _on_key(self, event):
        if event.key in ("enter", "escape"):
            self.plt.close(self.fig)
        elif event.key in ("backspace", "delete", "u"):
            self._undo()

    def _undo(self):
        if self.points:
            self.points.pop()
            self._redraw()

    def _redraw(self):
        p = np.array(self.points) if self.points else np.empty((0, 2))
        self.marks.set_data(p[:, 0] if len(p) else [], p[:, 1] if len(p) else [])
        for t in self.labels:
            t.remove()
        self.labels = [
            self.ax.annotate(str(i + 1), xy, xytext=(8, 8), textcoords="offset points",
                             color="#eb6834", fontsize=11, weight="bold")
            for i, xy in enumerate(self.points)
        ]
        self.fig.canvas.draw_idle()

    def run(self) -> list[tuple[float, float]]:
        self.plt.show()
        return self.points


def _require_gui() -> None:
    import matplotlib

    if matplotlib.get_backend().lower() == "agg":
        raise ConfigError(
            "no interactive matplotlib backend is available, so points cannot be "
            "clicked. Install a GUI backend (e.g. `pip install PyQt5`) or write "
            "section.start_px / end_px into the config by hand."
        )


def pick_points(video: str | Path, n_points: Optional[int], prompt: str,
                t_s: float = 0.0, shown: Sequence[Sequence[Sequence[float]]] = ()
                ) -> list[tuple[float, float]]:
    """Open a frame and return the points the user clicks.

    ``shown`` are cross-sections clicked already, drawn for reference.
    """
    _require_gui()
    frame = read_frame_at(video, t_s)
    title = (
        f"{prompt}\n"
        f"left click = add    right click / backspace = undo    enter = done"
    )
    return _Picker(frame, title, n_points, shown).run()


def confirm_water_outline(video: str | Path, polygon: Sequence[Sequence[float]],
                          section: Sequence[Sequence[float]], t_s: float = 0.0,
                          note: str = "") -> bool:
    """Show an automatic water outline on the frame; True if the user accepts it.

    Enter or ``y`` accepts; ``n`` or escape -- or closing the window -- turns
    it down, and the outline is then clicked by hand instead.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    _require_gui()
    frame = read_frame_at(video, t_s)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(frame[:, :, ::-1])
    ax.set_axis_off()
    ax.add_patch(Polygon(np.asarray(polygon, float), closed=True, facecolor="#1baf7a",
                         edgecolor="#1baf7a", alpha=0.3, lw=2))
    sec = np.asarray(section, float)
    ax.plot(sec[:, 0], sec[:, 1], "o-", color="#eb6834", markersize=8,
            markeredgecolor="white", markeredgewidth=1.5, lw=2)
    ax.set_title(
        "The WATER was outlined automatically (green). It should cover the moving water "
        "around the section but not the banks;\nit need not be exact, and too much water "
        "is better than too little.   enter / y = use it    n / escape = outline it by hand"
        + (f"\n{note}" if note else ""),
        fontsize=11, loc="left")
    choice = {"accept": False}

    def _on_key(event):
        if event.key in ("enter", "y"):
            choice["accept"] = True
            plt.close(fig)
        elif event.key in ("n", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()
    return choice["accept"]


def automatic_water_outline(
    video: str | Path,
    t_s: float,
    section: Sequence[Sequence[float]],
    downstream: Optional[Sequence[float]],
    fps: Optional[float] = None,
) -> Optional[list[tuple[float, float]]]:
    """Find the water, offer it to the user, and return it if accepted.

    None when the detector finds nothing it trusts -- the reason is logged --
    or when the user turns the outline down; either way the caller falls back
    to clicking the outline by hand.
    """
    from .watermask import detect_water

    log.info("looking for the water...")
    try:
        det = detect_water(video, t_s, section, downstream, fps=fps)
    except (ConfigError, ValueError) as exc:
        log.warning("could not outline the water automatically (%s); outline it by hand "
                    "instead, or press enter to skip", exc)
        return None
    if det.flow_agrees is False:
        log.warning("the water appears to move towards the other side of the cross-section "
                    "from the downstream point you clicked; if that point is on the wrong "
                    "side, run `stiv pick` again (stiv run also flips a reversed profile, "
                    "with flow.auto_flip)")
    if not det.ok:
        log.warning("could not outline the water automatically: %s. Outline it by hand "
                    "instead, or press enter to skip", det.message)
        return None
    note = "" if det.confidence == "good" else f"Check it carefully: {det.message}."
    if det.confidence != "good":
        log.warning("the automatic water outline needs checking: %s", det.message)
    if confirm_water_outline(video, det.polygon_px, section, t_s, note):
        log.info("using the automatic water outline (%s)", det.message)
        return list(det.polygon_px)
    log.info("automatic water outline turned down; outline it by hand instead")
    return None


def build_config_interactively(
    video: str | Path,
    out_dir: str = OutputConfig.out_dir,
    t_s: float = 0.0,
    with_gcps: bool = False,
    n_stations: int = SectionConfig.n_stations,
    config_path: Optional[str | Path] = None,
    auto_water: bool = True,
    fps: Optional[float] = None,
    n_sections: int = 1,
) -> dict:
    """Walk the user through defining a survey, returning a raw config dict.

    The frame shown is the one at ``t_s``, which also becomes ``video.start_s``:
    the stabiliser measures drift relative to the first analysed frame, so the
    points must be clicked on that frame or they are off by however far the
    aircraft has drifted in between.

    ``video.path`` is written relative to the folder of ``config_path`` (the
    file the config will be saved as) when the video is inside it, otherwise
    absolute -- :meth:`Config.from_yaml` resolves relative paths against the
    config file, not the working directory.  The scale is left as a ``null``
    placeholder (or, with ``with_gcps``, the world coordinates of the control
    points), which the config refuses to run with until it is filled in.

    With ``auto_water`` the water is outlined automatically after the
    downstream click and offered for confirmation; it is clicked by hand only
    if that finds nothing or the user turns it down.

    ``n_sections`` cross-sections are clicked one after another, each window
    showing those already in, and written as ``sections`` to be analysed
    together.  One downstream point serves them all, so it is asked for below
    the last of them.
    """
    info = probe(video)
    if info.duration_s and t_s >= info.duration_s:
        raise ConfigError(
            f"-t {t_s:g} s is past the end of the video ({info.duration_s:.1f} s long)"
        )

    if n_sections < 1:
        raise ConfigError(f"n_sections must be 1 or more, got {n_sections}")
    sections: list[list[tuple[float, float]]] = []
    for k in range(n_sections):
        which = ("the measurement cross-section" if n_sections == 1
                 else f"cross-section {k + 1} of {n_sections}")
        # Passed only once there is something to show.
        extra = {"shown": list(sections)} if sections else {}
        pts = pick_points(
            video, 2,
            f"Click the TWO ENDS of {which} (bank to bank, across the flow).",
            t_s, **extra,
        )
        if len(pts) < 2:
            what = "a cross-section" if n_sections == 1 else f"cross-section {k + 1}"
            raise ConfigError(f"{what} needs two endpoints; "
                              f"{'none were' if not pts else 'only one was'} selected")
        sections.append(pts[:2])
    pts = sections[0]

    extra = {"shown": list(sections)} if n_sections > 1 else {}
    down = pick_points(
        video, 1,
        "Click one point DOWNSTREAM of the cross-section "
        "(this fixes which way the flow runs)." if n_sections == 1 else
        "Click one point DOWNSTREAM of ALL the cross-sections "
        "(this fixes which way the flow runs).",
        t_s, **extra,
    )

    gcps = None
    if with_gcps:
        gcps = pick_points(
            video, None,
            "Click at least FOUR ground control points, then press enter. "
            "Note their real-world coordinates - you will need to type them into "
            "the config.",
            t_s,
        )
        if len(gcps) < 4:
            raise ConfigError(f"a homography needs at least 4 GCPs; {len(gcps)} were clicked")

    water = None
    if auto_water:
        water = automatic_water_outline(video, t_s, pts[:2], down[0] if down else None, fps)
    if water is None:
        water = pick_points(
            video, None,
            "Optional: outline the WATER by clicking around its edge, then press "
            "enter. This keeps the stabiliser from tracking the moving surface. "
            "Press enter immediately to skip.",
            t_s,
        )

    video_entry = (path_for_config(video, config_path) if config_path is not None
                   else str(Path(video).expanduser().resolve()))
    return survey_dict(
        video_entry,
        start_s=t_s,
        duration_s=default_duration_s(info.duration_s, t_s),
        gcp_px=gcps,                 # gcp_world is left as null placeholders
        start_px=pts[0],
        end_px=pts[1],
        n_stations=n_stations,
        name=Path(str(video)).stem,
        downstream_px=down[0] if down else None,
        water_mask_px=[water] if len(water) >= 3 else None,
        out_dir=out_dir,
        sections=([{"start_px": a, "end_px": b} for a, b in sections]
                  if n_sections > 1 else None),
    )
