"""Diagnostic figures for a STIV run.

Six outputs, each answering a question an operator actually asks:

``overview.png``            did I measure where I meant to?
``velocity_map.png``        how fast, and which way, is the water going where?
``velocity_profile.png``    what is the velocity across the section?
``sti_panel.png``           do the space-time images show usable streaks?
``sti_detail_*.png``        why did *this* station give *that* number?
``discharge.png``           how does velocity combine with depth into flow?

and, when LS-PIV or particle tracking ran,

``correlation_panel.png``   does each station's correlation have one clear peak?
``tracks.png``              what did the tracker follow, and how fast?

When methods were compared, the profile carries each of them.

These are report figures written to disk, so they deliberately commit to a
single light appearance rather than adapting to a viewer theme.  Colours are
the validated reference palette; magnitude uses a single-hue blue ramp and the
space-time images stay greyscale, since they are photographs rather than
encodings of an abstract quantity.

Each ``plot_*`` function writes to a file path, or -- for the PDF report --
to an open multi-page PDF, one page per figure.

Figures are built on :class:`matplotlib.figure.Figure` with an Agg canvas of
their own rather than through ``pyplot``, so importing this module never
changes the process-wide backend -- in a notebook, a later inline or ipympl
figure keeps working after a run has written its report.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .analysis import StivResult
from .config import METHOD_LABELS
from .sti import to_display

# --- palette ---------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_1 = "#2a78d6"   # blue    - velocity
SERIES_2 = "#eb6834"   # orange  - overlays on greyscale, depth/area
SERIES_3 = "#1baf7a"   # aqua
CRITICAL = "#d03b3b"   # flagged / failed stations
WARNING = "#fab219"

#: The reported profile is always drawn in SERIES_1; the methods compared with
#: it take these, in the order the config lists them.
COMPARE_COLOURS = (SERIES_2, SERIES_3)

#: Single-hue sequential ramp for coherency (0 -> 1).
COHERENCY_CMAP = LinearSegmentedColormap.from_list(
    "stiv_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "figure.constrained_layout.use": True,
}


#: Narrowest overview figure, inches, that still holds its three-entry legend.
_MIN_OVERVIEW_WIDTH_IN = 6.5


def _style():
    return matplotlib.rc_context(_RC)


def _save(fig: Figure, target, dpi: int, **kwargs):
    """Write ``fig`` to ``target``: a file path, or an open multi-page PDF.

    The PDF report (:mod:`stiv.pdf_report`) passes a
    :class:`~matplotlib.backends.backend_pdf.PdfPages`, and each figure becomes
    a page of it, as vector graphics.  Returns ``target``.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    if isinstance(target, PdfPages):
        target.savefig(fig, dpi=dpi, **kwargs)
    else:
        fig.savefig(target, dpi=dpi, **kwargs)
    return target


def _figure(**kwargs) -> Figure:
    """A figure on its own Agg canvas, unknown to pyplot and its backend.

    Create it inside :func:`_style` -- rcParams are read when the figure is
    built and again when it is saved.
    """
    fig = Figure(**kwargs)
    FigureCanvasAgg(fig)
    return fig


def _velocity_label(st, digits: int = 2) -> str:
    """A station's reported velocity, or what stands in for it."""
    if np.isfinite(st.velocity_ms):
        return f"{st.velocity_ms:.{digits}f} m/s"
    raw = getattr(st, "velocity_raw_ms", math.nan)
    if np.isfinite(raw):
        return f"excluded (raw {raw:.{digits}f})"
    return "no value"


# --------------------------------------------------------------------------- #
def _what_measures(result: StivResult) -> str:
    """What each station was measured in, for a title."""
    if result.method == "lspiv" and result.correlations:
        return f"{result.correlations[0].layout.ia_size_m:.2f} m LS-PIV interrogation areas"
    if result.method == "ptv":
        return f"{result.config.stiv.search_line_length_m:.1f} m particle-tracking boxes"
    return f"{result.config.stiv.search_line_length_m:.1f} m search lines"


def _station_outlines(result: StivResult) -> list[np.ndarray]:
    """Image outline of each station's interrogation area (LS-PIV) or box (PTV)."""
    rect = result.geometry.rectifier
    if result.method == "lspiv":
        return [rect.world_to_image(c.layout.corners_world(ia_only=True))
                for c in result.correlations]
    if result.method == "ptv":
        return [rect.world_to_image(b.corners_world()) for b in result.boxes]
    return []


def plot_overview(result: StivResult, path: Path, dpi: int = 150) -> Path:
    """First frame with the cross-section, search lines and station numbering."""
    geom = result.geometry
    title = (f"{result.config.section.name}  -  {geom.width_m:.1f} m, "
             f"{len(geom.search_lines)} stations, {_what_measures(result)}")
    return _overview([result], path, dpi, title)


def plot_survey_overview(results: Sequence[StivResult], path: Path, dpi: int = 150) -> Path:
    """:func:`plot_overview` of every cross-section of a survey, on one frame."""
    n = sum(len(r.stations) for r in results)
    title = f"{len(results)} cross-sections  -  {n} stations, {_what_measures(results[0])}"
    return _overview(results, path, dpi, title, names=True)


def _overview(results: Sequence[StivResult], path: Path, dpi: int, title: str,
              names: bool = False) -> Path:
    frame = results[0].reference_frame_bgr
    rgb = frame[:, :, ::-1] if frame.ndim == 3 else frame
    h, w = rgb.shape[:2]

    # About 110 frame pixels to the inch, capped at 13 x 9 in.  A small frame
    # -- a 336 px thermal sensor, a 480 px proxy -- is scaled up to at least
    # _MIN_OVERVIEW_WIDTH_IN, aspect kept: any narrower and the legend beneath
    # the image is wider than the figure, and constrained layout gives up.
    scale = max(1.0 / 110.0, _MIN_OVERVIEW_WIDTH_IN / w)
    with _style():
        fig = _figure(figsize=(min(13, w * scale), min(9, h * scale)))
        ax = fig.subplots()
        ax.imshow(rgb)
        ax.set_axis_off()
        ax.grid(False)

        for result in results:
            geom = result.geometry
            outlines = _station_outlines(result)
            if outlines:
                for poly, st in zip(outlines, result.stations):
                    q = np.vstack([poly, poly[:1]])
                    ax.plot(q[:, 0], q[:, 1], color=SERIES_2 if st.failed else SERIES_3,
                            lw=1.3, alpha=0.95)
            else:
                for sl, st in zip(geom.search_lines, result.stations):
                    pts = sl.samples_px
                    colour = SERIES_2 if st.failed else SERIES_3
                    ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=1.6, alpha=0.95,
                            solid_capstyle="round")

            ax.plot([geom.start_px[0], geom.end_px[0]], [geom.start_px[1], geom.end_px[1]],
                    color=SERIES_1, lw=2.4, alpha=0.95, solid_capstyle="round")
            ax.scatter(*geom.start_px, s=48, color=SERIES_1, zorder=5,
                       edgecolor=SURFACE, linewidth=1.5)
            ax.scatter(*geom.end_px, s=48, color=SERIES_1, zorder=5,
                       edgecolor=SURFACE, linewidth=1.5)
            start_label = f"{result.config.section.name}  0 m" if names else "0 m"
            ax.annotate(start_label, geom.start_px, color=SURFACE, fontsize=9, weight="bold",
                        xytext=(6, 6), textcoords="offset points",
                        path_effects=_outline())
            ax.annotate(f"{geom.width_m:.1f} m", geom.end_px, color=SURFACE, fontsize=9,
                        weight="bold", xytext=(6, 6), textcoords="offset points",
                        path_effects=_outline())

            # Arrow showing the assumed downstream direction.
            centre = 0.5 * (np.asarray(geom.start_px) + np.asarray(geom.end_px))
            tip = geom.rectifier.world_to_image(
                [geom.rectifier.image_to_world([centre])[0] + geom.flow_dir_world *
                 max(2.0, 0.15 * geom.width_m)]
            )[0]
            ax.annotate("", xy=tip, xytext=centre,
                        arrowprops=dict(arrowstyle="-|>", color=SERIES_1, lw=2.2,
                                        shrinkA=0, shrinkB=0))

        what = {"lspiv": "interrogation area", "ptv": "tracking box"}.get(
            results[0].method, "search line")
        handles = [
            Line2D([], [], color=SERIES_1, lw=2.4, label="cross-section / flow direction"),
            Line2D([], [], color=SERIES_3, lw=1.8, label=f"{what} (measured)"),
        ]
        if any(s.failed for r in results for s in r.stations):
            handles.append(Line2D([], [], color=SERIES_2, lw=1.8, label=f"{what} (failed QC)"))
        if what == "search line":
            ax.legend(handles=handles, loc="lower center", ncol=len(handles),
                      bbox_to_anchor=(0.5, -0.06), labelcolor=INK_2)
        else:
            # The longer names of LS-PIV's and PTV's outlines do not fit three
            # abreast under a small frame, so they take two rows, hung from
            # the image's lower edge rather than standing on it.
            ax.legend(handles=handles, loc="upper center", ncol=min(2, len(handles)),
                      bbox_to_anchor=(0.5, -0.01), labelcolor=INK_2)

        ax.set_title(title, loc="left")
        _save(fig, path, dpi, bbox_inches="tight")
    return path


def _outline():
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=2.5, foreground="#00000090")]


# --------------------------------------------------------------------------- #
#: Longest arrow on the velocity map, as a fraction of the section width.
_ARROW_FRACTION = 0.25
#: Most stations given a printed value on the velocity map.
_MAX_ARROW_LABELS = 8


def _nice_speed(x: float) -> float:
    """The largest 1-2-5 step (0.1, 0.2, 0.5, 1, 2, ...) not above ``x``."""
    if not (np.isfinite(x) and x > 0):
        return 1.0
    exp = math.floor(math.log10(x))
    for m in (5.0, 2.0, 1.0):
        if m * 10.0 ** exp <= x:
            return m * 10.0 ** exp
    return 10.0 ** exp


def plot_velocity_map(result: StivResult, path: Path, dpi: int = 150) -> Path:
    """The frame with one flow arrow per station, its length proportional to speed.

    Arrows are built in world coordinates -- from each station along the flow
    direction, ``velocity * k`` metres long -- and only then mapped into the
    image, so they stay proportional to speed in an oblique (GCP) view too.
    The fastest arrow is about a quarter of the section width; a reference
    arrow gives the scale.  A station that failed quality control and was
    interpolated is drawn hollow; one that was excluded is a cross.
    """
    return _velocity_map([result], path, dpi)


def plot_survey_velocity_map(results: Sequence[StivResult], path: Path, dpi: int = 150) -> Path:
    """:func:`plot_velocity_map` of every cross-section of a survey, on one frame.

    One arrow scale serves every section, so arrows compare across sections
    as well as along each: the fastest is a quarter of the narrowest section.
    """
    return _velocity_map(results, path, dpi)


def _velocity_map(results: Sequence[StivResult], path: Path, dpi: int) -> Path:
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle

    frame = results[0].reference_frame_bgr
    rgb = frame[:, :, ::-1] if frame.ndim == 3 else frame
    h, w = rgb.shape[:2]
    multi = len(results) > 1

    secs = []
    for result in results:
        geom = result.geometry
        # Paired as the overview pairs them, so a result trimmed to fewer
        # stations than its geometry still lines up.
        pairs = list(zip(geom.search_lines, result.stations))
        v = np.array([st.velocity_ms for _, st in pairs], float)
        failed = np.array([st.failed for _, st in pairs], bool)
        shown = np.isfinite(v)
        # LS-PIV and PTV measured the component across the flow too, so their
        # arrows point where the water went; an interpolated station has no
        # direction of its own and points down the flow.
        cross = np.array([st.velocity_cross_ms for _, st in pairs], float)
        cross = np.where(np.isfinite(cross) & ~failed, cross, 0.0)
        flow_dir = np.asarray(geom.flow_dir_world, float)
        secs.append({
            "result": result,
            "geom": geom,
            "rect": geom.rectifier,
            "flow_dir": flow_dir,
            "across": np.array([-flow_dir[1], flow_dir[0]]),
            "cross": cross,
            "v": v,
            "shown": shown,
            "measured": shown & ~failed,
            "interp": failed & np.array(["interpolated" in st.flags for _, st in pairs], bool) & shown,
            "excluded": ~shown,
            "base_w": np.array([sl.centre_world for sl, _ in pairs], float).reshape(-1, 2),
            "base_px": np.array([sl.centre_px for sl, _ in pairs], float).reshape(-1, 2),
        })

    all_v = np.concatenate([sec["v"][sec["shown"]] for sec in secs])
    all_speed = np.concatenate([np.hypot(sec["v"], sec["cross"])[sec["shown"]] for sec in secs])
    vmax = float(np.max(all_speed)) if all_speed.size else 0.0
    width_m = min(sec["geom"].width_m for sec in secs)
    k = _ARROW_FRACTION * width_m / vmax if vmax > 0 else 0.0

    def _tips(sec: dict, scale: float) -> np.ndarray:
        vv = np.where(sec["shown"], sec["v"], 0.0)
        cc = np.where(sec["shown"], sec["cross"], 0.0)
        return sec["rect"].world_to_image(
            sec["base_w"] + (vv * scale)[:, None] * sec["flow_dir"][None, :]
            + (cc * scale)[:, None] * sec["across"][None, :])

    def _inside(sec: dict) -> bool:
        t = sec["tips"][sec["shown"]]
        return not len(t) or bool(
            ((t[:, 0] >= 0) & (t[:, 0] < w) & (t[:, 1] >= 0) & (t[:, 1] < h)).all())

    # Shrink the arrows until the longest stays inside the frame.
    for sec in secs:
        sec["tips"] = _tips(sec, k)
    for _ in range(12):
        if all(_inside(sec) for sec in secs):
            break
        k *= 0.8
        for sec in secs:
            sec["tips"] = _tips(sec, k)

    # Shaft width, as a fraction of the axes width: thin enough that adjacent
    # arrow heads (3.6 shafts wide) do not run into each other.
    spacings = []
    for sec in secs:
        sec["d"] = sec["tips"] - sec["base_px"]
        bp = sec["base_px"]
        spacings.append(float(np.linalg.norm(bp[-1] - bp[0])) / (len(bp) - 1)
                        if len(bp) > 1 else 0.05 * w)
    shaft = float(np.clip(0.17 * min(spacings) / w, 0.0012, 0.0045))

    scale = max(1.0 / 110.0, _MIN_OVERVIEW_WIDTH_IN / w)
    halo = [pe.withStroke(linewidth=2.8, foreground="#000000a0")]
    with _style():
        fig = _figure(figsize=(min(13, w * scale), min(9, h * scale)))
        ax = fig.subplots()
        ax.imshow(rgb)
        # A light scrim so the arrows read over whitewater and glare.
        ax.add_patch(Rectangle((-0.5, -0.5), w, h, color="black", alpha=0.18, lw=0, zorder=1))
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_axis_off()
        ax.grid(False)

        q = None
        for sec in secs:
            geom, v, shown = sec["geom"], sec["v"], sec["shown"]
            base_px, tips, d = sec["base_px"], sec["tips"], sec["d"]
            measured, interp, excluded = sec["measured"], sec["interp"], sec["excluded"]

            ax.plot([geom.start_px[0], geom.end_px[0]], [geom.start_px[1], geom.end_px[1]],
                    color=SERIES_1, lw=1.8, alpha=0.95, solid_capstyle="round", zorder=2)
            ax.scatter(base_px[:, 0], base_px[:, 1], s=10, color=SURFACE, edgecolor=INK,
                       linewidth=0.6, zorder=3)
            if multi:
                ax.annotate(sec["result"].config.section.name, geom.start_px, color=SURFACE,
                            fontsize=9, weight="bold", xytext=(-6, 6), textcoords="offset points",
                            ha="right", path_effects=_outline(), zorder=7)

            # The profile drawn on the river: a light band from the section out to
            # the arrow tips, over each unbroken run of stations.
            runs = np.split(np.arange(len(v)), np.flatnonzero(np.diff(shown.astype(int)) != 0) + 1)
            for idx in runs:
                if len(idx) < 2 or not shown[idx[0]]:
                    continue
                poly = np.vstack([base_px[idx], tips[idx][::-1]])
                ax.fill(poly[:, 0], poly[:, 1], color=SERIES_2, alpha=0.20, lw=0, zorder=2)
                ax.plot(tips[idx, 0], tips[idx, 1], color=SURFACE, lw=1.0, ls=(0, (3, 2)),
                        alpha=0.9, zorder=3, path_effects=halo)

            if measured.any():
                q1 = ax.quiver(base_px[measured, 0], base_px[measured, 1], d[measured, 0],
                               d[measured, 1], angles="xy", scale_units="xy", scale=1.0,
                               color=SERIES_2, edgecolor=SURFACE, linewidth=0.7, width=shaft,
                               headwidth=3.6, headlength=4.2, headaxislength=3.8, zorder=5)
                q = q or q1
            if interp.any():
                q2 = ax.quiver(base_px[interp, 0], base_px[interp, 1], d[interp, 0], d[interp, 1],
                               angles="xy", scale_units="xy", scale=1.0, facecolor="none",
                               edgecolor=CRITICAL, linewidth=1.3, width=shaft, headwidth=3.6,
                               headlength=4.2, headaxislength=3.8, zorder=5)
                q = q or q2
            if excluded.any():
                ax.scatter(base_px[excluded, 0], base_px[excluded, 1], s=42, marker="X",
                           color=CRITICAL, edgecolor=SURFACE, linewidth=0.8, zorder=6)

            # Values on a few evenly spaced arrows.
            lab = np.flatnonzero(shown)
            if len(lab) > _MAX_ARROW_LABELS:
                lab = lab[np.unique(np.linspace(0, len(lab) - 1, _MAX_ARROW_LABELS).round().astype(int))]
            for i in lab:
                n = d[i] / max(float(np.linalg.norm(d[i])), 1e-9)
                ax.annotate(f"{v[i]:.2f}", tips[i], xytext=(7 * n[0], -7 * n[1]),
                            textcoords="offset points", ha="center", va="center",
                            color=SURFACE, fontsize=8, weight="bold", path_effects=_outline(),
                            zorder=7)

        if q is not None and k > 0:
            sec = secs[0]
            ref = _nice_speed(0.6 * vmax)
            centre = 0.5 * (np.asarray(sec["geom"].start_world, float)
                            + np.asarray(sec["geom"].end_world, float))
            ends = sec["rect"].world_to_image(np.array([centre, centre + sec["flow_dir"] * ref * k]))
            ref_px = float(np.linalg.norm(ends[1] - ends[0]))
            qk = ax.quiverkey(q, 0.04, 0.05, ref_px, f"{ref:g} m/s", labelpos="E",
                              coordinates="axes", color=SERIES_2, labelcolor=SURFACE,
                              fontproperties={"size": 9, "weight": "bold"})
            qk.text.set_path_effects(_outline())

        handles = [Line2D([], [], color=SERIES_2, lw=2.4, marker=">", markersize=7,
                          label="surface velocity (arrow length ∝ speed)")]
        if any(sec["interp"].any() for sec in secs):
            handles.append(Line2D([], [], color=CRITICAL, lw=1.4, marker=">", markersize=7,
                                  markerfacecolor="none", label="failed QC (interpolated)"))
        if any(sec["excluded"].any() for sec in secs):
            handles.append(Line2D([], [], color=CRITICAL, lw=0, marker="X", markersize=7,
                                  label="no velocity (excluded)"))
        ax.legend(handles=handles, loc="lower center", ncol=len(handles),
                  bbox_to_anchor=(0.5, -0.06), labelcolor=INK_2)

        stats = (f"mean {all_v.mean():.2f} m/s, max {all_v.max():.2f} m/s"
                 if all_v.size else "no velocities passed quality control")
        what = (f"{len(results)} cross-sections" if multi
                else results[0].config.section.name)
        ax.set_title(f"Surface velocity  -  {what}   ({stats})", loc="left")
        _save(fig, path, dpi, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------- #
def plot_velocity_profile(result: StivResult, path: Path, dpi: int = 150) -> Path:
    """Surface velocity across the section, with its uncertainty band.

    A station that failed quality control is drawn one of two ways.  With
    ``quality.interpolate_flagged`` on it carries the ``interpolated`` flag and
    its reported value is its neighbours' interpolation, so it sits on the
    profile line.  With it off the station is excluded -- the profile has a gap
    there -- and its raw, untrusted estimate is drawn hollow, off the line, so
    it can be inspected without being mistaken for a measurement.
    """
    x = result.stations_m
    v = result.velocities
    sd = np.array([s.velocity_std_ms for s in result.stations])
    coh = np.array([s.coherency for s in result.stations])
    failed = np.array([s.failed for s in result.stations])
    interp = failed & np.array(["interpolated" in s.flags for s in result.stations])
    excluded = failed & ~interp
    raw = np.array([getattr(s, "velocity_raw_ms", math.nan) for s in result.stations],
                   dtype=float)
    # Where an excluded station's value lives: the raw estimate, unless the
    # station still reports one of its own.
    v_excl = np.where(np.isfinite(v), v, raw)
    excl_shown = excluded & np.isfinite(v_excl)
    excl_blank = excluded & ~np.isfinite(v_excl)
    # The profile line only joins reported values; excluded stations are gaps.
    v_line = np.where(excluded, np.nan, v)

    with _style():
        fig = _figure(figsize=(9.5, 5.6))
        ax, axc = fig.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3.0, 1.0]},
        )

        band = np.where(np.isfinite(sd), sd, 0.0)
        ax.fill_between(x, v_line - band, v_line + band, color=SERIES_1, alpha=0.16,
                        linewidth=0)
        ax.plot(x, v_line, color=SERIES_1, lw=2.0, solid_capstyle="round", zorder=3)
        ax.scatter(x[~failed], v[~failed], s=30, color=SERIES_1, zorder=4,
                   edgecolor=SURFACE, linewidth=1.2)
        # Each compared method as a thinner line of its own colour; only the
        # reported profile carries the quality-control marks.
        for colour, other in zip(COMPARE_COLOURS, result.comparisons.values()):
            xo = np.array([s.station_m for s in other], float)
            vo = np.array([s.velocity_ms for s in other], float)
            ax.plot(xo, vo, color=colour, lw=1.4, ls=(0, (5, 2)), marker="o",
                    markersize=3.5, zorder=2)
        if interp.any():
            ax.scatter(x[interp], v[interp], s=46, marker="X", color=CRITICAL, zorder=5,
                       edgecolor=SURFACE, linewidth=1.2)
        if excl_shown.any():
            ax.scatter(x[excl_shown], v_excl[excl_shown], s=46, marker="X",
                       facecolor="none", edgecolor=CRITICAL, linewidth=1.2, zorder=5)
        if excl_blank.any():
            # No number at all: mark the station along the foot of the axes.
            ax.scatter(x[excl_blank], np.zeros(int(excl_blank.sum())), s=40, marker="|",
                       color=CRITICAL, transform=ax.get_xaxis_transform(), zorder=5,
                       clip_on=False)

        finite = v_line[np.isfinite(v_line)]
        if finite.size:
            i = int(np.nanargmax(np.where(np.isfinite(v_line), v_line, -np.inf)))
            ax.annotate(f"max {v[i]:.2f} m/s", (x[i], v[i]), xytext=(0, 12),
                        textcoords="offset points", ha="center",
                        color=INK, fontsize=9)
            mean = float(np.mean(finite))
            ax.axhline(mean, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.annotate(f"mean {mean:.2f} m/s", (x[-1], mean), xytext=(-4, 5),
                        textcoords="offset points", ha="right", color=INK_2, fontsize=8.5)

        ax.set_ylabel("surface velocity  (m/s)")
        ax.set_title(
            f"Streamwise surface velocity - {result.config.section.name}"
            f"   ({result.duration_s:.0f} s, {result.n_frames} frames)",
            loc="left",
        )
        ax.margins(x=0.02)

        spread = {
            "stiv": ("±1σ from the spectral peak" if result.config.stiv.estimator == "fft"
                     else "±1σ from the angle histogram"),
            "lspiv": "±1σ of the one-second sub-ensembles",
            "ptv": "±1σ of the tracks",
        }[result.method]
        handles = [Line2D([], [], color=SERIES_1, lw=2.0, marker="o", markersize=5,
                          markeredgecolor=SURFACE,
                          label=f"{METHOD_LABELS[result.method]} surface velocity"),
                   Line2D([], [], color=SERIES_1, lw=6, alpha=0.25, label=spread)]
        for colour, m in zip(COMPARE_COLOURS, result.comparisons):
            handles.append(Line2D([], [], color=colour, lw=1.4, ls=(0, (5, 2)),
                                  marker="o", markersize=3.5,
                                  label=f"{METHOD_LABELS[m]} (compared)"))
        if interp.any():
            handles.append(Line2D([], [], color=CRITICAL, lw=0, marker="X", markersize=7,
                                  label="failed QC (interpolated)"))
        if excl_shown.any():
            handles.append(Line2D([], [], lw=0, marker="X", markersize=7,
                                  markerfacecolor="none", markeredgecolor=CRITICAL,
                                  label="failed QC (excluded; raw value)"))
        if excl_blank.any():
            handles.append(Line2D([], [], color=CRITICAL, lw=0, marker="|", markersize=8,
                                  label="failed QC (excluded; no value)"))
        ax.legend(handles=handles, loc="best", labelcolor=INK_2, fontsize=8.5)

        # The method's own quality measure shares the x-axis in its own panel
        # rather than a second y-scale on the same one: coherency for STIV,
        # the correlation peak ratio for LS-PIV, the track count for PTV.
        q = result.config.quality
        if result.method == "lspiv":
            quality = np.array([s.peak_ratio for s in result.stations], float)
            threshold, label, top = q.min_peak_ratio, "peak ratio", None
        elif result.method == "ptv":
            quality = np.array([s.n_measurements for s in result.stations], float)
            threshold, label, top = q.min_tracks, "tracks", None
        else:
            quality, threshold, label, top = coh, q.min_mean_coherency, "coherency", 1.05
        quality = np.where(np.isfinite(quality), quality, 0.0)
        axc.bar(x, quality, width=_bar_width(x), color=SERIES_1, alpha=0.55, linewidth=0)
        axc.axhline(threshold, color=CRITICAL, lw=1.2, ls=(0, (4, 3)))
        axc.annotate("QC threshold", (x[0], threshold),
                     xytext=(2, 4), textcoords="offset points", color=CRITICAL, fontsize=8)
        axc.set_ylim(0, top if top is not None else max(1.15 * float(quality.max(initial=0.0)),
                                                         1.5 * threshold))
        axc.set_ylabel(label)
        axc.set_xlabel(f"distance along cross-section  (m)   -   "
                       f"0 m at image point ({result.geometry.start_px[0]:.0f}, "
                       f"{result.geometry.start_px[1]:.0f})")

        _save(fig, path, dpi)
    return path


def _bar_width(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.5
    # Leave a small surface gap between adjacent bars.
    return float(np.min(np.diff(x))) * 0.8


# --------------------------------------------------------------------------- #
def plot_sti_panel(result: StivResult, path: Path, max_panels: int = 24,
                   dpi: int = 150) -> Path:
    """Contact sheet of space-time images, in the manner of Fig. 4 of the paper."""
    n = len(result.stis)
    if n == 0:
        raise ValueError("no space-time images to plot")
    idx = np.unique(np.linspace(0, n - 1, min(n, max_panels)).round().astype(int))

    ncols = int(min(6, len(idx)))
    nrows = int(math.ceil(len(idx) / ncols))

    with _style():
        fig = _figure(figsize=(2.05 * ncols, 2.5 * nrows))
        axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()

        for ax in axes:
            ax.set_axis_off()
            ax.grid(False)

        stations = result.stations_for("stiv")
        for ax, i in zip(axes, idx):
            s = result.stis[i]
            st = stations[i]
            ax.imshow(to_display(s.data), cmap="gray", aspect="auto",
                      extent=[0, s.length_m, s.duration_s, 0])
            ax.set_axis_on()
            ax.grid(False)
            ax.tick_params(labelsize=7, length=2)
            ax.set_title(
                f"{st.station_m:.1f} m\n"
                f"{_velocity_label(st)}   C={st.coherency:.2f}",
                fontsize=8, color=CRITICAL if st.failed else INK,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(AXIS)

        fig.suptitle(
            "Space-time images along the cross-section  "
            "(horizontal: metres downstream, vertical: seconds)",
            fontsize=10, color=INK, x=0.01, ha="left",
        )
        _save(fig, path, dpi)
    return path


# --------------------------------------------------------------------------- #
def plot_sti_detail(result: StivResult, index: int, path: Path, dpi: int = 150) -> Path:
    """Full orientation diagnosis of one station: Figs. 8, 9 and 11 together."""
    s = result.stis[index]
    st = result.stations_for("stiv")[index]
    field = result.fields[index]
    est = result.estimates[index]
    spectral = result.config.stiv.estimator == "fft"

    with _style():
        fig = _figure(figsize=(13.5, 4.4))
        ax0, ax1, ax2 = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

        extent = [0, s.length_m, s.duration_s, 0]
        ax0.imshow(to_display(s.data), cmap="gray", aspect="auto", extent=extent)
        ax0.grid(False)
        ax0.set_title(f"STI with orientation vectors  -  station {st.station_m:.2f} m",
                      loc="left")
        ax0.set_xlabel("downstream distance (m)")
        ax0.set_ylabel("time (s)")

        if field is not None:
            # Fig. 8: one vector per local window, drawn only where the pattern
            # is coherent enough to be trusted.
            xs = field.centres_x * s.sx_m_per_px
            ts = field.centres_t * s.st_s_per_px
            X, T = np.meshgrid(xs, ts)
            keep = field.coherency > max(0.3, result.config.stiv.min_coherency)
            scale_t = 0.035 * s.duration_s
            dx = np.sin(field.phi) * scale_t / s.st_s_per_px * s.sx_m_per_px
            dt = np.cos(field.phi) * scale_t
            ax0.quiver(
                X[keep], T[keep], dx[keep], dt[keep],
                angles="xy", scale_units="xy", scale=1.0,
                color=SERIES_2, width=0.004, headwidth=0, headlength=0, headaxislength=0,
                alpha=0.9,
            )
            ax0.set_xlim(extent[0], extent[1])
            ax0.set_ylim(extent[2], extent[3])

            im = ax1.imshow(field.coherency, cmap=COHERENCY_CMAP, aspect="auto",
                            vmin=0, vmax=1, extent=extent)
            ax1.grid(False)
            ax1.set_title("Coherency  C  (Eq. 11)", loc="left")
            ax1.set_xlabel("downstream distance (m)")
            cb = fig.colorbar(im, ax=ax1, pad=0.02)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=8, length=2)

        if est is not None and est.hist.sum() > 0:
            # Fig. 11: the summed, coherency-weighted angle histogram.
            deg = np.degrees(est.bin_centres)
            k = s.velocity_per_unit_slope()
            ax2.fill_between(deg, est.hist, color=SERIES_1, alpha=0.25, linewidth=0)
            ax2.plot(deg, est.hist, color=SERIES_1, lw=1.8)
            if np.isfinite(st.phi_rad):
                ax2.axvline(math.degrees(st.phi_rad), color=SERIES_2, lw=1.8)
                ax2.annotate(
                    f"  φ = {math.degrees(st.phi_rad):.1f}°\n"
                    f"  tan φ = {st.tan_phi:.2f}\n"
                    f"  U = {_velocity_label(st, 3)}",
                    (math.degrees(st.phi_rad), ax2.get_ylim()[1]),
                    va="top", color=INK, fontsize=9,
                )
            if spectral:
                ax2.set_title("Spectral power by streak angle  (FFT)", loc="left")
                ax2.set_ylabel("mean spectral power")
            else:
                ax2.set_title("Orientation-angle histogram  (Eq. 14)", loc="left")
                ax2.set_ylabel("summed frequency")
            ax2.set_xlabel("orientation angle φ  (degrees from time axis)")
            ax2.annotate(
                f"search range {math.degrees(est.phi_min):.0f}° to "
                f"{math.degrees(est.phi_max):.0f}°  "
                f"(U = {k * math.tan(est.phi_min):.2f} to {k * math.tan(est.phi_max):.2f} m/s)",
                (0.5, -0.28), xycoords="axes fraction", ha="center",
                color=MUTED, fontsize=8,
            )

        flags = ", ".join(st.flags) if st.flags else "none"
        fig.suptitle(
            f"Station {st.index} at {st.station_m:.2f} m   |   "
            f"Sx = {s.sx_m_per_px * 100:.2f} cm/px, St = {s.st_s_per_px * 1000:.1f} ms/px, "
            f"Sx/St = {s.velocity_per_unit_slope():.2f} m/s per unit slope   |   flags: {flags}",
            fontsize=9.5, color=INK_2, x=0.01, ha="left",
        )
        _save(fig, path, dpi)
    return path


# --------------------------------------------------------------------------- #
def plot_discharge(result: StivResult, path: Path, dpi: int = 150) -> Path:
    """Velocity, channel shape and panel discharge, stacked on a shared axis."""
    d = result.discharge
    if d is None:
        raise ValueError("no discharge result to plot")

    with _style():
        fig = _figure(figsize=(9.5, 7.6))
        ax0, ax1, ax2 = fig.subplots(
            3, 1, sharex=True, gridspec_kw={"height_ratios": [1.3, 1.2, 1.0]},
        )

        ax0.plot(d.stations_m, d.surface_velocity_ms, color=SERIES_1, lw=2.0,
                 marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0,
                 label=f"surface ({METHOD_LABELS[result.method]})")
        ax0.plot(d.stations_m, d.depth_averaged_ms, color=SERIES_3, lw=2.0,
                 ls=(0, (5, 2)), label=f"depth-averaged (α = {d.alpha:g})")
        ax0.set_ylabel("velocity  (m/s)")
        ax0.legend(loc="best", labelcolor=INK_2, fontsize=8.5)
        ax0.set_title(
            f"Discharge  {d.discharge_m3s:.3f} m³/s   |   "
            f"area {d.area_m2:.2f} m²   |   mean U {d.mean_velocity_ms:.3f} m/s",
            loc="left",
        )

        ax1.fill_between(d.stations_m, -d.depths_m, 0, color=SERIES_1, alpha=0.20,
                         linewidth=0)
        ax1.plot(d.stations_m, -d.depths_m, color=SERIES_1, lw=1.8)
        ax1.axhline(0, color=AXIS, lw=1.2)
        ax1.set_ylabel("depth below surface  (m)")
        ax1.annotate(f"max depth {d.max_depth_m:.2f} m",
                     (d.stations_m[int(np.argmax(d.depths_m))], -d.max_depth_m),
                     xytext=(0, -12), textcoords="offset points", ha="center",
                     color=INK_2, fontsize=8.5)

        ax2.bar(d.stations_m, d.panel_discharge_m3s, width=_bar_width(d.stations_m),
                color=SERIES_2, alpha=0.85, linewidth=0)
        ax2.set_ylabel("panel Q  (m³/s)")
        ax2.set_xlabel("distance along cross-section  (m)")

        _save(fig, path, dpi)
    return path


# --------------------------------------------------------------------------- #
def plot_correlation_panel(result: StivResult, path: Path, max_panels: int = 24,
                           dpi: int = 150) -> Path:
    """Contact sheet of LS-PIV's ensemble correlation planes, one per station.

    Each plane is drawn on velocity axes -- downstream across, cross-stream up
    -- so the peak sits at the station's velocity; one clear, round peak is a
    good measurement, a ridge or several peaks a doubtful one.
    """
    corrs = result.correlations
    if not corrs:
        raise ValueError("no LS-PIV correlations to plot")
    stations = result.stations_for("lspiv")
    n = len(corrs)
    idx = np.unique(np.linspace(0, n - 1, min(n, max_panels)).round().astype(int))
    ncols = int(min(6, len(idx)))
    nrows = int(math.ceil(len(idx) / ncols))
    # The planes are in the patch's own along/across axes, laid out with the
    # flow direction the section started with.  If the direction was reversed
    # since, both axes are -- across being square to along.
    flipped = float(np.dot(corrs[0].layout.along_world, result.geometry.flow_dir_world)) < 0

    with _style():
        fig = _figure(figsize=(2.05 * ncols, 2.3 * nrows))
        axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
        for ax in axes:
            ax.set_axis_off()
            ax.grid(False)
        for ax, i in zip(axes, idx):
            c, st = corrs[i], stations[i]
            m = c.layout.margin_n
            per = c.layout.gsd_m / (c.pair_step * c.st_s_per_px)     # m/s per sample
            lim = (m + 0.5) * per
            # Row 0 is the most negative cross-stream offset, column 0 the
            # most negative downstream one: origin "lower" puts them right.
            ax.imshow(c.plane[::-1, ::-1] if flipped else c.plane, cmap=COHERENCY_CMAP, vmin=0.0,
                      vmax=max(float(c.plane.max()), 1e-6), origin="lower",
                      extent=[-lim, lim, -lim, lim], interpolation="nearest")
            ax.axhline(0, color=SURFACE, lw=0.5, alpha=0.7)
            ax.axvline(0, color=SURFACE, lw=0.5, alpha=0.7)
            if np.isfinite(st.velocity_raw_ms):
                ax.scatter([st.velocity_raw_ms],
                           [st.velocity_cross_ms if np.isfinite(st.velocity_cross_ms) else 0.0],
                           s=26, marker="+", color=SERIES_2, linewidth=1.4, zorder=3)
            ax.set_axis_on()
            ax.grid(False)
            ax.tick_params(labelsize=7, length=2)
            ax.set_title(f"{st.station_m:.1f} m\n{_velocity_label(st)}   R={st.peak_ratio:.1f}",
                         fontsize=8, color=CRITICAL if st.failed else INK)
            for spine in ax.spines.values():
                spine.set_edgecolor(AXIS)
        fig.suptitle(
            "LS-PIV ensemble correlation at each station  (horizontal: downstream "
            "velocity, vertical: cross-stream velocity, m/s; + the peak; R the peak ratio)",
            fontsize=10, color=INK, x=0.01, ha="left",
        )
        _save(fig, path, dpi)
    return path


#: Most tracks drawn on the tracks figure; a random subset beyond that.
_MAX_TRACKS_DRAWN = 4000


def plot_tracks(result: StivResult, path: Path, dpi: int = 150) -> Path:
    """The reference frame with every PTV track that counted, coloured by speed.

    Tracks are drawn where they ran in the reference frame -- the camera's
    own motion removed -- over the stations' boxes.
    """
    from matplotlib.collections import LineCollection

    tracks = result.tracks
    if tracks is None:
        raise ValueError("no PTV tracks to plot")
    frame = result.reference_frame_bgr
    rgb = frame[:, :, ::-1] if frame.ndim == 3 else frame
    h, w = rgb.shape[:2]
    rect = result.geometry.rectifier

    keep = np.arange(len(tracks))
    if len(keep) > _MAX_TRACKS_DRAWN:
        keep = np.sort(np.random.default_rng(0).choice(keep, _MAX_TRACKS_DRAWN, replace=False))
    seg = np.stack([tracks.start_px[keep], tracks.end_px[keep]], axis=1) if len(keep) else \
        np.empty((0, 2, 2))
    speed = np.linalg.norm(tracks.velocity_world[keep], axis=1) if len(keep) else np.empty(0)

    scale = max(1.0 / 110.0, _MIN_OVERVIEW_WIDTH_IN / w)
    with _style():
        fig = _figure(figsize=(min(13, w * scale), min(9, h * scale)))
        ax = fig.subplots()
        ax.imshow(rgb)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_axis_off()
        ax.grid(False)
        for b in result.boxes:
            q = rect.world_to_image(b.corners_world())
            q = np.vstack([q, q[:1]])
            ax.plot(q[:, 0], q[:, 1], color=SURFACE, lw=0.8, alpha=0.6)
        if len(seg):
            vmax = float(np.percentile(speed, 98)) or 1.0
            lc = LineCollection(seg, array=speed, cmap=COHERENCY_CMAP, norm=None,
                                linewidths=1.2, alpha=0.95)
            lc.set_clim(0.0, vmax)
            ax.add_collection(lc)
            cb = fig.colorbar(lc, ax=ax, pad=0.01, shrink=0.8)
            cb.set_label("track speed  (m/s)")
            cb.outline.set_visible(False)
        ax.set_title(
            f"Particle tracks  -  {result.config.section.name}   ({len(tracks)} tracks of "
            f"{tracks.duration_s:.2f} s{', a sample drawn' if len(keep) < len(tracks) else ''})",
            loc="left",
        )
        _save(fig, path, dpi, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------- #
def save_all(result: StivResult, out_dir: Path, dpi: int = 150,
             detail_stations: Optional[Sequence[int]] = None) -> list[Path]:
    """Write the full figure set; returns the paths written.

    The STI figures are drawn whenever STIV ran, reported or compared; the
    correlation panel when LS-PIV did, and the tracks when PTV did.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written.append(plot_overview(result, out_dir / "overview.png", dpi))
    written.append(plot_velocity_map(result, out_dir / "velocity_map.png", dpi))
    written.append(plot_velocity_profile(result, out_dir / "velocity_profile.png", dpi))
    if result.stis:
        written.append(plot_sti_panel(result, out_dir / "sti_panel.png", dpi=dpi))
    if result.correlations:
        written.append(plot_correlation_panel(result, out_dir / "correlation_panel.png", dpi=dpi))
    if result.tracks is not None:
        written.append(plot_tracks(result, out_dir / "tracks.png", dpi))

    if result.discharge is not None:
        written.append(plot_discharge(result, out_dir / "discharge.png", dpi))

    if result.stis:
        if detail_stations is None:
            n = len(result.stis)
            detail_stations = sorted({0, n // 2, n - 1}) if n else []
        detail_dir = out_dir / "sti_detail"
        detail_dir.mkdir(exist_ok=True)
        for i in detail_stations:
            if 0 <= i < len(result.stis) and (result.fields[i] is not None
                                              or result.estimates[i] is not None):
                written.append(
                    plot_sti_detail(result, i, detail_dir / f"station_{i:03d}.png", dpi)
                )

    return written
