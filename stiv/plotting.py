"""Diagnostic figures for a STIV run.

Five outputs, each answering a question an operator actually asks:

``overview.png``            did I measure where I meant to?
``velocity_profile.png``    what is the velocity across the section?
``sti_panel.png``           do the space-time images show usable streaks?
``sti_detail_*.png``        why did *this* station give *that* number?
``discharge.png``           how does velocity combine with depth into flow?

These are report figures written to disk, so they deliberately commit to a
single light appearance rather than adapting to a viewer theme.  Colours are
the validated reference palette; magnitude uses a single-hue blue ramp and the
space-time images stay greyscale, since they are photographs rather than
encodings of an abstract quantity.

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
def plot_overview(result: StivResult, path: Path, dpi: int = 150) -> Path:
    """First frame with the cross-section, search lines and station numbering."""
    frame = result.reference_frame_bgr
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

        geom = result.geometry
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
        ax.annotate("0 m", geom.start_px, color=SURFACE, fontsize=9, weight="bold",
                    xytext=(6, 6), textcoords="offset points",
                    path_effects=_outline())
        ax.annotate(f"{geom.width_m:.1f} m", geom.end_px, color=SURFACE, fontsize=9,
                    weight="bold", xytext=(6, 6), textcoords="offset points",
                    path_effects=_outline())

        # Arrow showing the assumed downstream direction.
        centre = 0.5 * (np.asarray(geom.start_px) + np.asarray(geom.end_px))
        tip = result.geometry.rectifier.world_to_image(
            [geom.rectifier.image_to_world([centre])[0] + geom.flow_dir_world *
             max(2.0, 0.15 * geom.width_m)]
        )[0]
        ax.annotate("", xy=tip, xytext=centre,
                    arrowprops=dict(arrowstyle="-|>", color=SERIES_1, lw=2.2,
                                    shrinkA=0, shrinkB=0))

        handles = [
            Line2D([], [], color=SERIES_1, lw=2.4, label="cross-section / flow direction"),
            Line2D([], [], color=SERIES_3, lw=1.8, label="search line (measured)"),
        ]
        if any(s.failed for s in result.stations):
            handles.append(Line2D([], [], color=SERIES_2, lw=1.8, label="search line (failed QC)"))
        ax.legend(handles=handles, loc="lower center", ncol=len(handles),
                  bbox_to_anchor=(0.5, -0.06), labelcolor=INK_2)

        ax.set_title(
            f"{result.config.section.name}  -  {geom.width_m:.1f} m, "
            f"{len(geom.search_lines)} stations, "
            f"{result.config.stiv.search_line_length_m:.1f} m search lines",
            loc="left",
        )
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _outline():
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=2.5, foreground="#00000090")]


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

        handles = [Line2D([], [], color=SERIES_1, lw=2.0, marker="o", markersize=5,
                          markeredgecolor=SURFACE, label="STIV surface velocity"),
                   Line2D([], [], color=SERIES_1, lw=6, alpha=0.25,
                          label="±1σ from the angle histogram")]
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

        # Coherency shares the x-axis in its own panel rather than a second
        # y-scale on the same one.
        axc.bar(x, coh, width=_bar_width(x), color=SERIES_1, alpha=0.55, linewidth=0)
        axc.axhline(result.config.quality.min_mean_coherency, color=CRITICAL,
                    lw=1.2, ls=(0, (4, 3)))
        axc.annotate("QC threshold", (x[0], result.config.quality.min_mean_coherency),
                     xytext=(2, 4), textcoords="offset points", color=CRITICAL, fontsize=8)
        axc.set_ylim(0, 1.05)
        axc.set_ylabel("coherency")
        axc.set_xlabel(f"distance along cross-section  (m)   -   "
                       f"0 m at image point ({result.geometry.start_px[0]:.0f}, "
                       f"{result.geometry.start_px[1]:.0f})")

        fig.savefig(path, dpi=dpi)
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

        for ax, i in zip(axes, idx):
            s = result.stis[i]
            st = result.stations[i]
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
        fig.savefig(path, dpi=dpi)
    return path


# --------------------------------------------------------------------------- #
def plot_sti_detail(result: StivResult, index: int, path: Path, dpi: int = 150) -> Path:
    """Full orientation diagnosis of one station: Figs. 8, 9 and 11 together."""
    s = result.stis[index]
    st = result.stations[index]
    field = result.fields[index]
    est = result.estimates[index]

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
            ax2.set_title("Orientation-angle histogram  (Eq. 14)", loc="left")
            ax2.set_xlabel("orientation angle φ  (degrees from time axis)")
            ax2.set_ylabel("summed frequency")
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
        fig.savefig(path, dpi=dpi)
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
                 label="surface (STIV)")
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

        fig.savefig(path, dpi=dpi)
    return path


# --------------------------------------------------------------------------- #
def save_all(result: StivResult, out_dir: Path, dpi: int = 150,
             detail_stations: Optional[Sequence[int]] = None) -> list[Path]:
    """Write the full figure set; returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written.append(plot_overview(result, out_dir / "overview.png", dpi))
    written.append(plot_velocity_profile(result, out_dir / "velocity_profile.png", dpi))
    written.append(plot_sti_panel(result, out_dir / "sti_panel.png", dpi=dpi))

    if result.discharge is not None:
        written.append(plot_discharge(result, out_dir / "discharge.png", dpi))

    if detail_stations is None:
        n = len(result.stations)
        detail_stations = sorted({0, n // 2, n - 1}) if n else []
    detail_dir = out_dir / "sti_detail"
    detail_dir.mkdir(exist_ok=True)
    for i in detail_stations:
        if 0 <= i < len(result.stis) and result.fields[i] is not None:
            written.append(
                plot_sti_detail(result, i, detail_dir / f"station_{i:03d}.png", dpi)
            )

    return written
