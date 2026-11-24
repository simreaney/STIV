"""One PDF of a run, to keep, print or hand in: ``report.pdf``.

Everything a reader needs to judge the numbers, in reading order:

1. **Summary** -- the headline velocities (and discharge), what was measured
   and how, and every warning the run raised.
2. **Figures** -- the flow map, the velocity profile, the discharge (if
   computed), the section check, the space-time images and the per-station
   diagnostics: the same figures as the PNGs, as vector pages.
3. **Stations** -- the velocity table.
4. **Configuration** -- ``config_used.yaml`` in full, so the run can be
   repeated from the report alone.

A survey of several cross-sections gets one report covering them all
(:func:`write_survey_pdf_report`): a summary comparing the sections, the flow
map and section check with every section on one frame, then the summary,
figures and stations of each section in turn, and the configuration once.

Written with matplotlib's own PDF backend, so it needs nothing beyond the
package's dependencies.  Pages are A4 for the text and each figure's own size
for the figures.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from . import __version__, plotting
from .analysis import StivResult, SurveyResult
from .config import Config, yaml_text

#: A4 portrait, inches.
A4 = (8.27, 11.69)
#: Resolution the photographs inside the figures are resampled to.  Sharp in
#: print at page size, and it keeps a report of 5.4K drone footage near 5 MB
#: where 150 would make it nearly 10: the frame is stored losslessly, twice.
REPORT_DPI = 110
_MARGIN_IN = 0.75
_SANS = "DejaVu Sans"
_MONO = "DejaVu Sans Mono"


def write_pdf_report(result: StivResult, cfg: Config, path: str | Path,
                     dpi: int = REPORT_DPI, created: Optional[datetime] = None) -> Path:
    """Write the report for ``result`` to ``path``; returns ``path``.

    ``dpi`` is the resolution the video frame and the space-time images are
    resampled to inside the figures -- lines and text stay vector.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = created or datetime.now()
    name = result.config.section.name
    metadata = {
        "Title": f"STIV report - {name}",
        "Subject": "River surface velocity by space-time image velocimetry",
        "Creator": f"stiv {__version__}",
        "CreationDate": created,
    }
    with PdfPages(path, metadata=metadata) as pdf:
        _section_pages(pdf, result, cfg, created, dpi)
        _configuration(pdf, cfg, name)
    return path


def write_survey_pdf_report(survey: SurveyResult, cfg: Config, path: str | Path,
                            dpi: int = REPORT_DPI, created: Optional[datetime] = None) -> Path:
    """Write one report covering every cross-section of ``survey``; returns ``path``.

    ``cfg`` is the survey's own config.  A survey of one section gets the
    report :func:`write_pdf_report` writes.
    """
    if len(survey.results) == 1:
        return write_pdf_report(survey.results[0], cfg, path, dpi, created)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = created or datetime.now()
    label = f"{len(survey.results)} cross-sections"
    metadata = {
        "Title": f"STIV report - {label}",
        "Subject": "River surface velocity by space-time image velocimetry",
        "Creator": f"stiv {__version__}",
        "CreationDate": created,
    }
    with PdfPages(path, metadata=metadata) as pdf:
        _survey_summary(pdf, survey, cfg, created)
        plotting.plot_survey_velocity_map(survey.results, pdf, dpi)
        plotting.plot_survey_overview(survey.results, pdf, dpi)
        for r in survey.results:
            _section_pages(pdf, r, r.config, created, dpi, in_survey=True,
                           omit_warnings=survey.shared_warnings)
        _configuration(pdf, cfg, label)
    return path


def _section_pages(pdf: PdfPages, result: StivResult, cfg: Config, created: datetime, dpi: int,
                   in_survey: bool = False, omit_warnings: Sequence[str] = ()) -> None:
    """A section's summary, figures and station table."""
    _summary(pdf, result, cfg, created, in_survey, omit_warnings)

    plotting.plot_velocity_map(result, pdf, dpi)
    plotting.plot_velocity_profile(result, pdf, dpi)
    if result.discharge is not None:
        plotting.plot_discharge(result, pdf, dpi)
    plotting.plot_overview(result, pdf, dpi)
    if result.stis:
        plotting.plot_sti_panel(result, pdf, dpi=dpi)
    n = len(result.stations)
    for i in sorted({0, n // 2, n - 1}) if n else []:
        if 0 <= i < len(result.stis) and result.fields[i] is not None:
            plotting.plot_sti_detail(result, i, pdf, dpi)

    _stations(pdf, result)


# --------------------------------------------------------------------------- #
# text pages
# --------------------------------------------------------------------------- #
class _Pages:
    """A4 pages of flowing text, a new page whenever one fills up."""

    def __init__(self, pdf: PdfPages, title: str, footer: str):
        self.pdf = pdf
        self.title = title
        self.footer = footer
        self.fig: Optional[Figure] = None
        self.y = 0.0
        self.page = 0

    # figure-fraction geometry
    @staticmethod
    def _dy(points: float) -> float:
        return points / (A4[1] * 72)

    @property
    def left(self) -> float:
        return _MARGIN_IN / A4[0]

    @property
    def bottom(self) -> float:
        return _MARGIN_IN / A4[1]

    def _new_page(self) -> None:
        self.close_page()
        with plotting._style():
            self.fig = Figure(figsize=A4)
            FigureCanvasAgg(self.fig)
        self.page += 1
        self.y = 1 - _MARGIN_IN / A4[1]
        title = self.title if self.page == 1 else f"{self.title} (continued)"
        self.text(title, size=13, weight="bold", gap=6)
        self.rule()

    def close_page(self) -> None:
        if self.fig is not None:
            self.fig.text(1 - self.left, self.bottom * 0.55, self.footer, ha="right", va="bottom",
                          fontsize=7, color=plotting.MUTED, family=_SANS)
            with plotting._style():
                self.pdf.savefig(self.fig)
            self.fig = None

    def _room(self, points: float) -> None:
        if self.fig is None or self.y - self._dy(points) < self.bottom:
            self._new_page()

    def text(self, text: str, size: float = 9, weight: str = "normal", colour: str = plotting.INK,
             mono: bool = False, indent_in: float = 0.0, gap: float = 0.0) -> None:
        line = size * 1.35
        self._room(line)
        self.y -= self._dy(line)
        self.fig.text(self.left + indent_in / A4[0], self.y, text, fontsize=size,
                      fontweight=weight, color=colour, family=_MONO if mono else _SANS,
                      va="baseline")
        self.y -= self._dy(gap)

    def wrapped(self, text: str, size: float = 9, mono: bool = False, indent_in: float = 0.0,
                colour: str = plotting.INK, bullet: bool = False) -> None:
        """``text`` wrapped to the page; a bullet hangs in the margin of its block."""
        if bullet:
            indent_in += 0.18
        # Characters per line across the text width, at this size.
        usable_pt = (A4[0] - 2 * _MARGIN_IN - indent_in) * 72
        per_char = size * (0.6 if mono else 0.52)
        width = max(20, int(usable_pt / per_char))
        lines = textwrap.wrap(text, width=width, break_long_words=True,
                              break_on_hyphens=False) or [""]
        for k, line in enumerate(lines):
            self.text(line, size=size, mono=mono, indent_in=indent_in, colour=colour)
            if bullet and k == 0:
                self.fig.text(self.left + (indent_in - 0.16) / A4[0], self.y, "•", fontsize=size,
                              color=colour, family=_SANS, va="baseline")

    def pair(self, key: str, value: str) -> None:
        """A label and its value, the value wrapped in its own column."""
        size = 9
        usable_pt = (A4[0] - 2 * _MARGIN_IN - 1.9) * 72
        lines = textwrap.wrap(value, width=max(20, int(usable_pt / (size * 0.52)))) or [""]
        for k, line in enumerate(lines):
            self._room(size * 1.35)
            self.y -= self._dy(size * 1.35)
            if k == 0:
                self.fig.text(self.left, self.y, key, fontsize=size, color=plotting.INK_2,
                              family=_SANS, va="baseline")
            self.fig.text(self.left + 1.9 / A4[0], self.y, line, fontsize=size,
                          color=plotting.INK, family=_SANS, va="baseline")

    def heading(self, text: str) -> None:
        self._room(40)
        self.gap(10)
        self.text(text, size=11, weight="bold", gap=3)

    def rule(self) -> None:
        self.y -= self._dy(4)
        self.fig.add_artist(_hline(self.left, 1 - self.left, self.y))
        self.y -= self._dy(6)

    def gap(self, points: float) -> None:
        self.y -= self._dy(points)

    def tiles(self, items: list[tuple[str, str, str]]) -> None:
        """A row of headline numbers in boxes."""
        height_pt = 50
        self._room(height_pt + 10)
        top = self.y - self._dy(4)
        bottom = top - self._dy(height_pt)
        width = (1 - 2 * self.left - 0.02 * (len(items) - 1)) / max(len(items), 1)
        for k, (label, value, sub) in enumerate(items):
            x = self.left + k * (width + 0.02)
            self.fig.add_artist(FancyBboxPatch(
                (x, bottom), width, top - bottom, boxstyle="round,pad=0,rounding_size=0.006",
                transform=self.fig.transFigure, facecolor="#f1f0ec", edgecolor=plotting.GRID,
                linewidth=0.8))
            self.fig.text(x + 0.012, top - self._dy(13), label.upper(), fontsize=6.5,
                          color=plotting.INK_2, family=_SANS)
            self.fig.text(x + 0.012, top - self._dy(31), value, fontsize=15, fontweight="bold",
                          color=plotting.INK, family=_SANS)
            if sub:
                self.fig.text(x + 0.012, top - self._dy(44), sub, fontsize=6.5,
                              color=plotting.MUTED, family=_SANS)
        self.y = bottom - self._dy(8)


def _hline(x0: float, x1: float, y: float):
    from matplotlib.lines import Line2D

    return Line2D([x0, x1], [y, y], color=plotting.AXIS, linewidth=0.8)


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}" if x is not None and np.isfinite(x) else "-"


def _summary(pdf: PdfPages, result: StivResult, cfg: Config, created: datetime,
             in_survey: bool = False, omit_warnings: Sequence[str] = ()) -> None:
    """A section's summary page.

    In a survey's report (``in_survey``) the page is titled by the section,
    ``omit_warnings`` -- those the survey's own summary has listed -- are
    left out, and so is the guide to the figures, which it gives once.
    """
    name = result.config.section.name
    footer = f"STIV report  -  {name}  -  stiv {__version__}"
    title = f"Cross-section: {name}" if in_survey else "Surface velocity report"
    pages = _Pages(pdf, title, footer)
    pages.text(f"{name}   -   {Path(cfg.video.path).name}   -   "
               f"{created:%d %B %Y, %H:%M}", size=9, colour=plotting.INK_2, gap=4)

    v = result.velocities
    finite = v[np.isfinite(v)]
    n_ok = sum(1 for s in result.stations if not s.failed)
    tiles = []
    if finite.size:
        tiles += [("Mean surface velocity", f"{finite.mean():.2f} m/s", f"over {finite.size} stations"),
                  ("Max surface velocity", f"{finite.max():.2f} m/s", "")]
    else:
        tiles.append(("Surface velocity", "none", "no station passed quality control"))
    if result.discharge is not None:
        tiles.append(("Discharge", f"{result.discharge.discharge_m3s:.3f} m³/s",
                      f"area {result.discharge.area_m2:.2f} m²"))
    tiles.append(("Stations passing QC", f"{n_ok} / {len(result.stations)}", ""))
    pages.tiles(tiles)

    pages.heading("The survey")
    geom = result.geometry
    start = cfg.video.start_s
    pages.pair("Video", Path(cfg.video.path).name)
    pages.pair("Analysed", f"{start:.1f} s to {start + result.duration_s:.1f} s: "
                           f"{result.n_frames} frames at {result.fps:.2f} fps "
                           f"(frame step {cfg.video.frame_step})")
    pages.pair("Scale", geom.rectifier.describe())
    pages.pair("Cross-section", f"{geom.width_m:.2f} m wide, {len(result.stations)} stations, "
                                f"{cfg.stiv.search_line_length_m:g} m search lines")
    pages.pair("Expected velocity", f"{cfg.stiv.velocity_range_ms[0]:g} to "
                                    f"{cfg.stiv.velocity_range_ms[1]:g} m/s")
    pages.pair("Camera motion", result.stabilisation.summary())
    note = getattr(result, "water_mask_note", "")
    if note:
        pages.pair("Water mask", note)
    if result.discharge is not None:
        d = result.discharge
        pages.pair("Discharge", f"{d.discharge_m3s:.3f} m³/s from {d.area_m2:.2f} m² at a mean "
                                f"velocity of {d.mean_velocity_ms:.3f} m/s (velocity index "
                                f"{d.alpha:g})")
    pages.pair("Software", f"stiv {__version__}; analysis took {result.elapsed_s:.1f} s")

    warnings = [w for w in result.warnings if w not in set(omit_warnings)]
    if in_survey:
        heading = ("Warnings for this section" if warnings else
                   "Warnings for this section: none beyond those for the whole survey")
    else:
        heading = "Warnings" if warnings else "Warnings: none"
    pages.heading(heading)
    for w in warnings:
        pages.wrapped(w, bullet=True, colour=plotting.INK)
        pages.gap(3)

    pages.heading("Run report")
    # The console summary, less its warnings: they are listed above.
    for line in result.summary().split("\n\nwarning:")[0].splitlines():
        pages.wrapped(line, size=7.5, mono=True, colour=plotting.INK_2)

    if not in_survey:
        _figure_guide(pages)
    pages.close_page()


def _figure_guide(pages: _Pages, survey: bool = False) -> None:
    pages.heading("Reading the figures")
    for text in (
        "Flow map: an arrow per station, its length proportional to the surface velocity; the "
        "key gives the scale. Hollow red arrows failed quality control and were interpolated; "
        "a cross is a station with no velocity."
        + (" The survey's flow map draws every section at one scale, so arrows compare "
           "between sections." if survey else ""),
        "Velocity profile: the surface velocity across the section with its spread, and the "
        "coherency of each station below -- 1 for a clean streak pattern, 0 for none.",
        "Section check: where the cross-section and the search lines were laid on the frame.",
        "Space-time images: the streaks the velocities were measured from. Their angle is the "
        "speed; no streaks, no measurement.",
    ):
        pages.wrapped(text, bullet=True, colour=plotting.INK_2)
        pages.gap(2)


def _survey_summary(pdf: PdfPages, survey: SurveyResult, cfg: Config, created: datetime) -> None:
    results = survey.results
    first = results[0]
    label = f"{len(results)} cross-sections"
    pages = _Pages(pdf, "Surface velocity report", f"STIV report  -  {label}  -  stiv {__version__}")
    pages.text(f"{label}   -   {Path(cfg.video.path).name}   -   {created:%d %B %Y, %H:%M}",
               size=9, colour=plotting.INK_2, gap=4)

    n = sum(len(r.stations) for r in results)
    n_ok = sum(1 for r in results for s in r.stations if not s.failed)
    tiles = [("Cross-sections", str(len(results)), "analysed together"),
             ("Stations passing QC", f"{n_ok} / {n}", "over every section")]
    n_q = sum(1 for r in results if r.discharge is not None)
    if n_q:
        tiles.append(("Discharge", f"{n_q} of {len(results)}", "sections with depths"))
    pages.tiles(tiles)

    pages.heading("The survey")
    start = cfg.video.start_s
    pages.pair("Video", Path(cfg.video.path).name)
    pages.pair("Analysed", f"{start:.1f} s to {start + first.duration_s:.1f} s: "
                           f"{first.n_frames} frames at {first.fps:.2f} fps "
                           f"(frame step {cfg.video.frame_step}), read once for every section")
    pages.pair("Scale", first.geometry.rectifier.describe())
    pages.pair("Expected velocity", f"{cfg.stiv.velocity_range_ms[0]:g} to "
                                    f"{cfg.stiv.velocity_range_ms[1]:g} m/s")
    pages.pair("Camera motion", first.stabilisation.summary())
    if first.water_mask_note:
        pages.pair("Water mask", first.water_mask_note)
    pages.pair("Software", f"stiv {__version__}; analysis took {survey.elapsed_s:.1f} s")

    pages.heading("The cross-sections")
    pages.wrapped("Each section has pages of its own after the two figures that follow. "
                  "Clean stations passed quality control with no caveats.", colour=plotting.INK_2)
    pages.gap(4)
    width = max(8, *(len(nm) for nm in survey.names))
    has_q = n_q > 0
    pages.text(f"{'section':<{width}} {'width':>7} {'clean':>7} {'mean U':>7} {'max U':>7}"
               + (f" {'Q':>8}" if has_q else ""), size=8, mono=True, weight="bold")
    pages.text(f"{'':<{width}} {'(m)':>7} {'':>7} {'(m/s)':>7} {'(m/s)':>7}"
               + (f" {'(m³/s)':>8}" if has_q else ""), size=8, mono=True, colour=plotting.MUTED,
               gap=2)
    for r in results:
        v = r.velocities
        finite = v[np.isfinite(v)]
        clean = f"{sum(1 for s in r.stations if s.ok)}/{len(r.stations)}"
        row = (f"{r.config.section.name:<{width}} {r.geometry.width_m:7.2f} {clean:>7} "
               f"{_fmt(finite.mean() if finite.size else None):>7} "
               f"{_fmt(finite.max() if finite.size else None):>7}")
        if has_q:
            row += f" {_fmt(r.discharge.discharge_m3s if r.discharge is not None else None):>8}"
        pages.text(row, size=8, mono=True)

    pages.heading("Warnings for the whole survey" if survey.shared_warnings
                  else "Warnings for the whole survey: none")
    for w in survey.shared_warnings:
        pages.wrapped(w, bullet=True, colour=plotting.INK)
        pages.gap(3)

    _figure_guide(pages, survey=True)
    pages.close_page()


def _stations(pdf: PdfPages, result: StivResult) -> None:
    pages = _Pages(pdf, "Velocity by station",
                   f"STIV report  -  {result.config.section.name}  -  stiv {__version__}")
    pages.wrapped("Distance is measured along the cross-section from its start. ± is the "
                  "spread of the station's angle estimate. Flags explain a station that failed "
                  "or was changed by quality control.", colour=plotting.INK_2)
    pages.gap(6)
    header = f"{'station':>8} {'velocity':>9} {'±':>7} {'coherency':>9} {'tan phi':>8}  flags"
    pages.text(header, size=8, mono=True, weight="bold")
    pages.text(f"{'(m)':>8} {'(m/s)':>9} {'(m/s)':>7}", size=8, mono=True, colour=plotting.MUTED,
               gap=2)
    for s in result.stations:
        flags = ", ".join(s.flags)
        colour = plotting.CRITICAL if s.failed else plotting.INK
        pages.text(f"{s.station_m:8.2f} {_fmt(s.velocity_ms):>9} {_fmt(s.velocity_std_ms):>7} "
                   f"{_fmt(s.coherency, 2):>9} {_fmt(s.tan_phi, 2):>8}  {flags}",
                   size=8, mono=True, colour=colour)
    pages.close_page()


def _configuration(pdf: PdfPages, cfg: Config, name: Optional[str] = None) -> None:
    pages = _Pages(pdf, "Configuration used",
                   f"STIV report  -  {name or cfg.section.name}  -  stiv {__version__}")
    pages.wrapped("The settings this run used, as written to config_used.yaml. Running "
                  "`stiv run` on them repeats the analysis.", colour=plotting.INK_2)
    pages.gap(6)
    for line in yaml_text(cfg.to_dict()).splitlines():
        indent = len(line) - len(line.lstrip(" "))
        pages.wrapped(line.strip(), size=7.5, mono=True, indent_in=indent * 0.06)
    pages.close_page()
