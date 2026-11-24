"""The STIV pipeline: video in, cross-section velocity profile out.

Order of operations, following Fig. 5 of the paper with the drone-specific
additions:

1. lay out the cross-section and its streamwise search lines (:mod:`geometry`);
2. stream the video, tracking camera motion (:mod:`stabilise`) and reading one
   STI row per frame per line (:mod:`sti`);
3. condition each STI and run the structure-tensor orientation analysis over a
   grid of local windows (:mod:`orientation`);
4. reduce each STI's windows to a single angle by the coherency-weighted
   average of Eq. (14), and convert to velocity by Eq. (16);
5. quality-control the profile and, if depths are available, integrate it into
   a discharge (:mod:`discharge`).

STIV is one of three velocity methods (``velocimetry.method``).  LS-PIV
(:mod:`lspiv`) and particle tracking (:mod:`ptv`) collect from the same single
pass over the video -- ortho-image patches and feature tracks in place of, or
beside, the STIs -- and their per-station velocity vectors go through the same
quality control and discharge.  Methods listed in ``velocimetry.compare`` are
run alongside the reported one and reported next to it.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import geometry, lspiv, orientation, ptv, sti as sti_mod
from .config import (
    METHOD_LABELS, Config, ConfigError, FlowConfig, StabiliseConfig, is_auto_water_mask,
)
from .discharge import DischargeResult, compute_discharge
from .geometry import SectionGeometry, StationBox
from .lspiv import CorrelationResult
from .orientation import AngleEstimate, OrientationField, SpectralProfile
from .ptv import TrackSet
from .stabilise import StabilisationReport, Stabiliser, build_mask
from .sti import SpaceTimeImage
from .vectors import VectorEstimate, robust_spread
from .video import FrameSource

log = logging.getLogger(__name__)


#: Flags meaning "this station produced no trustworthy number at all".  Only
#: these make a station eligible for replacement by interpolation.
FAILURE_FLAGS = frozenset(
    {
        "no_orientation",
        "sti_too_small",
        "low_valid_fraction",
        "low_coherency",
        "high_uncertainty",
        "out_of_frame",
        # LS-PIV
        "no_correlation_peak",
        "low_peak_ratio",
        "displacement_at_search_edge",
        # PTV
        "too_few_tracks",
    }
)

#: Flags meaning "the number is real but near the edge of the method's range".
#: A genuinely slow station by the bank is the common case, and interpolating
#: it away from its neighbours would be worse than reporting it.
ADVISORY_FLAGS = frozenset(
    {"tan_phi_low", "tan_phi_high", "interpolated", "median_filtered",
     # LS-PIV and PTV
     "displacement_low", "cross_flow_high"}
)

#: LS-PIV: a displacement below this many samples per pair is resolved only
#: to a few percent by the sub-pixel peak fit.
_LOW_DISPLACEMENT_SAMPLES = 0.5
#: LS-PIV and PTV: a vector further than this off the flow direction suggests
#: the direction is wrong, or the station is in an eddy.
_CROSS_FLOW_DEG = 30.0

#: A station whose samples left the frame for more than this fraction of the
#: STI is failed: the gaps are filled with the column mean, and past a few
#: percent that fill starts to draw structure of its own.
_MAX_OUT_OF_FRAME_FRACTION = 0.02
#: The high-uncertainty test compares a station's spread with the larger of its
#: own velocity and this fraction of the section's median speed, so a slow but
#: well-resolved station by the bank is not condemned for a small denominator.
_UNCERTAINTY_SCALE_FRACTION = 0.25
#: Warn about the measurement as a whole when more than this fraction of the
#: stations fail quality control.
_FAILED_FRACTION_WARNING = 0.3
#: Median |tan phi| outside this band earns a frame_step suggestion.  Fig. 7 of
#: the paper is accurate over roughly 0.1-7; this is the comfortable middle.
_TAN_PHI_ADVICE_BAND = (0.2, 4.0)
#: Fewest usable values a median-filter window must hold for the filter to
#: move its station.  The median of two values is their mean, which a single
#: spike drags as far as it likes; from three up a lone outlier cannot win.
_MIN_MEDIAN_VALUES = 3
#: Fewest ground control points that leave a homography any redundancy.  Four
#: determine it exactly, so their residuals are zero whatever their errors.
_MIN_REDUNDANT_GCPS = 5


@dataclass
class StationResult:
    """Velocity and quality metrics at one vertical of the cross-section."""

    index: int
    station_m: float
    x_world: float
    y_world: float
    u_px: float
    v_px: float
    phi_rad: float
    tan_phi: float
    velocity_ms: float
    #: Velocity before quality control replaced or dropped it.
    velocity_raw_ms: float
    coherency: float
    valid_fraction: float
    n_windows: int
    #: Standard deviation of the window angles in the Eq. (14) histogram, rad.
    phi_std_rad: float
    #: The same spread carried through Eq. (16), m/s: how much the individual
    #: windows *scatter* about the station's velocity.  Despite the name it is
    #: not the standard error of that velocity, which is smaller -- by a factor
    #: that depends on how many of the overlapping windows are independent,
    #: which nothing here estimates.  NaN where the reported velocity is not
    #: this station's own measurement (interpolated, median-filtered or
    #: failed), because then the spread describes a number that is no longer
    #: reported.  The name is kept for the constructor, ``velocity.csv`` and the
    #: iOS export; :attr:`velocity_spread_ms` is the same value under an honest
    #: name.
    velocity_std_ms: float
    sx_m_per_px: float
    st_s_per_px: float
    flags: list[str] = field(default_factory=list)
    #: The method that measured it: ``stiv``, ``lspiv`` or ``ptv``.  The STI
    #: fields above (``phi_rad``, ``tan_phi``, ``coherency``,
    #: ``valid_fraction``, ``n_windows``, ``sx_m_per_px``) are NaN or 0 for
    #: the other two.
    method: str = "stiv"
    #: LS-PIV and PTV: the velocity component across the flow direction, m/s,
    #: positive to the left looking downstream.  NaN for STIV, which measures
    #: only along its search line.
    velocity_cross_ms: float = math.nan
    #: LS-PIV: ratio of the correlation peak to the next highest.
    peak_ratio: float = math.nan
    #: LS-PIV: pairs correlated; PTV: tracks counted.  0 for STIV, whose
    #: count is ``n_windows``.
    n_measurements: int = 0

    @property
    def velocity_spread_ms(self) -> float:
        """:attr:`velocity_std_ms` by what it is: the window spread, m/s."""
        return self.velocity_std_ms

    @velocity_spread_ms.setter
    def velocity_spread_ms(self, value: float) -> None:
        self.velocity_std_ms = value

    def add_flag(self, flag: str) -> None:
        """Flag the station, once: a flag already present is not added again."""
        if flag not in self.flags:
            self.flags.append(flag)

    @property
    def failed(self) -> bool:
        """True when the station produced nothing usable."""
        return bool(FAILURE_FLAGS & set(self.flags)) or not np.isfinite(self.velocity_ms)

    @property
    def ok(self) -> bool:
        """True when the station produced a measurement with no caveats."""
        return not self.flags and np.isfinite(self.velocity_ms)


@dataclass
class StivResult:
    """Everything a run produced."""

    config: Config
    geometry: SectionGeometry
    stations: list[StationResult]
    stis: list[SpaceTimeImage]
    fields: list[Optional[OrientationField]]
    estimates: list[Optional[AngleEstimate]]
    stabilisation: StabilisationReport
    n_frames: int
    duration_s: float
    fps: float
    st_s_per_px: float
    reference_frame_bgr: np.ndarray
    discharge: Optional[DischargeResult] = None
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    #: What became of ``stabilise.water_mask_px: auto``; empty when the config
    #: gave polygons, or none.
    water_mask_note: str = ""
    #: The method that measured ``stations``: ``stiv``, ``lspiv`` or ``ptv``.
    #: ``stis``, ``fields`` and ``estimates`` are STIV's whenever STIV ran,
    #: whether reported or compared, and empty when it did not.
    method: str = "stiv"
    #: The profile of each method in ``velocimetry.compare``, quality-
    #: controlled the same way, and its discharge (None if not computed).
    comparisons: dict[str, list[StationResult]] = field(default_factory=dict)
    comparison_discharge: dict[str, Optional[DischargeResult]] = field(default_factory=dict)
    #: LS-PIV's ensemble correlation at each station, when it ran.
    correlations: list[CorrelationResult] = field(default_factory=list)
    #: PTV's tracks through this section's station boxes, and the boxes.
    tracks: Optional[TrackSet] = None
    boxes: list[StationBox] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def methods(self) -> list[str]:
        """Every method run: the reported one first."""
        return [self.method, *self.comparisons]

    def stations_for(self, method: str) -> list[StationResult]:
        """The profile measured by ``method``, reported or compared."""
        if method == self.method:
            return self.stations
        if method in self.comparisons:
            return self.comparisons[method]
        raise KeyError(f"{method!r} was not run; this result has {self.methods}")

    def discharge_for(self, method: str) -> Optional[DischargeResult]:
        if method == self.method:
            return self.discharge
        return self.comparison_discharge.get(method)

    @property
    def velocities(self) -> np.ndarray:
        """The reported profile, m/s.

        NaN where a station failed quality control and was not interpolated;
        its raw value is still in ``StationResult.velocity_raw_ms``.  Nothing
        downstream -- discharge, summary, report -- ever sees a failed value.
        """
        return np.array([s.velocity_ms for s in self.stations])

    @property
    def stations_m(self) -> np.ndarray:
        return np.array([s.station_m for s in self.stations])

    def to_frame(self, method: Optional[str] = None) -> pd.DataFrame:
        """One row per station of ``method``'s profile (the reported one by default).

        A run that is STIV alone gives exactly the columns it always has.
        Any other method, or a comparison, adds ``method``,
        ``velocity_cross_ms``, ``peak_ratio`` and ``n_measurements`` after
        them, and -- for the reported profile -- a ``velocity_<method>_ms``
        column for each method compared.
        """
        method = self.method if method is None else method
        stations = self.stations_for(method)
        extended = self.method != "stiv" or bool(self.comparisons)
        rows = []
        for s in stations:
            rows.append(
                {
                    "index": s.index,
                    "station_m": s.station_m,
                    "x_world_m": s.x_world,
                    "y_world_m": s.y_world,
                    "u_px": s.u_px,
                    "v_px": s.v_px,
                    "phi_deg": math.degrees(s.phi_rad) if np.isfinite(s.phi_rad) else np.nan,
                    "tan_phi": s.tan_phi,
                    "velocity_ms": s.velocity_ms,
                    "velocity_raw_ms": s.velocity_raw_ms,
                    # Column name kept for the iOS export and existing scripts;
                    # it is the window spread, see StationResult.
                    "velocity_std_ms": s.velocity_spread_ms,
                    "coherency": s.coherency,
                    "valid_fraction": s.valid_fraction,
                    "n_windows": s.n_windows,
                    "sx_m_per_px": s.sx_m_per_px,
                    "st_s_per_px": s.st_s_per_px,
                    "flags": ";".join(s.flags),
                }
            )
            if extended:
                rows[-1].update({
                    "method": s.method,
                    "velocity_cross_ms": s.velocity_cross_ms,
                    "peak_ratio": s.peak_ratio,
                    "n_measurements": s.n_measurements,
                })
        df = pd.DataFrame(rows)
        if method == self.method:
            for m, other in self.comparisons.items():
                by_index = {s.index: s.velocity_ms for s in other}
                df[f"velocity_{m}_ms"] = [by_index.get(s.index, np.nan) for s in stations]
        return df

    def summary(self, omit_warnings: Sequence[str] = ()) -> str:
        """The run report.  ``omit_warnings`` are left out of it: a survey's
        report lists the warnings its sections share once, not under each."""
        v = self.velocities
        clean = [s for s in self.stations if s.ok]
        failed = [s for s in self.stations if s.failed]
        filled = [s for s in failed if "interpolated" in s.flags]
        finite = v[np.isfinite(v)]
        lines = [
            f"section        {self.config.section.name}: "
            f"{self.geometry.width_m:.2f} m wide, {len(self.stations)} stations",
            f"rectification  {self.geometry.rectifier.describe()}",
            f"frames         {self.n_frames} over {self.duration_s:.2f} s "
            f"(St = {self.st_s_per_px * 1000:.1f} ms/row, {self.fps:.2f} fps source)",
            f"flow direction {self.geometry.flow_bearing_deg:+.1f} deg in world frame",
            self.stabilisation.summary(),
            *([f"water mask     {self.water_mask_note}"] if self.water_mask_note else []),
            f"stations       {len(clean)}/{len(self.stations)} clean, "
            f"{len(failed)} failed ({len(filled)} interpolated)",
        ]
        if self.method != "stiv" or self.comparisons:
            lines.insert(1, f"method         {METHOD_LABELS[self.method]}"
                            + (f", compared with {', '.join(METHOD_LABELS[m] for m in self.comparisons)}"
                               if self.comparisons else ""))
        if finite.size:
            lines += [
                f"surface U      mean {finite.mean():.3f}  median {np.median(finite):.3f}  "
                f"max {finite.max():.3f} m/s",
            ]
        if self.comparisons:
            lines.append(self.comparison_table())
        if self.discharge is not None:
            lines.append("")
            lines.append(self.discharge.summary())
        warnings = [w for w in self.warnings if w not in set(omit_warnings)]
        if warnings:
            lines.append("")
            lines += [f"warning: {w}" for w in warnings]
        return "\n".join(lines)

    def comparison_table(self) -> str:
        """Each method's profile side by side: clean stations, speeds, discharge."""
        has_q = any(self.discharge_for(m) is not None for m in self.methods)
        out = [f"{'':14} {'method':<8} {'clean':>7} {'mean U':>7} {'max U':>7}"
               + (f" {'Q (m3/s)':>9}" if has_q else "")]
        for m in self.methods:
            st = self.stations_for(m)
            v = np.array([s.velocity_ms for s in st], dtype=float)
            fin = v[np.isfinite(v)]
            q = self.discharge_for(m)
            row = (f"{'':14} {METHOD_LABELS[m]:<8} "
                   f"{sum(1 for s in st if s.ok):>3}/{len(st):<3} "
                   f"{(f'{fin.mean():.3f}' if fin.size else '-'):>7} "
                   f"{(f'{fin.max():.3f}' if fin.size else '-'):>7}")
            if has_q:
                row += f" {(f'{q.discharge_m3s:.3f}' if q is not None else '-'):>9}"
            out.append(row)
        return "\n".join(out)


# --------------------------------------------------------------------------- #
@dataclass
class SurveyResult:
    """Every cross-section of a survey, analysed together in one pass.

    ``results`` holds one :class:`StivResult` per section, in the order the
    config lists them, each complete on its own: its ``config`` is the
    single-section config of :meth:`Config.section_configs`, and its warnings
    include those about the clip as a whole.  A config with one ``section``
    gives a survey of one.
    """

    config: Config
    results: list[StivResult]
    #: Warnings about the clip as a whole -- the video, the rectification, the
    #: camera motion -- which every section's own list repeats.
    shared_warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, i: int) -> StivResult:
        return self.results[i]

    @property
    def names(self) -> list[str]:
        return [r.config.section.name for r in self.results]

    def to_frame(self) -> pd.DataFrame:
        """One row per section: its size, its velocities and its discharge."""
        rows = []
        for r in self.results:
            v = r.velocities
            finite = v[np.isfinite(v)]
            d = r.discharge
            rows.append(
                {
                    "section": r.config.section.name,
                    "width_m": r.geometry.width_m,
                    "n_stations": len(r.stations),
                    "n_failed": sum(1 for s in r.stations if s.failed),
                    "n_interpolated": sum(1 for s in r.stations
                                          if s.failed and "interpolated" in s.flags),
                    "mean_surface_velocity_ms": float(finite.mean()) if finite.size else np.nan,
                    "median_surface_velocity_ms": float(np.median(finite)) if finite.size else np.nan,
                    "max_surface_velocity_ms": float(finite.max()) if finite.size else np.nan,
                    "flow_bearing_deg": r.geometry.flow_bearing_deg,
                    "discharge_m3s": d.discharge_m3s if d is not None else np.nan,
                    "area_m2": d.area_m2 if d is not None else np.nan,
                    "mean_velocity_ms": d.mean_velocity_ms if d is not None else np.nan,
                    "n_warnings": len([w for w in r.warnings if w not in self.shared_warnings]),
                }
            )
            if r.method != "stiv" or r.comparisons:
                rows[-1]["method"] = r.method
                for m in r.comparisons:
                    v = np.array([s.velocity_ms for s in r.comparisons[m]], dtype=float)
                    v = v[np.isfinite(v)]
                    q = r.comparison_discharge.get(m)
                    rows[-1][f"mean_surface_velocity_{m}_ms"] = float(v.mean()) if v.size else np.nan
                    rows[-1][f"discharge_{m}_m3s"] = q.discharge_m3s if q is not None else np.nan
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """The run report: a table of the sections, then each one in turn."""
        if len(self.results) == 1:
            return self.results[0].summary()
        first = self.results[0]
        lines = [
            f"survey         {len(self.results)} cross-sections, analysed together",
            f"frames         {first.n_frames} over {first.duration_s:.2f} s "
            f"(St = {first.st_s_per_px * 1000:.1f} ms/row, {first.fps:.2f} fps source)",
            f"rectification  {first.geometry.rectifier.describe()}",
            first.stabilisation.summary(),
            *([f"water mask     {first.water_mask_note}"] if first.water_mask_note else []),
            "",
        ]
        width = max(8, *(len(n) for n in self.names))
        has_q = any(r.discharge is not None for r in self.results)
        head = (f"{'section':<{width}}  {'width':>7}  {'clean':>7}  {'mean U':>7}  {'max U':>7}"
                + ("  {:>9}".format("Q") if has_q else ""))
        units = (f"{'':<{width}}  {'(m)':>7}  {'':>7}  {'(m/s)':>7}  {'(m/s)':>7}"
                 + ("  {:>9}".format("(m3/s)") if has_q else ""))
        lines += [head, units]
        for r in self.results:
            v = r.velocities
            finite = v[np.isfinite(v)]
            clean = f"{sum(1 for s in r.stations if s.ok)}/{len(r.stations)}"
            mean = f"{finite.mean():.3f}" if finite.size else "-"
            vmax = f"{finite.max():.3f}" if finite.size else "-"
            row = (f"{r.config.section.name:<{width}}  {r.geometry.width_m:7.2f}  {clean:>7}  "
                   f"{mean:>7}  {vmax:>7}")
            if has_q:
                row += f"  {r.discharge.discharge_m3s:9.3f}" if r.discharge is not None else f"  {'-':>9}"
            lines.append(row)
        for r in self.results:
            lines += ["", f"--- {r.config.section.name} " + "-" * max(3, 60 - len(r.config.section.name)),
                      r.summary(omit_warnings=self.shared_warnings)]
        if self.shared_warnings:
            lines.append("")
            lines += [f"warning: {w}" for w in self.shared_warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def run(cfg: Config, progress: bool = True) -> StivResult:
    """Execute a full STIV analysis of the cross-section ``cfg`` describes.

    A config listing several ``sections`` is analysed with :func:`run_survey`,
    which returns a result for each.
    """
    if cfg.sections is not None and len(cfg.sections) != 1:
        raise ConfigError(
            f"this config lists {len(cfg.sections)} cross-sections under `sections`; analyse "
            f"them with stiv.analysis.run_survey, which returns a result for each"
        )
    return run_survey(cfg, progress).results[0]


def run_survey(cfg: Config, progress: bool = True) -> SurveyResult:
    """Analyse every cross-section of ``cfg`` in one pass over the video.

    Reading the frames and tracking the camera's motion is most of a run's
    time, and both are the same whichever section is being measured, so they
    are done once: each frame is sampled along every section's search lines
    together, against the one stabilisation.  Each section is then analysed
    exactly as it would be on its own -- its own flow direction, quality
    control and discharge.

    The one difference from running a section alone is the stabiliser, which
    keeps off the water around *every* section's search lines rather than one
    section's, so it tracks a slightly different set of bank features.  The
    velocities move by the stabiliser's own noise -- a couple of mm/s on the
    synthetic test river -- and with stabilisation off they are identical.
    """
    t_start = time.perf_counter()
    configs = cfg.section_configs()
    multi = cfg.sections is not None
    methods = cfg.velocimetry.methods
    primary = cfg.velocimetry.method
    use_stiv = "stiv" in methods

    def _named(c: Config, text: str) -> str:
        return f"cross-section {c.section.name!r}: {text}" if multi else text

    source = FrameSource(cfg.video)
    log.info("video: %s", source.info)
    if source.n_frames is None:
        log.info("using every frame to the end of the video, step %d (St = %.4f s)",
                 source.step, source.st_s_per_px)
    else:
        log.info("using %d frames, step %d (St = %.4f s)",
                 source.n_frames, source.step, source.st_s_per_px)
    log.info("velocity: %s", ", ".join(METHOD_LABELS[m] for m in methods))

    # An STI shorter than the analysis window fails at every station, and
    # used to be reported as a texture or velocity-range problem afterwards.
    need = orientation.min_sti_size(
        cfg.stiv.window_px, cfg.stiv.window_type, cfg.stiv.gradient_order
    )
    if use_stiv and source.n_frames is not None and source.n_frames < need:
        raise ConfigError(
            f"the clip gives only {source.n_frames} STI rows (frames left after "
            f"video.start_s, duration_s and frame_step: {source.step}), but a "
            f"{cfg.stiv.window_px} px {cfg.stiv.window_type} window needs at least "
            f"{need}; use a longer video.duration_s, a smaller video.frame_step, or "
            f"a smaller stiv.window_px"
        )

    geoms = [geometry.build_section(c, source.shape) for c in configs]
    log.info("rectification: %s", geoms[0].rectifier.describe())

    # Each section's warnings in the order a lone section has always listed
    # them: its own layout first, the clip's next, then its own analysis.
    own: list[list[str]] = []
    for c, geom in zip(configs, geoms):
        log.info(
            "%s%.2f m wide, %d stations, %.2f m search lines along %+.1f deg",
            f"section {c.section.name!r}: " if multi else "section ",
            geom.width_m, len(geom.search_lines), c.stiv.search_line_length_m,
            geom.flow_bearing_deg,
        )
        w: list[str] = []
        if use_stiv:
            narrow = [sl.index for sl in geom.search_lines if sl.n_samples < need]
            if narrow and len(narrow) == len(geom.search_lines):
                raise ConfigError(_named(c,
                    f"the search lines have only {min(sl.n_samples for sl in geom.search_lines)} "
                    f"samples, but a {c.stiv.window_px} px {c.stiv.window_type} window "
                    f"needs at least {need}; lengthen stiv.search_line_length_m, raise "
                    f"stiv.n_samples, or use a smaller stiv.window_px"
                ))
            if narrow:
                w.append(
                    f"{len(narrow)} search line(s) have fewer than the {need} samples a "
                    f"{c.stiv.window_px} px window needs (stations {narrow[:5]}"
                    f"{'...' if len(narrow) > 5 else ''}) and will fail; lengthen "
                    f"stiv.search_line_length_m or use a smaller stiv.window_px"
                )
            outside = geometry.check_lines_in_frame(geom, source.shape)
            if outside:
                w.append(
                    f"{len(outside)} search line(s) extend outside the frame "
                    f"(stations {outside[:5]}{'...' if len(outside) > 5 else ''}); "
                    f"shorten stiv.search_line_length_m or move the section"
                )
        own.append(w)

    # LS-PIV's patches and PTV's boxes, laid out once the sections are.
    plans = _plan_vector_methods(cfg, configs, geoms, source, methods, own)

    shared: list[str] = []
    # One rectifier serves every section, so its fit is judged once.
    _rectification_warnings(geoms[0], shared)

    stab_cfg, water_note = cfg.stabilise, ""
    if is_auto_water_mask(cfg.stabilise.water_mask_px):
        # The outline is of the water, not of a section: the first section
        # calibrates it, and it then serves them all.
        stab_cfg, water_note = _automatic_water_mask(configs[0], shared)

    got = _collect(cfg, source, geoms, progress, stab_cfg, methods, plans)
    stab_report, reference_bgr = got.report, got.reference_bgr
    shared.extend(source.warnings)
    shared_after = stab_report.warnings()

    results: list[StivResult] = []
    for si, (c, geom, warnings) in enumerate(zip(configs, geoms, own)):
        warnings.extend(shared)
        notes: dict[str, list[str]] = {m: [] for m in methods}
        runs: dict[str, list[StationResult]] = {}
        stis: list[SpaceTimeImage] = got.stis[si] if got.stis is not None else []
        fields: list[Optional[OrientationField]] = []
        estimates: list[Optional[AngleEstimate]] = []

        stiv_flipped = False
        if use_stiv:
            spectra: list[Optional[SpectralProfile]] = []
            stations, fields, estimates = _analyse_stis(c, stis, geom, spectra)
            # Direction first: it can replace every station's estimate, and
            # quality control must judge the estimates that are actually going
            # to be reported.
            stiv_flipped = _direction_check(c, stations, stis, fields, estimates,
                                            notes["stiv"], spectra)
            runs["stiv"] = stations

        vector_ests: dict[str, list[VectorEstimate]] = {}
        if "lspiv" in methods:
            vector_ests["lspiv"] = [lspiv.estimate(r) for r in got.correlations[si]]
        if "ptv" in methods:
            boxes = plans.boxes[si]
            keep = ptv.accepted(c, got.tracks, boxes[0].along_world) if boxes else None
            vector_ests["ptv"] = [ptv.estimate(c, got.tracks, b, keep) for b in boxes]

        # The reported method decides which way is downstream.
        if primary == "stiv":
            flipped = stiv_flipped
        else:
            flipped = _vector_direction(c, vector_ests[primary], geom.flow_dir_world,
                                        notes[primary])
        if flipped:
            # The stations now report the flow as running the other way along the
            # search lines, so the direction reported and drawn follows them.  The
            # lines themselves are already laid out and sampled, and nothing reads
            # this back to lay them out again.
            geom.flow_dir_world = -geom.flow_dir_world
            bearing = geom.flow_bearing_deg + 180.0
            geom.flow_bearing_deg = bearing - 360.0 if bearing > 180.0 else bearing

        for m, ests in vector_ests.items():
            runs[m] = _vector_stations(c, geom, ests, m, source.st_s_per_px)
            if m != primary:
                _direction_agreement(m, runs[m], primary, notes[m])
        if use_stiv and primary != "stiv" and stiv_flipped != flipped:
            # STIV settled the direction its own way; report its velocities
            # along the direction the reported method chose, and say so.
            for st in runs["stiv"]:
                for name in ("velocity_ms", "velocity_raw_ms", "phi_rad", "tan_phi"):
                    setattr(st, name, -getattr(st, name))
            notes["stiv"].append(
                f"STIV and {METHOD_LABELS[primary]} disagree about which way the water "
                f"flows, so STIV's velocities, reported along {METHOD_LABELS[primary]}'s "
                f"direction, are negative; check the flow direction"
            )

        for m in methods:
            _quality_control(c, runs[m], notes[m])
            if m == "stiv":
                _range_advice(c, runs[m], source, notes[m])
            elif m == "lspiv":
                tip = lspiv.advice(c, vector_ests[m], plans.pair_steps[si])
                if tip:
                    notes[m].append(tip)

        warnings.extend(notes[primary])
        for m in methods[1:]:
            warnings.extend(f"{METHOD_LABELS[m]}: {w}" for w in notes[m])
        warnings.extend(shared_after)

        disch = _discharge(c, runs[primary], geom, warnings)
        comparisons = {m: runs[m] for m in methods[1:]}
        comparison_q: dict[str, Optional[DischargeResult]] = {}
        for m in methods[1:]:
            extra: list[str] = []
            comparison_q[m] = _discharge(c, runs[m], geom, extra)
            warnings.extend(f"{METHOD_LABELS[m]}: {w}" for w in extra)

        n_frames = stis[0].data.shape[0] if stis else got.n_rows
        results.append(StivResult(
            config=c,
            geometry=geom,
            stations=runs[primary],
            stis=stis,
            fields=fields,
            estimates=estimates,
            stabilisation=stab_report,
            n_frames=n_frames,
            duration_s=n_frames * source.st_s_per_px,
            fps=source.fps,
            st_s_per_px=source.st_s_per_px,
            reference_frame_bgr=reference_bgr,
            discharge=disch,
            warnings=warnings,
            water_mask_note=water_note,
            method=primary,
            comparisons=comparisons,
            comparison_discharge=comparison_q,
            correlations=got.correlations[si] if got.correlations is not None else [],
            tracks=_section_tracks(got.tracks, plans.boxes[si]) if plans.boxes else None,
            boxes=plans.boxes[si] if plans.boxes else [],
        ))

    elapsed = time.perf_counter() - t_start
    for r in results:
        r.elapsed_s = elapsed
    return SurveyResult(config=cfg, results=results,
                        shared_warnings=shared + shared_after, elapsed_s=elapsed)


def _automatic_water_mask(cfg: Config, warnings: list[str]) -> tuple[StabiliseConfig, str]:
    """Resolve ``stabilise.water_mask_px: auto`` into polygons, or into none.

    Returns the stabiliser settings to track with, and a line for the run
    summary.  ``cfg`` itself keeps ``auto``, so the ``config_used.yaml`` a run
    writes detects again on a re-run rather than freezing this outline.

    Unlike the notebook wizard and ``stiv pick``, nobody is here to look at the
    outline, so only a ``good`` one is used.  Without one the analysis goes on
    as it would with no water mask at all -- the corridor around the search
    lines is still kept off the water -- and the warnings say so.
    """
    without = replace(cfg.stabilise, water_mask_px=None)
    if not cfg.stabilise.enabled:
        return without, "auto: not needed, camera-motion compensation is off"
    if cfg.stabilise.static_mask_px:
        return without, "auto: not used, stabilise.static_mask_px already names the ground to track"

    from .watermask import detect_water

    try:
        det = detect_water(cfg.video.path, cfg.video.start_s,
                           [cfg.section.start_px, cfg.section.end_px],
                           cfg.flow.downstream_px, fps=cfg.video.fps_override)
    except (ConfigError, ValueError) as exc:
        reason, det = f"the detector stopped: {exc}", None
    else:
        reason = det.message

    if det is not None and det.flow_agrees is False:
        warnings.append(
            "the automatic water outline found the water moving towards the other side "
            "of the cross-section from flow.downstream_px; check that point is "
            "downstream"
        )
    if det is not None and det.confidence == "good":
        log.info("water mask: %s", det.message)
        return replace(cfg.stabilise, water_mask_px=[det.polygon_px]), f"auto: {det.message}"

    what = "was not reliable enough to use" if det is not None and det.ok else "failed"
    warnings.append(
        f"stabilise.water_mask_px is auto, but the automatic water outline {what} "
        f"({reason}), so no water mask was used: only the corridor around the search "
        f"lines was kept off the water. If the run also warns about features on the "
        f"water, outline it by hand instead (stiv pick, or polygons in the config)"
    )
    return without, f"auto: not used ({reason})"


def _rectification_warnings(geom: SectionGeometry, warnings: list[str]) -> None:
    """Flag a ground-control fit that will not support the velocities it scales.

    Velocity is linear in ``Sx``, so an error in the image-to-world map passes
    straight into every station with no attenuation and nothing downstream can
    detect it.  The residuals are the only evidence available.
    """
    rect = geom.rectifier
    if not isinstance(rect, geometry.HomographyRectifier):
        return

    if rect.n_gcp < _MIN_REDUNDANT_GCPS:
        warnings.append(
            f"the homography was fitted to only {rect.n_gcp} ground control points, "
            f"the minimum it needs, so it passes through every one of them exactly: "
            f"its RMSE of {rect.rmse_m:.3f} m says nothing about their accuracy, and "
            f"a blunder in any one goes undetected while scaling every velocity. "
            f"Survey at least {_MIN_REDUNDANT_GCPS} (6 or more is better) so the fit "
            f"has redundancy and a bad point can be rejected."
        )

    tolerance_m = rect.inlier_threshold_m
    if rect.rmse_m > tolerance_m:
        warnings.append(
            f"the {rect.n_inliers} inlier ground control points fit to {rect.rmse_m:.3f} m "
            f"RMSE, worse than the {tolerance_m:.3f} m expected of {geometry._RANSAC_PX:.0f}-pixel "
            f"clicking accuracy; every velocity scales linearly with this error. "
            f"Re-check the survey coordinates and the clicked pixels."
        )
    if rect.n_inliers < rect.n_gcp:
        warnings.append(
            f"{rect.n_gcp - rect.n_inliers} of {rect.n_gcp} ground control "
            f"point(s) were rejected as outliers by the homography fit; check "
            f"them for a transcription error or a mismatched point order"
        )


def _section_tracks(tracks: Optional[TrackSet], boxes: Sequence[StationBox]) -> Optional[TrackSet]:
    """The tracks whose midpoints lie in one section's station boxes."""
    if tracks is None or not len(tracks) or not boxes:
        return tracks
    mid = tracks.mid_world
    return tracks.subset(np.any([b.contains_world(mid) for b in boxes], axis=0))


@dataclass
class _VectorPlans:
    """LS-PIV's patches and pair steps, and PTV's boxes, for each section."""

    layouts: list[list[lspiv.PatchLayout]] = field(default_factory=list)
    pair_steps: list[int] = field(default_factory=list)
    boxes: list[list[StationBox]] = field(default_factory=list)


def _plan_vector_methods(
    cfg: Config, configs: Sequence[Config], geoms: Sequence[SectionGeometry],
    source: FrameSource, methods: Sequence[str], own: list[list[str]],
) -> _VectorPlans:
    """Lay out LS-PIV's patches and PTV's boxes, and check the clip is long enough."""
    plans = _VectorPlans()
    h, w = source.shape
    multi = cfg.sections is not None

    def _off_frame(corners_px: list[np.ndarray]) -> list[int]:
        return [i for i, q in enumerate(corners_px)
                if (q[:, 0] < 0).any() or (q[:, 0] > w - 1).any()
                or (q[:, 1] < 0).any() or (q[:, 1] > h - 1).any()]

    for c, geom, warnings in zip(configs, geoms, own):
        rect = geom.rectifier
        if "lspiv" in methods:
            layouts, k = lspiv.plan_patches(c, geom, source.st_s_per_px)
            plans.layouts.append(layouts)
            plans.pair_steps.append(k)
            if layouts:
                lay = layouts[0]
                log.info("%sLS-PIV: %d-sample interrogation areas (%.2f m) at %.3f m, "
                         "pair step %d, search +/- %d samples",
                         f"section {c.section.name!r}: " if multi else "",
                         lay.ia_n, lay.ia_size_m, lay.gsd_m, k, lay.margin_n)
                off = _off_frame([rect.world_to_image(la.corners_world()) for la in layouts])
                if off:
                    warnings.append(
                        f"{len(off)} LS-PIV patch(es) extend outside the frame (stations "
                        f"{off[:5]}{'...' if len(off) > 5 else ''}); their missing samples "
                        f"weaken the correlation"
                    )
        if "ptv" in methods:
            boxes = geometry.station_boxes(geom, c.stiv.search_line_length_m)
            plans.boxes.append(boxes)
            off = _off_frame([rect.world_to_image(b.corners_world()) for b in boxes])
            if off:
                warnings.append(
                    f"{len(off)} PTV station box(es) extend outside the frame (stations "
                    f"{off[:5]}{'...' if len(off) > 5 else ''}); shorten "
                    f"stiv.search_line_length_m or move the section"
                )

    n = source.n_frames
    if n is not None and "lspiv" in methods and plans.pair_steps:
        need_lspiv = max(plans.pair_steps) + 2
        if n < need_lspiv:
            raise ConfigError(
                f"the clip gives only {n} frames (after video.start_s, duration_s and "
                f"frame_step: {source.step}), but LS-PIV with a pair step of "
                f"{max(plans.pair_steps)} needs at least {need_lspiv}; use a longer "
                f"video.duration_s or a smaller lspiv.pair_step"
            )
    if n is not None and "ptv" in methods and n < cfg.ptv.track_frames + 1:
        raise ConfigError(
            f"the clip gives only {n} frames, but PTV tracks each feature for "
            f"ptv.track_frames: {cfg.ptv.track_frames}; use a longer video.duration_s "
            f"or fewer ptv.track_frames"
        )
    return plans


@dataclass
class _Collected:
    """What the single pass over the video gathered, per section."""

    stis: Optional[list[list[SpaceTimeImage]]]
    correlations: Optional[list[list[CorrelationResult]]]
    tracks: Optional[TrackSet]
    report: StabilisationReport
    reference_bgr: np.ndarray
    n_rows: int


def _collect(
    cfg: Config, source: FrameSource, geoms: Sequence[SectionGeometry], progress: bool,
    stabilise_cfg: Optional[StabiliseConfig] = None,
    methods: Sequence[str] = ("stiv",),
    plans: Optional[_VectorPlans] = None,
) -> _Collected:
    """Single streaming pass over the video, feeding every method at once.

    Each frame is read and registered against the reference once, then handed
    to a collector per method: the STIs of every section's search lines, the
    LS-PIV patches of every station, the PTV tracks through every station box.
    Only the collectors the methods need are built.  ``stabilise_cfg`` is
    ``cfg.stabilise`` with an automatic water mask resolved; ``cfg.stabilise``
    itself when omitted.
    """
    plans = plans or _VectorPlans()
    stab_cfg = stabilise_cfg if stabilise_cfg is not None else cfg.stabilise
    collectors: list = []
    exclude: list[np.ndarray] = []

    sti_collector = None
    if "stiv" in methods:
        lines = [sl for g in geoms for sl in g.search_lines]
        sti_collector = sti_mod.StiCollector(lines, source.n_frames, source.st_s_per_px)
        collectors.append(sti_collector)
        exclude.append(sti_collector.all_pts)
    patch_collector = None
    if "lspiv" in methods:
        layouts = [lay for group in plans.layouts for lay in group]
        steps = [k for group, k in zip(plans.layouts, plans.pair_steps) for _ in group]
        patch_collector = lspiv.PatchCollector(layouts, steps, source.st_s_per_px, cfg.lspiv)
        collectors.append(patch_collector)
        exclude.append(patch_collector.all_pts)
    track_collector = None
    if "ptv" in methods:
        boxes = [b for group in plans.boxes for b in group]
        track_collector = ptv.TrackCollector(boxes, geoms[0].rectifier, source.st_s_per_px,
                                             cfg.ptv, source.shape)
        collectors.append(track_collector)
        exclude.append(track_collector.all_pts)
    # The water the methods watch is kept out of the stabiliser's features.
    exclude_px = exclude[0] if len(exclude) == 1 else np.vstack(exclude)

    probe = np.vstack([np.vstack([g.start_px, g.end_px]) for g in geoms])
    stabiliser: Optional[Stabiliser] = None
    reference_bgr = np.zeros((*source.shape, 3), np.uint8)

    bar = None
    if progress:
        try:
            # tqdm.auto: a progress widget in a notebook, a text bar elsewhere.
            from tqdm.auto import tqdm

            bar = tqdm(total=source.n_frames, unit="frame",
                       desc="STI" if sti_collector is not None else "frames", leave=False)
        except ImportError:
            bar = None

    n_rows = 0
    try:
        for i, frame_bgr in enumerate(source.frames(colour=True)):
            grey = source.to_grey(frame_bgr)
            if i == 0:
                reference_bgr = frame_bgr.copy()
                mask = build_mask(stab_cfg, source.shape, exclude_px=exclude_px)
                stabiliser = Stabiliser(stab_cfg, grey, mask)
                if stabiliser.enabled:
                    log.info("stabilisation: tracking %d static features", stabiliser.report.n_features)
                M = None
            else:
                M = stabiliser.update(grey, probe_px=probe) if stabiliser.enabled else None
            for col in collectors:
                col.add_frame(grey, M)
            n_rows += 1
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()

    report = stabiliser.report if stabiliser else StabilisationReport(enabled=False)

    def _split(items: list, groups: Sequence[Sequence]) -> list[list]:
        out, k = [], 0
        for g in groups:
            out.append(items[k:k + len(g)])
            k += len(g)
        return out

    return _Collected(
        stis=(_split(sti_collector.finish(), [g.search_lines for g in geoms])
              if sti_collector is not None else None),
        correlations=(_split(patch_collector.finish(), plans.layouts)
                      if patch_collector is not None else None),
        tracks=track_collector.finish() if track_collector is not None else None,
        report=report,
        reference_bgr=reference_bgr,
        n_rows=sti_collector.n_rows if sti_collector is not None else n_rows,
    )


def _collect_stis(
    cfg: Config, source: FrameSource, geoms: Sequence[SectionGeometry], progress: bool,
    stabilise_cfg: Optional[StabiliseConfig] = None,
) -> tuple[list[list[SpaceTimeImage]], StabilisationReport, np.ndarray]:
    """The STIs alone, as :func:`_collect` gathers them for STIV."""
    got = _collect(cfg, source, geoms, progress, stabilise_cfg)
    return got.stis, got.report, got.reference_bgr


def _eq14(cfg: Config, field_: OrientationField, phi_min: float, phi_max: float) -> AngleEstimate:
    """Run Eq. (14) over one orientation field for a given angle range."""
    return orientation.mean_orientation_angle(
        field_,
        phi_min,
        phi_max,
        coherency_levels=cfg.stiv.coherency_levels,
        n_bins=cfg.stiv.n_angle_bins,
        min_coherency=cfg.stiv.min_coherency,
        estimator=cfg.stiv.estimator,
    )


def _angle(
    cfg: Config, field_: Optional[OrientationField], spectrum: Optional[SpectralProfile],
    phi_min: float, phi_max: float,
) -> AngleEstimate:
    """One STI's angle over a range: from its spectrum for ``stiv.estimator:
    fft``, from its orientation field by Eq. (14) otherwise."""
    if spectrum is not None:
        return orientation.spectral_estimate(spectrum, phi_min, phi_max)
    return _eq14(cfg, field_, phi_min, phi_max)


def _estimate_angle(
    cfg: Config, s: SpaceTimeImage, field_: Optional[OrientationField], v_lo: float,
    v_hi: float, spectrum: Optional[SpectralProfile] = None,
) -> AngleEstimate:
    """Estimate one STI's angle for a given velocity range."""
    phi_min, phi_max = orientation.angle_range_from_velocity(
        v_lo, v_hi, s.sx_m_per_px, s.st_s_per_px
    )
    return _angle(cfg, field_, spectrum, phi_min, phi_max)


def _apply_estimate(
    st: StationResult,
    s: SpaceTimeImage,
    est: Optional[AngleEstimate],
    negate: bool = False,
) -> None:
    """Write an angle estimate onto a station, converting it to a velocity.

    ``negate`` is used when the search range was mirrored to follow flow that
    runs against the assumed downstream direction: reversing the search line
    negates phi, and Eq. (16) is odd in phi, so the velocity negates with it.
    """
    if est is None or not np.isfinite(est.phi):
        st.phi_rad = math.nan
        st.tan_phi = math.nan
        st.velocity_ms = math.nan
        st.velocity_raw_ms = math.nan
        st.velocity_spread_ms = math.nan
        st.phi_std_rad = math.nan
        st.coherency = 0.0
        st.valid_fraction = 0.0 if est is None else est.valid_fraction
        st.n_windows = 0
        if "sti_too_small" not in st.flags:
            st.add_flag("no_orientation")
        return

    sign = -1.0 if negate else 1.0
    phi = sign * est.phi
    st.phi_rad = phi
    st.tan_phi = float(math.tan(phi))
    st.velocity_ms = orientation.velocity_from_angle(phi, s.sx_m_per_px, s.st_s_per_px)
    st.velocity_raw_ms = st.velocity_ms
    # Propagate the angle spread through Eq. (16): dU/dphi = k * sec^2(phi).
    k = s.sx_m_per_px / s.st_s_per_px
    st.velocity_spread_ms = abs(k / max(math.cos(phi) ** 2, 1e-6)) * est.phi_std
    st.phi_std_rad = est.phi_std
    st.coherency = est.mean_coherency
    st.valid_fraction = est.valid_fraction
    st.n_windows = est.n_used


def _analyse_stis(
    cfg: Config, stis: Sequence[SpaceTimeImage], geom: SectionGeometry,
    spectra: Optional[list[Optional[SpectralProfile]]] = None,
) -> tuple[list[StationResult], list[Optional[OrientationField]], list[Optional[AngleEstimate]]]:
    """Run the orientation analysis on every STI and convert to velocity.

    With ``stiv.estimator: fft`` each STI's spectrum stands in for its
    orientation field, whose entry is then None; the spectra are appended to
    ``spectra`` when it is given (None elsewhere), for the direction check.
    """
    stations: list[StationResult] = []
    fields: list[Optional[OrientationField]] = []
    estimates: list[Optional[AngleEstimate]] = []
    use_fft = cfg.stiv.estimator == "fft"

    v_lo, v_hi = cfg.stiv.velocity_range_ms

    for s in stis:
        sl = s.line
        st = StationResult(
            index=sl.index,
            station_m=sl.station_m,
            x_world=float(sl.centre_world[0]),
            y_world=float(sl.centre_world[1]),
            u_px=float(sl.centre_px[0]),
            v_px=float(sl.centre_px[1]),
            phi_rad=math.nan,
            tan_phi=math.nan,
            velocity_ms=math.nan,
            velocity_raw_ms=math.nan,
            coherency=0.0,
            valid_fraction=0.0,
            n_windows=0,
            phi_std_rad=math.nan,
            velocity_std_ms=math.nan,
            sx_m_per_px=s.sx_m_per_px,
            st_s_per_px=s.st_s_per_px,
        )

        # Conditioning and orientation run in float64 throughout: the Swift
        # port is held to these numbers to 1e-9.
        conditioned = sti_mod.preprocess(s.data, cfg.stiv)

        field_ = None
        spectrum = None
        est = None
        try:
            if use_fft:
                spectrum = orientation.spectral_profile(conditioned)
            else:
                field_ = orientation.structure_tensor_field(
                    conditioned,
                    window_px=cfg.stiv.window_px,
                    window_type=cfg.stiv.window_type,
                    overlap=cfg.stiv.window_overlap,
                    gradient_order=cfg.stiv.gradient_order,
                )
            est = _estimate_angle(cfg, s, field_, v_lo, v_hi, spectrum)
        except ValueError as exc:
            st.add_flag("sti_too_small")
            log.warning("station %d: %s", sl.index, exc)

        # From here on the conditioned STI is only drawn and exported, for
        # which float32 is ample.  It is written back over the raw samples --
        # a view into the collector's single buffer -- so the result holds one
        # float32 copy of the STIs rather than a float64 one beside the raw.
        if (
            isinstance(s.data, np.ndarray)
            and s.data.dtype == np.float32
            and s.data.flags.writeable
            and s.data.shape == conditioned.shape
        ):
            s.data[...] = conditioned
        else:
            s.data = conditioned.astype(np.float32)
        del conditioned

        _apply_estimate(st, s, est)

        if s.out_of_frame_fraction > _MAX_OUT_OF_FRAME_FRACTION:
            st.add_flag("out_of_frame")

        stations.append(st)
        fields.append(field_)
        estimates.append(est)
        if spectra is not None:
            spectra.append(spectrum)

    return stations, fields, estimates


# --------------------------------------------------------------------------- #
# quality control
# --------------------------------------------------------------------------- #
def _report_velocity(st: StationResult, velocity_ms: float) -> None:
    """Replace a station's reported velocity with one it did not measure.

    Everything that describes the reported number is kept consistent with it:
    ``tan_phi`` and ``phi_rad`` are recomputed through Eq. (16), and the
    spreads are cleared, since they describe the measurement that was replaced
    rather than the value now reported -- an interpolated value's uncertainty
    is the interpolation error, which nothing here can estimate.  The raw
    measurement stays in ``velocity_raw_ms``; the diagnostics that explain why
    it was replaced (coherency, valid fraction) are left alone.
    """
    st.velocity_ms = float(velocity_ms)
    if st.method != "stiv":
        # LS-PIV and PTV have no STI angle to keep in step.
        pass
    elif np.isfinite(velocity_ms) and st.sx_m_per_px > 0:
        st.tan_phi = float(velocity_ms) * st.st_s_per_px / st.sx_m_per_px
        st.phi_rad = math.atan(st.tan_phi)
    else:
        st.tan_phi = math.nan
        st.phi_rad = math.nan
    st.velocity_spread_ms = math.nan
    st.phi_std_rad = math.nan


def _quality_control(cfg: Config, stations: list[StationResult], warnings: list[str]) -> None:
    """Flag doubtful stations, then fill or blank the failed ones.

    The order is deliberate, and the Swift port mirrors it:

    1. flag every station on its measured values;
    2. median-filter the *usable* stations only (if asked), so no failed
       value can leak into a good neighbour;
    3. interpolate the failed stations from the *filtered* usable values, so
       a spike the filter removed cannot come back through a filled gap;
    4. blank to NaN whatever failed and is still unfilled (its raw value stays
       in ``velocity_raw_ms``), so no failed number can reach the discharge
       or the summary.
    """
    q = cfg.quality
    lo, hi = q.tan_phi_range

    # The uncertainty test is relative to the section's own velocity scale, so
    # that a genuinely slow but well-resolved station near the bank is not
    # condemned merely for having a small denominator.
    v_all = np.array([abs(s.velocity_ms) for s in stations], dtype=float)
    finite = v_all[np.isfinite(v_all)]
    v_scale = _UNCERTAINTY_SCALE_FRACTION * float(np.median(finite)) if finite.size else 0.0

    for s in stations:
        # The STI tests below skip LS-PIV and PTV stations by themselves:
        # their valid fraction, coherency and tan(phi) are NaN.
        if s.method == "lspiv" and np.isfinite(s.peak_ratio) and s.peak_ratio < q.min_peak_ratio:
            s.add_flag("low_peak_ratio")
        if s.method == "ptv" and s.n_measurements < q.min_tracks:
            s.add_flag("too_few_tracks")
        if np.isfinite(s.valid_fraction) and s.valid_fraction < q.min_valid_fraction:
            s.add_flag("low_valid_fraction")
        if np.isfinite(s.coherency) and s.coherency < q.min_mean_coherency:
            s.add_flag("low_coherency")
        if np.isfinite(s.velocity_spread_ms) and np.isfinite(s.velocity_ms):
            reference = max(abs(s.velocity_ms), v_scale)
            if reference > 0 and s.velocity_spread_ms > q.max_relative_uncertainty * reference:
                s.add_flag("high_uncertainty")
        if np.isfinite(s.tan_phi):
            a = abs(s.tan_phi)
            if a > hi:
                # Steep streaks: too little spatial displacement per frame is
                # resolvable.  Section 2.3 tested only up to tan phi = 7.
                s.add_flag("tan_phi_high")
            elif a < lo:
                s.add_flag("tan_phi_low")

    if q.median_filter_stations and q.median_filter_stations > 1:
        _median_filter(stations, int(q.median_filter_stations))

    if q.interpolate_flagged:
        _interpolate_flagged(stations)

    unfilled = [s for s in stations if s.failed and "interpolated" not in s.flags]
    for s in unfilled:
        _report_velocity(s, math.nan)
    if unfilled and q.interpolate_flagged:
        n_usable = sum(1 for s in stations if not s.failed)
        warnings.append(
            f"{len(unfilled)} failed station(s) could not be interpolated -- only "
            f"{n_usable} station(s) measured a usable velocity, and interpolation "
            f"needs two -- so they report no velocity"
        )

    # A clip or a search line too small for the window fails every station it
    # touches, and is not a texture or velocity-range problem: say so plainly.
    need = orientation.min_sti_size(
        cfg.stiv.window_px, cfg.stiv.window_type, cfg.stiv.gradient_order
    )
    too_small = [s for s in stations if "sti_too_small" in s.flags]
    if too_small:
        warnings.append(
            f"{len(too_small)} of {len(stations)} stations produced a space-time "
            f"image too small for the {cfg.stiv.window_px} px {cfg.stiv.window_type} "
            f"window, which needs at least {need} rows (frames) and {need} columns "
            f"(samples); use more frames (video.duration_s, video.frame_step), "
            f"longer search lines, or a smaller stiv.window_px"
        )

    n_bad = sum(1 for s in stations if s.failed and "sti_too_small" not in s.flags)
    if n_bad > _FAILED_FRACTION_WARNING * max(1, len(stations)):
        method = stations[0].method if stations else "stiv"
        if method == "ptv":
            check = ("that the water carries trackable features (foam, debris, boils), "
                     "that stiv.velocity_range_ms brackets the true velocity, and that "
                     "the flow direction is right")
        elif method == "lspiv":
            check = ("that the water surface has visible texture, that "
                     "stiv.velocity_range_ms brackets the true velocity, and that "
                     "glare is not hiding the surface")
        else:
            check = ("that the water surface has visible texture, that "
                     "stiv.velocity_range_ms brackets the true velocity, and that the "
                     "search lines are parallel to the flow")
        warnings.append(
            f"{n_bad} of {len(stations)} stations failed quality control; check {check}"
        )


def _median_filter(stations: list[StationResult], k: int) -> None:
    """Median-filter the velocity profile across the *usable* stations.

    Only stations that did not fail take part, either as inputs or as outputs:
    a failed station's raw value is exactly what the filter must not spread
    into its neighbours.  It runs before interpolation, so the gaps are then
    filled from the filtered values.  The window is ``k`` stations wide by
    index and centred; failed stations and the ends of the section leave it
    holding fewer values, and a station whose window holds fewer than
    ``_MIN_MEDIAN_VALUES`` is left as measured rather than replaced by what
    would really be the mean of two.

    Any station the filter actually moves is flagged.  The reported velocity is
    then no longer the one that station measured, and a silently substituted
    number that still reports ``ok`` is worse than no filter at all.
    """
    if k % 2 == 0:
        log.warning(
            "quality.median_filter_stations = %d is even, but a centred median "
            "needs an odd window; using %d stations", k, k + 1,
        )
    v = np.array([s.velocity_ms for s in stations], dtype=float)
    usable = np.array([not s.failed for s in stations], dtype=bool) & np.isfinite(v)
    half = k // 2
    for i, s in enumerate(stations):
        if not usable[i]:
            continue
        lo, hi = max(0, i - half), min(len(v), i + half + 1)
        window = v[lo:hi][usable[lo:hi]]
        if window.size < _MIN_MEDIAN_VALUES:
            continue
        after = float(np.median(window))
        if after != v[i]:
            _report_velocity(s, after)
            s.add_flag("median_filtered")


def _interpolate_flagged(stations: list[StationResult]) -> int:
    """Replace *failed* stations with an interpolation of their neighbours.

    Advisory flags are deliberately left alone: a station reading a genuinely
    slow patch near the bank is measuring correctly, and pulling it towards its
    faster neighbours would introduce error rather than remove it.

    The usable values are taken as they stand -- after the median filter, when
    there is one.  Failed stations beyond the outermost usable one take that
    station's value.  Nothing happens with fewer than two usable stations.
    Returns the number of stations filled.
    """
    x = np.array([s.station_m for s in stations], dtype=float)
    v = np.array([s.velocity_ms for s in stations], dtype=float)
    failed = np.array([s.failed for s in stations], dtype=bool)
    usable = ~failed & np.isfinite(v)

    if usable.sum() < 2 or not failed.any():
        return 0
    filled = np.interp(x, x[usable], v[usable])
    for s, val, bad in zip(stations, filled, failed):
        if bad:
            _report_velocity(s, float(val))
            s.add_flag("interpolated")
    return int(failed.sum())


def _reverse_advice(flow: FlowConfig) -> str:
    """How to point the search lines downstream, given how they were pointed."""
    if flow.reverse:
        advice = "set flow.reverse: false (it is currently true)"
    else:
        advice = "set flow.reverse: true"
    if flow.direction == "auto":
        if flow.downstream_px is None:
            advice += ", or give flow.downstream_px"
        else:
            advice += ", or check that flow.downstream_px really is downstream of the section"
    elif flow.direction == "points":
        advice += ", or swap flow.from_px and flow.to_px"
    else:
        advice += ", or add 180 to flow.angle_deg"
    return advice


def _direction_check(
    cfg: Config,
    stations: list[StationResult],
    stis: Sequence[SpaceTimeImage],
    fields: list[Optional[OrientationField]],
    estimates: list[Optional[AngleEstimate]],
    warnings: list[str],
    spectra: Optional[Sequence[Optional[SpectralProfile]]] = None,
) -> bool:
    """Detect and, if allowed, correct search lines that point upstream.

    Returns True when the stations were re-analysed with the range mirrored,
    i.e. when the flow is now reported as running against the search lines.

    Simply negating the velocities would not be enough.  ``velocity_range_ms``
    is nearly always asymmetric -- something like [-0.5, 5] -- so if the lines
    face upstream the true angle lies *outside* the search range of Eq. (14).
    Every station then reports a value pinned near the range boundary, and
    flipping the sign would turn that garbage into plausible-looking positive
    numbers.

    Instead the angle range is mirrored and the stations re-estimated.  That is
    cheap: reversing the search line mirrors the STI in space, which negates
    phi and leaves the structure tensor untouched, so the existing orientation
    fields can be reused and only the histogram of Eq. (14) is recomputed.

    The sign cannot be read off the first pass.  ``velocity_raw_ms`` came from
    the configured -- and nearly always asymmetric -- range, so when the lines
    face upstream the true angle does not merely sit near that range's boundary
    but can fall entirely outside it.  Then *no* window lands in range, every
    station comes back NaN, and the one clue to the direction is gone precisely
    when it is needed.  So the direction is probed over the full angle range
    instead, which has no assumed direction to be wrong about.
    """
    limit = 0.5 * math.pi - 1e-3
    if spectra is None:
        spectra = [None] * len(stis)
    probe: list[float] = []
    for s, field_, spec in zip(stis, fields, spectra):
        if field_ is None and spec is None:
            continue
        est = _angle(cfg, field_, spec, -limit, limit)
        if np.isfinite(est.phi):
            probe.append(
                orientation.velocity_from_angle(est.phi, s.sx_m_per_px, s.st_s_per_px)
            )
    if not probe:
        return False

    # Upper median for an even count, matching StivKit's `directionCheck` so
    # the two implementations decide identically on the same input.
    med = float(sorted(probe)[len(probe) // 2])
    if med >= 0:
        return False

    msg = (
        f"the median velocity along the search lines is {med:.3f} m/s, i.e. the "
        f"flow runs opposite to the assumed downstream direction"
    )
    advice = _reverse_advice(cfg.flow)
    if not cfg.flow.auto_flip:
        warnings.append(f"{msg}; {advice}.")
        return False

    v_lo, v_hi = cfg.stiv.velocity_range_ms
    structural = {"sti_too_small", "out_of_frame"}
    for st, s, field_, spec in zip(stations, stis, fields, spectra):
        if field_ is None and spec is None:
            continue
        # Mirror the search range about zero and re-run Eq. (14).
        est = _estimate_angle(cfg, s, field_, -v_hi, -v_lo, spec)
        # Every flag but the structural ones came from the now-discarded first
        # pass.  Drop them so that quality control -- which runs after this,
        # once -- judges only the estimate that will actually be reported.
        st.flags = [f for f in st.flags if f in structural]
        _apply_estimate(st, s, est, negate=True)
        # Kept mirrored too, so the STI detail figure draws the angle, the
        # histogram and the search range of the velocity the station reports.
        estimates[st.index] = replace(
            est, phi=-est.phi, tan_phi=-est.tan_phi,
            hist=est.hist[::-1], bin_centres=-est.bin_centres[::-1],
            phi_min=-est.phi_max, phi_max=-est.phi_min,
        )

    warnings.append(
        f"{msg}; the search range was mirrored and the stations re-analysed "
        f"(flow.auto_flip). To point the search lines the right way in the "
        f"first place, {advice}."
    )
    return True


def _vector_direction(
    cfg: Config, ests: Sequence[VectorEstimate], flow_dir: np.ndarray, warnings: list[str]
) -> bool:
    """Whether LS-PIV or PTV found the water running against ``flow_dir``.

    A 2-D method needs none of the STI's mirrored re-analysis: its vectors
    carry their own sign, so the stations are simply projected onto the
    reversed direction.  Returns True when the direction should be reversed,
    which ``flow.auto_flip: false`` forbids (the warning is still given).
    """
    along = [float(e.velocity_world @ flow_dir) for e in ests if e.ok]
    if not along:
        return False
    # Upper median for an even count, as _direction_check takes it.
    med = float(sorted(along)[len(along) // 2])
    if med >= 0:
        return False
    msg = (
        f"the median velocity along the flow direction is {med:.3f} m/s, i.e. the "
        f"flow runs opposite to the assumed downstream direction"
    )
    advice = _reverse_advice(cfg.flow)
    if not cfg.flow.auto_flip:
        warnings.append(f"{msg}; {advice}.")
        return False
    warnings.append(
        f"{msg}; the velocities are reported along the reversed direction "
        f"(flow.auto_flip). To set the direction the right way in the first place, "
        f"{advice}."
    )
    return True


def _direction_agreement(method: str, stations: Sequence[StationResult], primary: str,
                         warnings: list[str]) -> None:
    """Warn when a compared 2-D method sees the water run the other way."""
    v = sorted(s.velocity_ms for s in stations if np.isfinite(s.velocity_ms))
    if v and v[len(v) // 2] < 0:
        warnings.append(
            f"the median velocity is {v[len(v) // 2]:.3f} m/s: {METHOD_LABELS[method]} "
            f"sees the water flow the opposite way to {METHOD_LABELS[primary]}; check "
            f"the flow direction"
        )


def _vector_stations(
    cfg: Config, geom: SectionGeometry, ests: Sequence[VectorEstimate], method: str,
    st_s_per_px: float,
) -> list[StationResult]:
    """LS-PIV or PTV estimates as stations, along the section's final flow direction.

    The reported velocity is the component along ``geom.flow_dir_world`` --
    the one the discharge needs -- and the component across it is kept in
    ``velocity_cross_ms``.  The spread is that of the individual pairs or
    tracks along the flow, as STIV's is the spread of its windows.
    """
    along = np.asarray(geom.flow_dir_world, dtype=float)
    across = np.array([-along[1], along[0]])
    stations: list[StationResult] = []
    for sl, e in zip(geom.search_lines, ests):
        st = StationResult(
            index=sl.index,
            station_m=sl.station_m,
            x_world=float(sl.centre_world[0]),
            y_world=float(sl.centre_world[1]),
            u_px=float(sl.centre_px[0]),
            v_px=float(sl.centre_px[1]),
            phi_rad=math.nan,
            tan_phi=math.nan,
            velocity_ms=math.nan,
            velocity_raw_ms=math.nan,
            coherency=math.nan,
            valid_fraction=math.nan,
            n_windows=0,
            phi_std_rad=math.nan,
            velocity_std_ms=math.nan,
            sx_m_per_px=math.nan,
            st_s_per_px=st_s_per_px,
            flags=list(e.flags),
            method=method,
            peak_ratio=e.peak_ratio,
            n_measurements=e.n_used,
        )
        if e.ok:
            st.velocity_ms = st.velocity_raw_ms = float(e.velocity_world @ along)
            st.velocity_cross_ms = float(e.velocity_world @ across)
            st.velocity_spread_ms = robust_spread(e.samples_world @ along)
            if (method == "lspiv" and np.isfinite(e.displacement_px)
                    and e.displacement_px < _LOW_DISPLACEMENT_SAMPLES):
                st.add_flag("displacement_low")
        if e.out_of_frame_fraction > _MAX_OUT_OF_FRAME_FRACTION:
            st.add_flag("out_of_frame")
        stations.append(st)

    # Cross-flow is judged only on stations moving fast enough for their
    # direction to mean something: at a near-still bank it is noise.
    speeds = [math.hypot(s.velocity_ms, s.velocity_cross_ms) for s in stations
              if np.isfinite(s.velocity_ms)]
    floor = _UNCERTAINTY_SCALE_FRACTION * float(np.median(speeds)) if speeds else math.inf
    limit = math.tan(math.radians(_CROSS_FLOW_DEG))
    for s in stations:
        if (np.isfinite(s.velocity_ms) and math.hypot(s.velocity_ms, s.velocity_cross_ms) >= floor
                and abs(s.velocity_cross_ms) > limit * abs(s.velocity_ms)):
            s.add_flag("cross_flow_high")
    return stations


def _discharge(cfg: Config, stations: Sequence[StationResult], geom: SectionGeometry,
               warnings: list[str]) -> Optional[DischargeResult]:
    """Integrate a profile into a discharge, if the config asks for one."""
    if not cfg.discharge.enabled:
        return None
    # A discharge that cannot be computed -- every station failed, say --
    # must not take the velocity profile down with it.
    try:
        disch = compute_discharge(
            cfg.discharge,
            [s.station_m for s in stations],
            [s.velocity_ms for s in stations],
            geom.width_m,
        )
    except ConfigError as exc:
        warnings.append(f"discharge was not computed: {exc}")
        return None
    warnings.extend(disch.warnings)
    return disch


def _range_advice(
    cfg: Config, stations: list[StationResult], source: FrameSource, warnings: list[str]
) -> None:
    """Suggest a ``frame_step`` that puts ``tan phi`` in the paper's sweet spot.

    Accuracy is best when ``tan phi`` is around 1 (Fig. 7), i.e. when a surface
    feature moves about one STI column per row.  Since ``tan phi = U St / Sx``
    and ``St = frame_step / fps``, the step can be scaled to move it there.

    Judged on every station's *measured* angle, failed or not -- it is when a
    run fails wholesale that this advice matters most -- recovered from
    ``velocity_raw_ms`` since the reported angle of a filled station is not a
    measurement.
    """
    tans = np.array(
        [
            abs(s.velocity_raw_ms * s.st_s_per_px / s.sx_m_per_px)
            for s in stations
            if np.isfinite(s.velocity_raw_ms) and s.sx_m_per_px > 0
        ]
    )
    if tans.size == 0:
        return
    med = float(np.median(tans))
    if med <= 0:
        return

    band_lo, band_hi = _TAN_PHI_ADVICE_BAND
    if band_lo <= med <= band_hi:
        return

    if med > band_hi:
        suggested = max(1, int(round(source.step / med)))
        if suggested < source.step:
            warnings.append(
                f"median tan(phi) is {med:.1f}, near the top of the range the method "
                f"was validated over (Fig. 7 of the paper stops at 7). Features move "
                f"far between STI rows -- try video.frame_step: {suggested}."
            )
            return
        # Already reading every frame, so St cannot shrink any further and
        # only Sx is left.  A longer search line does not change Sx when the
        # sample count is automatic -- the samples stay about a pixel apart and
        # the line merely gains columns -- so that is not offered.
        sx = float(np.median([s.sx_m_per_px for s in stations]))
        n_cols = max(16, int(round(cfg.stiv.search_line_length_m / (sx * med))) + 1)
        longer = (
            " A longer search line does not help: with automatic stiv.n_samples "
            "it only adds columns." if cfg.stiv.n_samples is None else ""
        )
        warnings.append(
            f"median tan(phi) is {med:.1f} with video.frame_step already 1, near the "
            f"top of the range the method was validated over (Fig. 7 of the paper "
            f"stops at 7): features move about {med:.0f} STI columns per frame. "
            f"Since tan(phi) = U St / Sx, what helps is a shorter St -- record at a "
            f"higher frame rate -- or a coarser Sx: fly higher, or set "
            f"stiv.n_samples to about {n_cols} so the samples are spaced wider "
            f"(only if the surface texture is coarser than that spacing).{longer}"
        )
    else:
        suggested = max(1, int(round(source.step / med)))
        warnings.append(
            f"median tan(phi) is {med:.2f}: features barely move between frames, "
            f"so velocity resolution is poor. Try video.frame_step: {suggested} "
            f"to space the samples further apart in time."
        )
