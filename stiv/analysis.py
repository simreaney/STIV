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
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import geometry, orientation, sti as sti_mod
from .config import Config
from .discharge import DischargeResult, compute_discharge
from .geometry import SectionGeometry
from .orientation import AngleEstimate, OrientationField
from .stabilise import StabilisationReport, Stabiliser, build_mask
from .sti import SpaceTimeImage
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
    }
)

#: Flags meaning "the number is real but near the edge of the method's range".
#: A genuinely slow station by the bank is the common case, and interpolating
#: it away from its neighbours would be worse than reporting it.
ADVISORY_FLAGS = frozenset(
    {"tan_phi_low", "tan_phi_high", "interpolated", "median_filtered"}
)


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
    phi_std_rad: float
    #: 1-sigma velocity spread implied by the angle histogram, m/s.
    velocity_std_ms: float
    sx_m_per_px: float
    st_s_per_px: float
    flags: list[str] = field(default_factory=list)

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

    # ------------------------------------------------------------------ #
    @property
    def velocities(self) -> np.ndarray:
        return np.array([s.velocity_ms for s in self.stations])

    @property
    def stations_m(self) -> np.ndarray:
        return np.array([s.station_m for s in self.stations])

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for s in self.stations:
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
                    "velocity_std_ms": s.velocity_std_ms,
                    "coherency": s.coherency,
                    "valid_fraction": s.valid_fraction,
                    "n_windows": s.n_windows,
                    "sx_m_per_px": s.sx_m_per_px,
                    "st_s_per_px": s.st_s_per_px,
                    "flags": ";".join(s.flags),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        v = self.velocities
        clean = [s for s in self.stations if s.ok]
        failed = [s for s in self.stations if s.failed]
        finite = v[np.isfinite(v)]
        lines = [
            f"section        {self.config.section.name}: "
            f"{self.geometry.width_m:.2f} m wide, {len(self.stations)} stations",
            f"rectification  {self.geometry.rectifier.describe()}",
            f"frames         {self.n_frames} over {self.duration_s:.2f} s "
            f"(St = {self.st_s_per_px * 1000:.1f} ms/row, {self.fps:.2f} fps source)",
            f"flow direction {self.geometry.flow_bearing_deg:+.1f} deg in world frame",
            self.stabilisation.summary(),
            f"stations       {len(clean)}/{len(self.stations)} clean, "
            f"{len(failed)} failed and interpolated",
        ]
        if finite.size:
            lines += [
                f"surface U      mean {finite.mean():.3f}  median {np.median(finite):.3f}  "
                f"max {finite.max():.3f} m/s",
            ]
        if self.discharge is not None:
            lines.append("")
            lines.append(self.discharge.summary())
        if self.warnings:
            lines.append("")
            lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def run(cfg: Config, progress: bool = True) -> StivResult:
    """Execute a full STIV analysis described by ``cfg``."""
    t_start = time.perf_counter()
    warnings: list[str] = []

    source = FrameSource(cfg.video)
    log.info("video: %s", source.info)
    log.info("using %d frames, step %d (St = %.4f s)", source.n_frames, source.step, source.st_s_per_px)

    geom = geometry.build_section(cfg, source.shape)
    log.info("rectification: %s", geom.rectifier.describe())
    log.info(
        "section %.2f m wide, %d stations, %.2f m search lines along %+.1f deg",
        geom.width_m, len(geom.search_lines), cfg.stiv.search_line_length_m, geom.flow_bearing_deg,
    )

    outside = geometry.check_lines_in_frame(geom, source.shape)
    if outside:
        warnings.append(
            f"{len(outside)} search line(s) extend outside the frame "
            f"(stations {outside[:5]}{'...' if len(outside) > 5 else ''}); "
            f"shorten stiv.search_line_length_m or move the section"
        )

    _rectification_warnings(geom, warnings)

    stis, stab_report, reference_bgr = _collect_stis(cfg, source, geom, progress)

    stations, fields, estimates = _analyse_stis(cfg, stis, geom)

    # Direction first: it can replace every station's estimate, and quality
    # control must judge the estimates that are actually going to be reported.
    _direction_check(cfg, stations, stis, fields, estimates, warnings)
    _quality_control(cfg, stations, warnings)
    _range_advice(cfg, stations, source, warnings)
    warnings.extend(stab_report.warnings())

    disch = None
    if cfg.discharge.enabled:
        disch = compute_discharge(
            cfg.discharge,
            [s.station_m for s in stations],
            [s.velocity_ms for s in stations],
            geom.width_m,
        )

    return StivResult(
        config=cfg,
        geometry=geom,
        stations=stations,
        stis=stis,
        fields=fields,
        estimates=estimates,
        stabilisation=stab_report,
        n_frames=stis[0].data.shape[0] if stis else 0,
        duration_s=(stis[0].data.shape[0] * source.st_s_per_px) if stis else 0.0,
        fps=source.fps,
        st_s_per_px=source.st_s_per_px,
        reference_frame_bgr=reference_bgr,
        discharge=disch,
        warnings=warnings,
        elapsed_s=time.perf_counter() - t_start,
    )


def _rectification_warnings(geom: SectionGeometry, warnings: list[str]) -> None:
    """Flag a ground-control fit that will not support the velocities it scales.

    Velocity is linear in ``Sx``, so an error in the image-to-world map passes
    straight into every station with no attenuation and nothing downstream can
    detect it.  The residuals are the only evidence available.
    """
    rect = geom.rectifier
    if not isinstance(rect, geometry.HomographyRectifier):
        return

    tolerance_m = rect.inlier_threshold_m
    if rect.rmse_m > tolerance_m:
        warnings.append(
            f"the ground control points fit to {rect.rmse_m:.3f} m RMSE, worse "
            f"than the {tolerance_m:.3f} m expected of {geometry._RANSAC_PX:.0f}-pixel "
            f"clicking accuracy; every velocity scales linearly with this error. "
            f"Re-check the survey coordinates and the clicked pixels."
        )
    if rect.n_inliers < rect.n_gcp:
        warnings.append(
            f"{rect.n_gcp - rect.n_inliers} of {rect.n_gcp} ground control "
            f"point(s) were rejected as outliers by the homography fit; check "
            f"them for a transcription error or a mismatched point order"
        )


def _collect_stis(
    cfg: Config, source: FrameSource, geom: SectionGeometry, progress: bool
) -> tuple[list[SpaceTimeImage], StabilisationReport, np.ndarray]:
    """Single streaming pass over the video, building every STI at once.

    Returns the STIs, the stabilisation diagnostics, and the reference frame
    (kept for the overview plot).
    """
    collector = sti_mod.StiCollector(geom.search_lines, source.n_frames, source.st_s_per_px)
    probe = np.vstack([geom.start_px[None, :], geom.end_px[None, :]])

    stabiliser: Optional[Stabiliser] = None
    reference_bgr = np.zeros((*source.shape, 3), np.uint8)

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=source.n_frames, unit="frame", desc="STI", leave=False)
        except ImportError:
            bar = None

    try:
        for i, frame_bgr in enumerate(source.frames(colour=True)):
            grey = source.to_grey(frame_bgr)
            if i == 0:
                reference_bgr = frame_bgr.copy()
                mask = build_mask(cfg.stabilise, source.shape, exclude_px=collector.all_pts)
                stabiliser = Stabiliser(cfg.stabilise, grey, mask)
                if stabiliser.enabled:
                    log.info("stabilisation: tracking %d static features", stabiliser.report.n_features)
                collector.add_frame(grey, None)
            else:
                M = stabiliser.update(grey, probe_px=probe) if stabiliser.enabled else None
                collector.add_frame(grey, M)
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()

    report = stabiliser.report if stabiliser else StabilisationReport(enabled=False)
    return collector.finish(), report, reference_bgr


def _estimate_angle(
    cfg: Config, s: SpaceTimeImage, field_: OrientationField, v_lo: float, v_hi: float
) -> AngleEstimate:
    """Run Eq. (14) over one orientation field for a given velocity range."""
    phi_min, phi_max = orientation.angle_range_from_velocity(
        v_lo, v_hi, s.sx_m_per_px, s.st_s_per_px
    )
    return orientation.mean_orientation_angle(
        field_,
        phi_min,
        phi_max,
        coherency_levels=cfg.stiv.coherency_levels,
        n_bins=cfg.stiv.n_angle_bins,
        min_coherency=cfg.stiv.min_coherency,
        estimator=cfg.stiv.estimator,
    )


def _apply_estimate(
    cfg: Config,
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
        st.velocity_std_ms = math.nan
        st.phi_std_rad = math.nan
        st.coherency = 0.0
        st.valid_fraction = 0.0 if est is None else est.valid_fraction
        st.n_windows = 0
        if "sti_too_small" not in st.flags and "no_orientation" not in st.flags:
            st.flags.append("no_orientation")
        return

    sign = -1.0 if negate else 1.0
    phi = sign * est.phi
    st.phi_rad = phi
    st.tan_phi = float(math.tan(phi))
    st.velocity_ms = orientation.velocity_from_angle(phi, s.sx_m_per_px, s.st_s_per_px)
    st.velocity_raw_ms = st.velocity_ms
    # Propagate the angle spread through Eq. (16): dU/dphi = k * sec^2(phi).
    k = s.sx_m_per_px / s.st_s_per_px
    st.velocity_std_ms = abs(k / max(math.cos(phi) ** 2, 1e-6)) * est.phi_std
    st.phi_std_rad = est.phi_std
    st.coherency = est.mean_coherency
    st.valid_fraction = est.valid_fraction
    st.n_windows = est.n_used


def _analyse_stis(
    cfg: Config, stis: Sequence[SpaceTimeImage], geom: SectionGeometry
) -> tuple[list[StationResult], list[Optional[OrientationField]], list[Optional[AngleEstimate]]]:
    """Run the orientation analysis on every STI and convert to velocity."""
    stations: list[StationResult] = []
    fields: list[Optional[OrientationField]] = []
    estimates: list[Optional[AngleEstimate]] = []

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

        s.data = sti_mod.preprocess(s.data, cfg.stiv)

        field_ = None
        est = None
        try:
            field_ = orientation.structure_tensor_field(
                s.data,
                window_px=cfg.stiv.window_px,
                window_type=cfg.stiv.window_type,
                overlap=cfg.stiv.window_overlap,
                gradient_order=cfg.stiv.gradient_order,
            )
            est = _estimate_angle(cfg, s, field_, v_lo, v_hi)
        except ValueError as exc:
            st.flags.append("sti_too_small")
            log.warning("station %d: %s", sl.index, exc)

        _apply_estimate(cfg, st, s, est)

        if s.out_of_frame_fraction > 0.02:
            st.flags.append("out_of_frame")

        stations.append(st)
        fields.append(field_)
        estimates.append(est)

    return stations, fields, estimates


# --------------------------------------------------------------------------- #
# quality control
# --------------------------------------------------------------------------- #
def _quality_control(cfg: Config, stations: list[StationResult], warnings: list[str]) -> None:
    q = cfg.quality
    lo, hi = q.tan_phi_range

    # The uncertainty test is relative to the section's own velocity scale, so
    # that a genuinely slow but well-resolved station near the bank is not
    # condemned merely for having a small denominator.
    v_all = np.array([abs(s.velocity_ms) for s in stations], dtype=float)
    finite = v_all[np.isfinite(v_all)]
    v_scale = 0.25 * float(np.median(finite)) if finite.size else 0.0

    for s in stations:
        if np.isfinite(s.valid_fraction) and s.valid_fraction < q.min_valid_fraction:
            s.flags.append("low_valid_fraction")
        if np.isfinite(s.coherency) and s.coherency < q.min_mean_coherency:
            s.flags.append("low_coherency")
        if np.isfinite(s.velocity_std_ms) and np.isfinite(s.velocity_ms):
            reference = max(abs(s.velocity_ms), v_scale)
            if reference > 0 and s.velocity_std_ms > q.max_relative_uncertainty * reference:
                s.flags.append("high_uncertainty")
        if np.isfinite(s.tan_phi):
            a = abs(s.tan_phi)
            if a > hi:
                # Steep streaks: too little spatial displacement per frame is
                # resolvable.  Section 2.3 tested only up to tan phi = 7.
                s.flags.append("tan_phi_high")
            elif a < lo:
                s.flags.append("tan_phi_low")

    if q.median_filter_stations and q.median_filter_stations > 1:
        _median_filter(stations, int(q.median_filter_stations))

    if q.interpolate_flagged:
        _interpolate_flagged(stations)

    n_bad = sum(1 for s in stations if s.failed)
    if n_bad > 0.3 * max(1, len(stations)):
        warnings.append(
            f"{n_bad} of {len(stations)} stations failed quality control; "
            f"check that the water surface has visible texture, that "
            f"stiv.velocity_range_ms brackets the true velocity, and that the "
            f"search lines are parallel to the flow"
        )


def _median_filter(stations: list[StationResult], k: int) -> None:
    """Median-filter the velocity profile across stations.

    Any station the filter actually moves is flagged.  The reported velocity is
    then no longer the one that station measured, and a silently substituted
    number that still reports ``ok`` is worse than no filter at all.
    """
    v = np.array([s.velocity_ms for s in stations], dtype=float)
    half = k // 2
    out = v.copy()
    for i in range(len(v)):
        lo, hi = max(0, i - half), min(len(v), i + half + 1)
        window = v[lo:hi]
        window = window[np.isfinite(window)]
        if window.size:
            out[i] = float(np.median(window))
    for s, before, after in zip(stations, v, out):
        s.velocity_ms = float(after)
        unchanged = before == after or (np.isnan(before) and np.isnan(after))
        if not unchanged and "median_filtered" not in s.flags:
            s.flags.append("median_filtered")


def _interpolate_flagged(stations: list[StationResult]) -> None:
    """Replace *failed* stations with an interpolation of their neighbours.

    Advisory flags are deliberately left alone: a station reading a genuinely
    slow patch near the bank is measuring correctly, and pulling it towards its
    faster neighbours would introduce error rather than remove it.
    """
    x = np.array([s.station_m for s in stations], dtype=float)
    v = np.array([s.velocity_ms for s in stations], dtype=float)
    failed = np.array([s.failed for s in stations], dtype=bool)
    usable = ~failed & np.isfinite(v)

    if usable.sum() < 2 or not failed.any():
        return
    filled = np.interp(x, x[usable], v[usable])
    for s, val, bad in zip(stations, filled, failed):
        if bad:
            s.velocity_ms = float(val)
            if "interpolated" not in s.flags:
                s.flags.append("interpolated")


def _direction_check(
    cfg: Config,
    stations: list[StationResult],
    stis: Sequence[SpaceTimeImage],
    fields: list[Optional[OrientationField]],
    estimates: list[Optional[AngleEstimate]],
    warnings: list[str],
) -> None:
    """Detect and, if allowed, correct search lines that point upstream.

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
    probe: list[float] = []
    for s, field_ in zip(stis, fields):
        if field_ is None:
            continue
        est = orientation.mean_orientation_angle(
            field_,
            -limit,
            limit,
            coherency_levels=cfg.stiv.coherency_levels,
            n_bins=cfg.stiv.n_angle_bins,
            min_coherency=cfg.stiv.min_coherency,
            estimator=cfg.stiv.estimator,
        )
        if np.isfinite(est.phi):
            probe.append(
                orientation.velocity_from_angle(est.phi, s.sx_m_per_px, s.st_s_per_px)
            )
    if not probe:
        return

    # Upper median for an even count, matching StivKit's `directionCheck` so
    # the two implementations decide identically on the same input.
    med = float(sorted(probe)[len(probe) // 2])
    if med >= 0:
        return

    msg = (
        f"the median velocity along the search lines is {med:.3f} m/s, i.e. the "
        f"flow runs opposite to the assumed downstream direction"
    )
    if not cfg.flow.auto_flip:
        warnings.append(msg + "; set flow.reverse: true, or give flow.downstream_px.")
        return

    v_lo, v_hi = cfg.stiv.velocity_range_ms
    structural = {"sti_too_small", "out_of_frame"}
    for st, s, field_ in zip(stations, stis, fields):
        if field_ is None:
            continue
        # Mirror the search range about zero and re-run Eq. (14).
        est = _estimate_angle(cfg, s, field_, -v_hi, -v_lo)
        # Every flag but the structural ones came from the now-discarded first
        # pass.  Drop them so that quality control -- which runs after this,
        # once -- judges only the estimate that will actually be reported.
        st.flags = [f for f in st.flags if f in structural]
        _apply_estimate(cfg, st, s, est, negate=True)
        estimates[st.index] = est

    warnings.append(
        msg + "; the search range was mirrored and the stations re-analysed "
        "(flow.auto_flip). Set flow.reverse: true or give flow.downstream_px "
        "to point the search lines the right way in the first place."
    )


def _range_advice(
    cfg: Config, stations: list[StationResult], source: FrameSource, warnings: list[str]
) -> None:
    """Suggest a ``frame_step`` that puts ``tan phi`` in the paper's sweet spot.

    Accuracy is best when ``tan phi`` is around 1 (Fig. 7), i.e. when a surface
    feature moves about one STI column per row.  Since ``tan phi = U St / Sx``
    and ``St = frame_step / fps``, the step can be scaled to move it there.
    """
    tans = np.array([abs(s.tan_phi) for s in stations if np.isfinite(s.tan_phi)])
    if tans.size == 0:
        return
    med = float(np.median(tans))
    if med <= 0:
        return

    if 0.2 <= med <= 4.0:
        return

    suggested = max(1, int(round(source.step / med)))
    if suggested == source.step:
        return

    if med > 4.0:
        warnings.append(
            f"median tan(phi) is {med:.1f}, near the top of the range the method "
            f"was validated over (Fig. 7 of the paper stops at 7). Features move "
            f"far between frames -- try video.frame_step: {suggested}, or a "
            f"longer stiv.search_line_length_m."
        )
    else:
        warnings.append(
            f"median tan(phi) is {med:.2f}: features barely move between frames, "
            f"so velocity resolution is poor. Try video.frame_step: {suggested} "
            f"to space the samples further apart in time."
        )
