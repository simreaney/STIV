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
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import geometry, orientation, sti as sti_mod
from .config import Config, ConfigError, FlowConfig, StabiliseConfig, is_auto_water_mask
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

    # ------------------------------------------------------------------ #
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
        return pd.DataFrame(rows)

    def summary(self) -> str:
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
    if source.n_frames is None:
        log.info("using every frame to the end of the video, step %d (St = %.4f s)",
                 source.step, source.st_s_per_px)
    else:
        log.info("using %d frames, step %d (St = %.4f s)",
                 source.n_frames, source.step, source.st_s_per_px)

    # An STI shorter than the analysis window fails at every station, and
    # used to be reported as a texture or velocity-range problem afterwards.
    need = orientation.min_sti_size(
        cfg.stiv.window_px, cfg.stiv.window_type, cfg.stiv.gradient_order
    )
    if source.n_frames is not None and source.n_frames < need:
        raise ConfigError(
            f"the clip gives only {source.n_frames} STI rows (frames left after "
            f"video.start_s, duration_s and frame_step: {source.step}), but a "
            f"{cfg.stiv.window_px} px {cfg.stiv.window_type} window needs at least "
            f"{need}; use a longer video.duration_s, a smaller video.frame_step, or "
            f"a smaller stiv.window_px"
        )

    geom = geometry.build_section(cfg, source.shape)
    log.info("rectification: %s", geom.rectifier.describe())
    log.info(
        "section %.2f m wide, %d stations, %.2f m search lines along %+.1f deg",
        geom.width_m, len(geom.search_lines), cfg.stiv.search_line_length_m, geom.flow_bearing_deg,
    )

    narrow = [sl.index for sl in geom.search_lines if sl.n_samples < need]
    if narrow and len(narrow) == len(geom.search_lines):
        raise ConfigError(
            f"the search lines have only {min(sl.n_samples for sl in geom.search_lines)} "
            f"samples, but a {cfg.stiv.window_px} px {cfg.stiv.window_type} window "
            f"needs at least {need}; lengthen stiv.search_line_length_m, raise "
            f"stiv.n_samples, or use a smaller stiv.window_px"
        )
    if narrow:
        warnings.append(
            f"{len(narrow)} search line(s) have fewer than the {need} samples a "
            f"{cfg.stiv.window_px} px window needs (stations {narrow[:5]}"
            f"{'...' if len(narrow) > 5 else ''}) and will fail; lengthen "
            f"stiv.search_line_length_m or use a smaller stiv.window_px"
        )

    outside = geometry.check_lines_in_frame(geom, source.shape)
    if outside:
        warnings.append(
            f"{len(outside)} search line(s) extend outside the frame "
            f"(stations {outside[:5]}{'...' if len(outside) > 5 else ''}); "
            f"shorten stiv.search_line_length_m or move the section"
        )

    _rectification_warnings(geom, warnings)

    stab_cfg, water_note = cfg.stabilise, ""
    if is_auto_water_mask(cfg.stabilise.water_mask_px):
        stab_cfg, water_note = _automatic_water_mask(cfg, warnings)

    stis, stab_report, reference_bgr = _collect_stis(cfg, source, geom, progress, stab_cfg)
    warnings.extend(source.warnings)

    stations, fields, estimates = _analyse_stis(cfg, stis, geom)

    # Direction first: it can replace every station's estimate, and quality
    # control must judge the estimates that are actually going to be reported.
    if _direction_check(cfg, stations, stis, fields, estimates, warnings):
        # The stations now report the flow as running the other way along the
        # search lines, so the direction reported and drawn follows them.  The
        # lines themselves are already laid out and sampled, and nothing reads
        # this back to lay them out again.
        geom.flow_dir_world = -geom.flow_dir_world
        bearing = geom.flow_bearing_deg + 180.0
        geom.flow_bearing_deg = bearing - 360.0 if bearing > 180.0 else bearing
    _quality_control(cfg, stations, warnings)
    _range_advice(cfg, stations, source, warnings)
    warnings.extend(stab_report.warnings())

    disch = None
    if cfg.discharge.enabled:
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
        else:
            warnings.extend(disch.warnings)

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
        water_mask_note=water_note,
    )


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


def _collect_stis(
    cfg: Config, source: FrameSource, geom: SectionGeometry, progress: bool,
    stabilise_cfg: Optional[StabiliseConfig] = None,
) -> tuple[list[SpaceTimeImage], StabilisationReport, np.ndarray]:
    """Single streaming pass over the video, building every STI at once.

    ``stabilise_cfg`` is ``cfg.stabilise`` with an automatic water mask
    resolved; ``cfg.stabilise`` itself when omitted.  Returns the STIs, the
    stabilisation diagnostics, and the reference frame (kept for the overview
    plot).
    """
    stab_cfg = stabilise_cfg if stabilise_cfg is not None else cfg.stabilise
    collector = sti_mod.StiCollector(geom.search_lines, source.n_frames, source.st_s_per_px)
    probe = np.vstack([geom.start_px[None, :], geom.end_px[None, :]])

    stabiliser: Optional[Stabiliser] = None
    reference_bgr = np.zeros((*source.shape, 3), np.uint8)

    bar = None
    if progress:
        try:
            # tqdm.auto: a progress widget in a notebook, a text bar elsewhere.
            from tqdm.auto import tqdm

            bar = tqdm(total=source.n_frames, unit="frame", desc="STI", leave=False)
        except ImportError:
            bar = None

    try:
        for i, frame_bgr in enumerate(source.frames(colour=True)):
            grey = source.to_grey(frame_bgr)
            if i == 0:
                reference_bgr = frame_bgr.copy()
                mask = build_mask(stab_cfg, source.shape, exclude_px=collector.all_pts)
                stabiliser = Stabiliser(stab_cfg, grey, mask)
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


def _estimate_angle(
    cfg: Config, s: SpaceTimeImage, field_: OrientationField, v_lo: float, v_hi: float
) -> AngleEstimate:
    """Run Eq. (14) over one orientation field for a given velocity range."""
    phi_min, phi_max = orientation.angle_range_from_velocity(
        v_lo, v_hi, s.sx_m_per_px, s.st_s_per_px
    )
    return _eq14(cfg, field_, phi_min, phi_max)


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

        # Conditioning and orientation run in float64 throughout: the Swift
        # port is held to these numbers to 1e-9.
        conditioned = sti_mod.preprocess(s.data, cfg.stiv)

        field_ = None
        est = None
        try:
            field_ = orientation.structure_tensor_field(
                conditioned,
                window_px=cfg.stiv.window_px,
                window_type=cfg.stiv.window_type,
                overlap=cfg.stiv.window_overlap,
                gradient_order=cfg.stiv.gradient_order,
            )
            est = _estimate_angle(cfg, s, field_, v_lo, v_hi)
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
    if np.isfinite(velocity_ms) and st.sx_m_per_px > 0:
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
        warnings.append(
            f"{n_bad} of {len(stations)} stations failed quality control; "
            f"check that the water surface has visible texture, that "
            f"stiv.velocity_range_ms brackets the true velocity, and that the "
            f"search lines are parallel to the flow"
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
    probe: list[float] = []
    for s, field_ in zip(stis, fields):
        if field_ is None:
            continue
        est = _eq14(cfg, field_, -limit, limit)
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
    for st, s, field_ in zip(stations, stis, fields):
        if field_ is None:
            continue
        # Mirror the search range about zero and re-run Eq. (14).
        est = _estimate_angle(cfg, s, field_, -v_hi, -v_lo)
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
