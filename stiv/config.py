"""Configuration schema for a STIV run.

Everything the analysis needs is described by :class:`Config`, which is normally
loaded from a YAML file.  Nested dataclasses mirror the YAML structure, so a
config file reads like the analysis it describes::

    video:      {path: flight.mp4, start: 0.0, duration: 30.0}
    rectify:    {method: scale, m_per_px: 0.021}
    section:    {start_px: [310, 880], end_px: [1640, 830], n_stations: 40}
    flow:       {direction: auto}
    stiv:       {search_line_length_m: 4.0, window_px: 31}
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

Point = Sequence[float]


class ConfigError(ValueError):
    """Raised when a configuration file is missing or internally inconsistent."""


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass
class VideoConfig:
    """Which frames to read out of the movie file.

    ``frame_step`` is the single most important tuning knob in STIV.  It sets
    the time scale ``St = frame_step / fps`` (s/pixel) of the space-time image,
    and therefore where the measured velocities land on the ``tan phi`` axis.
    Fujita et al. (2007), section 2.3, show the orientation analysis is accurate
    for ``tan phi`` between 0.1 and 7; :mod:`stiv.analysis` warns when a run
    falls outside that window and suggests a better ``frame_step``.
    """

    path: str = ""
    start_s: float = 0.0
    duration_s: Optional[float] = None
    frame_step: int = 1
    max_frames: Optional[int] = None
    #: Optional override when the container reports a wrong frame rate.
    fps_override: Optional[float] = None
    #: Weights used to collapse BGR to the scalar "brightness" g(x, t).
    #: The default is Rec.601 luma. ``[0, 0, 1]`` isolates the red channel,
    #: which can help when the water surface pattern is turbidity-driven.
    channel_weights: Optional[Sequence[float]] = None


@dataclass
class RectifyConfig:
    """Mapping between image pixels and physical distance on the water surface.

    Two methods are supported:

    ``scale``
        A single ground sample distance in m/pixel.  Correct for a nadir
        (straight-down) drone shot, which is the usual UAV gauging geometry.
        Either give ``m_per_px`` directly or let it be derived from
        ``altitude_m`` plus the camera intrinsics.

    ``gcp``
        A plane-to-plane homography fitted to >= 4 ground control points whose
        image and world coordinates are both known.  This is the general case
        of Eq. (15) in the paper, specialised to the water-surface plane, and
        handles oblique views.
    """

    method: str = "scale"  # "scale" | "gcp"

    # --- method: scale ----------------------------------------------------- #
    m_per_px: Optional[float] = None
    altitude_m: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    focal_length_mm: Optional[float] = None
    #: Horizontal field of view; an alternative to sensor/focal length.
    hfov_deg: Optional[float] = None

    # --- method: gcp ------------------------------------------------------- #
    #: Image coordinates [[u, v], ...] of >= 4 control points.
    gcp_px: Optional[Sequence[Point]] = None
    #: Matching world coordinates [[X, Y], ...] in metres, on the water plane.
    gcp_world: Optional[Sequence[Point]] = None

    def resolve_m_per_px(self, image_width_px: int) -> Optional[float]:
        """Return the ground sample distance, deriving it from optics if needed."""
        if self.m_per_px is not None:
            return float(self.m_per_px)
        if self.altitude_m is None:
            return None
        if self.hfov_deg is not None:
            import math

            ground_width = 2.0 * self.altitude_m * math.tan(math.radians(self.hfov_deg) / 2.0)
            return ground_width / image_width_px
        if self.sensor_width_mm is not None and self.focal_length_mm is not None:
            ground_width = self.altitude_m * self.sensor_width_mm / self.focal_length_mm
            return ground_width / image_width_px
        return None


@dataclass
class SectionConfig:
    """The measurement cross-section, and where velocity is sampled along it.

    The section is drawn on the image (``start_px`` -> ``end_px``); its physical
    length follows from :class:`RectifyConfig`.  Stations are spaced evenly in
    *world* space, so an oblique view still yields evenly spaced verticals.
    """

    start_px: Optional[Point] = None
    end_px: Optional[Point] = None
    n_stations: int = 30
    #: Skip this fraction of the section at each end (banks are usually noisy).
    edge_trim: float = 0.0
    #: Name written into the output tables.
    name: str = "section"


@dataclass
class FlowConfig:
    """Orientation of the streamwise search lines.

    STIV is a 1-D method: it measures the velocity component along the search
    line, so the line must be parallel to the main flow (paper, section 2.1).

    ``auto``
        Perpendicular to the cross-section, evaluated in world space.
    ``angle``
        A fixed bearing ``angle_deg``, measured counter-clockwise from the
        world +X axis.
    ``points``
        Parallel to ``from_px`` -> ``to_px``, which point downstream.
    """

    direction: str = "auto"  # "auto" | "angle" | "points"
    angle_deg: Optional[float] = None
    from_px: Optional[Point] = None
    to_px: Optional[Point] = None
    #: A point in the image that lies downstream of the section; disambiguates
    #: the sign of the "auto" normal.
    downstream_px: Optional[Point] = None
    #: Flip the assumed downstream direction.
    reverse: bool = False
    #: If the coherency-weighted median velocity comes out negative, flip the
    #: whole set and say so in the log rather than reporting negative flow.
    auto_flip: bool = True


@dataclass
class StabiliseConfig:
    """Compensation for drone motion.

    A hovering multirotor still drifts and yaws by a few pixels, and STIV reads
    that apparent motion as water velocity.  Static features on the bank are
    tracked with pyramidal Lucas-Kanade optical flow and a per-frame transform
    back to the reference frame is fitted; the search-line sample points are
    then pushed through that transform instead of warping whole frames.
    """

    enabled: bool = True
    #: "similarity" (rotation + translation + uniform scale), "affine" or
    #: "homography".  Similarity is the most stable with few bank features.
    model: str = "similarity"
    max_features: int = 1200
    quality_level: float = 0.01
    min_distance_px: int = 12
    #: Polygons [[ [u,v], ... ], ...] enclosing *static* ground to track.
    #: If empty, everything outside ``water_mask_px`` is used.
    static_mask_px: Optional[Sequence[Sequence[Point]]] = None
    #: Polygons enclosing water, excluded from tracking.  If neither mask is
    #: given, a corridor around the search lines is excluded automatically.
    water_mask_px: Optional[Sequence[Sequence[Point]]] = None
    #: Re-detect features when fewer than this fraction survive tracking.
    redetect_fraction: float = 0.5
    #: Reject the transform (and hold the previous one) above this RANSAC
    #: reprojection error, in pixels.
    max_reprojection_px: float = 3.0


@dataclass
class StiConfig:
    """Space-time image construction and the orientation analysis on it."""

    # --- search line ------------------------------------------------------- #
    #: Streamwise length of each search line, in metres.  Long enough to hold
    #: several surface features, short enough that the velocity is uniform
    #: along it.  A few metres suits most rivers.
    search_line_length_m: float = 4.0
    #: Samples along the search line.  ``None`` picks roughly one sample per
    #: image pixel, which is the natural resolution limit.
    n_samples: Optional[int] = None

    # --- STI preprocessing ------------------------------------------------- #
    #: Subtract the time-mean of each spatial column.  Removes static texture
    #: (rocks, glare, bank) that would otherwise draw a vertical -- i.e. zero
    #: velocity -- stripe through the STI.  Keep this on.
    remove_static_background: bool = True
    #: Normalise each STI to zero mean / unit variance.
    normalise: bool = True
    #: Contrast-limited adaptive histogram equalisation on the STI.
    clahe: bool = False
    #: Gaussian smoothing sigma (pixels) applied to the STI.  Section 2.3 of
    #: the paper shows this materially improves accuracy on smooth patterns.
    smooth_sigma: float = 0.0

    # --- structure tensor -------------------------------------------------- #
    #: Order of the central-difference gradient operator: 4 for Eq. (12),
    #: 6 for the six-point stencil of Eq. (13), or 2 for a plain difference.
    gradient_order: int = 4
    #: Side length of the local integration area A of Eqs. (7)-(9).  The paper
    #: uses 31 x 31 pixels with 50% overlap.
    window_px: int = 31
    #: "box" reproduces the paper's rectangular area A; "gaussian" uses a
    #: smoothly tapered window, which yields a better-conditioned tensor.
    window_type: str = "box"
    #: Overlap between neighbouring windows (the paper uses 0.5).
    window_overlap: float = 0.5

    # --- angle averaging (Eq. 14) ------------------------------------------ #
    #: Velocity search range, in m/s.  Sets phi_min and phi_max, which the
    #: paper uses to discard vectors corrupted by obstacles (Fig. 12).
    #: Allow a small negative bound so a genuinely still reach is not biased.
    velocity_range_ms: Sequence[float] = (-0.5, 8.0)
    #: Coherency thresholds C of Eq. (14); the cumulative histograms of
    #: orientation angle for C_i >= C are summed (Figs. 10 and 11).
    coherency_levels: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8)
    #: Bins in the orientation-angle histogram.
    n_angle_bins: int = 180
    #: Discard windows below this coherency before histogramming at all.
    min_coherency: float = 0.0
    #: "eq14" is the paper's weighted mean.  "peak" instead takes the
    #: parabolically interpolated mode of the summed histogram, which is more
    #: robust when a strong secondary peak (wind waves, glare) is present.
    estimator: str = "eq14"


@dataclass
class QualityConfig:
    """Thresholds used to flag, and optionally drop, unreliable stations."""

    #: Minimum fraction of windows falling inside [phi_min, phi_max].
    min_valid_fraction: float = 0.15
    #: Minimum coherency-weighted mean coherency for a station.
    min_mean_coherency: float = 0.10
    #: Maximum velocity uncertainty as a fraction of the station's velocity.
    #: This is usually the criterion that catches a search line straddling the
    #: bank: the angle histogram goes flat, so sigma blows up even though mean
    #: coherency may still scrape past its own threshold.  The comparison is
    #: made against the section's own velocity scale, so it transfers between a
    #: 0.3 m/s brook and a 4 m/s flood without retuning.
    max_relative_uncertainty: float = 0.5
    #: Flag stations outside the accuracy range demonstrated in the paper.
    tan_phi_range: Sequence[float] = (0.05, 8.0)
    #: Replace flagged stations by an interpolation of their neighbours.
    interpolate_flagged: bool = True
    #: Median-filter window (stations) applied across the profile; 0 disables.
    median_filter_stations: int = 0


@dataclass
class DischargeConfig:
    """Velocity-area discharge integration.

    Surface velocity is converted to depth-averaged velocity with the velocity
    index ``alpha`` (0.85 is the conventional default for the logarithmic
    profile), then integrated over the wetted cross-section by the mid-section
    method.
    """

    enabled: bool = False
    #: CSV with columns ``distance_m,depth_m`` measured along the section from
    #: ``section.start_px``.  Depths are interpolated onto the stations.
    bathymetry_csv: Optional[str] = None
    #: Or give the depths inline, one per station.
    depths_m: Optional[Sequence[float]] = None
    alpha: float = 0.85
    #: Water-surface elevation, if the bathymetry is given as bed elevation
    #: rather than depth.  When set, depth = water_level_m - elevation.
    water_level_m: Optional[float] = None
    #: Fraction of the near-bank velocity retained at the waterline.
    bank_coefficient: float = 0.7


@dataclass
class OutputConfig:
    out_dir: str = "stiv_output"
    save_sti_images: bool = True
    save_diagnostics: bool = True
    dpi: int = 150
    #: Write the per-window orientation/coherency fields to a .npz archive.
    save_fields: bool = False


@dataclass
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    rectify: RectifyConfig = field(default_factory=RectifyConfig)
    section: SectionConfig = field(default_factory=SectionConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    stabilise: StabiliseConfig = field(default_factory=StabiliseConfig)
    stiv: StiConfig = field(default_factory=StiConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    discharge: DischargeConfig = field(default_factory=DischargeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(raw)
        cfg.resolve_paths(path.parent)
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(
                f"unknown top-level config key(s): {sorted(unknown)}. "
                f"Expected one of {sorted(f.name for f in fields(cls))}"
            )
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            section_raw = raw.get(f.name)
            if section_raw is None:
                kwargs[f.name] = f.default_factory()  # type: ignore[misc]
                continue
            if not isinstance(section_raw, dict):
                raise ConfigError(f"config section '{f.name}' must be a mapping")
            sub_cls = f.default_factory  # type: ignore[misc]
            kwargs[f.name] = _build(sub_cls().__class__, section_raw, f.name)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def resolve_paths(self, base: Path) -> None:
        """Make relative paths in the config relative to the config file."""

        def _abs(p: Optional[str]) -> Optional[str]:
            if not p:
                return p
            q = Path(p)
            return str(q if q.is_absolute() else (base / q).resolve())

        self.video.path = _abs(self.video.path) or ""
        self.discharge.bathymetry_csv = _abs(self.discharge.bathymetry_csv)
        self.output.out_dir = _abs(self.output.out_dir) or "stiv_output"

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        v, r, s, st = self.video, self.rectify, self.section, self.stiv

        if not v.path:
            raise ConfigError("video.path is required")
        if v.frame_step < 1:
            raise ConfigError("video.frame_step must be >= 1")

        if r.method not in {"scale", "gcp"}:
            raise ConfigError("rectify.method must be 'scale' or 'gcp'")
        if r.method == "gcp":
            if not r.gcp_px or not r.gcp_world:
                raise ConfigError("rectify.method='gcp' needs gcp_px and gcp_world")
            if len(r.gcp_px) != len(r.gcp_world):
                raise ConfigError("gcp_px and gcp_world must have the same length")
            if len(r.gcp_px) < 4:
                raise ConfigError("a homography needs at least 4 ground control points")
        elif r.m_per_px is None and r.altitude_m is None:
            raise ConfigError(
                "rectify.method='scale' needs m_per_px, or altitude_m plus "
                "either hfov_deg or (sensor_width_mm and focal_length_mm)"
            )

        if s.start_px is None or s.end_px is None:
            raise ConfigError(
                "section.start_px and section.end_px are required "
                "(run `stiv pick <video>` to click them on a frame)"
            )
        if s.n_stations < 2:
            raise ConfigError("section.n_stations must be >= 2")
        if not 0.0 <= s.edge_trim < 0.5:
            raise ConfigError("section.edge_trim must be in [0, 0.5)")

        if self.flow.direction not in {"auto", "angle", "points"}:
            raise ConfigError("flow.direction must be 'auto', 'angle' or 'points'")
        if self.flow.direction == "angle" and self.flow.angle_deg is None:
            raise ConfigError("flow.direction='angle' needs flow.angle_deg")
        if self.flow.direction == "points" and (self.flow.from_px is None or self.flow.to_px is None):
            raise ConfigError("flow.direction='points' needs flow.from_px and flow.to_px")

        if st.search_line_length_m <= 0:
            raise ConfigError("stiv.search_line_length_m must be > 0")
        if st.window_px < 5 or st.window_px % 2 == 0:
            raise ConfigError("stiv.window_px must be an odd integer >= 5")
        if st.window_type not in {"box", "gaussian"}:
            raise ConfigError("stiv.window_type must be 'box' or 'gaussian'")
        if not 0.0 <= st.window_overlap < 1.0:
            raise ConfigError("stiv.window_overlap must be in [0, 1)")
        if st.gradient_order not in {2, 4, 6}:
            raise ConfigError("stiv.gradient_order must be 2, 4 or 6")
        if st.estimator not in {"eq14", "peak"}:
            raise ConfigError("stiv.estimator must be 'eq14' or 'peak'")
        lo, hi = st.velocity_range_ms
        if lo >= hi:
            raise ConfigError("stiv.velocity_range_ms must be [min, max] with min < max")
        if not st.coherency_levels:
            raise ConfigError("stiv.coherency_levels must not be empty")

        if self.stabilise.model not in {"similarity", "affine", "homography"}:
            raise ConfigError("stabilise.model must be 'similarity', 'affine' or 'homography'")

        d = self.discharge
        if d.enabled and d.bathymetry_csv is None and d.depths_m is None:
            raise ConfigError("discharge.enabled needs bathymetry_csv or depths_m")
        if d.enabled and not 0.0 < d.alpha <= 1.5:
            raise ConfigError("discharge.alpha should be a velocity index in (0, 1.5]")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    def dump_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build(cls: type, raw: dict[str, Any], name: str):
    valid = {f.name for f in fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(
            f"unknown key(s) in config section '{name}': {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
    return cls(**copy.deepcopy(raw))


def _asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(o) for o in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
