"""Configuration schema for a STIV run.

Everything the analysis needs is described by :class:`Config`, which is normally
loaded from a YAML file.  Nested dataclasses mirror the YAML structure, so a
config file reads like the analysis it describes::

    video:      {path: flight.mp4, start_s: 0.0, duration_s: 30.0}
    rectify:    {method: scale, m_per_px: 0.021}
    section:    {start_px: [310, 880], end_px: [1640, 830], n_stations: 40}
    flow:       {direction: auto}
    stiv:       {search_line_length_m: 4.0, window_px: 31}

:func:`survey_dict` is the one place a *new* survey's raw config is put
together, so ``stiv init``, ``stiv pick`` and the notebook wizard all start from
the same schema defaults.
"""

from __future__ import annotations

import copy
import datetime
import math
import numbers
import re
import typing
from collections.abc import Sequence as _AbcSequence
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
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
    #: Desktop only -- the iOS port has no CLAHE, so a survey.yaml exported
    #: from the phone never sets it, and setting it here makes a desktop run
    #: diverge from what the phone would have reported.
    clahe: bool = False
    #: Gaussian smoothing sigma (pixels) applied to the STI.  Section 2.3 of
    #: the paper shows this materially improves accuracy on smooth patterns,
    #: and it is also what keeps the gradient below the Nyquist limit of the
    #: surface texture: a feature moving `tan phi` columns per row aliases once
    #: `tan phi` approaches a third of the finest texture wavelength, which
    #: biases the velocity *low* by tens of percent with no drop in coherency
    #: to warn you.  1.5 px suppresses that while costing nothing when the
    #: pattern is already well sampled; 0.0 disables smoothing entirely.
    smooth_sigma: float = 1.5

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
        path = Path(path).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        try:
            raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader)
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse {path}{_yaml_error_detail(exc)}") from exc
        except UnicodeDecodeError as exc:
            raise ConfigError(f"{path} is not a UTF-8 text file") from exc
        # Paths are resolved against the config file *before* validating, so the
        # up-front checks (the bathymetry file exists, say) look where the run
        # itself will look.
        return cls.from_dict(raw if raw is not None else {}, base=path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base: Optional[str | Path] = None) -> "Config":
        """Build and validate a config from its raw mapping.

        Relative paths are resolved against ``base`` when it is given (that is
        what :meth:`from_yaml` passes: the config file's folder), and otherwise
        left relative to the working directory.
        """
        if not isinstance(raw, dict):
            raise ConfigError(
                f"a config must be a mapping of sections ({', '.join(f.name for f in fields(cls))}), "
                f"not {type(raw).__name__}"
            )
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
        cfg.normalise_types()
        cfg.resolve_paths(Path(base) if base is not None else None)
        cfg.validate()
        return cfg

    def resolve_paths(self, base: Optional[Path] = None) -> None:
        """Expand ``~`` and make relative paths relative to ``base``.

        ``base`` is the config file's folder for a config read from YAML; with
        no ``base`` only the ``~`` is expanded.
        """

        def _abs(p: Optional[str]) -> Optional[str]:
            if not p:
                return p
            q = Path(p).expanduser()
            if base is None or q.is_absolute():
                return str(q)
            return str((base / q).resolve())

        self.video.path = _abs(self.video.path) or ""
        self.discharge.bathymetry_csv = _abs(self.discharge.bathymetry_csv)
        self.output.out_dir = _abs(self.output.out_dir) or "stiv_output"

    def normalise_types(self) -> None:
        """Coerce every field to the type the schema declares, or say why not.

        YAML is loosely typed: PyYAML reads ``15e-1`` and ``6e1`` as strings
        (a YAML 1.1 float needs a decimal point), ``30.0`` where a count is
        meant is a float, and nothing stops a list turning up where a number
        should be.  Left alone, those surface as a ``TypeError`` from deep in
        the analysis -- or, worse, not at all.
        """
        for sec in fields(self):
            obj = getattr(self, sec.name)
            hints = _type_hints(type(obj))
            for f in fields(obj):
                where = f"{sec.name}.{f.name}"
                value = getattr(obj, f.name)
                setattr(obj, f.name, _coerce(value, hints[f.name], where, _default_of(f)))

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        self.normalise_types()
        v, r, s, st, q = self.video, self.rectify, self.section, self.stiv, self.quality

        if not v.path:
            raise ConfigError("video.path is required")
        if v.frame_step < 1:
            raise ConfigError("video.frame_step must be >= 1")
        # Both of these are silently ignored by the frame reader when zero,
        # which is the opposite of what anyone writing a zero would expect.
        if v.max_frames is not None and v.max_frames < 2:
            raise ConfigError(
                "video.max_frames must be >= 2 (STIV needs at least two frames), "
                "or null for no limit"
            )
        if v.fps_override is not None and not v.fps_override > 0:
            raise ConfigError(
                "video.fps_override must be > 0, or null to use the frame rate the "
                "video file reports"
            )
        if v.channel_weights is not None:
            _check_len(v.channel_weights, 3, "video.channel_weights", "3 numbers, in BGR order")

        if r.method not in {"scale", "gcp"}:
            raise ConfigError("rectify.method must be 'scale' or 'gcp'")
        if r.method == "gcp":
            if not r.gcp_px or not r.gcp_world:
                raise ConfigError(
                    "rectify.method='gcp' needs gcp_px and gcp_world: fill in the "
                    "image position of each control point and its surveyed world "
                    "position in metres, in the same order"
                )
            if len(r.gcp_px) != len(r.gcp_world):
                raise ConfigError("gcp_px and gcp_world must have the same length")
            if len(r.gcp_px) < 4:
                raise ConfigError("a homography needs at least 4 ground control points")
            for i, (p, w) in enumerate(zip(r.gcp_px, r.gcp_world)):
                _check_len(p, 2, f"rectify.gcp_px[{i}]")
                _check_len(w, 2, f"rectify.gcp_world[{i}]")
        elif r.m_per_px is not None:
            if not r.m_per_px > 0:
                raise ConfigError("rectify.m_per_px must be > 0")
        elif r.altitude_m is None:
            raise ConfigError(
                "rectify.m_per_px is not set: fill in the ground sample distance "
                "(metres per pixel), or give altitude_m plus either hfov_deg or "
                "(sensor_width_mm and focal_length_mm)"
            )
        else:
            if not r.altitude_m > 0:
                raise ConfigError("rectify.altitude_m must be > 0")
            has_lens = r.sensor_width_mm is not None and r.focal_length_mm is not None
            if r.hfov_deg is None and not has_lens:
                raise ConfigError(
                    "rectify.altitude_m alone does not give a scale: add the camera's "
                    "hfov_deg, or sensor_width_mm and focal_length_mm"
                )
            if r.hfov_deg is not None and not 0.0 < r.hfov_deg < 180.0:
                raise ConfigError("rectify.hfov_deg must be in (0, 180) degrees")

        if s.start_px is None or s.end_px is None:
            raise ConfigError(
                "section.start_px and section.end_px are not set: fill in the two "
                "ends of the cross-section in image pixels "
                "(or run `stiv pick <video>` to click them on a frame)"
            )
        _check_len(s.start_px, 2, "section.start_px")
        _check_len(s.end_px, 2, "section.end_px")
        if s.n_stations < 2:
            raise ConfigError("section.n_stations must be >= 2")
        if not 0.0 <= s.edge_trim < 0.5:
            raise ConfigError("section.edge_trim must be in [0, 0.5)")

        fl = self.flow
        if fl.direction not in {"auto", "angle", "points"}:
            raise ConfigError("flow.direction must be 'auto', 'angle' or 'points'")
        if fl.direction == "angle" and fl.angle_deg is None:
            raise ConfigError("flow.direction='angle' needs flow.angle_deg")
        if fl.direction == "points" and (fl.from_px is None or fl.to_px is None):
            raise ConfigError("flow.direction='points' needs flow.from_px and flow.to_px")
        for name in ("from_px", "to_px", "downstream_px"):
            if getattr(fl, name) is not None:
                _check_len(getattr(fl, name), 2, f"flow.{name}")

        if st.search_line_length_m <= 0:
            raise ConfigError("stiv.search_line_length_m must be > 0")
        if st.n_samples is not None and st.n_samples < 16:
            raise ConfigError("stiv.n_samples must be >= 16, or null for one per image pixel")
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
        _check_len(st.velocity_range_ms, 2, "stiv.velocity_range_ms", "[min, max]")
        lo, hi = st.velocity_range_ms
        if lo >= hi:
            raise ConfigError("stiv.velocity_range_ms must be [min, max] with min < max")
        if not st.coherency_levels:
            raise ConfigError("stiv.coherency_levels must not be empty")
        if st.n_angle_bins < 2:
            raise ConfigError("stiv.n_angle_bins must be >= 2")

        _check_len(q.tan_phi_range, 2, "quality.tan_phi_range", "[min, max]")
        if not 0.0 <= q.tan_phi_range[0] < q.tan_phi_range[1]:
            raise ConfigError("quality.tan_phi_range must be [min, max] with 0 <= min < max")
        # An even window is accepted, as it always has been: the filter is
        # centred, so it widens it to the next odd number and says so in the log.
        if q.median_filter_stations < 0:
            raise ConfigError(
                "quality.median_filter_stations must be a number of stations, "
                "or 0 to disable the filter"
            )

        stab = self.stabilise
        if stab.model not in {"similarity", "affine", "homography"}:
            raise ConfigError("stabilise.model must be 'similarity', 'affine' or 'homography'")
        for name in ("static_mask_px", "water_mask_px"):
            for i, poly in enumerate(getattr(stab, name) or []):
                if len(poly) < 3:
                    raise ConfigError(
                        f"stabilise.{name}[{i}] must be a polygon of at least 3 [x, y] points"
                    )
                for j, p in enumerate(poly):
                    _check_len(p, 2, f"stabilise.{name}[{i}][{j}]")

        d = self.discharge
        if d.enabled:
            if d.depths_m is not None:
                # Checked here rather than after the video has been processed,
                # which is the first point the analysis itself would notice.
                if len(d.depths_m) != s.n_stations:
                    raise ConfigError(
                        f"discharge.depths_m has {len(d.depths_m)} entries but "
                        f"section.n_stations is {s.n_stations}; give one depth per "
                        f"station, or use bathymetry_csv to interpolate instead"
                    )
            elif d.bathymetry_csv is None:
                raise ConfigError("discharge.enabled needs bathymetry_csv or depths_m")
            else:
                _check_bathymetry(d.bathymetry_csv, d.water_level_m)
        if d.enabled and not 0.0 < d.alpha <= 1.5:
            raise ConfigError("discharge.alpha should be a velocity index in (0, 1.5]")
        if d.enabled and not 0.0 < d.bank_coefficient <= 1.0:
            raise ConfigError(
                "discharge.bank_coefficient is the fraction of the near-bank "
                "velocity retained at the waterline, so it must be in (0, 1]"
            )
        if st.smooth_sigma < 0.0:
            raise ConfigError("stiv.smooth_sigma must be >= 0")
        if not 0.0 <= st.min_coherency <= 1.0:
            raise ConfigError("stiv.min_coherency must be in [0, 1]")
        if any(not 0.0 <= c <= 1.0 for c in st.coherency_levels):
            raise ConfigError("stiv.coherency_levels must all be in [0, 1]")

        if self.output.dpi < 1:
            raise ConfigError("output.dpi must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    def dump_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml_text(self.to_dict()), encoding="utf-8")


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


class _ConfigLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` without two YAML 1.1 surprises.

    Base-60 numbers: ``name: 12:30`` would load as the integer 750, and
    ``start_s: 1:30`` as 90.  And timestamps: ``name: 2024-05-01`` would load
    as a ``datetime.date``.  No config field is a date or a sexagesimal number,
    so both are left as the text that was typed, and a numeric field given one
    is refused with a message rather than silently converted.
    """


#: PyYAML's int and float patterns (``yaml.resolver.Resolver``), minus the
#: sexagesimal alternatives.
_NO_BASE60 = {
    "tag:yaml.org,2002:int": re.compile(
        r"""^(?:[-+]?0b[0-1_]+
            |[-+]?0[0-7_]+
            |[-+]?(?:0|[1-9][0-9_]*)
            |[-+]?0x[0-9a-fA-F_]+)$""",
        re.X,
    ),
    "tag:yaml.org,2002:float": re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
            |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
}
# Rebuilt in place, so the resolvers keep PyYAML's order of precedence.
_ConfigLoader.yaml_implicit_resolvers = {
    first: [(tag, _NO_BASE60.get(tag, rx)) for tag, rx in resolvers
            if tag != "tag:yaml.org,2002:timestamp"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _yaml_error_detail(exc: yaml.YAMLError) -> str:
    """`` at line L, column C: problem`` for a YAML syntax error."""
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
    problem = getattr(exc, "problem", None) or str(exc)
    return f"{where}: {problem}"


_HINTS: dict[type, dict[str, Any]] = {}


def _type_hints(cls: type) -> dict[str, Any]:
    # The module uses postponed annotations, so field types are strings until
    # resolved; resolving is slow enough to be worth doing once per class.
    if cls not in _HINTS:
        _HINTS[cls] = typing.get_type_hints(cls)
    return _HINTS[cls]


def _default_of(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return MISSING


def _coerce(value: Any, tp: Any, where: str, default: Any = MISSING) -> Any:
    """Convert ``value`` to the schema type ``tp``, or raise a ConfigError."""
    origin = typing.get_origin(tp)
    if origin is Union:  # Optional[X]
        if value is None:
            return None
        inner = [a for a in typing.get_args(tp) if a is not type(None)]
        return _coerce(value, inner[0], where)

    if value is None:
        hint = (
            f"; delete the line to use the default ({_asdict(default)!r})"
            if default is not MISSING and default is not None else ""
        )
        raise ConfigError(f"{where} is empty (null): fill it in{hint}")

    if origin in (_AbcSequence, list, tuple):
        # Code building a config often holds its points as numpy arrays.
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (str, bytes)) or not isinstance(value, _AbcSequence):
            raise ConfigError(f"{where} must be a list, got {value!r}")
        (item_tp,) = typing.get_args(tp)[:1] or (Any,)
        return [_coerce(v, item_tp, f"{where}[{i}]") for i, v in enumerate(value)]

    if tp is bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        raise ConfigError(f"{where} must be true or false, got {value!r}")

    if tp is int:
        x = value
        if isinstance(x, str):
            x = _parse_number(x, where, "a whole number")
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise ConfigError(f"{where} must be a whole number, got {value!r}")
        if isinstance(x, numbers.Integral):
            return int(x)
        if math.isfinite(x) and float(x).is_integer():
            return int(x)
        raise ConfigError(f"{where} must be a whole number, got {value!r}")

    if tp is float:
        x = _parse_number(value, where) if isinstance(value, str) else value
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise ConfigError(f"{where} must be a number, got {value!r}")
        x = float(x)
        if math.isnan(x):
            raise ConfigError(f"{where} must be a number, got NaN")
        return x

    if tp is str:
        if isinstance(value, str):
            return value
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return str(value)
        # A date typed into a name, from a mapping loaded by plain yaml.safe_load.
        if isinstance(value, (datetime.date, datetime.time)):
            return value.isoformat()
        raise ConfigError(f"{where} must be text, got {value!r}")

    return value


def _parse_number(text: str, where: str, kind: str = "a number") -> float:
    try:
        return float(text.strip())
    except ValueError:
        raise ConfigError(f"{where} must be {kind}, got {text!r}") from None


def _check_len(seq: Sequence, n: int, where: str, what: Optional[str] = None) -> None:
    if len(seq) != n:
        shape = what or ("[x, y]" if n == 2 else f"{n} numbers")
        raise ConfigError(f"{where} must be {shape}, got {len(seq)} value(s): {list(seq)!r}")


def _check_bathymetry(path: str, water_level_m: Optional[float]) -> None:
    """Read the bathymetry file now, so a bad one fails before the video is read."""
    if not Path(path).is_file():
        raise ConfigError(f"discharge.bathymetry_csv not found: {path}")
    from .discharge import load_bathymetry  # deferred: discharge imports this module

    try:
        dist, _ = load_bathymetry(path, water_level_m)
    except ConfigError:
        raise
    except Exception as exc:  # pandas' parse errors are not ConfigErrors
        raise ConfigError(f"could not read discharge.bathymetry_csv {path}: {exc}") from exc
    if len(dist) < 2:
        raise ConfigError(f"discharge.bathymetry_csv {path} needs at least two rows")


def _asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(o) for o in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# writing survey files
# --------------------------------------------------------------------------- #
class _SurveyDumper(yaml.SafeDumper):
    """Writes a point as ``[310.0, 880.0]`` rather than two lines of dashes."""


def _represent_list(dumper: yaml.SafeDumper, data: list):
    flat = all(not isinstance(v, (list, tuple, dict)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flat)


_SurveyDumper.add_representer(list, _represent_list)
_SurveyDumper.add_representer(tuple, _represent_list)


def yaml_text(raw: dict[str, Any]) -> str:
    """Render a raw config mapping as YAML, keeping the schema's key order."""
    return yaml.dump(raw, Dumper=_SurveyDumper, sort_keys=False, default_flow_style=False)


def path_for_config(target: str | Path, config_path: str | Path) -> str:
    """How to spell ``target`` inside a config file saved at ``config_path``.

    :meth:`Config.from_yaml` resolves a relative path against the config
    file's folder, not the shell's working directory, so writing a path the way
    it was typed (``footage/clip.mp4`` from ``-o configs/survey.yaml``) points
    somewhere else entirely.  A target inside the config's folder is written
    relative to it, so the two can be moved together; anything else is
    written absolute.
    """
    t = Path(target).expanduser().resolve()
    base = Path(config_path).expanduser().resolve().parent
    try:
        return t.relative_to(base).as_posix()
    except ValueError:
        return str(t)


#: The paper analysed 30 s of footage; 20-40 s of steady hover is plenty.
DEFAULT_DURATION_S = 30.0

#: Surface speeds, m/s, at which a ``frame_step`` suggestion is tabulated.
FRAME_STEP_SPEEDS_MS = (0.5, 1.0, 2.0, 3.0)


def default_duration_s(video_duration_s: Optional[float], start_s: float = 0.0) -> float:
    """A sensible ``video.duration_s`` for a new survey of this clip."""
    if video_duration_s and video_duration_s > 0:
        remaining = video_duration_s - start_s
        if remaining > 0:
            return round(min(DEFAULT_DURATION_S, remaining), 3)
    return DEFAULT_DURATION_S


def suggest_frame_step(m_per_px: float, fps: float, velocity_ms: float) -> int:
    """The ``video.frame_step`` that puts ``tan phi`` near 1 at ``velocity_ms``.

    ``tan phi = U * St / Sx`` with ``St = frame_step / fps`` and ``Sx`` about
    one image pixel, so ``tan phi = 1`` at ``frame_step = m_per_px * fps / U``.
    """
    return max(1, round(m_per_px * fps / velocity_ms))


def _point(p: Optional[Point]) -> Optional[list]:
    if p is None:
        return None
    return [None if v is None else round(float(v), 1) for v in p]


def survey_dict(
    video_path: str,
    *,
    start_s: float = VideoConfig.start_s,
    duration_s: Optional[float] = VideoConfig.duration_s,
    frame_step: int = VideoConfig.frame_step,
    fps_override: Optional[float] = VideoConfig.fps_override,
    m_per_px: Optional[float] = None,
    altitude_m: Optional[float] = None,
    hfov_deg: Optional[float] = None,
    gcp_px: Optional[Sequence[Point]] = None,
    gcp_world: Optional[Sequence[Point]] = None,
    start_px: Optional[Point] = None,
    end_px: Optional[Point] = None,
    n_stations: int = SectionConfig.n_stations,
    edge_trim: float = SectionConfig.edge_trim,
    name: str = SectionConfig.name,
    downstream_px: Optional[Point] = None,
    stabilise: bool = StabiliseConfig.enabled,
    water_mask_px: Optional[Sequence[Sequence[Point]]] = None,
    search_line_length_m: float = StiConfig.search_line_length_m,
    window_px: int = StiConfig.window_px,
    velocity_range_ms: Sequence[float] = StiConfig.velocity_range_ms,
    bathymetry_csv: Optional[str] = None,
    alpha: float = DischargeConfig.alpha,
    bank_coefficient: float = DischargeConfig.bank_coefficient,
    out_dir: str = OutputConfig.out_dir,
) -> dict[str, Any]:
    """The raw config of a new survey, with every default taken from the schema.

    ``stiv init``, ``stiv pick`` and the notebook wizard all build their
    config here, so a survey started from any of them analyses the same way.
    Whatever the caller does not know yet is written as ``null`` -- the scale
    and the section ends above all -- and :meth:`Config.validate` refuses to
    run until it has been filled in, rather than running on a made-up value.

    The scale comes from ``gcp_px``/``gcp_world`` if given, else from
    ``altitude_m`` and ``hfov_deg``, else from ``m_per_px``.
    """
    if gcp_px is not None:
        rectify: dict[str, Any] = {
            "method": "gcp",
            "gcp_px": [_point(p) for p in gcp_px],
            "gcp_world": [_point(p) for p in gcp_world]
            if gcp_world is not None else [[None, None] for _ in gcp_px],
        }
    elif altitude_m is not None:
        rectify = {"method": "scale", "altitude_m": altitude_m, "hfov_deg": hfov_deg}
    else:
        rectify = {"method": "scale", "m_per_px": m_per_px}

    stab: dict[str, Any] = {"enabled": bool(stabilise)}
    if water_mask_px:
        stab["water_mask_px"] = [[_point(p) for p in poly] for poly in water_mask_px]

    video: dict[str, Any] = {
        "path": str(video_path),
        "start_s": float(start_s),
        "duration_s": duration_s,
        "frame_step": int(frame_step),
    }
    if fps_override is not None:
        video["fps_override"] = float(fps_override)

    return {
        "video": video,
        "rectify": rectify,
        "section": {
            "start_px": _point(start_px),
            "end_px": _point(end_px),
            "n_stations": int(n_stations),
            "edge_trim": float(edge_trim),
            "name": name,
        },
        "flow": {"direction": "auto", "downstream_px": _point(downstream_px)},
        "stabilise": stab,
        "stiv": {
            "search_line_length_m": float(search_line_length_m),
            "window_px": int(window_px),
            "velocity_range_ms": [float(v) for v in velocity_range_ms],
        },
        "discharge": {
            "enabled": bathymetry_csv is not None,
            "bathymetry_csv": bathymetry_csv,
            "alpha": float(alpha),
            "bank_coefficient": float(bank_coefficient),
        },
        "output": {"out_dir": str(out_dir)},
    }
