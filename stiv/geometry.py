"""Image <-> world mapping, and construction of the STIV search lines.

Two coordinate systems are used throughout the package:

image
    ``(u, v)`` = (column, row) in pixels, floating point, OpenCV convention,
    origin at the top-left of the frame.

world
    ``(X, Y)`` in metres on the horizontal water-surface plane.  The origin is
    arbitrary; only distances and directions matter.

A :class:`Rectifier` converts between the two.  All geometric reasoning --
station spacing, search-line length, the "perpendicular to the section"
flow direction -- happens in world coordinates, so an oblique view is handled
correctly rather than approximately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import Config, ConfigError, FlowConfig, RectifyConfig, SectionConfig

ArrayLike = Sequence[Sequence[float]] | np.ndarray


# --------------------------------------------------------------------------- #
# rectification
# --------------------------------------------------------------------------- #
class Rectifier:
    """Base class for image <-> world mappings on the water-surface plane."""

    #: Nominal ground sample distance, m/pixel, used for defaults and reports.
    nominal_m_per_px: float

    def image_to_world(self, pts_px: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def world_to_image(self, pts_world: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


class ScaleRectifier(Rectifier):
    """Uniform scaling: the nadir drone case.

    World coordinates are the image coordinates times a ground sample distance,
    with the row axis negated so that world Y increases upwards on screen and
    the frame is right-handed.  That keeps signed rotations (used for the
    "perpendicular to the section" flow direction) behaving conventionally.
    """

    def __init__(self, m_per_px: float, origin_px: Sequence[float] = (0.0, 0.0)):
        if not m_per_px > 0:
            raise ConfigError(f"m_per_px must be positive, got {m_per_px}")
        self.m_per_px = float(m_per_px)
        self.nominal_m_per_px = float(m_per_px)
        self.origin_px = np.asarray(origin_px, dtype=float)

    def image_to_world(self, pts_px: ArrayLike) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_px, dtype=float))
        d = p - self.origin_px
        return np.column_stack([d[:, 0] * self.m_per_px, -d[:, 1] * self.m_per_px])

    def world_to_image(self, pts_world: ArrayLike) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_world, dtype=float))
        return np.column_stack(
            [p[:, 0] / self.m_per_px + self.origin_px[0],
             -p[:, 1] / self.m_per_px + self.origin_px[1]]
        )

    def describe(self) -> str:
        return f"uniform scale, {self.m_per_px * 100:.3f} cm/pixel (nadir view assumed)"


#: Clicking accuracy a ground control point is expected to be good to, pixels.
_RANSAC_PX = 3.0


def _provisional_scale(gcp_px: ArrayLike, gcp_world: ArrayLike) -> float:
    """Rough metres-per-pixel from the control points, before any fit.

    Only used to express a pixel tolerance in the world units that
    :func:`cv2.findHomography` measures its residuals in.  The median over all
    point pairs is robust to a single bad point, which is the whole reason the
    tolerance is wanted.
    """
    p = np.atleast_2d(np.asarray(gcp_px, dtype=float))
    w = np.atleast_2d(np.asarray(gcp_world, dtype=float))
    ratios = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            d_px = float(np.linalg.norm(p[i] - p[j]))
            if d_px > 1e-9:
                ratios.append(float(np.linalg.norm(w[i] - w[j])) / d_px)
    if not ratios:
        raise ConfigError("the ground control points are coincident in the image")
    scale = float(np.median(ratios))
    if not scale > 0:
        raise ConfigError(
            "the ground control points are coincident in world coordinates; "
            "check that gcp_world is not all the same point"
        )
    return scale


class HomographyRectifier(Rectifier):
    """Plane-to-plane projective mapping fitted to ground control points.

    This is Eq. (15) of the paper restricted to the water-surface plane: with
    ``Z`` fixed at the water level the collinearity equations collapse to a
    3x3 homography, which can be fitted directly from >= 4 point pairs without
    knowing the camera pose or focal length.
    """

    def __init__(self, gcp_px: ArrayLike, gcp_world: ArrayLike, image_shape: tuple[int, int]):
        src = np.asarray(gcp_px, dtype=np.float64).reshape(-1, 1, 2)
        dst = np.asarray(gcp_world, dtype=np.float64).reshape(-1, 1, 2)
        if len(src) < 4:
            raise ConfigError("a homography needs at least 4 ground control points")

        # cv2.findHomography measures its RANSAC reprojection error in the
        # *destination* space -- which here is world metres, not pixels.  A bare
        # threshold of 3.0 therefore admits a three-metre blunder as an inlier,
        # i.e. disables the robust fit entirely at any realistic ground sample
        # distance.  Convert the intended three-pixel tolerance using a scale
        # taken from the control points themselves, by median over point pairs
        # so that one mis-surveyed point cannot inflate it.
        method = cv2.RANSAC if len(src) >= 5 else 0
        self.inlier_threshold_m = _RANSAC_PX * _provisional_scale(gcp_px, gcp_world)
        H, mask = cv2.findHomography(src, dst, method, self.inlier_threshold_m)
        if H is None:
            raise ConfigError(
                "could not fit a homography to the given ground control points; "
                "check that they are not collinear and that the image/world "
                "orders match"
            )
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.n_gcp = len(src)
        inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(src), bool)
        self.n_inliers = int(inliers.sum())

        # Residuals tell the user how good the survey/clicking was.  Every
        # point keeps its residual, so a rejected blunder still shows; the RMSE
        # is over the inliers alone, as the fit is, so the blunder it ignored
        # does not make a good fit look bad.
        pred = self.image_to_world(np.asarray(gcp_px, dtype=float))
        self.residuals_m = np.linalg.norm(pred - np.asarray(gcp_world, dtype=float), axis=1)
        self.rmse_m = float(np.sqrt(np.mean(self.residuals_m[inliers] ** 2)))

        h, w = image_shape[:2]
        self.nominal_m_per_px = self._local_scale(np.array([[w / 2.0, h / 2.0]]))[0]

    def image_to_world(self, pts_px: ArrayLike) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_px, dtype=np.float64)).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)

    def world_to_image(self, pts_world: ArrayLike) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_world, dtype=np.float64)).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H_inv).reshape(-1, 2)

    def _local_scale(self, pts_px: ArrayLike) -> np.ndarray:
        """Local ground sample distance (m/pixel) at the given image points."""
        p = np.atleast_2d(np.asarray(pts_px, dtype=float))
        a = self.image_to_world(p)
        b = self.image_to_world(p + np.array([1.0, 0.0]))
        c = self.image_to_world(p + np.array([0.0, 1.0]))
        return 0.5 * (np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - a, axis=1))

    def describe(self) -> str:
        return (
            f"homography from {self.n_gcp} GCPs ({self.n_inliers} inliers), "
            f"inlier RMSE {self.rmse_m:.3f} m, "
            f"~{self.nominal_m_per_px * 100:.2f} cm/pixel at frame centre"
        )


def build_rectifier(cfg: RectifyConfig, image_shape: tuple[int, int]) -> Rectifier:
    """Instantiate the rectifier described by ``cfg``."""
    h, w = image_shape[:2]
    if cfg.method == "gcp":
        return HomographyRectifier(cfg.gcp_px, cfg.gcp_world, image_shape)

    m_per_px = cfg.resolve_m_per_px(w)
    if m_per_px is None:
        raise ConfigError(
            "could not determine m_per_px: give rectify.m_per_px, or "
            "rectify.altitude_m together with hfov_deg or "
            "(sensor_width_mm and focal_length_mm)"
        )
    return ScaleRectifier(m_per_px, origin_px=(w / 2.0, h / 2.0))


# --------------------------------------------------------------------------- #
# cross-section and search lines
# --------------------------------------------------------------------------- #
@dataclass
class SearchLine:
    """One streamwise search line: the seed of a single space-time image.

    Attributes
    ----------
    index
        Station number along the cross-section, 0 at ``section.start_px``.
    station_m
        Distance along the cross-section from its start, in metres.
    centre_world, centre_px
        Where the line crosses the section.
    samples_px
        ``(n_samples, 2)`` image coordinates sampled uniformly *in world
        space*, so ``Sx`` is constant along the STI even in an oblique view.
    sx_m_per_px
        Spatial scale of the STI x-axis, metres per STI column.  This is the
        ``S_x`` of Eq. (16).
    """

    index: int
    station_m: float
    centre_world: np.ndarray
    centre_px: np.ndarray
    samples_px: np.ndarray
    samples_world: np.ndarray
    sx_m_per_px: float

    @property
    def n_samples(self) -> int:
        return len(self.samples_px)


@dataclass
class SectionGeometry:
    """Everything about where the measurement happens, in both frames."""

    start_world: np.ndarray
    end_world: np.ndarray
    start_px: np.ndarray
    end_px: np.ndarray
    width_m: float
    flow_dir_world: np.ndarray
    search_lines: list[SearchLine]
    rectifier: Rectifier
    #: Angle of the flow direction, degrees CCW from world +X.
    flow_bearing_deg: float

    @property
    def stations_m(self) -> np.ndarray:
        return np.array([sl.station_m for sl in self.search_lines])

    @property
    def station_spacing_m(self) -> float:
        """Distance between neighbouring stations; the width for a lone one."""
        x = self.stations_m
        return float(np.median(np.diff(x))) if len(x) > 1 else float(self.width_m)


@dataclass
class StationBox:
    """A flow-aligned rectangle of water surface centred on one station.

    LS-PIV samples a grid in it and PTV counts the tracks that pass through
    it.  ``along_world`` is the downstream direction the section was laid out
    with and ``across_world`` is perpendicular to it, so the box is square to
    the flow even where the flow crosses the section at an angle.
    """

    index: int
    centre_world: np.ndarray
    along_world: np.ndarray
    across_world: np.ndarray
    half_length_m: float
    half_width_m: float

    def corners_world(self) -> np.ndarray:
        """``(4, 2)`` corners, in order round the box."""
        a = self.along_world * self.half_length_m
        c = self.across_world * self.half_width_m
        return np.array([self.centre_world + s * a + t * c
                         for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    def contains_world(self, pts_world: ArrayLike) -> np.ndarray:
        """Which of ``(n, 2)`` world points lie inside the box."""
        d = np.asarray(pts_world, dtype=float).reshape(-1, 2) - self.centre_world
        return ((np.abs(d @ self.along_world) <= self.half_length_m)
                & (np.abs(d @ self.across_world) <= self.half_width_m))


def station_boxes(geom: SectionGeometry, length_m: float,
                  width_m: Optional[float] = None) -> list[StationBox]:
    """One :class:`StationBox` per station, ``length_m`` long down the flow.

    ``width_m`` defaults to the station spacing, so neighbouring boxes tile the
    section without overlapping.
    """
    along = _unit(np.asarray(geom.flow_dir_world, dtype=float))
    across = np.array([-along[1], along[0]])
    w = geom.station_spacing_m if width_m is None else float(width_m)
    return [
        StationBox(index=sl.index, centre_world=np.asarray(sl.centre_world, dtype=float),
                   along_world=along, across_world=across,
                   half_length_m=0.5 * float(length_m), half_width_m=0.5 * w)
        for sl in geom.search_lines
    ]


def finest_m_per_px(rect: Rectifier, centre_w: np.ndarray, along: np.ndarray,
                    across: np.ndarray, half_m: float) -> float:
    """Smallest metres-per-pixel anywhere along the two axes of a box."""
    c = np.asarray(centre_w, dtype=float)
    px_per_m = max(
        _max_px_per_m(rect, c - along * half_m, c + along * half_m),
        _max_px_per_m(rect, c - across * half_m, c + across * half_m),
    )
    return 1.0 / px_per_m if px_per_m > 0 else math.inf


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ConfigError("degenerate direction vector (zero length)")
    return v / n


def resolve_flow_direction(
    cfg: FlowConfig,
    section_dir_world: np.ndarray,
    section_centre_world: np.ndarray,
    rectifier: Rectifier,
) -> np.ndarray:
    """Return the unit downstream direction in world coordinates."""
    if cfg.direction == "angle":
        a = math.radians(float(cfg.angle_deg))
        d = np.array([math.cos(a), math.sin(a)])
    elif cfg.direction == "points":
        p = rectifier.image_to_world([cfg.from_px, cfg.to_px])
        d = _unit(p[1] - p[0])
    else:  # auto: perpendicular to the section, in world space
        d = np.array([-section_dir_world[1], section_dir_world[0]])
        if cfg.downstream_px is not None:
            # Point the normal at the user's downstream marker.
            hint = rectifier.image_to_world([cfg.downstream_px])[0]
            if np.dot(hint - section_centre_world, d) < 0:
                d = -d

    d = _unit(np.asarray(d, dtype=float))
    if cfg.reverse:
        d = -d
    return d


#: Points along a search line at which the local image scale is evaluated when
#: choosing how many samples it needs.  A homography's scale along a line is
#: monotonic, so the endpoints alone would do; the interior points cost nothing
#: and keep the rule honest for any other rectifier.
_SCALE_PROBES = 33
#: Relative difference between the largest and the mean image scale along a
#: line below which the scale is taken to be uniform (a nadir view).
_UNIFORM_SCALE_RTOL = 1e-6


def _max_px_per_m(rect: Rectifier, a_w: np.ndarray, b_w: np.ndarray) -> float:
    """Largest image distance, pixels per metre, anywhere along a world segment."""
    d = b_w - a_w
    length = float(np.linalg.norm(d))
    if length <= 0:
        return 0.0
    u = d / length
    h = 1e-3 * length
    s = np.linspace(0.0, length, _SCALE_PROBES)
    centres = a_w[None, :] + s[:, None] * u[None, :]
    fwd = rect.world_to_image(centres + h * u[None, :])
    back = rect.world_to_image(centres - h * u[None, :])
    return float(np.max(np.linalg.norm(fwd - back, axis=1)) / (2.0 * h))


def default_n_samples(rect: Rectifier, a_w: np.ndarray, b_w: np.ndarray) -> int:
    """Samples for a search line from ``a_w`` to ``b_w`` when none are configured.

    One STI column per image pixel is the natural resolution limit -- *where
    the image is finest along the line*.  In an oblique view the near end of a
    line covers more pixels per metre than the far end, so counting pixels
    end to end (the mean spacing) leaves the near field undersampled: 1.25 px
    per column in the test geometry, which aliases the finest texture there.
    So the count follows the largest local pixels-per-metre along the line,
    rounded as before; in a nadir view that is the same number as the mean.
    """
    length = float(np.linalg.norm(b_w - a_w))
    ab_px = rect.world_to_image([a_w, b_w])
    span_mean = float(np.linalg.norm(ab_px[1] - ab_px[0]))
    span_max = length * _max_px_per_m(rect, a_w, b_w)
    # Uniform scale gives the same span both ways up to rounding error, and
    # that error must not flip a count that sits on a half: nadir counts are
    # pinned to the original rule.
    span = span_max if span_max > span_mean * (1.0 + _UNIFORM_SCALE_RTOL) else span_mean
    return int(np.clip(int(round(span)) + 1, 16, 4096))


def build_section(cfg: Config, image_shape: tuple[int, int]) -> SectionGeometry:
    """Lay out the cross-section and its search lines from the config."""
    rect = build_rectifier(cfg.rectify, image_shape)
    sec: SectionConfig = cfg.section

    p_world = rect.image_to_world([sec.start_px, sec.end_px])
    start_w, end_w = p_world[0], p_world[1]
    span = end_w - start_w
    width_m = float(np.linalg.norm(span))
    if width_m <= 0:
        raise ConfigError("section.start_px and section.end_px map to the same world point")
    sec_dir = span / width_m

    section_centre_w = 0.5 * (start_w + end_w)
    flow_dir = resolve_flow_direction(cfg.flow, sec_dir, section_centre_w, rect)

    # Station positions along the section, evenly spaced in world space.
    t0, t1 = sec.edge_trim, 1.0 - sec.edge_trim
    fracs = np.linspace(t0, t1, sec.n_stations)

    length_m = float(cfg.stiv.search_line_length_m)
    half = 0.5 * length_m

    lines: list[SearchLine] = []
    for i, f in enumerate(fracs):
        centre_w = start_w + span * f
        a_w = centre_w - flow_dir * half
        b_w = centre_w + flow_dir * half

        n = cfg.stiv.n_samples
        if n is None:
            n = default_n_samples(rect, a_w, b_w)
        if n < 16:
            raise ConfigError(f"stiv.n_samples must be >= 16, got {n}")

        s = np.linspace(0.0, 1.0, n)[:, None]
        samples_w = a_w[None, :] * (1.0 - s) + b_w[None, :] * s
        samples_px = rect.world_to_image(samples_w)

        lines.append(
            SearchLine(
                index=i,
                station_m=float(f * width_m),
                centre_world=centre_w,
                centre_px=rect.world_to_image([centre_w])[0],
                samples_px=samples_px,
                samples_world=samples_w,
                sx_m_per_px=length_m / (n - 1),
            )
        )

    return SectionGeometry(
        start_world=start_w,
        end_world=end_w,
        start_px=np.asarray(sec.start_px, dtype=float),
        end_px=np.asarray(sec.end_px, dtype=float),
        width_m=width_m,
        flow_dir_world=flow_dir,
        search_lines=lines,
        rectifier=rect,
        flow_bearing_deg=float(math.degrees(math.atan2(flow_dir[1], flow_dir[0]))),
    )


def check_lines_in_frame(
    geom: SectionGeometry, image_shape: tuple[int, int], margin: int = 2
) -> list[int]:
    """Return the indices of search lines that leave the frame."""
    h, w = image_shape[:2]
    bad = []
    for sl in geom.search_lines:
        u, v = sl.samples_px[:, 0], sl.samples_px[:, 1]
        if (u < margin).any() or (u > w - 1 - margin).any() or \
           (v < margin).any() or (v > h - 1 - margin).any():
            bad.append(sl.index)
    return bad
