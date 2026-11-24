"""Particle tracking velocimetry (PTV) at the stations of a section, KLT style.

Corners on the water -- foam, leaves, debris, the crests of boils -- are found
with ``cv2.goodFeaturesToTrack`` inside the stations' boxes and followed from
frame to frame by pyramidal Lucas-Kanade optical flow (the KLT tracker, as in
KLT-IV), each step checked by tracking back.  After ``ptv.track_frames``
frames a track ends: its two ends are carried back into the reference frame
through the inverse of the stabilisation transform, which removes the camera's
own motion, mapped to the water surface by the rectifier, and the distance
between them over the time between gives one velocity.  New features are then
detected and the next tracks begin.

Each station's velocity is the median of the tracks whose midpoints fall in its
box, after dropping tracks that are too fast for ``stiv.velocity_range_ms``,
run too far across the flow axis, or barely move -- static texture such as
glare or a rock.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import Config, PtvConfig
from .geometry import Rectifier, StationBox
from .stabilise import _to_u8, _u8_scaling, track_forward_backward, transform_points
from .vectors import VectorEstimate

log = logging.getLogger(__name__)

#: Pixels round the stations' boxes that the tracker works in, so a feature
#: near a box edge can be followed out of it.
_ROI_PAD_PX = 48


@dataclass
class TrackSet:
    """Every finished track, in the reference frame."""

    start_px: np.ndarray       # (n, 2) reference-frame image pixels
    end_px: np.ndarray         # (n, 2)
    start_world: np.ndarray    # (n, 2) metres
    end_world: np.ndarray      # (n, 2)
    duration_s: float

    def __len__(self) -> int:
        return len(self.start_px)

    @property
    def velocity_world(self) -> np.ndarray:
        return (self.end_world - self.start_world) / self.duration_s

    @property
    def mid_world(self) -> np.ndarray:
        return 0.5 * (self.start_world + self.end_world)

    def subset(self, keep: np.ndarray) -> "TrackSet":
        return TrackSet(self.start_px[keep], self.end_px[keep], self.start_world[keep],
                        self.end_world[keep], self.duration_s)


class TrackCollector:
    """Detects and tracks features inside every station box of a survey."""

    def __init__(self, boxes: Sequence[StationBox], rectifier: Rectifier,
                 st_s_per_px: float, cfg: PtvConfig, image_shape: tuple[int, int]):
        self.cfg = cfg
        self.rect = rectifier
        self.st_s_per_px = float(st_s_per_px)
        self.shape = tuple(image_shape[:2])
        self.polys_px = [rectifier.world_to_image(b.corners_world()) for b in boxes]
        self.all_pts = np.vstack(self.polys_px) if self.polys_px else np.empty((0, 2))
        self.n_rows = 0
        self._u8 = None
        self._prev: Optional[np.ndarray] = None
        self._prev_M = np.eye(3)
        self._pts = np.empty((0, 2), np.float32)
        self._start_ref = np.empty((0, 2))
        self._age = 0
        self._starts: list[np.ndarray] = []
        self._ends: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def _roi(self, *Ms: np.ndarray) -> tuple[int, int, int, int]:
        """Pixel bounds ``(x0, y0, x1, y1)`` covering the boxes under every ``M``."""
        h, w = self.shape
        pts = np.vstack([transform_points(M, self.all_pts) for M in Ms])
        x0 = int(max(0, math.floor(pts[:, 0].min()) - _ROI_PAD_PX))
        y0 = int(max(0, math.floor(pts[:, 1].min()) - _ROI_PAD_PX))
        x1 = int(min(w, math.ceil(pts[:, 0].max()) + _ROI_PAD_PX + 1))
        y1 = int(min(h, math.ceil(pts[:, 1].max()) + _ROI_PAD_PX + 1))
        return x0, y0, x1, y1

    def _detect(self, cur: np.ndarray, M: np.ndarray) -> None:
        h, w = self.shape
        mask = np.zeros((h, w), np.uint8)
        for poly in self.polys_px:
            q = transform_points(M, poly)
            cv2.fillPoly(mask, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], 255)
        found = cv2.goodFeaturesToTrack(
            cur, maxCorners=int(self.cfg.max_features), qualityLevel=self.cfg.quality_level,
            minDistance=int(self.cfg.min_distance_px), mask=mask, blockSize=7,
        )
        if found is None:
            self._pts = np.empty((0, 2), np.float32)
            self._start_ref = np.empty((0, 2))
        else:
            self._pts = found.reshape(-1, 2).astype(np.float32)
            self._start_ref = transform_points(np.linalg.inv(M), self._pts)
        self._age = 0

    def add_frame(self, grey: np.ndarray, transform: Optional[np.ndarray] = None) -> None:
        if not self.polys_px:
            self.n_rows += 1
            return
        M = np.eye(3) if transform is None else np.asarray(transform, dtype=float)
        if self._u8 is None and self._prev is None:
            self._u8 = _u8_scaling(grey)
        cur = _to_u8(grey, self._u8)

        if self._prev is not None and len(self._pts):
            x0, y0, x1, y1 = self._roi(self._prev_M, M)
            off = np.array([x0, y0], np.float32)
            p = self._pts - off
            nxt, ok = track_forward_backward(
                np.ascontiguousarray(self._prev[y0:y1, x0:x1]),
                np.ascontiguousarray(cur[y0:y1, x0:x1]),
                p, p, self.cfg.fb_error_px,
            )
            self._pts = (nxt[ok] + off).astype(np.float32)
            self._start_ref = self._start_ref[ok]
            self._age += 1
            if self._age >= self.cfg.track_frames:
                if len(self._pts):
                    self._starts.append(self._start_ref)
                    self._ends.append(transform_points(np.linalg.inv(M), self._pts))
                self._pts = np.empty((0, 2), np.float32)

        if not len(self._pts):
            self._detect(cur, M)
        self._prev, self._prev_M = cur, M
        self.n_rows += 1

    def finish(self) -> TrackSet:
        if self._starts:
            s = np.vstack(self._starts)
            e = np.vstack(self._ends)
        else:
            s = e = np.empty((0, 2))
        return TrackSet(
            start_px=s, end_px=e,
            start_world=self.rect.image_to_world(s) if len(s) else np.empty((0, 2)),
            end_world=self.rect.image_to_world(e) if len(e) else np.empty((0, 2)),
            duration_s=self.cfg.track_frames * self.st_s_per_px,
        )


def accepted(cfg: Config, tracks: TrackSet, along_world: np.ndarray) -> np.ndarray:
    """Which tracks survive the speed, direction and static-texture tests."""
    if not len(tracks):
        return np.zeros(0, bool)
    v = tracks.velocity_world
    speed = np.linalg.norm(v, axis=1)
    v_lo, v_hi = cfg.stiv.velocity_range_ms
    fast = max(abs(v_lo), abs(v_hi))
    # On the flow *axis*, either way along it: the direction check, not this
    # test, decides which way is downstream.
    cos_axis = np.abs(v @ along_world) / np.maximum(speed, 1e-12)
    moved = np.linalg.norm(tracks.end_px - tracks.start_px, axis=1)
    return ((speed <= fast)
            & (cos_axis >= math.cos(math.radians(cfg.ptv.max_angle_deg)))
            & (moved >= cfg.ptv.min_displacement_px))


#: Fewest tracks in a box for the across-box fit; fewer take the median.
_MIN_FIT_TRACKS = 8
#: Tracks further than this many robust standard deviations from the box's
#: median are left out of the fit.
_FIT_CLIP = 3.0


def _at_centre(values: np.ndarray, offset: np.ndarray, half_width: float) -> float:
    """The value at zero offset of a robust straight-line fit of ``values``.

    Tracks are rarely spread evenly across a box: texture is richer on one side,
    or features are harder to keep in the faster water.  Across a velocity
    gradient -- the flanks of the channel -- the median of the tracks then leans
    towards the better-populated side.  A line fitted across the box and read
    at its centre does not, to first order.  With few tracks, or tracks bunched
    in one part of the box, the median is used instead.
    """
    med = float(np.median(values))
    if len(values) < _MIN_FIT_TRACKS:
        return med
    mad = float(np.median(np.abs(values - med)))
    keep = np.abs(values - med) <= _FIT_CLIP * 1.4826 * max(mad, 1e-12)
    x, y = offset[keep], values[keep]
    if len(y) < _MIN_FIT_TRACKS or float(np.std(x)) < 0.2 * half_width:
        return med
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept)


def estimate(cfg: Config, tracks: TrackSet, box: StationBox,
             keep: Optional[np.ndarray] = None) -> VectorEstimate:
    """The velocity of one station from the tracks through its box.

    Each component is the across-box fit of :func:`_at_centre`, which falls
    back to the median of the tracks.
    """
    if keep is None:
        keep = accepted(cfg, tracks, box.along_world)
    inside = keep & box.contains_world(tracks.mid_world) if len(tracks) else keep
    v = tracks.velocity_world[inside] if len(tracks) else np.empty((0, 2))
    est = VectorEstimate(velocity_world=np.full(2, np.nan), samples_world=v,
                         n_used=int(inside.sum()))
    if len(v) == 0:
        return est
    offset = (tracks.mid_world[inside] - box.centre_world) @ box.across_world
    along = _at_centre(v @ box.along_world, offset, box.half_width_m)
    across = _at_centre(v @ box.across_world, offset, box.half_width_m)
    est.velocity_world = along * box.along_world + across * box.across_world
    moved = np.linalg.norm(tracks.end_px[inside] - tracks.start_px[inside], axis=1)
    est.displacement_px = float(np.median(moved))
    return est
