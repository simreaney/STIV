"""Construction and conditioning of space-time images.

Section 2.1 of the paper: for a search line laid along the main flow, the
brightness profile is read from ``N`` consecutive frames and stacked, giving
``g(x, t)`` -- space across, time down.  Surface features advecting along the
line draw inclined streaks whose slope is the velocity.

Two departures from the 2007 implementation are worth noting, both aimed at
drone footage:

* Samples are spaced uniformly in **world** coordinates rather than in image
  pixels, so ``Sx`` in Eq. (16) is genuinely constant along the line even in an
  oblique view.  The paper notes (section 2.1, on Fig. 3) that "the horizontal
  length of the respective STI is not constant because of the image
  obliqueness"; sampling through the rectifier removes that caveat.
* Each frame's sample points are pushed through the stabilisation transform, so
  the STI follows a fixed patch of ground rather than a fixed patch of sensor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .config import StiConfig
from .geometry import SearchLine
from .stabilise import transform_points

log = logging.getLogger(__name__)


@dataclass
class SpaceTimeImage:
    """One STI plus the scales needed to turn its slope into a velocity."""

    data: np.ndarray          # (n_time, n_space), float32
    line: SearchLine
    sx_m_per_px: float        # metres per column  -> S_x of Eq. (16)
    st_s_per_px: float        # seconds per row    -> S_t of Eq. (16)
    #: Fraction of samples that fell outside the frame at some point.
    out_of_frame_fraction: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def duration_s(self) -> float:
        return self.data.shape[0] * self.st_s_per_px

    @property
    def length_m(self) -> float:
        return (self.data.shape[1] - 1) * self.sx_m_per_px

    def velocity_per_unit_slope(self) -> float:
        """``Sx / St``: the velocity corresponding to ``tan phi = 1``."""
        return self.sx_m_per_px / self.st_s_per_px


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def sample_bilinear(image: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``image`` at ``(n, 2)`` float ``(u, v)`` coordinates.

    Points outside the frame come back as NaN so the caller can tell a genuinely
    dark pixel from a missing one.
    """
    h, w = image.shape[:2]
    u = np.asarray(pts_px[:, 0], dtype=np.float64)
    v = np.asarray(pts_px[:, 1], dtype=np.float64)

    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    uc = np.clip(u, 0, w - 1.000001)
    vc = np.clip(v, 0, h - 1.000001)

    u0 = np.floor(uc).astype(np.intp)
    v0 = np.floor(vc).astype(np.intp)
    u1 = np.minimum(u0 + 1, w - 1)
    v1 = np.minimum(v0 + 1, h - 1)
    fu = uc - u0
    fv = vc - v0

    img = image.astype(np.float32, copy=False)
    top = img[v0, u0] * (1.0 - fu) + img[v0, u1] * fu
    bot = img[v1, u0] * (1.0 - fu) + img[v1, u1] * fu
    out = top * (1.0 - fv) + bot * fv

    return np.where(inside, out, np.nan)


class StiCollector:
    """Accumulates one STI row per frame for every search line at once.

    All lines are sampled from a single flattened coordinate array, so the whole
    cross-section costs one pass over each frame.
    """

    def __init__(self, lines: Sequence[SearchLine], n_frames: int, st_s_per_px: float):
        self.lines = list(lines)
        self.st_s_per_px = float(st_s_per_px)
        self.counts = [sl.n_samples for sl in self.lines]
        self.offsets = np.cumsum([0] + self.counts)
        self.all_pts = np.vstack([sl.samples_px for sl in self.lines])
        self.buffer = np.full((n_frames, self.offsets[-1]), np.nan, dtype=np.float32)
        self.n_rows = 0

    def add_frame(self, grey: np.ndarray, transform: Optional[np.ndarray] = None) -> None:
        if self.n_rows >= self.buffer.shape[0]:
            return
        pts = self.all_pts if transform is None else transform_points(transform, self.all_pts)
        self.buffer[self.n_rows] = sample_bilinear(grey, pts)
        self.n_rows += 1

    def finish(self) -> list[SpaceTimeImage]:
        buf = self.buffer[: self.n_rows]
        out = []
        for i, sl in enumerate(self.lines):
            block = buf[:, self.offsets[i] : self.offsets[i + 1]]
            frac = float(np.isnan(block).mean())
            out.append(
                SpaceTimeImage(
                    data=block.copy(),
                    line=sl,
                    sx_m_per_px=sl.sx_m_per_px,
                    st_s_per_px=self.st_s_per_px,
                    out_of_frame_fraction=frac,
                )
            )
        return out


# --------------------------------------------------------------------------- #
# conditioning
# --------------------------------------------------------------------------- #
def preprocess(sti: np.ndarray, cfg: StiConfig) -> np.ndarray:
    """Condition an STI before the orientation analysis.

    The background removal is the important step for real footage.  Anything
    that does not move -- a rock breaking the surface, a bank in the corner of
    an oblique view, a fixed specular highlight -- writes a perfectly vertical
    stripe into the STI.  Vertical means ``tan phi = 0``, so such features vote
    hard for zero velocity and drag the Eq. (14) average down.  Subtracting each
    column's time-mean removes them; subtracting each row's spatial mean removes
    frame-wide exposure flicker, which writes horizontal stripes instead.
    """
    a = np.array(sti, dtype=np.float64, copy=True)

    if np.isnan(a).any():
        a = _fill_nan_columns(a)

    if cfg.remove_static_background:
        a -= np.nanmean(a, axis=0, keepdims=True)   # static texture -> vertical stripes
        a -= np.nanmean(a, axis=1, keepdims=True)   # exposure flicker -> horizontal stripes

    if cfg.normalise:
        s = float(np.nanstd(a))
        a = (a - float(np.nanmean(a))) / s if s > 1e-12 else a - float(np.nanmean(a))

    if cfg.clahe:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if hi - lo > 1e-12:
            u8 = ((a - lo) * (255.0 / (hi - lo))).astype(np.uint8)
            u8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
            a = u8.astype(np.float64) / 255.0 * (hi - lo) + lo

    if cfg.smooth_sigma and cfg.smooth_sigma > 0:
        # Section 2.3: smoothing markedly improves accuracy on smooth (Case C)
        # patterns, and is what lets STIV work without a high-end camera.
        a = gaussian_filter(a, sigma=float(cfg.smooth_sigma), mode="nearest")

    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _fill_nan_columns(a: np.ndarray) -> np.ndarray:
    """Replace NaNs (samples that left the frame) with the column time-mean."""
    col_mean = np.nanmean(a, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, float(np.nanmean(a)) if np.isfinite(np.nanmean(a)) else 0.0)
    idx = np.where(np.isnan(a))
    a = a.copy()
    a[idx] = np.take(col_mean, idx[1])
    return np.nan_to_num(a, nan=0.0)


def to_display(sti: np.ndarray) -> np.ndarray:
    """Stretch an STI to 0-255 uint8 for saving as a PNG."""
    a = np.asarray(sti, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi - lo < 1e-12:
        lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
