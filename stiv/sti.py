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

    #: (n_time, n_space).  Raw float32 samples as collected; float64 only
    #: transiently, while :mod:`stiv.analysis` conditions and analyses it; and
    #: the conditioned STI stored back as float32 afterwards, since by then it is
    #: kept only for display and export.
    data: np.ndarray
    line: SearchLine
    sx_m_per_px: float        # metres per column  -> S_x of Eq. (16)
    st_s_per_px: float        # seconds per row    -> S_t of Eq. (16)
    #: Fraction of samples that fell outside the frame at some point.
    out_of_frame_fraction: float = 0.0

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

    Only the four neighbours of each sample are read and converted to float.
    Converting the whole frame first (``image.astype(np.float32)``) gives the
    same numbers but copies every pixel of a 4K frame to read a few thousand.
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

    # Gathered as float32 exactly as the whole-frame conversion would have
    # produced them, so the result is bit-identical (and a float64 frame from
    # custom channel weights is rounded the same way it always was).
    p00 = image[v0, u0].astype(np.float32)
    p01 = image[v0, u1].astype(np.float32)
    p10 = image[v1, u0].astype(np.float32)
    p11 = image[v1, u1].astype(np.float32)
    top = p00 * (1.0 - fu) + p01 * fu
    bot = p10 * (1.0 - fu) + p11 * fu
    out = top * (1.0 - fv) + bot * fv

    return np.where(inside, out, np.nan)


#: Rows per block when the number of frames is not known in advance.  Large
#: enough that the per-block overhead is nothing, small enough that the unused
#: tail of the last block is a few megabytes at most.
_CHUNK_ROWS = 256


class StiCollector:
    """Accumulates one STI row per frame for every search line at once.

    All lines are sampled from a single flattened coordinate array, so the whole
    cross-section costs one pass over each frame.

    When ``n_frames`` is known the buffer is allocated once, exactly, and the
    STIs handed back by :meth:`finish` are views into it -- no second copy.
    When it is not (``None``: a raw ``.h264``/``.mjpeg`` stream reports no frame
    count, and no duration or frame cap was given) rows go into fixed-size
    blocks that are joined once at the end.  Preallocating for "as many frames
    as there might be" is not an option: that was 10^9 rows, i.e. terabytes.
    """

    def __init__(self, lines: Sequence[SearchLine], n_frames: Optional[int], st_s_per_px: float):
        self.lines = list(lines)
        self.st_s_per_px = float(st_s_per_px)
        self.counts = [sl.n_samples for sl in self.lines]
        self.offsets = np.cumsum([0] + self.counts)
        self.all_pts = np.vstack([sl.samples_px for sl in self.lines])
        self.n_frames = None if n_frames is None else int(n_frames)
        self._chunks: list[np.ndarray] = []
        self.buffer: Optional[np.ndarray] = None
        if self.n_frames is not None:
            self.buffer = np.full((self.n_frames, int(self.offsets[-1])), np.nan, dtype=np.float32)
        self.n_rows = 0

    def add_frame(self, grey: np.ndarray, transform: Optional[np.ndarray] = None) -> None:
        if self.n_frames is not None and self.n_rows >= self.n_frames:
            return
        pts = self.all_pts if transform is None else transform_points(transform, self.all_pts)
        row = sample_bilinear(grey, pts)
        if self.n_frames is not None:
            self.buffer[self.n_rows] = row
        else:
            k = self.n_rows % _CHUNK_ROWS
            if k == 0:
                self._chunks.append(
                    np.full((_CHUNK_ROWS, int(self.offsets[-1])), np.nan, dtype=np.float32)
                )
            self._chunks[-1][k] = row
        self.n_rows += 1

    def finish(self) -> list[SpaceTimeImage]:
        if self.n_frames is None:
            # One join, then views as below.  The blocks are dropped straight
            # away, so the peak is the data twice over only for that moment.
            if self._chunks:
                buf = np.concatenate(self._chunks, axis=0)[: self.n_rows]
            else:
                buf = np.empty((0, int(self.offsets[-1])), dtype=np.float32)
            self._chunks = []
        else:
            buf = self.buffer[: self.n_rows]
        out = []
        for i, sl in enumerate(self.lines):
            # A view, not a copy: every STI shares the one buffer, which is
            # freed once the analysis has replaced each STI's data.
            block = buf[:, self.offsets[i] : self.offsets[i + 1]]
            frac = float(np.isnan(block).mean()) if block.size else 0.0
            out.append(
                SpaceTimeImage(
                    data=block,
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

    Always returns a new float64 array, whatever ``sti`` was: the orientation
    analysis that follows must see full precision (the Swift port is held to
    the Python numbers to 1e-9), and ``sti`` may be a view into a shared buffer.
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
