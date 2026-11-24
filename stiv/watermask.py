"""Automatic outline of the water surface, for the stabiliser's water mask.

The water mask has one job: keep the drone-motion stabiliser from locking onto
features on the moving river (:func:`stiv.stabilise.build_mask`).  Drawing it
by hand is the fiddliest click in a survey, and it can be found automatically
from the one thing that tells water from bank in drone footage -- the water
moves and the bank does not:

1. Read a handful of frames a fraction of a second apart from the start of the
   analysis window, at a reduced working resolution.
2. Register each to the first with the stabiliser itself, so the aircraft's
   own drift is removed exactly as the analysis will remove it.
3. Measure the motion that is left between consecutive frames with dense
   optical flow.  On bank it is close to zero; on water it is the surface
   velocity times the frame gap.  A pair whose banks still appear to move was
   registered onto the water and is dropped; the registration is then run a
   second time with the water found so far kept out of it.
4. Calibrate the threshold between the two from the cross-section already
   clicked: its middle is water, and the ground just beyond its ends is bank.
   A scene in which the two do not move clearly differently is reported as a
   failure rather than outlined on a guess.
5. Keep the moving region that crosses the section, smooth it, and simplify
   its outline to a polygon.

The outline is a *suggestion*: the wizard shows it on the frame and the user
confirms it.  Erring towards too much water is harmless (fewer features, all of
them on static ground); erring onto textured bank only costs features too, and
the run report says when there are too few.  As a by-product, the direction the
water moves is compared with the downstream point the user clicked, which
catches a downstream point on the wrong side of the section.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import ConfigError, StabiliseConfig
from .stabilise import Stabiliser, build_mask

#: Water must move at least this many times as much as the bank around the
#: section ends before an outline is offered.
MIN_CONTRAST = 2.0
#: ... and by at least this many working-resolution pixels between frames.
MIN_WATER_MOTION_PX = 0.3
#: Floor on the bank motion used to set the threshold, pixels: registration and
#: optical flow both leave a few tenths of a pixel on perfectly static ground.
_BANK_FLOOR_PX = 0.1
#: Most vertices the returned polygon may have.
_MAX_VERTICES = 48


@dataclass
class WaterDetection:
    """What :func:`detect_water` found.

    ``polygon_px`` is in full-resolution image coordinates, ready for
    ``stabilise.water_mask_px``; it is empty when ``confidence`` is
    ``"failed"``.  ``mask`` is the detected region at the working resolution
    ``scale`` (``mask`` pixel = image pixel * ``scale``).
    """

    polygon_px: list[tuple[float, float]]
    confidence: str                     # "good" | "weak" | "failed"
    message: str
    mask: Optional[np.ndarray] = None
    scale: float = 1.0
    water_fraction: float = 0.0
    #: Median motion over the water / over the bank, working pixels per gap.
    water_motion_px: float = math.nan
    bank_motion_px: float = math.nan
    #: Whether the water moves towards the downstream point; None if unclear.
    flow_agrees: Optional[bool] = None

    @property
    def ok(self) -> bool:
        return self.confidence != "failed" and len(self.polygon_px) >= 3

    @property
    def contrast(self) -> float:
        return self.water_motion_px / max(self.bank_motion_px, _BANK_FLOOR_PX)


def _failed(message: str, **kw) -> WaterDetection:
    return WaterDetection(polygon_px=[], confidence="failed", message=message, **kw)


# --------------------------------------------------------------------------- #
def detect_water(
    video_path: str | Path,
    start_s: float,
    section_px: Sequence[Sequence[float]],
    downstream_px: Optional[Sequence[float]] = None,
    *,
    gap_s: float = 0.5,
    n_pairs: int = 4,
    work_width: int = 960,
    fps: Optional[float] = None,
) -> WaterDetection:
    """Outline the moving water around a cross-section.

    Parameters
    ----------
    video_path, start_s
        The clip, and the time of the frame the outline is drawn on -- the
        analysis start, which is the stabiliser's reference frame.
    section_px
        The two ends of the cross-section, full-resolution image pixels.
    downstream_px
        The downstream point, if known; used only for the direction check.
    gap_s, n_pairs
        Frames are compared ``gap_s`` apart, ``n_pairs`` times.
    work_width
        Width the frames are reduced to before any of this, pixels.
    fps
        The frame rate, when the file does not record it.
    """
    if len(section_px) != 2:
        raise ValueError("section_px must be the two ends of the cross-section")
    frames, scale = _read_frames(video_path, start_s, gap_s, n_pairs, work_width, fps)
    if len(frames) < 2:
        return _failed("the clip is too short after the start time to see the water move")

    ref = frames[0]
    h, w = ref.shape
    a = np.asarray(section_px[0], float) * scale
    b = np.asarray(section_px[1], float) * scale
    if float(np.linalg.norm(b - a)) < 4:
        return _failed("the cross-section is too short to calibrate against", scale=scale)
    # Labelled samples from the section the user clicked: its middle is
    # water, and the ground just beyond its two ends is bank.
    water_pts = a + np.linspace(0.2, 0.8, 41)[:, None] * (b - a)
    ext = np.concatenate([np.linspace(-0.30, -0.06, 13), np.linspace(1.06, 1.30, 13)])
    bank_pts = a + ext[:, None] * (b - a)
    if _sample(ref, water_pts).size < 5:
        return _failed("the cross-section lies outside the frame", scale=scale)

    # Pass 1 keeps the stabiliser off a corridor around the section only, as
    # the analysis does when no water mask is given.  On a narrow-banked scene
    # the water elsewhere can still outvote the bank, so pass 2 registers again
    # with the water pass 1 found excluded, and keeps whichever is cleaner.
    along = (b - a) / float(np.linalg.norm(b - a))
    normal = np.array([-along[1], along[0]])
    corridor = np.array([a + 2 * normal, b + 2 * normal, b - 2 * normal, a - 2 * normal])
    first = _measure(frames, corridor, None, water_pts, bank_pts)
    if isinstance(first, str):
        return _failed(first, scale=scale)
    best = first
    if first.region is not None:
        second = _measure(frames, corridor, first.region, water_pts, bank_pts)
        if not isinstance(second, str) and second.region is not None \
                and second.contrast >= first.contrast:
            best = second
    m = best
    stats = dict(scale=scale, water_motion_px=m.water_level, bank_motion_px=m.bank_level)

    if m.region is None:
        return _failed(
            "the water around the cross-section does not move clearly more than the "
            "banks beyond its ends (water {:.2f} px, bank {:.2f} px between frames {:g} s "
            "apart)".format(m.water_level, m.bank_level, gap_s),
            **stats,
        )

    # -- keep what crosses the section, as one outline ------------------ #
    _, labels = cv2.connectedComponents(m.region.astype(np.uint8))
    hit = {int(labels[int(round(y)), int(round(x))]) for x, y in water_pts
           if 0 <= x < w and 0 <= y < h}
    hit.discard(0)
    if not hit:
        return _failed("no moving region crosses the cross-section", **stats)
    keep = np.isin(labels, sorted(hit)).astype(np.uint8)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    poly = _simplify(contour)
    filled = np.zeros_like(keep)
    cv2.fillPoly(filled, [poly.astype(np.int32)], 1)
    water_fraction = float(filled.mean())

    flow_agrees = None
    if downstream_px is not None:
        flow_agrees = _flow_agrees(m.flow, filled.astype(bool) & (m.mag > m.threshold), a, b,
                                   np.asarray(downstream_px, float) * scale, m.water_level)

    centre = tuple(float(v) for v in 0.5 * (a + b))
    concerns = []
    if cv2.pointPolygonTest(poly.reshape(-1, 1, 2).astype(np.float32), centre, False) < 0:
        concerns.append("the outline misses the middle of the cross-section")
    if water_fraction < 0.03:
        concerns.append("very little of the frame was found to be water")
    if water_fraction > 0.97:
        concerns.append("almost the whole frame was found to be water")
    if m.n_used < (len(frames) - 1) / 2:
        concerns.append(f"the drone's drift could be removed cleanly from only {m.n_used} "
                        f"of {len(frames) - 1} frame pairs")
    if len(contours) > 1 and sum(cv2.contourArea(c) for c in contours) > 1.25 * cv2.contourArea(contour):
        concerns.append("the water is split into several pieces; only the largest is outlined")

    polygon = [(float(x) / scale, float(y) / scale) for x, y in poly]
    common = dict(mask=filled.astype(bool), water_fraction=water_fraction,
                  flow_agrees=flow_agrees, **stats)
    if concerns:
        return WaterDetection(polygon, "weak", "; ".join(concerns), **common)
    return WaterDetection(
        polygon, "good",
        f"water outlined ({100 * water_fraction:.0f}% of the frame; it moves about "
        f"{m.contrast:.0f}x as much as the banks)",
        **common,
    )


@dataclass
class _Motion:
    """Residual (drift-removed) motion between frames, at working resolution."""

    mag: np.ndarray            # smoothed per-pixel motion, px per gap
    flow: np.ndarray           # (h, w, 2) median flow vectors
    water_level: float
    bank_level: float
    threshold: float
    region: Optional[np.ndarray]   # moving region, or None if water ~ bank
    n_used: int

    @property
    def contrast(self) -> float:
        return self.water_level / max(self.bank_level, _BANK_FLOOR_PX)


def _measure(frames: list[np.ndarray], corridor: np.ndarray, exclude: Optional[np.ndarray],
             water_pts: np.ndarray, bank_pts: np.ndarray) -> "_Motion | str":
    """Register ``frames`` to the first and measure the motion left over.

    Returns a reason string instead when nothing could be measured.
    """
    ref = frames[0]
    h, w = ref.shape
    sigma = max(1.5, 0.006 * w)
    mask = build_mask(StabiliseConfig(), ref.shape, exclude_px=corridor,
                      exclude_dilate_px=max(8, int(0.04 * w)))
    if exclude is not None:
        grown = cv2.dilate(exclude.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        mask[grown > 0] = 0
    stab = Stabiliser(StabiliseConfig(), ref, mask)
    if not stab.enabled:
        return ("too few static features to remove the drone's drift, so moving water "
                "cannot be told from a moving camera")

    # Consecutive frames, both brought into the reference frame's coordinates,
    # so every comparison spans one gap however long the run of frames is.
    pairs = []
    ones = np.ones_like(ref)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    prev, prev_valid = ref, np.ones(ref.shape, bool)
    for cur in frames[1:]:
        before = stab.report.n_failed
        M = stab.update(cur)
        if stab.report.n_failed > before:
            prev = None
            continue
        aligned = cv2.warpPerspective(cur, M, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(ones, M, (w, h), flags=flags, borderValue=0) > 0
        if prev is not None:
            flow = _flow(prev, aligned)
            both = valid & prev_valid
            mag = np.linalg.norm(flow, axis=2)
            mag_s = cv2.GaussianBlur(np.where(both, mag, 0.0).astype(np.float32), (0, 0), sigma)
            water_k, bank_k = _levels(mag_s, water_pts, bank_pts)
            mag[~both] = np.nan
            flow[~both] = np.nan
            pairs.append((mag, flow, water_k, bank_k))
        prev, prev_valid = aligned, valid
    if not pairs:
        return "the drone's drift could not be removed from enough frames"

    # A pair whose banks appear to move was registered onto the water (or a
    # gust moved the aircraft between the two frames); it says nothing about
    # which pixels are water.  Fall back to every pair only if none is clean.
    clean = [p for p in pairs if p[2] >= MIN_CONTRAST * max(p[3], _BANK_FLOOR_PX)]
    used = clean or pairs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # all-NaN border pixels
        mag = np.nanmedian(np.stack([p[0] for p in used]), axis=0)
        flow = np.nanmedian(np.stack([p[1] for p in used]), axis=0)
    mag = cv2.GaussianBlur(np.where(np.isfinite(mag), mag, 0.0).astype(np.float32), (0, 0), sigma)
    water_level, bank_level = _levels(mag, water_pts, bank_pts)
    threshold = math.sqrt(max(water_level, 0.0) * max(bank_level, _BANK_FLOOR_PX))
    region = None
    if water_level >= MIN_WATER_MOTION_PX and water_level >= MIN_CONTRAST * max(bank_level, _BANK_FLOOR_PX):
        region = _clean_region(mag > threshold, w)
    return _Motion(mag, flow, water_level, bank_level, threshold, region, len(clean))


def _levels(mag: np.ndarray, water_pts: np.ndarray, bank_pts: np.ndarray) -> tuple[float, float]:
    """Median motion at the water samples, and at the bank samples."""
    water = _sample(mag, water_pts)
    bank = _sample(mag, bank_pts)
    water_level = float(np.median(water)) if water.size else 0.0
    # Section ends at the frame edge leave no bank beyond them in view; the
    # quietest tenth of the frame stands in for it.
    bank_level = float(np.median(bank)) if bank.size >= 5 else float(np.percentile(mag, 10))
    return water_level, bank_level


# --------------------------------------------------------------------------- #
def _read_frames(path, start_s: float, gap_s: float, n_pairs: int, work_width: int,
                 fps: Optional[float]) -> tuple[list[np.ndarray], float]:
    """Grey frames at ``start_s`` and every ``gap_s`` after, reduced in size."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ConfigError(f"could not open video: {path}")
    try:
        rate = fps or cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not rate > 0:
            rate = 30.0
        gap = max(1, int(round(gap_s * rate)))
        if start_s > 0:
            # The same seek read_frame_at() makes, so frame 0 is the picker's.
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        frames: list[np.ndarray] = []
        scale = 1.0
        for i in range(n_pairs * gap + 1):
            if i % gap:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            scale = min(1.0, work_width / grey.shape[1])
            if scale < 1.0:
                grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(grey)
        return frames, scale
    finally:
        cap.release()


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense optical flow from ``a`` to ``b``, pixels.

    DIS rather than Farneback: its coarse-to-fine patch search follows the
    tens of pixels a fast river moves between frames half a second apart,
    where Farneback's estimate breaks down and bleeds onto the banks.
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(a, b, None)


def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Values of ``img`` at the in-frame points of ``pts`` (nearest pixel)."""
    h, w = img.shape
    pts = np.round(np.asarray(pts, float)).astype(int)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[inside]
    return img[pts[:, 1], pts[:, 0]]


def _clean_region(binary: np.ndarray, width: int) -> np.ndarray:
    """Close gaps, drop specks and fill holes in a thresholded motion map."""
    def _k(frac: float) -> np.ndarray:
        n = max(3, int(frac * width) | 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

    m = binary.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _k(0.03))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _k(0.015))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def _simplify(contour: np.ndarray) -> np.ndarray:
    """A polygon of at most ``_MAX_VERTICES`` vertices tracing ``contour``."""
    perimeter = cv2.arcLength(contour, True)
    eps = 0.003 * perimeter
    poly = cv2.approxPolyDP(contour, eps, True)
    while len(poly) > _MAX_VERTICES:
        eps *= 1.4
        poly = cv2.approxPolyDP(contour, eps, True)
    return poly.reshape(-1, 2).astype(float)


def _flow_agrees(flow: np.ndarray, water: np.ndarray, a: np.ndarray, b: np.ndarray,
                 downstream: np.ndarray, water_level: float) -> Optional[bool]:
    """Does the water move to the downstream point's side of the section?"""
    along = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-along[1], along[0]])
    if np.dot(downstream - 0.5 * (a + b), normal) < 0:
        normal = -normal
    v = flow[water]
    v = v[np.all(np.isfinite(v), axis=1)]
    if len(v) < 20:
        return None
    proj = float(np.median(v @ normal))
    if abs(proj) < 0.3 * water_level:
        return None
    return proj > 0
