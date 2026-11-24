"""Compensation for drone motion between frames.

A hovering multirotor still drifts, yaws and rocks by a few pixels over a
30-second clip.  STIV cannot tell that apparent motion from water motion: a
steady 1 px/frame drift along the search line adds ``Sx * fps`` m/s to every
station, which at a typical 2 cm/px and 30 fps is 0.6 m/s of pure error.  So
for UAV footage this step matters as much as the orientation analysis itself.

The approach is the standard one: track distinctive features on *static* ground
with pyramidal Lucas-Kanade optical flow, fit a per-frame transform back to a
reference frame, and use it to follow the same patch of ground through the
clip.  Three details are worth flagging:

* Every frame is matched against the **reference** frame, using the previous
  frame's tracks only as an initial guess.  Chaining frame-to-frame transforms
  instead would accumulate drift, and drift is exactly the error mode that
  masquerades as flow.
* Nothing is ever warped.  The estimated transform is applied to the *search
  line sample points*, which is a handful of coordinates per frame rather than
  a few million pixels.
* Features die as the aircraft drifts, so the set is topped back up from the
  current frame (``stabilise.redetect_fraction``) and the new corners carried
  back into reference coordinates.  Without it the surviving set thins until
  the fit fails and the last good transform is held for the rest of the clip --
  which reads as uncompensated camera motion, i.e. as flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import StabiliseConfig

log = logging.getLogger(__name__)

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
#: Fewest features a transform may rest on -- counted twice: among the tracks
#: that survive forward-backward checking, and again among the RANSAC inliers
#: of the fit itself.  A similarity has four degrees of freedom, so two points
#: determine it exactly and three or four barely constrain it.
_MIN_INLIERS = 6
#: Fewest tracks, as a fraction of those that survived forward-backward
#: checking, that must agree with a fitted transform for it to be accepted.
#: RANSAC returns the *largest* consensus, which is not the same as a good
#: one: with most tracks on moving water or lost in blur, a handful of points
#: can agree by accident, or a coherent patch of water texture can outvote the
#: bank.  Below this the fit is treated as failed and the previous transform
#: held.  Deliberately lower than the 0.5 at which the run report starts
#: warning, so that a scene with some water in the mask degrades to a warning
#: rather than to held transforms.
_MIN_INLIER_FRACTION = 0.3
#: Forward-backward tracking error, pixels, above which a track is discarded:
#: tracked from the reference to the current frame and back, a feature on
#: static ground lands where it started, while one on water or in a smear of
#: motion blur does not.
_MAX_FB_ERROR_PX = 1.5
#: How far, in pixels, the automatic water corridor around the search lines is
#: grown before features are looked for outside it.  Generous on purpose: a
#: feature on the water just beside a search line moves with the flow, and a
#: few of those can drag the fitted transform.
_CORRIDOR_DILATE_PX = 60
#: Frames a reference feature may fail to track before it is retired and its
#: slot given to a fresh corner.  Short enough to keep the set healthy, long
#: enough that a feature briefly lost to blur or a passing bird recovers.
_MAX_DEAD_FRAMES = 5
#: Minimum frames between top-ups.  Corner detection is not cheap, and the
#: features injected by one attempt need a frame or two to prove themselves, so
#: re-running it every frame pays full price for nothing.
_REDETECT_INTERVAL = 15
#: Consecutive top-ups that find nothing before the attempt is abandoned.  Once
#: the aircraft has drifted off the reference frame there is no more static
#: ground to find, and retrying for the rest of the clip is pure cost.
_REDETECT_GIVE_UP = 2


@dataclass
class StabilisationReport:
    """Per-frame diagnostics, summarised in the run report."""

    n_frames: int = 0
    n_failed: int = 0
    n_features: int = 0
    #: Times the tracked set was topped back up after features died.
    n_redetections: int = 0
    #: Per-frame RMS reprojection error of the fitted transform over its
    #: RANSAC inliers, pixels; NaN where no transform was accepted, so that the
    #: summary's median residual describes only the transforms actually used.
    rmse_px: list[float] = field(default_factory=list)
    #: Per-frame displacement of the section centre, pixels.
    shift_px: list[float] = field(default_factory=list)
    #: Per-frame fraction of the forward-backward-consistent tracks that agree
    #: with the fitted transform -- rejected fits included, since a low value
    #: is the evidence of tracks on water -- and 0 where none could be fitted.
    inlier_fraction: list[float] = field(default_factory=list)
    enabled: bool = True
    #: True when stabilisation was *requested* but had to be switched off --
    #: as opposed to ``enabled: false`` in the config, which is a choice.
    auto_disabled: bool = False
    #: Why it was switched off, when ``auto_disabled``.
    disabled_reason: str = ""

    def summary(self) -> str:
        if self.auto_disabled:
            return (
                f"stabilisation: DISABLED AUTOMATICALLY ({self.disabled_reason}); "
                f"camera motion is NOT compensated"
            )
        if not self.enabled:
            return "stabilisation: disabled"
        if not self.rmse_px:
            return "stabilisation: no frames processed"
        redetected = (
            f", {self.n_redetections} feature top-up(s)" if self.n_redetections else ""
        )
        # Accepted transforms only: rejected fits are recorded as NaN.
        accepted = [r for r in self.rmse_px if np.isfinite(r)]
        residual = (
            f"median residual {np.median(accepted):.2f} px" if accepted
            else "no transform accepted"
        )
        return (
            f"stabilisation: {self.n_features} features, {residual}, "
            f"max frame shift {np.max(self.shift_px):.1f} px, "
            f"{self.n_failed}/{self.n_frames} frames fell back to the previous "
            f"transform{redetected}"
        )

    def warnings(self) -> list[str]:
        out = []
        if self.auto_disabled:
            # Not merely "disabled": the user asked for compensation and is
            # not getting it, and every velocity carries the aircraft's drift.
            out.append(
                f"STABILISATION WAS SWITCHED OFF: {self.disabled_reason}, so the "
                f"camera motion could not be tracked and every velocity includes "
                f"any drift of the aircraft (at 2 cm/px and 30 fps, one pixel per "
                f"frame is 0.6 m/s). Give stabilise.static_mask_px around textured "
                f"static ground, or set stabilise.enabled: false if the camera "
                f"really was fixed"
            )
            return out
        if not self.enabled:
            return out
        if self.n_features < 30:
            out.append(
                f"only {self.n_features} trackable features were found on static "
                f"ground; give stabilise.static_mask_px, or disable stabilisation "
                f"if the camera is fixed"
            )
        if self.n_frames and self.n_failed > 0.1 * self.n_frames:
            out.append(
                f"{self.n_failed} of {self.n_frames} frames could not be "
                f"stabilised reliably; results may contain camera motion"
            )
        if self.inlier_fraction and np.median(self.inlier_fraction) < 0.5:
            out.append(
                "fewer than half the tracked features agree with the fitted "
                "camera motion; features are probably being picked up on the "
                "water surface -- supply stabilise.water_mask_px (a polygon, or auto)"
            )
        if self.shift_px and np.max(self.shift_px) > 100:
            out.append(
                f"camera moved up to {np.max(self.shift_px):.0f} px during the "
                f"clip; check that the search lines stay over water throughout"
            )
        return out


# --------------------------------------------------------------------------- #
def build_mask(
    cfg: StabiliseConfig,
    image_shape: tuple[int, int],
    exclude_px: Optional[np.ndarray] = None,
    exclude_dilate_px: int = _CORRIDOR_DILATE_PX,
) -> np.ndarray:
    """Build the uint8 mask of regions in which to look for static features.

    Priority: an explicit ``static_mask_px`` wins; otherwise the whole frame is
    used minus ``water_mask_px`` and minus a corridor around the search lines
    (``exclude_px``), which is water by construction.
    """
    h, w = image_shape[:2]
    if isinstance(cfg.water_mask_px, str):
        # `water_mask_px: auto` is resolved into polygons (or None) before the
        # stabiliser is built -- see stiv.analysis.  Iterating the word would
        # fill "polygons" of single characters.
        raise ValueError(
            f"water_mask_px {cfg.water_mask_px!r} must be resolved to polygons before "
            f"building the tracking mask"
        )

    if cfg.static_mask_px:
        mask = np.zeros((h, w), np.uint8)
        for poly in cfg.static_mask_px:
            cv2.fillPoly(mask, [np.asarray(poly, np.int32).reshape(-1, 1, 2)], 255)
    else:
        mask = np.full((h, w), 255, np.uint8)

    if cfg.water_mask_px:
        for poly in cfg.water_mask_px:
            cv2.fillPoly(mask, [np.asarray(poly, np.int32).reshape(-1, 1, 2)], 0)

    if not cfg.static_mask_px and exclude_px is not None and len(exclude_px) >= 3:
        hull = cv2.convexHull(np.asarray(exclude_px, np.float32).reshape(-1, 1, 2))
        blocker = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(blocker, hull.astype(np.int32), 255)
        k = 2 * int(exclude_dilate_px) + 1
        blocker = cv2.dilate(blocker, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mask[blocker > 0] = 0

    return mask


def _fit_transform(src: np.ndarray, dst: np.ndarray, model: str, threshold_px: float):
    """Fit ``src -> dst`` robustly; returns (3x3 matrix, inlier mask) or (None, None).

    ``threshold_px`` is the RANSAC inlier threshold, i.e.
    ``stabilise.max_reprojection_px``: a track further than that from the
    fitted motion does not count as agreeing with it.
    """
    if len(src) < _MIN_INLIERS:
        return None, None
    s = src.reshape(-1, 1, 2).astype(np.float32)
    d = dst.reshape(-1, 1, 2).astype(np.float32)
    thr = float(threshold_px)

    if model == "homography":
        M, inl = cv2.findHomography(s, d, cv2.RANSAC, thr, maxIters=2000, confidence=0.995)
        if M is None:
            return None, None
        return M, (inl.ravel().astype(bool) if inl is not None else None)

    if model == "affine":
        A, inl = cv2.estimateAffine2D(s, d, method=cv2.RANSAC, ransacReprojThreshold=thr)
    else:  # similarity: rotation + uniform scale + translation
        A, inl = cv2.estimateAffinePartial2D(s, d, method=cv2.RANSAC, ransacReprojThreshold=thr)
    if A is None:
        return None, None
    M = np.vstack([A, [0.0, 0.0, 1.0]])
    return M, (inl.ravel().astype(bool) if inl is not None else None)


def track_forward_backward(
    prev: np.ndarray, cur: np.ndarray, pts: np.ndarray, guess: np.ndarray, max_fb_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pyramidal Lucas-Kanade from ``prev`` to ``cur``, checked by tracking back.

    ``pts`` are float32 ``(n, 2)`` positions in ``prev`` and ``guess`` the
    starting estimate of where they are in ``cur``.  Returns their positions in
    ``cur`` and which of them to trust: tracked both ways, and landing within
    ``max_fb_px`` of where they started when tracked back.
    """
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        prev, cur,
        pts.reshape(-1, 1, 2),
        guess.reshape(-1, 1, 2).copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        **_LK_PARAMS,
    )
    nxt = nxt.reshape(-1, 2)
    ok = status.ravel().astype(bool)

    # Forward-backward consistency: track back and keep only the features
    # that land where they started.
    if ok.any():
        back, status_b, _ = cv2.calcOpticalFlowPyrLK(
            cur, prev,
            nxt.reshape(-1, 1, 2).astype(np.float32),
            pts.reshape(-1, 1, 2).copy(),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            **_LK_PARAMS,
        )
        fb_err = np.linalg.norm(back.reshape(-1, 2) - pts, axis=1)
        ok &= status_b.ravel().astype(bool) & (fb_err < max_fb_px)
    return nxt, ok


def transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 image transform to ``(n, 2)`` points."""
    p = np.asarray(pts, dtype=np.float64)
    ones = np.ones((len(p), 1))
    q = (M @ np.hstack([p, ones]).T).T
    w = q[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return q[:, :2] / w


class Stabiliser:
    """Tracks static ground and reports, per frame, where the reference went.

    Usage is streaming: construct with the reference frame, then call
    :meth:`update` once per subsequent frame.  Each call returns -- and leaves
    in ``M`` -- the 3x3 matrix mapping *reference* image coordinates to
    *current* frame coordinates.

    A transform is accepted only if at least ``_MIN_INLIERS`` tracks, and at
    least ``_MIN_INLIER_FRACTION`` of the tracks that survived the
    forward-backward check, lie within ``stabilise.max_reprojection_px`` of it.
    Otherwise the frame counts as failed and the previous transform is held.
    """

    def __init__(
        self,
        cfg: StabiliseConfig,
        reference_grey: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg
        self.report = StabilisationReport(enabled=cfg.enabled)
        self.M = np.eye(3)

        self._u8 = _u8_scaling(reference_grey)
        self.ref = _to_u8(reference_grey, self._u8)
        self.mask = mask

        if not cfg.enabled:
            self.ref_pts = np.empty((0, 2), np.float32)
            self.cur_pts = self.ref_pts
            self.alive = np.zeros(0, bool)
            return

        pts = cv2.goodFeaturesToTrack(
            self.ref,
            maxCorners=cfg.max_features,
            qualityLevel=cfg.quality_level,
            minDistance=cfg.min_distance_px,
            mask=mask,
            blockSize=7,
        )
        self.ref_pts = (
            pts.reshape(-1, 2).astype(np.float32) if pts is not None else np.empty((0, 2), np.float32)
        )
        self.cur_pts = self.ref_pts.copy()
        self.alive = np.ones(len(self.ref_pts), bool)
        #: Consecutive frames each reference feature has failed to track for.
        self.dead_streak = np.zeros(len(self.ref_pts), np.int32)
        #: How many features the set is topped back up to when it is depleted.
        self._n_target = len(self.ref_pts)
        self._redetect_wait = 0
        self._redetect_barren = 0
        self.report.n_features = len(self.ref_pts)

        if len(self.ref_pts) < _MIN_INLIERS:
            log.warning(
                "only %d static features found; stabilisation will be disabled",
                len(self.ref_pts),
            )
            self.cfg = StabiliseConfig(**{**cfg.__dict__, "enabled": False})
            self.report.enabled = False
            self.report.auto_disabled = True
            self.report.disabled_reason = (
                f"only {len(self.ref_pts)} trackable feature(s) found on static "
                f"ground, at least {_MIN_INLIERS} needed"
            )

    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def update(self, frame_grey: np.ndarray, probe_px: Optional[np.ndarray] = None) -> np.ndarray:
        """Register one frame against the reference; returns the new transform."""
        if not self.enabled:
            return self.M

        cur = _to_u8(frame_grey, self._u8)
        self.report.n_frames += 1

        # Predict where the reference features should be, then refine with LK
        # against the *reference* image so nothing accumulates frame to frame.
        guess = transform_points(self.M, self.ref_pts).astype(np.float32)
        guess = np.where(self.alive[:, None], self.cur_pts, guess).astype(np.float32)

        nxt, ok = track_forward_backward(self.ref, cur, self.ref_pts, guess, _MAX_FB_ERROR_PX)

        M, inliers = (None, None)
        if ok.sum() >= _MIN_INLIERS:
            M, inliers = _fit_transform(
                self.ref_pts[ok], nxt[ok], self.cfg.model, self.cfg.max_reprojection_px
            )

        rmse, inlier_frac = float("nan"), 0.0
        if M is not None:
            if inliers is None or len(inliers) != int(ok.sum()):
                inliers = np.ones(int(ok.sum()), bool)
            pred = transform_points(M, self.ref_pts[ok])
            resid = np.linalg.norm(pred - nxt[ok], axis=1)
            n_in = int(inliers.sum())
            inlier_frac = n_in / max(1, len(inliers))
            if n_in:
                # Over the inliers only, so it is a fit residual, not a measure
                # of how many tracks were on water; inlier_frac is that.  By
                # construction it cannot exceed max_reprojection_px by much, so
                # it is a diagnostic of accepted fits, never the acceptance test.
                rmse = float(np.sqrt(np.mean(resid[inliers] ** 2)))
            if n_in < _MIN_INLIERS or inlier_frac < _MIN_INLIER_FRACTION:
                # A bad fit is worse than no update: hold the previous transform.
                M = None

        if M is None:
            self.report.n_failed += 1
            # A rejected fit's residual describes a transform that was not
            # used; leave it out of the residual statistics.
            self.report.rmse_px.append(float("nan"))
            self.report.inlier_fraction.append(inlier_frac)
            self.report.shift_px.append(self._probe_shift(probe_px))
            # Keep the last good transform and re-seed from its prediction.
            self.cur_pts = transform_points(self.M, self.ref_pts).astype(np.float32)
            self.alive = np.zeros(len(self.ref_pts), bool)
            self.dead_streak = self.dead_streak + 1
            return self.M

        self.M = M
        self.report.rmse_px.append(rmse)
        self.report.inlier_fraction.append(inlier_frac)
        self.report.shift_px.append(self._probe_shift(probe_px))

        # Keep good tracks; re-inject lost ones at the model-predicted position
        # so they can recover without breaking the reference correspondence.
        alive = np.zeros(len(self.ref_pts), bool)
        alive[np.flatnonzero(ok)[inliers]] = True

        self.cur_pts = transform_points(self.M, self.ref_pts).astype(np.float32)
        self.cur_pts[alive] = nxt[alive]
        self.alive = alive
        self.dead_streak = np.where(alive, 0, self.dead_streak + 1)

        # Features die as the aircraft drifts: bank texture leaves the frame,
        # motion blur washes a corner out.  Without a top-up the surviving set
        # thins until the fit fails and the last good transform is held for the
        # rest of the clip -- which reads as uncompensated camera motion, i.e.
        # as flow.
        self._redetect_wait = max(0, self._redetect_wait - 1)
        if (
            self.cfg.redetect_fraction > 0.0
            and self._n_target > 0
            and self._redetect_wait == 0
            and self._redetect_barren < _REDETECT_GIVE_UP
            and int(alive.sum()) < self.cfg.redetect_fraction * self._n_target
        ):
            self._redetect(cur)
            self._redetect_wait = _REDETECT_INTERVAL

        return self.M

    # ------------------------------------------------------------------ #
    def _redetect(self, cur: np.ndarray) -> None:
        """Retire long-dead tracks and top the set back up from ``cur``.

        Corners are detected in the *current* frame -- the reference frame's
        own corners are the ones that just died -- and then carried back into
        reference coordinates through the inverse of the current transform, so
        the reference correspondence that keeps this drift-free is preserved.
        The static-ground mask is warped forward for the same reason.

        A feature whose back-projection falls outside the reference frame is
        dropped: there is no reference appearance to match it against.  That is
        a real limit of registering everything to one reference, and it is the
        price of not accumulating drift.
        """
        keep = self.dead_streak < _MAX_DEAD_FRAMES
        if not keep.all():
            self.ref_pts = self.ref_pts[keep]
            self.cur_pts = self.cur_pts[keep]
            self.alive = self.alive[keep]
            self.dead_streak = self.dead_streak[keep]

        room = int(self.cfg.max_features) - len(self.ref_pts)
        if room <= 0:
            self._redetect_barren += 1
            return
        try:
            m_inv = np.linalg.inv(self.M)
        except np.linalg.LinAlgError:
            self._redetect_barren += 1
            return

        mask = self.mask
        if mask is not None:
            h, w = cur.shape[:2]
            mask = cv2.warpPerspective(
                mask, self.M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
            )

        found = cv2.goodFeaturesToTrack(
            cur,
            maxCorners=room,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance_px,
            mask=mask,
            blockSize=7,
        )
        if found is None:
            self._redetect_barren += 1
            return
        cur_new = found.reshape(-1, 2).astype(np.float32)

        # Discard anything that duplicates a track already held.
        if len(self.cur_pts):
            gap = np.linalg.norm(
                cur_new[:, None, :] - self.cur_pts[None, :, :], axis=2
            ).min(axis=1)
            cur_new = cur_new[gap > self.cfg.min_distance_px]
        if not len(cur_new):
            self._redetect_barren += 1
            return

        ref_new = transform_points(m_inv, cur_new).astype(np.float32)
        h, w = self.ref.shape[:2]
        inside = (
            (ref_new[:, 0] >= 0) & (ref_new[:, 0] <= w - 1)
            & (ref_new[:, 1] >= 0) & (ref_new[:, 1] <= h - 1)
        )
        ref_new, cur_new = ref_new[inside], cur_new[inside]
        if not len(ref_new):
            self._redetect_barren += 1
            return

        self.ref_pts = np.vstack([self.ref_pts, ref_new]).astype(np.float32)
        self.cur_pts = np.vstack([self.cur_pts, cur_new]).astype(np.float32)
        self.alive = np.concatenate([self.alive, np.ones(len(ref_new), bool)])
        self.dead_streak = np.concatenate(
            [self.dead_streak, np.zeros(len(ref_new), np.int32)]
        )
        self._redetect_barren = 0
        self.report.n_redetections += 1

    # ------------------------------------------------------------------ #
    def _probe_shift(self, probe_px: Optional[np.ndarray]) -> float:
        if probe_px is None or len(probe_px) == 0:
            return 0.0
        moved = transform_points(self.M, probe_px)
        return float(np.max(np.linalg.norm(moved - np.asarray(probe_px, float), axis=1)))


def _u8_scaling(reference: np.ndarray) -> Optional[tuple[float, float]]:
    """Fix the float-to-byte mapping once, from the reference frame.

    Returns ``None`` when the input is already ``uint8`` and needs no mapping.

    Lucas-Kanade assumes brightness constancy: the same patch of ground must
    carry the same grey value in the reference frame and in the current one.
    Rescaling each frame by its *own* min and max breaks that assumption --
    one specular highlight drifting into view compresses the real texture into
    a fraction of the byte range, and the tracker loses every feature at once
    while the fitted transform quietly falls back to the previous frame's.  So
    the mapping is derived once, here, and applied unchanged to every frame.
    """
    if reference.dtype == np.uint8:
        return None
    a = np.asarray(reference, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return (lo, lo + 1.0)
    return (lo, hi)


def _to_u8(img: np.ndarray, scaling: Optional[tuple[float, float]]) -> np.ndarray:
    """Apply the fixed mapping from :func:`_u8_scaling`, clipping outliers."""
    if img.dtype == np.uint8:
        return img
    if scaling is None:
        return np.clip(np.asarray(img, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
    lo, hi = scaling
    a = np.asarray(img, dtype=np.float32)
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
