"""Tests for drone-motion compensation.

Camera motion is the dominant error source in UAV gauging -- at 2 cm/px and
30 fps, one pixel per frame of uncompensated drift is a spurious 0.6 m/s on
every station -- so the failure modes here matter more than their size
suggests.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stiv import stabilise as stabilise_mod
from stiv.config import StabiliseConfig
from stiv.stabilise import Stabiliser, _fit_transform, _to_u8, _u8_scaling


def _texture(low_contrast: bool = False) -> np.ndarray:
    rng = np.random.default_rng(3)
    a = cv2.GaussianBlur(rng.normal(128, 40, (400, 600)).clip(5, 250), (0, 0), 1.5)
    a = a.astype(np.float32)
    if low_contrast:
        a = 100.0 + a * (50.0 / 255.0)      # grey 100-150, as overcast water is
    return a


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class TestBrightnessConstancy:
    """Lucas-Kanade needs the same patch to carry the same value in the
    reference frame and in the current one.  Scaling each frame by its *own*
    min and max broke that: one specular highlight drifting into view
    compressed the real texture into a fraction of the byte range, every track
    died at once, and the fitted transform quietly fell back to the previous
    frame's -- reporting camera motion as flow.
    """

    def test_the_mapping_is_fixed_by_the_reference_frame(self):
        base = _texture(low_contrast=True)
        scaling = _u8_scaling(base)
        glinting = base.copy()
        glinting[10:40, 10:40] = 900.0

        # The same input value maps to the same byte in both frames.
        a = _to_u8(base, scaling)
        b = _to_u8(glinting, scaling)
        untouched = np.s_[100:300, 100:500]
        assert np.array_equal(a[untouched], b[untouched])

    def test_a_transient_highlight_does_not_break_tracking(self):
        base = _texture(low_contrast=True)
        moved = _shift(base, 4.0, -3.0)
        moved[10:40, 10:40] = 900.0             # a saturated glint drifts in

        s = Stabiliser(StabiliseConfig(), base)
        m = s.update(moved)

        assert m[0, 2] == pytest.approx(4.0, abs=0.2)
        assert m[1, 2] == pytest.approx(-3.0, abs=0.2)
        assert s.report.n_failed == 0
        assert s.report.inlier_fraction[-1] > 0.5

    def test_uint8_input_is_passed_through_untouched(self):
        grey = (_texture() * 0.5).astype(np.uint8)
        assert _u8_scaling(grey) is None
        assert _to_u8(grey, None) is grey


class TestFeatureRedetection:
    """``redetect_fraction`` was documented as the refresh policy but never
    read.  Features were detected once on frame 0 and never topped up, so on a
    long drifting clip the surviving set thinned until the fit failed and the
    last good transform was held for the rest of the run.
    """

    def test_the_tracked_set_is_topped_back_up(self):
        base = _texture()
        s = Stabiliser(StabiliseConfig(), base)
        n0 = len(s.ref_pts)
        assert n0 > 50

        # Retire most of the set, as a long drift would.
        s.dead_streak[: n0 - 10] = 99
        s.alive[:] = False
        s._redetect(_to_u8(base, s._u8))

        assert s.report.n_redetections == 1
        assert len(s.ref_pts) > 10

    def test_the_parallel_arrays_stay_the_same_length(self):
        """Where this kind of code breaks: four arrays indexed in lockstep."""
        base = _texture()
        s = Stabiliser(StabiliseConfig(), base)
        s.dead_streak[: len(s.ref_pts) // 2] = 99
        s.alive[:] = False
        s._redetect(_to_u8(base, s._u8))

        n = len(s.ref_pts)
        assert len(s.cur_pts) == n
        assert len(s.alive) == n
        assert len(s.dead_streak) == n

    def test_zero_disables_it(self):
        cfg = StabiliseConfig(redetect_fraction=0.0)
        base = _texture()
        s = Stabiliser(cfg, base)
        for _ in range(3):
            s.update(_shift(base, 1.0, 0.0))
        assert s.report.n_redetections == 0


class TestAutomaticDisable:
    """With too few features the stabiliser switched itself off, and the run
    report then said only "disabled" -- indistinguishable from a fixed camera
    the user had chosen not to stabilise -- with no warning at all.
    """

    def test_it_is_loud_and_distinguishable_from_a_choice(self):
        blank = np.full((200, 300), 128, np.uint8)
        s = Stabiliser(StabiliseConfig(enabled=True), blank)

        assert not s.enabled
        assert s.report.auto_disabled
        assert "DISABLED AUTOMATICALLY" in s.report.summary()
        (warning,) = s.report.warnings()
        assert "STABILISATION WAS SWITCHED OFF" in warning
        assert "static_mask_px" in warning

    def test_a_deliberate_choice_stays_quiet(self):
        s = Stabiliser(StabiliseConfig(enabled=False), _texture())
        assert not s.report.auto_disabled
        assert s.report.summary() == "stabilisation: disabled"
        assert s.report.warnings() == []


class TestFitAcceptance:
    """The RANSAC threshold was a hard-coded 3 px and the residual was taken
    over the RANSAC inliers only, which lie within 3 px by construction -- so
    with the default ``max_reprojection_px`` of 3 the bad-fit guard could never
    fire.  And the minimum-support test counted tracked points, not inliers.
    """

    def _rigged(self, monkeypatch, n_agree, n_scatter, shift=(5.0, 0.0), cfg=None):
        """A stabiliser whose next update sees exactly the tracks given."""
        s = Stabiliser(cfg or StabiliseConfig(redetect_fraction=0.0), _texture())
        n = n_agree + n_scatter
        assert len(s.ref_pts) >= n
        s.ref_pts = s.ref_pts[:n]
        s.cur_pts = s.cur_pts[:n]
        s.alive = s.alive[:n]
        s.dead_streak = s.dead_streak[:n]

        rng = np.random.default_rng(7)
        nxt = s.ref_pts.astype(np.float64).copy()
        nxt[:n_agree] += shift
        nxt[n_agree:] += rng.uniform(-40, 40, (n_scatter, 2))
        calls = []

        def fake_lk(prev, cur, pts, guess, **kw):
            calls.append(1)
            if len(calls) % 2:           # forward: where each feature went
                out = nxt
            else:                        # backward: every track is consistent
                out = s.ref_pts
            return (out.reshape(-1, 1, 2).astype(np.float32),
                    np.ones((n, 1), np.uint8), np.zeros((n, 1), np.float32))

        monkeypatch.setattr(stabilise_mod.cv2, "calcOpticalFlowPyrLK", fake_lk)
        return s

    def test_a_fit_resting_on_a_handful_of_points_is_rejected(self, monkeypatch):
        s = self._rigged(monkeypatch, n_agree=4, n_scatter=16)
        m = s.update(_texture().astype(np.uint8))
        assert s.report.n_failed == 1
        assert np.allclose(m, np.eye(3))               # the previous transform, held

    def test_a_minority_consensus_is_rejected(self, monkeypatch):
        s = self._rigged(monkeypatch, n_agree=8, n_scatter=32)
        s.update(_texture().astype(np.uint8))
        assert s.report.n_failed == 1
        assert s.report.inlier_fraction[-1] < 0.3

    def test_a_rejected_fit_stays_out_of_the_residual_statistics(self, monkeypatch):
        """Its near-zero residual describes a transform that was never used,
        and used to pull the summary's median residual down."""
        s = self._rigged(monkeypatch, n_agree=8, n_scatter=32)
        s.update(_texture().astype(np.uint8))
        assert np.isnan(s.report.rmse_px[-1])
        assert 0.0 < s.report.inlier_fraction[-1] < 0.3   # still the real fraction
        assert "no transform accepted" in s.report.summary()

    def test_a_clear_consensus_is_accepted(self, monkeypatch):
        s = self._rigged(monkeypatch, n_agree=30, n_scatter=10)
        m = s.update(_texture().astype(np.uint8))
        assert s.report.n_failed == 0
        assert m[0, 2] == pytest.approx(5.0, abs=1e-3)

    def test_the_ransac_threshold_is_max_reprojection_px(self):
        rng = np.random.default_rng(1)
        src = rng.uniform(0, 500, (60, 2))
        dst = src + 2.0 * rng.standard_normal((60, 2))   # ~2.8 px rms scatter
        _, tight = _fit_transform(src, dst, "similarity", 1.0)
        _, loose = _fit_transform(src, dst, "similarity", 6.0)
        assert tight.sum() < loose.sum()
