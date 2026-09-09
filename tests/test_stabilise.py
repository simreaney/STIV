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

from stiv.config import StabiliseConfig
from stiv.stabilise import Stabiliser, _to_u8, _u8_scaling


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
