"""Tests for the automatic water outline (stiv.watermask).

The synthetic river is the ground truth: its water is exactly the rows between
``bank_top_px`` and ``bank_bottom_px``.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stiv.config import Config, survey_dict
from stiv.synthetic import SyntheticRiver, render
from stiv.watermask import detect_water

RIVER = dict(width_px=320, height_px=200, duration_s=3.0, bank_top_px=50, bank_bottom_px=150,
             v_max_ms=0.8, drone_drift_px=20, drone_yaw_deg=1.0)


def _clip(tmp_path_factory, name, **kw):
    river = SyntheticRiver(**{**RIVER, **kw})
    return river, render(river, tmp_path_factory.mktemp(name) / "river", progress=False)


@pytest.fixture(scope="module")
def drifting(tmp_path_factory):
    return _clip(tmp_path_factory, "drift")


def _section(river):
    x = river.width_px // 2
    return [(x, river.bank_top_px), (x, river.bank_bottom_px)]


def _iou(polygon, river):
    truth = np.zeros((river.height_px, river.width_px), bool)
    truth[river.bank_top_px:river.bank_bottom_px + 1] = True
    pred = np.zeros_like(truth, np.uint8)
    cv2.fillPoly(pred, [np.round(np.asarray(polygon)).astype(np.int32)], 1)
    pred = pred.astype(bool)
    return (truth & pred).sum() / (truth | pred).sum()


def test_outlines_the_water_of_a_drifting_hover(drifting):
    river, clip = drifting
    right = (river.width_px - 1, 100)              # the river runs left to right
    det = detect_water(clip, 0.0, _section(river), right)
    assert det.ok and det.confidence == "good", det.message
    assert _iou(det.polygon_px, river) > 0.85
    assert det.flow_agrees is True
    assert det.contrast > 5


def test_a_downstream_point_on_the_wrong_side_is_caught(drifting):
    river, clip = drifting
    det = detect_water(clip, 0.0, _section(river), (0, 100))
    assert det.ok and det.flow_agrees is False


def test_later_start_time(drifting):
    river, clip = drifting
    det = detect_water(clip, 0.5, _section(river), None)
    assert det.ok and _iou(det.polygon_px, river) > 0.85
    assert det.flow_agrees is None                 # nothing to compare with


def test_still_water_is_refused_rather_than_guessed(tmp_path_factory):
    river, clip = _clip(tmp_path_factory, "still", v_max_ms=0.0)
    det = detect_water(clip, 0.0, _section(river), None)
    assert not det.ok and det.confidence == "failed"
    assert det.polygon_px == [] and "does not move" in det.message


def test_outline_is_a_valid_water_mask(drifting):
    river, clip = drifting
    det = detect_water(clip, 0.0, _section(river), None)
    a, b = _section(river)
    raw = survey_dict(str(clip), m_per_px=river.m_per_px, start_px=a, end_px=b,
                      water_mask_px=[det.polygon_px])
    cfg = Config.from_dict(raw)
    assert len(cfg.stabilise.water_mask_px[0]) == len(det.polygon_px) <= 48
