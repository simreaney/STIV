"""Tests for rectification and cross-section layout."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stiv.config import Config, ConfigError
from stiv.geometry import (
    HomographyRectifier,
    ScaleRectifier,
    build_section,
    check_lines_in_frame,
)

SHAPE = (540, 960)


def base_config(**overrides):
    raw = {
        "video": {"path": "dummy.mp4"},
        "rectify": {"method": "scale", "m_per_px": 0.02},
        "section": {"start_px": [480, 120], "end_px": [480, 420], "n_stations": 11},
        "flow": {"direction": "auto", "downstream_px": [900, 270]},
        "stiv": {"search_line_length_m": 3.0},
    }
    for k, v in overrides.items():
        raw.setdefault(k, {}).update(v)
    return Config.from_dict(raw)


class TestScaleRectifier:
    def test_round_trip(self):
        r = ScaleRectifier(0.02, origin_px=(480, 270))
        pts = np.array([[0.0, 0.0], [480.0, 270.0], [959.0, 539.0]])
        assert np.allclose(r.world_to_image(r.image_to_world(pts)), pts)

    def test_distances_scale(self):
        r = ScaleRectifier(0.05)
        w = r.image_to_world([[0, 0], [100, 0]])
        assert np.linalg.norm(w[1] - w[0]) == pytest.approx(5.0)

    def test_rejects_non_positive_scale(self):
        with pytest.raises(ConfigError, match="positive"):
            ScaleRectifier(0.0)


class TestHomographyRectifier:
    def test_recovers_a_known_similarity(self):
        # A pure scale+offset must be reproduced exactly by the homography.
        px = [[100, 100], [800, 100], [800, 400], [100, 400]]
        world = [[0.0, 0.0], [14.0, 0.0], [14.0, 6.0], [0.0, 6.0]]
        r = HomographyRectifier(px, world, SHAPE)
        assert r.rmse_m < 1e-6
        assert np.allclose(r.image_to_world(px), world, atol=1e-6)

    def test_round_trip_on_an_oblique_view(self):
        px = [[120, 460], [860, 470], [700, 230], [280, 225]]
        world = [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]]
        r = HomographyRectifier(px, world, SHAPE)
        probe = np.array([[400.0, 350.0], [500.0, 300.0]])
        assert np.allclose(r.world_to_image(r.image_to_world(probe)), probe, atol=1e-6)

    def test_local_scale_grows_with_range_in_an_oblique_view(self):
        # Ground sample distance must be coarser near the horizon.
        px = [[120, 460], [860, 470], [700, 230], [280, 225]]
        world = [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]]
        r = HomographyRectifier(px, world, SHAPE)
        near = r._local_scale([[490, 465]])[0]
        far = r._local_scale([[490, 228]])[0]
        assert far > near

    def test_rejects_too_few_points(self):
        with pytest.raises(ConfigError, match="at least 4"):
            HomographyRectifier([[0, 0], [1, 0], [0, 1]],
                                [[0, 0], [1, 0], [0, 1]], SHAPE)


class TestSection:
    def test_width_matches_the_scale(self):
        geom = build_section(base_config(), SHAPE)
        # 300 px at 2 cm/px
        assert geom.width_m == pytest.approx(6.0)

    def test_stations_are_evenly_spaced_in_world_space(self):
        geom = build_section(base_config(), SHAPE)
        spacing = np.diff(geom.stations_m)
        assert np.allclose(spacing, spacing[0])

    def test_sx_matches_the_requested_line_length(self):
        cfg = base_config(stiv={"search_line_length_m": 3.0, "n_samples": 151})
        geom = build_section(cfg, SHAPE)
        sl = geom.search_lines[0]
        assert sl.sx_m_per_px == pytest.approx(3.0 / 150)
        # And the sampled points really do span that distance.
        span = np.linalg.norm(sl.samples_world[-1] - sl.samples_world[0])
        assert span == pytest.approx(3.0)

    def test_samples_are_uniform_in_world_space(self):
        cfg = base_config(
            rectify={
                "method": "gcp",
                "gcp_px": [[120, 460], [860, 470], [700, 230], [280, 225]],
                "gcp_world": [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]],
            },
            section={"start_px": [300, 300], "end_px": [700, 320]},
        )
        geom = build_section(cfg, SHAPE)
        steps = np.linalg.norm(np.diff(geom.search_lines[0].samples_world, axis=0), axis=1)
        # Even under perspective, world spacing is constant -- that is the point
        # of sampling through the rectifier rather than along image pixels.
        assert np.allclose(steps, steps[0], rtol=1e-9)

    def test_edge_trim_pulls_stations_off_the_banks(self):
        geom = build_section(base_config(section={"edge_trim": 0.1}), SHAPE)
        assert geom.stations_m[0] == pytest.approx(0.6)
        assert geom.stations_m[-1] == pytest.approx(5.4)

    def test_downstream_hint_orients_the_flow(self):
        right = build_section(base_config(flow={"downstream_px": [900, 270]}), SHAPE)
        left = build_section(base_config(flow={"downstream_px": [50, 270]}), SHAPE)
        assert right.flow_dir_world[0] > 0
        assert left.flow_dir_world[0] < 0

    def test_flow_is_perpendicular_to_the_section(self):
        geom = build_section(base_config(), SHAPE)
        section_dir = geom.end_world - geom.start_world
        section_dir /= np.linalg.norm(section_dir)
        assert abs(float(np.dot(section_dir, geom.flow_dir_world))) < 1e-9

    def test_explicit_angle_is_honoured(self):
        geom = build_section(base_config(flow={"direction": "angle", "angle_deg": 30.0}), SHAPE)
        assert geom.flow_bearing_deg == pytest.approx(30.0)

    def test_reverse_flips_the_direction(self):
        a = build_section(base_config(), SHAPE)
        b = build_section(base_config(flow={"reverse": True}), SHAPE)
        assert np.allclose(a.flow_dir_world, -b.flow_dir_world)

    def test_detects_lines_leaving_the_frame(self):
        # A 40 m search line at 2 cm/px is 2000 px long: far wider than the frame.
        geom = build_section(base_config(stiv={"search_line_length_m": 40.0}), SHAPE)
        assert len(check_lines_in_frame(geom, SHAPE)) == len(geom.search_lines)

    def test_lines_inside_the_frame_are_not_flagged(self):
        geom = build_section(base_config(), SHAPE)
        assert check_lines_in_frame(geom, SHAPE) == []


class TestConfigValidation:
    def test_missing_section_is_rejected(self):
        with pytest.raises(ConfigError, match="start_px"):
            Config.from_dict({"video": {"path": "x.mp4"},
                              "rectify": {"method": "scale", "m_per_px": 0.02}})

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown key"):
            base_config(stiv={"windo_px": 31})

    def test_even_window_is_rejected(self):
        with pytest.raises(ConfigError, match="odd"):
            base_config(stiv={"window_px": 30})

    def test_gcp_without_world_coordinates_is_rejected(self):
        with pytest.raises(ConfigError, match="gcp_px and gcp_world"):
            base_config(rectify={"method": "gcp", "m_per_px": None})

    def test_inverted_velocity_range_is_rejected(self):
        with pytest.raises(ConfigError, match="velocity_range_ms"):
            base_config(stiv={"velocity_range_ms": [5.0, 1.0]})

    def test_gsd_can_be_derived_from_flight_geometry(self):
        cfg = base_config(rectify={"method": "scale", "m_per_px": None,
                                   "altitude_m": 60.0, "hfov_deg": 84.0})
        gsd = cfg.rectify.resolve_m_per_px(960)
        expected = 2 * 60.0 * math.tan(math.radians(84.0) / 2) / 960
        assert gsd == pytest.approx(expected)


class TestGcpOutlierRejection:
    """``cv2.findHomography`` measures its RANSAC error in the *destination*
    space.  The destination here is world metres, so a bare threshold of 3.0
    means three metres -- 150 px at 2 cm/px -- and the robust fit is off.
    """

    MPP = 0.02
    PX = np.array([[100, 100], [900, 100], [900, 700], [100, 700],
                   [500, 400], [300, 600]], float)

    def world(self):
        return np.column_stack([self.PX[:, 0] * self.MPP, -self.PX[:, 1] * self.MPP])

    def test_threshold_is_three_pixels_worth_of_metres(self):
        r = HomographyRectifier(self.PX, self.world(), SHAPE)
        assert r.inlier_threshold_m == pytest.approx(3.0 * self.MPP, rel=0.05)

    def test_a_blundered_control_point_is_rejected(self):
        blundered = self.world()
        blundered[4] += (2.0, 0.0)              # a 2 m (100 px) transcription error
        r = HomographyRectifier(self.PX, blundered, SHAPE)
        assert r.n_inliers == len(self.PX) - 1

    def test_rmse_is_over_the_inliers_only(self):
        """The fit is exact on the five good points; counting the rejected
        blunder gave 0.8 m and a warning that every velocity was that wrong."""
        blundered = self.world()
        blundered[4] += (2.0, 0.0)
        r = HomographyRectifier(self.PX, blundered, SHAPE)
        assert r.rmse_m < 1e-6
        assert r.residuals_m[4] == pytest.approx(2.0, rel=1e-3)   # still on show

    def test_ordinary_clicking_error_is_kept(self):
        jittered = self.world()
        jittered[4] += (2.0 * self.MPP, 0.0)    # 2 px, i.e. a steady hand
        r = HomographyRectifier(self.PX, jittered, SHAPE)
        assert r.n_inliers == len(self.PX)

    def test_coincident_image_points_are_rejected(self):
        px = np.zeros((4, 2))
        with pytest.raises(ConfigError, match="coincident"):
            HomographyRectifier(px, self.world()[:4], SHAPE)


class TestSampleDensity:
    """``n_samples`` was set so the *mean* pixel spacing along a line was one
    pixel.  In an oblique view the near end covers more pixels per metre than
    the far end, so the near field was undersampled.
    """

    OBLIQUE = {
        "method": "gcp",
        "gcp_px": [[100, 520], [860, 520], [600, 200], [360, 200]],
        "gcp_world": [[0.0, 0.0], [20.0, 0.0], [20.0, 30.0], [0.0, 30.0]],
    }

    def _oblique(self):
        cfg = base_config(
            rectify=self.OBLIQUE,
            section={"start_px": [300, 400], "end_px": [660, 400]},
            flow={"direction": "angle", "angle_deg": 90.0},
            stiv={"search_line_length_m": 8.0},
        )
        return build_section(cfg, SHAPE)

    def test_no_gap_wider_than_a_pixel_in_an_oblique_view(self):
        for sl in self._oblique().search_lines:
            gaps = np.linalg.norm(np.diff(sl.samples_px, axis=0), axis=1)
            # Within the rounding of the sample count; the mean rule gave 1.5.
            assert gaps.max() <= 1.0 + 0.5 / (sl.n_samples - 1) + 1e-9
            mean_rule = int(round(np.linalg.norm(sl.samples_px[-1] - sl.samples_px[0]))) + 1
            assert sl.n_samples > 1.4 * mean_rule

    @pytest.mark.parametrize("length_m", [1.0, 2.99, 3.0, 3.01, 4.37])
    def test_nadir_is_unchanged(self, length_m):
        geom = build_section(base_config(stiv={"search_line_length_m": length_m}), SHAPE)
        for sl in geom.search_lines:
            ends = np.linalg.norm(sl.samples_px[-1] - sl.samples_px[0])
            assert sl.n_samples == int(round(ends)) + 1
