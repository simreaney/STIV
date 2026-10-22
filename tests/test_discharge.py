"""Discharge with real bed surveys: dry banks, waterlines between survey points.

The rectangular-channel tests in ``test_sti_and_discharge.py`` are wet from
end to end, which hides everything that goes wrong at the waterline.
"""

from __future__ import annotations

import numpy as np
import pytest

from stiv.config import ConfigError, DischargeConfig
from stiv.discharge import _wetted_edges, compute_discharge, load_bathymetry


@pytest.fixture
def dry_banked_bed(tmp_path):
    """Bed elevations with the water (level 10 m) wet from 1 m to 9 m.

    Depths at the survey points are -0.5, 0.5, 1, 1, 0.5, -0.5, so the true
    wetted width is 8 m and the true area 0.25 + 1.5 + 2 + 1.5 + 0.25 = 5.5 m2.
    """
    p = tmp_path / "bed.csv"
    p.write_text("distance_m,bed_elevation_m\n0,10.5\n2,9.5\n4,9\n6,9\n8,9.5\n10,10.5\n")
    return p


@pytest.fixture
def wide_section(tmp_path):
    """Depths for a section clicked bank-top to bank-top: wet only 2.5-17.5 m."""
    p = tmp_path / "wide.csv"
    p.write_text("distance_m,depth_m\n0,-0.5\n2.5,0\n5,1\n10,1.5\n15,1\n17.5,0\n20,-0.5\n")
    return p


class TestWaterline:
    def test_edges_interpolate_between_dry_and_wet_points(self):
        d = np.array([0.0, 2.0, 4.0, 6.0])
        assert _wetted_edges(d, np.array([-0.5, 0.5, 1.0, -1.0]), (0, 6)) == pytest.approx((1.0, 5.0))

    def test_clipped_depths_used_to_snap_the_edge_to_the_dry_point(self):
        d = np.array([0.0, 2.0, 4.0])
        depth = np.array([-0.5, 0.5, -0.5])
        assert _wetted_edges(d, depth, (0, 4)) == pytest.approx((1.0, 3.0))
        assert _wetted_edges(d, np.maximum(depth, 0), (0, 4)) == pytest.approx((0.0, 4.0))

    def test_a_profile_wet_at_its_end_reaches_the_outer_bound(self):
        d = np.array([1.0, 2.0, 3.0])
        depth = np.array([0.4, 1.0, 0.4])
        assert _wetted_edges(d, depth, (0, 4)) == (1.0, 3.0)
        assert _wetted_edges(d, depth, (0, 4), outer=(0.0, 4.0)) == (0.0, 4.0)

    def test_nothing_wet_gives_the_fallback(self):
        assert _wetted_edges(np.arange(3.0), -np.ones(3), (0.0, 9.0)) == (0.0, 9.0)

    def test_bathymetry_is_returned_unclipped(self, dry_banked_bed):
        _, depth = load_bathymetry(dry_banked_bed, water_level_m=10.0)
        assert depth[0] == pytest.approx(-0.5)


class TestDryBanks:
    def test_bed_elevation_with_dry_banks(self, dry_banked_bed):
        """Was: wetted width 10 m and dry ground given a positive depth."""
        b = np.linspace(0.0, 10.0, 11)
        cfg = DischargeConfig(enabled=True, bathymetry_csv=str(dry_banked_bed),
                              water_level_m=10.0, alpha=1.0, bank_coefficient=1.0)
        r = compute_discharge(cfg, b, np.ones(11), 10.0)

        assert r.wetted_width_m == pytest.approx(8.0)
        assert r.area_m2 == pytest.approx(5.5)
        assert r.depths_m[0] == 0.0 and r.depths_m[-1] == 0.0
        # 1 m is the waterline itself.  Interpolating the clipped profile gave
        # it a depth of 0.25 m, half-way between the "0" bank and 0.5 m.
        assert r.depths_m[1] == pytest.approx(0.0)
        assert r.depths_m[2] == pytest.approx(0.5)
        # The stations at 1 m and 9 m *are* the waterlines: zero-depth
        # verticals owning no panel, so 2 m starts half-way, at 1.5 m.
        assert r.panel_width_m[1] == 0.0 and r.panel_width_m[2] == pytest.approx(1.0)
        assert r.panel_width_m.sum() == pytest.approx(7.0)

    def test_bank_coefficient_goes_on_the_outermost_wet_stations(self, wide_section):
        """Was: Q identical for 1.0 and 0.5, because it went on dry stations."""
        b = np.linspace(0.0, 20.0, 21)
        kw = dict(enabled=True, bathymetry_csv=str(wide_section), alpha=0.85)
        full = compute_discharge(DischargeConfig(bank_coefficient=1.0, **kw), b, np.ones(21), 20.0)
        half = compute_discharge(DischargeConfig(bank_coefficient=0.5, **kw), b, np.ones(21), 20.0)

        assert half.discharge_m3s < full.discharge_m3s
        assert half.area_m2 == pytest.approx(full.area_m2)
        # Stations 3 and 17 are the outermost in the water (2.5-17.5 m).
        assert half.depth_averaged_ms[3] == pytest.approx(0.5 * 0.85)
        assert half.depth_averaged_ms[17] == pytest.approx(0.5 * 0.85)
        assert half.depth_averaged_ms[10] == pytest.approx(0.85)

    def test_dry_stations_own_no_panel(self, wide_section):
        """Was: panel widths summing to 19 m over a 15 m wetted width."""
        b = np.linspace(0.0, 20.0, 21)
        cfg = DischargeConfig(enabled=True, bathymetry_csv=str(wide_section),
                              alpha=1.0, bank_coefficient=1.0)
        r = compute_discharge(cfg, b, np.ones(21), 20.0)

        assert r.wetted_width_m == pytest.approx(15.0)
        assert np.all(r.panel_width_m[:3] == 0) and np.all(r.panel_width_m[18:] == 0)
        # Half-way between the 2.5 m waterline and 3 m, to the 3.5 m midpoint.
        assert r.panel_width_m[3] == pytest.approx(0.75)
        # The wetted width less the two quarter-metre half-gaps at the banks.
        assert r.panel_width_m.sum() == pytest.approx(14.5)
        assert len(r.panel_width_m) == 21                  # still one row per station

    def test_inline_depths_with_dry_ends(self):
        b = np.linspace(0.0, 6.0, 7)
        depths = [-0.2, 0.0, 0.5, 1.0, 0.5, 0.0, -0.2]
        cfg = DischargeConfig(enabled=True, depths_m=depths, alpha=1.0, bank_coefficient=1.0)
        r = compute_discharge(cfg, b, np.ones(7), 6.0)
        assert r.wetted_width_m == pytest.approx(4.0)
        # Stations 1 and 5 are the waterlines themselves (depth 0).
        assert r.panel_width_m.tolist() == pytest.approx([0, 0, 1, 1, 1, 0, 0])
        assert r.area_m2 == pytest.approx(2.0)

    def test_inline_depths_wet_to_the_ends_reach_the_section_ends(self):
        """Stations trimmed off the banks: the waterline is the section end,
        a zero-depth vertical owning the half-gap up to the first station."""
        b = np.linspace(0.6, 5.4, 9)
        cfg = DischargeConfig(enabled=True, depths_m=[1.0] * 9, alpha=1.0, bank_coefficient=1.0)
        r = compute_discharge(cfg, b, np.ones(9), 6.0)
        assert r.wetted_width_m == pytest.approx(6.0)
        assert r.panel_width_m[0] == pytest.approx(0.3 + 0.3)     # 0.3 m to 0.9 m
        assert r.area_m2 == pytest.approx(6.0 - 0.3 - 0.3)


class TestContinuityAtTheWaterline:
    """A zero-depth station on the waterline counted as wet and took the bank
    panel; made a millimetre negative, the next station's full depth took a
    panel all the way to the bank instead: +6% area for 1 mm of depth.
    """

    @pytest.mark.parametrize("bank", [1.0, 0.7])
    def test_area_and_discharge_are_continuous_in_the_end_depth(self, bank):
        b = np.linspace(0.0, 6.0, 9)

        def run(e):
            depths = [e, 0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5, e]
            cfg = DischargeConfig(enabled=True, depths_m=depths, alpha=1.0,
                                  bank_coefficient=bank)
            return compute_discharge(cfg, b, np.ones(9), 6.0)

        at = run(0.0)
        for e in (-1e-3, 1e-3):
            assert run(e).area_m2 == pytest.approx(at.area_m2, abs=2e-3)
        # Drying out is smooth in Q too.  (Wetting is not, quite: a station
        # with any depth at all becomes the edge vertical and takes the bank
        # coefficient from its neighbour -- inherent in giving the coefficient
        # to one vertical, and documented in compute_discharge.)
        assert run(-1e-3).discharge_m3s == pytest.approx(at.discharge_m3s, abs=2e-3)

    def test_area_is_continuous_as_stations_slide_past_the_waterline(self, wide_section):
        """The CSV case: 15.096 / 14.900 / 15.096 m2 for offsets -1, 0, +1 mm."""
        areas = []
        for shift in (-1e-3, 0.0, 1e-3):
            b = np.linspace(0.0, 20.0, 9) + shift      # stations 2.5 m apart, one on each waterline
            cfg = DischargeConfig(enabled=True, bathymetry_csv=str(wide_section),
                                  alpha=1.0, bank_coefficient=1.0)
            areas.append(compute_discharge(cfg, b, np.ones(9), 20.0).area_m2)
        assert max(areas) - min(areas) < 5e-3

    def test_bank_coefficient_acts_when_the_ends_are_zero_depth(self):
        """With zero-depth end stations it multiplied zero-area panels: Q was
        5.1000 m3/s for both 0.7 and 1.0."""
        b = np.linspace(0.0, 6.0, 9)
        depths = [0.0, 0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5, 0.0]
        q = [compute_discharge(DischargeConfig(enabled=True, depths_m=depths, alpha=1.0,
                                               bank_coefficient=c), b, np.ones(9), 6.0)
             for c in (1.0, 0.7)]
        assert q[1].discharge_m3s < q[0].discharge_m3s
        assert q[1].area_m2 == pytest.approx(q[0].area_m2)
        assert q[1].depth_averaged_ms[1] == pytest.approx(0.7)


class TestGaps:
    def test_filling_a_gap_is_reported(self):
        b = np.linspace(0.0, 10.0, 11)
        v = np.ones(11)
        v[4] = np.nan
        cfg = DischargeConfig(enabled=True, depths_m=[2.0] * 11, alpha=1.0, bank_coefficient=1.0)
        r = compute_discharge(cfg, b, v, 10.0)
        assert r.discharge_m3s == pytest.approx(20.0)
        (msg,) = r.warnings
        assert "1 station(s) in the water had no velocity" in msg and "4.00" in msg

    def test_a_complete_profile_reports_nothing(self):
        cfg = DischargeConfig(enabled=True, depths_m=[2.0] * 3, alpha=1.0)
        assert compute_discharge(cfg, [0.0, 1.0, 2.0], [1.0] * 3, 2.0).warnings == []

    def test_no_velocity_at_all_is_an_error(self):
        cfg = DischargeConfig(enabled=True, depths_m=[2.0] * 3, alpha=1.0)
        with pytest.raises(ConfigError, match="no finite surface velocities"):
            compute_discharge(cfg, [0.0, 1.0, 2.0], [np.nan] * 3, 2.0)
