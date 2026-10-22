"""Quality control, interpolation and advice, exercised directly on stations.

The contract these pin: ``StivResult.velocities`` is the *reported* profile.
A station that failed and was not interpolated reports NaN (its raw value
stays in ``velocity_raw_ms``), so no failed number can reach the discharge,
the summary or the report.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
import pytest

from stiv import analysis
from stiv.analysis import (
    StationResult,
    StivResult,
    _interpolate_flagged,
    _median_filter,
    _quality_control,
    _range_advice,
    _rectification_warnings,
    _reverse_advice,
)
from stiv.config import Config, FlowConfig
from stiv.geometry import build_section
from stiv.stabilise import StabilisationReport

SX, ST = 0.02, 1 / 30          # 0.6 m/s per unit tan(phi)


def station(i: int, v: float, failed: bool = False, spread: float = 0.02) -> StationResult:
    """A station that measured ``v``; ``failed`` gives it a coherency below QC.

    The flag is set up front as well, for the tests that call the helpers
    directly rather than through ``_quality_control``.
    """
    tan = v * ST / SX
    return StationResult(
        index=i, station_m=0.5 * i, x_world=0.0, y_world=0.0, u_px=0.0, v_px=0.0,
        phi_rad=math.atan(tan), tan_phi=tan, velocity_ms=v, velocity_raw_ms=v,
        coherency=0.01 if failed else 0.9, valid_fraction=0.9, n_windows=50,
        phi_std_rad=0.01, velocity_std_ms=spread, sx_m_per_px=SX, st_s_per_px=ST,
        flags=["low_coherency"] if failed else [],
    )


def qc_config(**quality) -> Config:
    cfg = Config()
    for k, v in quality.items():
        setattr(cfg.quality, k, v)
    return cfg


def profile(values, failed=()):
    return [station(i, v, failed=i in failed) for i, v in enumerate(values)]


# --------------------------------------------------------------------------- #
class TestFlags:
    def test_low_coherency_fails_and_is_interpolated(self):
        st = profile([1.0, 1.2, 9.0, 1.4, 1.5], failed={2})
        warnings: list[str] = []
        _quality_control(qc_config(), st, warnings)
        assert "low_coherency" in st[2].flags and "interpolated" in st[2].flags
        assert st[2].failed and not st[2].ok
        assert st[2].velocity_ms == pytest.approx(1.3)
        assert st[2].velocity_raw_ms == 9.0
        assert all(s.ok for i, s in enumerate(st) if i != 2)

    def test_high_uncertainty_is_relative_to_the_section_scale(self):
        st = profile([1.0, 1.0, 0.05, 1.0, 1.0])
        st[2].velocity_spread_ms = 0.08              # 160% of itself, 32% of the scale
        st[3].velocity_spread_ms = 0.8               # 80% of itself
        _quality_control(qc_config(), st, [])
        assert "high_uncertainty" not in st[2].flags
        assert "high_uncertainty" in st[3].flags

    def test_tan_phi_band_is_advisory(self):
        st = profile([0.01, 1.0, 1.0, 1.0])
        _quality_control(qc_config(), st, [])
        assert st[0].flags == ["tan_phi_low"]
        assert not st[0].failed and st[0].velocity_ms == 0.01


class TestInterpolation:
    def test_interpolated_station_reports_consistent_angle_and_no_spread(self):
        """velocity.csv used to pair the filled velocity with the failed angle."""
        st = profile([1.0, 1.2, 9.0, 1.4, 1.5], failed={2})
        assert _interpolate_flagged(st) == 1
        s = st[2]
        assert s.tan_phi == pytest.approx(s.velocity_ms * ST / SX)
        assert s.phi_rad == pytest.approx(math.atan(s.tan_phi))
        assert math.isnan(s.velocity_spread_ms) and math.isnan(s.velocity_std_ms)
        assert math.isnan(s.phi_std_rad)
        # The diagnostics that explain the failure are kept.
        assert s.coherency == 0.01

    def test_ends_take_the_nearest_usable_value(self):
        st = profile([7.0, 1.0, 2.0, 7.0], failed={0, 3})
        _interpolate_flagged(st)
        assert [s.velocity_ms for s in st] == [1.0, 1.0, 2.0, 2.0]

    def test_needs_two_usable_stations(self):
        st = profile([1.0, 7.0, 7.0], failed={1, 2})
        assert _interpolate_flagged(st) == 0
        assert st[1].velocity_ms == 7.0               # untouched here ...

    def test_disabled_interpolation_blanks_failed_stations(self):
        """... and quality control then blanks it, rather than reporting 7 m/s."""
        st = profile([1.0, 1.2, 9.0, 1.4], failed={2})
        _quality_control(qc_config(interpolate_flagged=False), st, [])
        assert math.isnan(st[2].velocity_ms)
        assert math.isnan(st[2].tan_phi) and math.isnan(st[2].phi_rad)
        assert st[2].velocity_raw_ms == 9.0
        assert "interpolated" not in st[2].flags
        assert st[2].failed

    def test_too_few_usable_stations_blanks_and_warns(self):
        st = profile([1.0, 7.0, 7.0], failed={1, 2})
        warnings: list[str] = []
        _quality_control(qc_config(), st, warnings)
        assert np.isnan([st[1].velocity_ms, st[2].velocity_ms]).all()
        assert st[0].velocity_ms == 1.0
        assert any("could not be interpolated" in w for w in warnings)


class TestMedianFilter:
    def test_failed_values_do_not_leak_into_good_neighbours(self):
        """The audit case: k = 5 turned good station 1 into 2.95."""
        st = profile([1.0, 1.1, 5.0, 4.8, 1.2, 1.25, 1.3], failed={2, 3})
        _quality_control(qc_config(median_filter_stations=5), st, [])
        v = np.array([s.velocity_ms for s in st])
        assert np.all(v < 1.35)
        # Only stations 0 and 1 are usable in station 1's window: two values,
        # too few for a median, so it stays as measured.
        assert st[1].velocity_ms == 1.1 and "median_filtered" not in st[1].flags
        # Station 4's window holds 1.2, 1.25, 1.3.
        assert st[4].velocity_ms == pytest.approx(1.25)
        # The gap is filled between the *filtered* neighbours, 1.1 and 1.25.
        assert st[2].velocity_ms == pytest.approx(1.1 + 0.15 / 3)
        assert st[3].velocity_ms == pytest.approx(1.1 + 0.30 / 3)

    def test_a_filtered_spike_does_not_return_through_a_filled_gap(self):
        """Interpolating first filled station 4 with 2.0, from the raw 3.0
        that the filter then removed from station 3."""
        st = profile([1.0, 1.0, 1.0, 3.0, 9.0, 1.0, 1.0, 1.0], failed={4})
        _quality_control(qc_config(median_filter_stations=5), st, [])
        assert st[3].velocity_ms == 1.0 and "median_filtered" in st[3].flags
        assert st[4].velocity_ms == pytest.approx(1.0)
        assert st[4].flags.count("low_coherency") == 1

    def test_a_window_of_two_is_not_a_median(self):
        st = profile([1.0, 2.0, 9.0, 9.0, 9.0], failed={2, 3, 4})
        _median_filter(st, 5)
        # Station 0's window holds only stations 0 and 1: left as measured,
        # not replaced by their mean of 1.5.
        assert [st[0].velocity_ms, st[1].velocity_ms] == [1.0, 2.0]
        assert not any("median_filtered" in s.flags for s in st)

    def test_it_leaves_failed_stations_to_the_interpolation(self):
        st = profile([1.0, 1.1, 5.0, 1.2, 1.3], failed={2})
        _quality_control(qc_config(median_filter_stations=3, interpolate_flagged=False), st, [])
        assert math.isnan(st[2].velocity_ms)
        assert "median_filtered" not in st[2].flags

    def test_moved_stations_are_flagged_and_consistent(self):
        st = profile([1.0, 2.0, 1.0, 1.0])
        _median_filter(st, 3)
        assert st[1].velocity_ms == 1.0 and "median_filtered" in st[1].flags
        assert st[1].tan_phi == pytest.approx(1.0 * ST / SX)
        assert math.isnan(st[1].velocity_spread_ms)
        assert st[1].velocity_raw_ms == 2.0

    def test_an_even_window_is_logged(self, caplog):
        st = profile([1.0, 2.0, 1.0, 1.0])
        with caplog.at_level(logging.WARNING, logger="stiv.analysis"):
            _median_filter(st, 4)
        assert "even" in caplog.text


class TestFlagsAndFields:
    def test_flags_are_added_once(self):
        """Running quality control over already-flagged stations must not
        stack duplicate flags."""
        st = profile([1.0, 1.2, 9.0, 1.4, 1.5], failed={2})
        _quality_control(qc_config(), st, [])
        _quality_control(qc_config(), st, [])
        assert all(len(s.flags) == len(set(s.flags)) for s in st)
        st[0].add_flag("tan_phi_low")
        st[0].add_flag("tan_phi_low")
        assert st[0].flags.count("tan_phi_low") == 1

    def test_the_old_spread_name_still_works_both_ways(self):
        s = station(0, 1.0, spread=0.03)                 # built with velocity_std_ms=
        assert s.velocity_std_ms == s.velocity_spread_ms == 0.03
        s.velocity_spread_ms = 0.05
        assert s.velocity_std_ms == 0.05
        s.velocity_std_ms = 0.07
        assert s.velocity_spread_ms == 0.07


class TestFailureWarnings:
    def test_a_short_clip_is_not_blamed_on_texture(self):
        st = profile([math.nan] * 4)
        for s in st:
            s.flags.append("sti_too_small")
        warnings: list[str] = []
        _quality_control(qc_config(), st, warnings)
        assert any("too small for the 31 px box window" in w for w in warnings)
        assert not any("visible texture" in w for w in warnings)

    def test_widespread_failure_is_reported(self):
        st = profile([1.0, 7.0, 7.0, 7.0, 1.0], failed={1, 2, 3})
        warnings: list[str] = []
        _quality_control(qc_config(), st, warnings)
        assert any("3 of 5 stations failed" in w for w in warnings)


class TestSummary:
    def _result(self, stations) -> StivResult:
        cfg = Config()
        cfg.rectify.m_per_px = 0.02
        cfg.section.start_px, cfg.section.end_px = [100, 50], [100, 250]
        cfg.section.n_stations = len(stations)
        geom = build_section(cfg, (300, 400))
        return StivResult(
            config=cfg, geometry=geom, stations=stations, stis=[], fields=[],
            estimates=[], stabilisation=StabilisationReport(enabled=False),
            n_frames=100, duration_s=3.3, fps=30.0, st_s_per_px=ST,
            reference_frame_bgr=np.zeros((2, 2, 3), np.uint8),
        )

    def test_counts_failed_and_interpolated_separately(self):
        st = profile([1.0, 7.0, 1.2, 7.0, 1.4], failed={1, 3})
        _quality_control(qc_config(), st, [])
        assert "2 failed (2 interpolated)" in self._result(st).summary()

        st = profile([1.0, 7.0, 1.2, 7.0, 1.4], failed={1, 3})
        _quality_control(qc_config(interpolate_flagged=False), st, [])
        res = self._result(st)
        assert "2 failed (0 interpolated)" in res.summary()
        # The reported profile carries no failed value anywhere.
        assert np.isnan(res.velocities[[1, 3]]).all()
        assert "velocity_std_ms" in res.to_frame().columns


class TestRangeAdvice:
    def _stations(self, tan):
        return [station(i, tan * SX / ST) for i in range(5)]

    def test_steep_streaks_at_frame_step_one_still_get_advice(self):
        """At step 1 the suggested step equalled the current one, so: silence."""
        warnings: list[str] = []
        _range_advice(Config(), self._stations(6.0), SimpleNamespace(step=1), warnings)
        (msg,) = warnings
        assert "frame rate" in msg and "fly higher" in msg and "stiv.n_samples" in msg
        assert "does not help" in msg                  # the longer-line advice was wrong
        assert "video.frame_step:" not in msg

    def test_steep_streaks_with_a_frame_step_suggest_a_smaller_one(self):
        warnings: list[str] = []
        _range_advice(Config(), self._stations(6.0), SimpleNamespace(step=3), warnings)
        (msg,) = warnings
        assert "video.frame_step: 1" in msg
        assert "search_line_length_m" not in msg

    def test_shallow_streaks_suggest_a_larger_step(self):
        warnings: list[str] = []
        _range_advice(Config(), self._stations(0.1), SimpleNamespace(step=1), warnings)
        (msg,) = warnings
        assert "video.frame_step: 10" in msg

    def test_judged_on_measured_not_filled_angles(self):
        st = self._stations(6.0)
        for s in st:                                   # as if all were filled with 1 m/s
            analysis._report_velocity(s, 1.0)
        warnings: list[str] = []
        _range_advice(Config(), st, SimpleNamespace(step=1), warnings)
        assert warnings

    def test_quiet_in_the_sweet_spot(self):
        warnings: list[str] = []
        _range_advice(Config(), self._stations(1.0), SimpleNamespace(step=1), warnings)
        assert warnings == []


class TestReverseAdvice:
    def test_says_to_toggle_reverse_whichever_way_it_is(self):
        assert _reverse_advice(FlowConfig(reverse=False)).startswith("set flow.reverse: true")
        msg = _reverse_advice(FlowConfig(reverse=True))
        assert msg.startswith("set flow.reverse: false")
        assert "reverse: true" not in msg.replace("currently true", "")

    def test_names_the_setting_that_chose_the_direction(self):
        assert "give flow.downstream_px" in _reverse_advice(FlowConfig())
        assert "check that flow.downstream_px" in _reverse_advice(FlowConfig(downstream_px=[1, 2]))
        assert "swap flow.from_px" in _reverse_advice(FlowConfig(direction="points"))
        assert "180" in _reverse_advice(FlowConfig(direction="angle", angle_deg=10.0))


class TestGcpRedundancy:
    PX = [[100, 100], [800, 100], [800, 400], [100, 400], [450, 250], [300, 350]]

    def _geom(self, n):
        cfg = Config()
        cfg.rectify.method = "gcp"
        cfg.rectify.gcp_px = self.PX[:n]
        cfg.rectify.gcp_world = [[0.02 * u, -0.02 * v] for u, v in self.PX[:n]]
        cfg.section.start_px, cfg.section.end_px = [450, 150], [450, 350]
        cfg.section.n_stations = 5
        cfg.stiv.search_line_length_m = 1.0
        return build_section(cfg, (540, 960))

    def test_four_points_are_flagged_as_unverifiable(self):
        """Four GCPs fit exactly: a 2 m blunder still shows an RMSE of zero."""
        warnings: list[str] = []
        _rectification_warnings(self._geom(4), warnings)
        assert any("only 4 ground control points" in w for w in warnings)

    def test_five_or_more_are_not(self):
        warnings: list[str] = []
        _rectification_warnings(self._geom(6), warnings)
        assert warnings == []
