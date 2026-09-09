"""End-to-end tests against synthetic footage with a known velocity field.

These are the tests that matter: everything else checks a component in
isolation, whereas these check that the whole chain -- rectification, sampling,
stabilisation, orientation analysis, Eq. (16) -- returns the velocity that was
actually rendered into the video.
"""

from __future__ import annotations

import numpy as np
import pytest

from stiv import Config, run
from stiv.analysis import ADVISORY_FLAGS
from stiv.synthetic import SyntheticRiver, make_config, render


@pytest.fixture(scope="module")
def clean_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("clean")
    river = SyntheticRiver(width_px=720, height_px=420, duration_s=6.0,
                           bank_top_px=90, bank_bottom_px=330, v_max_ms=1.2)
    return river, render(river, d / "river", progress=False), d


@pytest.fixture(scope="module")
def drone_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("drone")
    river = SyntheticRiver(width_px=720, height_px=420, duration_s=6.0,
                           bank_top_px=90, bank_bottom_px=330, v_max_ms=1.2,
                           drone_drift_px=30.0, drone_sway_px=4.0, drone_yaw_deg=1.2)
    return river, render(river, d / "river", progress=False), d


def analyse(_river, _video, _out_dir, n_stations=15, **overrides):
    # Leading underscores keep these from colliding with config section names
    # passed through **overrides (``video=...``, ``flow=...`` and so on).
    raw = make_config(_river, _video, _out_dir, n_stations=n_stations)
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)
    return run(Config.from_dict(raw), progress=False)


def error_stats(result, river, n_stations=15):
    _, truth = river.truth_profile(n_stations)
    err = result.velocities - truth
    moving = truth > 0.2
    return {
        "rmse": float(np.sqrt(np.mean(err[moving] ** 2))),
        "bias": float(err[moving].mean()),
        "rel": float(np.sqrt(np.mean((err[moving] / truth[moving]) ** 2))),
        "max": float(np.abs(err[moving]).max()),
    }


class TestFixedCamera:
    def test_recovers_the_velocity_profile(self, clean_clip):
        river, video, d = clean_clip
        result = analyse(river, video, d / "out")
        stats = error_stats(result, river)
        assert stats["rmse"] < 0.05
        assert abs(stats["bias"]) < 0.03
        assert stats["rel"] < 0.10

    def test_profile_peaks_mid_channel(self, clean_clip):
        river, video, d = clean_clip
        v = analyse(river, video, d / "out2").velocities
        assert 4 <= int(np.argmax(v)) <= 10        # centre of 15 stations

    def test_velocities_are_positive_downstream(self, clean_clip):
        river, video, d = clean_clip
        result = analyse(river, video, d / "out3")
        assert np.nanmedian(result.velocities) > 0

    def test_reversed_flow_is_detected_and_corrected(self, clean_clip):
        """Search lines pointing upstream must still give the right magnitudes.

        The asymmetric velocity search range [-0.5, 4] does not contain the
        true angle when the lines face upstream, so merely negating the first
        pass would turn values pinned at the range boundary into
        plausible-looking positive numbers. The correction has to re-run
        Eq. (14) over a mirrored range.
        """
        river, video, d = clean_clip
        forward = analyse(river, video, d / "fwd")
        reversed_ = analyse(river, video, d / "rev", flow={"reverse": True})

        assert any("opposite to the assumed downstream" in w for w in reversed_.warnings)
        assert np.nanmedian(reversed_.velocities) > 0

        # The corrected profile must match the forward one, not merely its sign.
        stats = error_stats(reversed_, river)
        assert stats["rmse"] < 0.05
        both = np.isfinite(forward.velocities) & np.isfinite(reversed_.velocities)
        assert np.allclose(forward.velocities[both], reversed_.velocities[both], atol=0.05)

    def test_reversed_flow_is_corrected_when_no_window_is_in_range(self, clean_clip):
        """The first pass can come back entirely NaN, and must still recover.

        The test above leans on noise scattering *some* windows into the
        configured range, which leaves a negative median to read the direction
        from.  Narrow the range and nothing lands inside it at all: every
        station's first pass is NaN, so there is no sign to read and the
        correction never fires -- which reported a wholly failed
        cross-section rather than a reversed one.  The direction is therefore
        probed over the full angle range, which has no assumed direction to be
        wrong about.
        """
        river, video, d = clean_clip
        res = analyse(river, video, d / "rev_narrow",
                      flow={"reverse": True},
                      stiv={"velocity_range_ms": (0.3, 1.5)})

        assert any("opposite to the assumed downstream" in w for w in res.warnings)
        assert np.nanmedian(res.velocities) > 0
        measured = sum(1 for st in res.stations if not st.failed)
        assert measured > 0.5 * len(res.stations)

    def test_frame_step_does_not_change_the_answer(self, clean_clip):
        """tan(phi) doubles with frame_step, but Sx/St halves, so U is invariant.

        Compared only over stations that both runs actually measured -- a
        station that failed QC carries an interpolated fill value, which says
        nothing about whether Eq. (16) is scale-invariant.
        """
        river, video, d = clean_clip
        a = analyse(river, video, d / "out5", video={"frame_step": 1})
        b = analyse(river, video, d / "out6", video={"frame_step": 2})

        measured = np.array([not (x.failed or y.failed)
                             for x, y in zip(a.stations, b.stations)])
        assert measured.sum() >= 10
        assert np.allclose(a.velocities[measured], b.velocities[measured], atol=0.06)

        # Doubling the time step doubles the streak inclination.
        tan_a = np.nanmedian([abs(s.tan_phi) for s in a.stations])
        tan_b = np.nanmedian([abs(s.tan_phi) for s in b.stations])
        assert tan_b == pytest.approx(2 * tan_a, rel=0.15)

    def test_search_line_length_does_not_change_the_answer(self, clean_clip):
        river, video, d = clean_clip
        a = analyse(river, video, d / "out7", stiv={"search_line_length_m": 2.0})
        b = analyse(river, video, d / "out8", stiv={"search_line_length_m": 4.0})
        assert error_stats(a, river)["rmse"] < 0.06
        assert error_stats(b, river)["rmse"] < 0.06

    @pytest.mark.parametrize("estimator", ["eq14", "peak"])
    def test_both_angle_estimators_work(self, clean_clip, estimator):
        river, video, d = clean_clip
        result = analyse(river, video, d / f"out_{estimator}",
                         stiv={"estimator": estimator})
        assert error_stats(result, river)["rmse"] < 0.06


class TestDroneMotion:
    def test_stabilisation_recovers_the_profile(self, drone_clip):
        river, video, d = drone_clip
        result = analyse(river, video, d / "on", stabilise={"enabled": True})
        stats = error_stats(result, river)
        assert stats["rmse"] < 0.06
        assert result.stabilisation.enabled
        assert result.stabilisation.n_failed == 0

    def test_ignoring_camera_motion_is_much_worse(self, drone_clip):
        """The regression guard for the whole stabilisation module."""
        river, video, d = drone_clip
        on = error_stats(analyse(river, video, d / "a", stabilise={"enabled": True}), river)
        off = error_stats(analyse(river, video, d / "b", stabilise={"enabled": False}), river)
        assert off["rmse"] > 4 * on["rmse"]

    @pytest.mark.parametrize("model", ["similarity", "affine", "homography"])
    def test_all_transform_models_work(self, drone_clip, model):
        river, video, d = drone_clip
        result = analyse(river, video, d / model, stabilise={"enabled": True, "model": model})
        assert error_stats(result, river)["rmse"] < 0.10


class TestOutputsAndQuality:
    def test_result_frame_has_a_row_per_station(self, clean_clip):
        river, video, d = clean_clip
        result = analyse(river, video, d / "df", n_stations=9)
        df = result.to_frame()
        assert len(df) == 9
        assert {"station_m", "velocity_ms", "coherency", "flags"} <= set(df.columns)

    def test_discharge_integrates_the_profile(self, clean_clip):
        river, video, d = clean_clip
        result = analyse(
            river, video, d / "q", n_stations=15,
            discharge={"enabled": True, "depths_m": [1.5] * 15, "alpha": 0.85,
                       "bank_coefficient": 1.0},
        )
        q = result.discharge
        assert q is not None
        assert q.area_m2 == pytest.approx(river.channel_width_m * 1.5, rel=0.02)
        # For stations spanning the full wetted width at uniform spacing, the
        # mid-section panels reduce exactly to the trapezoid rule -- so an
        # independent trapezoidal integration of the measured profile is the
        # right thing to check the integration, alpha and depth against.
        expected = 0.85 * 1.5 * float(np.trapezoid(result.velocities, q.stations_m))
        assert q.discharge_m3s == pytest.approx(expected, rel=1e-6)
        # The stations against each bank own only half a panel.
        assert q.panel_width_m[0] == pytest.approx(q.panel_width_m[1] / 2)
        assert q.panel_width_m[-1] == pytest.approx(q.panel_width_m[-2] / 2)
        assert q.panel_width_m.sum() == pytest.approx(river.channel_width_m)

    def test_median_filter_flags_the_stations_it_moves(self, clean_clip):
        """A substituted velocity that still reports ``ok`` is worse than none.

        The filter is advisory, so it must not make a station eligible for
        interpolation either.
        """
        river, video, d = clean_clip
        res = analyse(river, video, d / "medfilt",
                      quality={"median_filter_stations": 3})

        moved = [st for st in res.stations if "median_filtered" in st.flags]
        assert moved, "the filter changed nothing, so this proves nothing"
        assert all(not st.ok for st in moved)
        assert all("median_filtered" in ADVISORY_FLAGS for _ in moved)
        assert not any(st.failed and st.flags == ["median_filtered"] for st in moved)

    def test_figures_are_written(self, clean_clip, tmp_path):
        from stiv import plotting

        river, video, d = clean_clip
        result = analyse(river, video, d / "fig", n_stations=9)
        written = plotting.save_all(result, tmp_path / "figs", dpi=60)
        assert {p.name for p in written} >= {
            "overview.png", "velocity_profile.png", "sti_panel.png"
        }
        assert all(p.stat().st_size > 1000 for p in written)

    def test_velocity_range_outside_the_truth_is_flagged(self, clean_clip):
        """If the search range excludes the real velocity, say so rather than
        returning a confident number from the wrong part of the histogram."""
        river, video, d = clean_clip
        result = analyse(river, video, d / "range",
                         stiv={"velocity_range_ms": [3.0, 6.0]})
        assert sum(s.failed for s in result.stations) > len(result.stations) // 2
