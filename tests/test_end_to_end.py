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
from stiv.config import ConfigError
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


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def oblique_clip(tmp_path_factory, clean_clip):
    """The clean clip seen obliquely: every frame warped by one homography.

    The far bank is compressed to about 60% of its nadir scale, so ``Sx`` is
    only constant along a search line if the samples really are spaced in
    world metres through the fitted homography.
    """
    import cv2

    from stiv.synthetic import _open_writer

    river, nadir, _ = clean_clip
    d = tmp_path_factory.mktemp("oblique")
    src = np.float32([[0, 0], [720, 0], [720, 420], [0, 420]])
    dst = np.float32([[130, 30], [590, 30], [720, 420], [0, 420]])
    H = cv2.getPerspectiveTransform(src, dst)

    cap = cv2.VideoCapture(str(nadir))
    writer, path = _open_writer(d / "oblique", river.fps, 720, 420)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.warpPerspective(frame, H, (720, 420), flags=cv2.INTER_LINEAR))
    writer.release()
    cap.release()

    def to_image(p):
        return cv2.perspectiveTransform(np.float64(p).reshape(-1, 1, 2), H).reshape(-1, 2).tolist()

    nadir_gcps = [[40, 40], [680, 40], [680, 380], [40, 380], [360, 60], [200, 370]]
    raw = make_config(river, path, d / "out", n_stations=15)
    raw["rectify"] = {
        "method": "gcp",
        "gcp_px": to_image(nadir_gcps),
        # World = the nadir view's own metres, so the truth profile still applies.
        "gcp_world": [[u * river.m_per_px, -v * river.m_per_px] for u, v in nadir_gcps],
    }
    start, end = to_image([[360, river.bank_top_px], [360, river.bank_bottom_px]])
    raw["section"].update(start_px=start, end_px=end)
    raw["flow"]["downstream_px"] = to_image([[719, 210]])[0]
    return river, raw


class TestObliqueView:
    def test_gcp_rectified_oblique_footage_recovers_the_profile(self, oblique_clip):
        river, raw = oblique_clip
        result = run(Config.from_dict(raw), progress=False)
        stats = error_stats(result, river)
        assert stats["rmse"] < 0.05
        assert stats["rel"] < 0.10
        # The lines run from near to far field, yet every one has a single Sx.
        sl = result.geometry.search_lines[0]
        steps = np.linalg.norm(np.diff(sl.samples_world, axis=0), axis=1)
        assert np.allclose(steps, sl.sx_m_per_px, rtol=1e-6)


class TestRunLevelFailures:
    def test_every_station_failing_does_not_lose_the_result(self, clean_clip):
        """A discharge error used to throw the whole velocity result away."""
        river, video, d = clean_clip
        res = analyse(river, video, d / "allfail",
                      stiv={"velocity_range_ms": [2.0, 2.1]},
                      discharge={"enabled": True, "depths_m": [1.0] * 15})
        assert all(s.failed for s in res.stations)
        assert np.isnan(res.velocities).all()          # no failed value is reported
        assert res.discharge is None
        assert any("discharge was not computed" in w for w in res.warnings)
        assert "15 failed (0 interpolated)" in res.summary()

    def test_a_clip_shorter_than_the_window_is_rejected_up_front(self, clean_clip):
        river, video, d = clean_clip
        with pytest.raises(ConfigError, match="STI rows"):
            analyse(river, video, d / "short", video={"duration_s": 0.5})

    def test_search_lines_shorter_than_the_window_are_rejected_up_front(self, clean_clip):
        river, video, d = clean_clip
        with pytest.raises(ConfigError, match="samples"):
            analyse(river, video, d / "narrow", stiv={"search_line_length_m": 0.4})

    def test_reverse_advice_says_to_toggle_it(self, clean_clip):
        river, video, d = clean_clip
        res = analyse(river, video, d / "noflip", flow={"reverse": True, "auto_flip": False})
        (msg,) = [w for w in res.warnings if "opposite to the assumed downstream" in w]
        assert "flow.reverse: false" in msg

    def test_filled_stations_report_a_consistent_angle(self, clean_clip):
        river, video, d = clean_clip
        res = analyse(river, video, d / "consistent")
        filled = [s for s in res.stations if "interpolated" in s.flags]
        assert filled, "nothing was interpolated, so this proves nothing"
        for s in filled:
            assert s.tan_phi * s.sx_m_per_px / s.st_s_per_px == pytest.approx(s.velocity_ms)
            assert np.isnan(s.velocity_std_ms)


class TestStabiliserSwitchedOff:
    def test_too_few_features_is_a_loud_warning(self, drone_clip):
        river, video, d = drone_clip
        res = analyse(river, video, d / "nofeatures",
                      stabilise={"enabled": True,
                                 "static_mask_px": [[[3, 3], [6, 3], [6, 6]]]})
        assert res.stabilisation.auto_disabled
        assert any("STABILISATION WAS SWITCHED OFF" in w for w in res.warnings)
        assert "DISABLED AUTOMATICALLY" in res.summary()


class TestRawStream:
    def test_a_stream_with_no_frame_count_is_read_to_its_end(self, clean_clip, tmp_path):
        """No frame count used to mean a 10**9-row buffer, i.e. terabytes."""
        import cv2

        river, video, d = clean_clip
        raw = tmp_path / "river.mjpeg"
        cap = cv2.VideoCapture(str(video))
        with raw.open("wb") as fh:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                fh.write(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
        cap.release()

        res = analyse(river, raw, d / "raw", video={"path": str(raw), "fps_override": river.fps})
        assert res.n_frames == river.n_frames
        assert error_stats(res, river)["rmse"] < 0.05
