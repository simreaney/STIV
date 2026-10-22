"""Tests for STI sampling/conditioning and the discharge integration."""

from __future__ import annotations

import numpy as np
import pytest

from stiv.config import ConfigError, DischargeConfig, StiConfig
from stiv.discharge import compute_discharge, mid_section_widths
from stiv.sti import preprocess, sample_bilinear, to_display


class TestBilinearSampling:
    def test_exact_at_pixel_centres(self):
        img = np.arange(20, dtype=np.float32).reshape(4, 5)
        got = sample_bilinear(img, np.array([[0.0, 0.0], [4.0, 3.0], [2.0, 1.0]]))
        assert got == pytest.approx([0.0, 19.0, 7.0])

    def test_interpolates_a_ramp_linearly(self):
        img = np.tile(np.arange(10, dtype=np.float32), (10, 1))
        got = sample_bilinear(img, np.array([[3.5, 5.0], [0.25, 2.0]]))
        assert got == pytest.approx([3.5, 0.25])

    def test_out_of_frame_returns_nan(self):
        img = np.ones((10, 10), np.float32)
        got = sample_bilinear(img, np.array([[-1.0, 5.0], [5.0, 12.0], [5.0, 5.0]]))
        assert np.isnan(got[0]) and np.isnan(got[1])
        assert got[2] == pytest.approx(1.0)


class TestPreprocess:
    def test_removes_a_static_vertical_stripe(self):
        """A rock in the water writes a vertical stripe that means 'zero flow'.

        Background removal has to erase it, otherwise those windows vote hard
        for tan(phi) = 0 and drag the Eq. (14) average down.
        """
        rng = np.random.default_rng(0)
        moving = rng.standard_normal((100, 100))
        stationary = moving.copy()
        stationary[:, 40] += 30.0            # a bright, unmoving feature

        cleaned = preprocess(stationary, StiConfig(remove_static_background=True))
        reference = preprocess(moving, StiConfig(remove_static_background=True))
        assert np.abs(cleaned[:, 40] - reference[:, 40]).max() < 1e-6

    def test_removes_frame_wide_exposure_flicker(self):
        rng = np.random.default_rng(1)
        base = rng.standard_normal((100, 100))
        flickering = base + np.linspace(0, 25, 100)[:, None]   # per-frame brightness drift
        a = preprocess(flickering, StiConfig(remove_static_background=True, normalise=False))
        b = preprocess(base, StiConfig(remove_static_background=True, normalise=False))
        assert np.abs(a - b).max() < 1e-6

    def test_normalisation_gives_unit_variance(self):
        rng = np.random.default_rng(2)
        # smooth_sigma is pinned off: the default Gaussian smoothing runs
        # *after* normalisation and would take the variance of white noise back
        # down well below one.  This isolates the normalisation step.
        out = preprocess(rng.standard_normal((60, 60)) * 17.0 + 3.0,
                         StiConfig(remove_static_background=False, normalise=True,
                                   smooth_sigma=0.0))
        assert out.std() == pytest.approx(1.0, rel=1e-6)

    def test_nans_are_filled(self):
        a = np.ones((40, 40))
        a[5:8, 10] = np.nan
        assert np.isfinite(preprocess(a, StiConfig())).all()

    def test_smoothing_reduces_high_frequency_content(self):
        rng = np.random.default_rng(3)
        raw = rng.standard_normal((80, 80))
        rough = preprocess(raw, StiConfig(smooth_sigma=0.0))
        smooth = preprocess(raw, StiConfig(smooth_sigma=2.0))
        assert np.abs(np.diff(smooth, axis=1)).mean() < np.abs(np.diff(rough, axis=1)).mean()


class TestDisplay:
    def test_stretches_to_full_range(self):
        out = to_display(np.linspace(-5, 5, 400).reshape(20, 20))
        assert out.dtype == np.uint8
        assert out.min() == 0 and out.max() == 255

    def test_handles_a_flat_image(self):
        assert to_display(np.zeros((10, 10))).max() == 0


class TestMidSection:
    def test_panels_tile_the_wetted_width(self):
        stations = np.array([1.0, 2.0, 3.0, 4.0])
        w = mid_section_widths(stations, 0.5, 4.5)
        assert w.sum() == pytest.approx(4.0)

    def test_interior_panel_spans_half_to_each_neighbour(self):
        w = mid_section_widths(np.array([0.0, 1.0, 3.0]), 0.0, 3.0)
        assert w[1] == pytest.approx(1.5)   # 0.5 left + 1.0 right

    def test_single_station_takes_the_whole_width(self):
        assert mid_section_widths(np.array([2.0]), 0.0, 5.0)[0] == pytest.approx(5.0)


class TestDischarge:
    def test_rectangular_channel_with_uniform_flow(self):
        """10 m wide, 2 m deep, 1 m/s surface, alpha = 1, no bank reduction.

        Q must be exactly 10 x 2 x 1 = 20 m3/s.
        """
        stations = np.linspace(0.0, 10.0, 11)
        cfg = DischargeConfig(enabled=True, depths_m=[2.0] * 11, alpha=1.0,
                              bank_coefficient=1.0)
        r = compute_discharge(cfg, stations, [1.0] * 11, 10.0)
        assert r.discharge_m3s == pytest.approx(20.0)
        assert r.area_m2 == pytest.approx(20.0)
        assert r.mean_velocity_ms == pytest.approx(1.0)

    def test_velocity_index_scales_the_discharge(self):
        stations = np.linspace(0.0, 10.0, 11)
        kw = dict(enabled=True, depths_m=[2.0] * 11, bank_coefficient=1.0)
        full = compute_discharge(DischargeConfig(alpha=1.0, **kw), stations, [1.0] * 11, 10.0)
        idx = compute_discharge(DischargeConfig(alpha=0.85, **kw), stations, [1.0] * 11, 10.0)
        assert idx.discharge_m3s == pytest.approx(0.85 * full.discharge_m3s)

    def test_bank_coefficient_reduces_the_edge_panels(self):
        stations = np.linspace(0.0, 10.0, 11)
        kw = dict(enabled=True, depths_m=[2.0] * 11, alpha=1.0)
        full = compute_discharge(DischargeConfig(bank_coefficient=1.0, **kw),
                                 stations, [1.0] * 11, 10.0)
        cut = compute_discharge(DischargeConfig(bank_coefficient=0.5, **kw),
                                stations, [1.0] * 11, 10.0)
        assert cut.discharge_m3s < full.discharge_m3s

    def test_triangular_channel_matches_the_analytic_area(self):
        # Depth rising linearly from 0 at each bank to 2 m mid-channel:
        # area = 0.5 * base * height = 0.5 * 10 * 2 = 10 m2.
        stations = np.linspace(0.0, 10.0, 101)
        depths = 2.0 * (1.0 - np.abs(stations - 5.0) / 5.0)
        cfg = DischargeConfig(enabled=True, depths_m=depths.tolist(), alpha=1.0,
                              bank_coefficient=1.0)
        r = compute_discharge(cfg, stations, [1.0] * 101, 10.0)
        assert r.area_m2 == pytest.approx(10.0, rel=0.01)

    def test_nan_velocities_are_interpolated(self):
        stations = np.linspace(0.0, 10.0, 11)
        v = [1.0] * 11
        v[5] = np.nan
        cfg = DischargeConfig(enabled=True, depths_m=[2.0] * 11, alpha=1.0,
                              bank_coefficient=1.0)
        assert compute_discharge(cfg, stations, v, 10.0).discharge_m3s == pytest.approx(20.0)

    def test_mismatched_depth_count_is_rejected(self):
        cfg = DischargeConfig(enabled=True, depths_m=[1.0, 2.0])
        with pytest.raises(ConfigError, match="depths_m"):
            compute_discharge(cfg, [0.0, 1.0, 2.0], [1.0, 1.0, 1.0], 2.0)

    def test_bathymetry_csv_is_interpolated_onto_stations(self, tmp_path):
        p = tmp_path / "bathy.csv"
        p.write_text("distance_m,depth_m\n0,0\n5,2\n10,0\n")
        cfg = DischargeConfig(enabled=True, bathymetry_csv=str(p), alpha=1.0,
                              bank_coefficient=1.0)
        r = compute_discharge(cfg, [2.5, 5.0, 7.5], [1.0, 1.0, 1.0], 10.0)
        assert r.depths_m == pytest.approx([1.0, 2.0, 1.0])
        assert r.wetted_width_m == pytest.approx(10.0)

    def test_bed_elevation_needs_a_water_level(self, tmp_path):
        p = tmp_path / "bed.csv"
        p.write_text("distance_m,bed_elevation_m\n0,10\n5,8\n10,10\n")
        cfg = DischargeConfig(enabled=True, bathymetry_csv=str(p))
        with pytest.raises(ConfigError, match="water_level_m"):
            compute_discharge(cfg, [5.0], [1.0], 10.0)

    def test_bed_elevation_converts_to_depth(self, tmp_path):
        p = tmp_path / "bed.csv"
        p.write_text("distance_m,bed_elevation_m\n0,10\n5,8\n10,10\n")
        cfg = DischargeConfig(enabled=True, bathymetry_csv=str(p), water_level_m=10.0,
                              alpha=1.0, bank_coefficient=1.0)
        r = compute_discharge(cfg, [5.0], [1.0], 10.0)
        assert r.depths_m[0] == pytest.approx(2.0)


class TestBankCoefficient:
    def test_it_does_not_shrink_the_reported_area(self):
        """It is a *velocity* coefficient, so the area must stay geometric.

        Applied to the panel width instead it yields an identical discharge --
        Q is linear in both -- while understating the cross-sectional area and
        so overstating Q/A.  Area is a number people check against an ADCP.
        """
        b = np.linspace(0, 20, 11)
        cfg = DischargeConfig(enabled=True, depths_m=[2.0] * 11,
                              alpha=0.85, bank_coefficient=0.7)
        r = compute_discharge(cfg, b, [1.0] * 11, 20.0)

        assert r.area_m2 == pytest.approx(20.0 * 2.0)
        assert r.panel_width_m.sum() == pytest.approx(r.wetted_width_m)

    def test_it_still_reduces_the_discharge(self):
        b = np.linspace(0, 20, 11)
        kw = dict(enabled=True, depths_m=[2.0] * 11, alpha=1.0)
        full = compute_discharge(DischargeConfig(bank_coefficient=1.0, **kw),
                                 b, [1.0] * 11, 20.0)
        reduced = compute_discharge(DischargeConfig(bank_coefficient=0.7, **kw),
                                    b, [1.0] * 11, 20.0)
        assert reduced.discharge_m3s < full.discharge_m3s
        # Only the two edge panels change, and only through their velocity.
        assert reduced.area_m2 == pytest.approx(full.area_m2)


def _old_sample_bilinear(image, pts_px):
    """The previous implementation, which converted the whole frame first."""
    h, w = image.shape[:2]
    u, v = pts_px[:, 0].astype(np.float64), pts_px[:, 1].astype(np.float64)
    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    uc, vc = np.clip(u, 0, w - 1.000001), np.clip(v, 0, h - 1.000001)
    u0, v0 = np.floor(uc).astype(np.intp), np.floor(vc).astype(np.intp)
    u1, v1 = np.minimum(u0 + 1, w - 1), np.minimum(v0 + 1, h - 1)
    fu, fv = uc - u0, vc - v0
    img = image.astype(np.float32, copy=False)
    top = img[v0, u0] * (1.0 - fu) + img[v0, u1] * fu
    bot = img[v1, u0] * (1.0 - fu) + img[v1, u1] * fu
    return np.where(inside, top * (1.0 - fv) + bot * fv, np.nan)


class TestSamplingWithoutAFrameCopy:
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
    def test_identical_to_converting_the_whole_frame(self, dtype):
        rng = np.random.default_rng(4)
        img = (rng.integers(0, 256, (90, 160)) * (0.37 if dtype != np.uint8 else 1)).astype(dtype)
        pts = np.column_stack([rng.uniform(-3, 163, 500), rng.uniform(-3, 93, 500)])
        assert np.array_equal(sample_bilinear(img, pts), _old_sample_bilinear(img, pts),
                              equal_nan=True)


class TestStiMemory:
    def _lines(self):
        from stiv.geometry import SearchLine

        out = []
        for i in range(3):
            n = 20 + i
            out.append(SearchLine(
                index=i, station_m=float(i), centre_world=np.zeros(2), centre_px=np.zeros(2),
                direction_world=np.array([1.0, 0.0]),
                samples_px=np.column_stack([np.arange(n, dtype=float), np.full(n, 2.0 + i)]),
                samples_world=np.zeros((n, 2)), sx_m_per_px=0.02, length_m=0.02 * (n - 1),
            ))
        return out

    def test_finished_stis_are_views_not_copies(self):
        """finish() copied every block out of the buffer: twice the memory."""
        from stiv.sti import StiCollector

        col = StiCollector(self._lines(), 50, 1 / 30)
        for k in range(50):
            col.add_frame(np.full((8, 40), k, np.uint8))
        stis = col.finish()
        assert all(np.shares_memory(s.data, col.buffer) for s in stis)
        assert [s.data.shape for s in stis] == [(50, 20), (50, 21), (50, 22)]
        assert np.array_equal(stis[1].data[:, 0], np.arange(50))

    def test_analysis_runs_in_float64_and_keeps_float32(self):
        """The orientation must see full precision; what is kept is for display."""
        from stiv import orientation
        from stiv.analysis import _analyse_stis
        from stiv.config import Config
        from stiv.sti import SpaceTimeImage

        rng = np.random.default_rng(5)
        raw = rng.standard_normal((120, 90)).astype(np.float32)
        s = SpaceTimeImage(data=raw.copy(), line=self._lines()[0], sx_m_per_px=0.02,
                           st_s_per_px=1 / 30)
        cfg = Config()
        _, fields, _ = _analyse_stis(cfg, [s], None)

        conditioned = preprocess(raw, cfg.stiv)
        assert conditioned.dtype == np.float64
        direct = orientation.structure_tensor_field(conditioned, window_px=cfg.stiv.window_px)
        assert np.array_equal(fields[0].phi, direct.phi)
        assert s.data.dtype == np.float32
        assert np.allclose(s.data, conditioned, atol=1e-5)
