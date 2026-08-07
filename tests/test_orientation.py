"""Tests for the structure-tensor core (section 2.2 of the paper)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from stiv.orientation import (
    angle_range_from_velocity,
    central_difference,
    mean_orientation_angle,
    structure_tensor_field,
    velocity_from_angle,
)

PHI_FULL = (-math.pi / 2 + 1e-4, math.pi / 2 - 1e-4)


def synthetic_sti(tan_phi, nt=200, nx=200, sigma=3.0, noise=0.0, seed=0):
    """An STI whose pattern advects at exactly ``tan_phi`` columns per row."""
    rng = np.random.default_rng(seed)
    length = nx + int(abs(tan_phi)) * nt + 8 * int(sigma) + 10
    base = gaussian_filter(rng.standard_normal(length), sigma)
    t = np.arange(nt)[:, None]
    x = np.arange(nx)[None, :]
    idx = np.mod(x - tan_phi * t, length)
    i0 = np.floor(idx).astype(int)
    f = idx - i0
    sti = base[i0] * (1 - f) + base[(i0 + 1) % length] * f
    if noise:
        sti = sti + noise * sti.std() * rng.standard_normal(sti.shape)
    return sti


# --------------------------------------------------------------------------- #
class TestGradients:
    @pytest.mark.parametrize("order", [2, 4, 6])
    def test_stencils_are_exact_on_a_linear_ramp(self, order):
        # A central difference of any order must return the exact slope.
        ramp = np.tile(np.arange(60, dtype=float) * 3.0, (40, 1))
        gx = central_difference(ramp, axis=1, order=order)
        interior = gx[:, order:-order]
        assert np.allclose(interior, 3.0)

    @pytest.mark.parametrize("order", [2, 4, 6])
    def test_constant_field_has_zero_gradient(self, order):
        flat = np.full((30, 30), 7.0)
        assert np.allclose(central_difference(flat, axis=0, order=order), 0.0)
        assert np.allclose(central_difference(flat, axis=1, order=order), 0.0)

    def test_unknown_order_rejected(self):
        with pytest.raises(ValueError, match="gradient order"):
            central_difference(np.zeros((20, 20)), axis=0, order=3)


class TestOrientation:
    @pytest.mark.parametrize("tan_phi", [0.1, 0.3, 0.5, 1.0, 2.0, 3.0])
    def test_recovers_known_inclination(self, tan_phi):
        """Reproduces Fig. 7: accurate across the validated tan-phi range."""
        field = structure_tensor_field(synthetic_sti(tan_phi))
        est = mean_orientation_angle(field, *PHI_FULL)
        assert est.tan_phi == pytest.approx(tan_phi, rel=0.10)

    @pytest.mark.parametrize("tan_phi", [-2.0, -0.5, 0.5, 2.0])
    def test_sign_follows_direction_of_travel(self, tan_phi):
        """A pattern moving upstream must give a negative angle, not |angle|.

        This is the branch that Eq. (6) leaves ambiguous: reading the major
        eigenvector instead of the minor one returns -1/tan(phi), which has the
        wrong sign for half the cases and the wrong magnitude for all of them.
        """
        field = structure_tensor_field(synthetic_sti(tan_phi))
        est = mean_orientation_angle(field, *PHI_FULL)
        assert np.sign(est.tan_phi) == np.sign(tan_phi)
        assert est.tan_phi == pytest.approx(tan_phi, rel=0.10)

    def test_stationary_pattern_reads_zero(self):
        field = structure_tensor_field(synthetic_sti(0.0))
        est = mean_orientation_angle(field, *PHI_FULL)
        assert abs(est.tan_phi) < 0.05

    def test_coherency_is_high_for_clean_streaks(self):
        field = structure_tensor_field(synthetic_sti(1.0))
        assert field.coherency.mean() > 0.9
        assert field.coherency.min() >= 0.0
        assert field.coherency.max() <= 1.0

    def test_coherency_is_low_for_isotropic_noise(self):
        rng = np.random.default_rng(3)
        field = structure_tensor_field(rng.standard_normal((200, 200)))
        assert field.coherency.mean() < 0.35

    def test_survives_heavy_noise(self):
        field = structure_tensor_field(synthetic_sti(1.5, noise=0.5))
        est = mean_orientation_angle(field, *PHI_FULL)
        assert est.tan_phi == pytest.approx(1.5, rel=0.15)

    def test_higher_gradient_order_helps_on_noisy_input(self):
        errs = {}
        for order in (2, 6):
            field = structure_tensor_field(synthetic_sti(3.0, noise=0.3), gradient_order=order)
            errs[order] = abs(mean_orientation_angle(field, *PHI_FULL).tan_phi - 3.0)
        assert errs[6] < errs[2]

    def test_rejects_sti_smaller_than_the_window(self):
        with pytest.raises(ValueError, match="too small"):
            structure_tensor_field(np.zeros((20, 20)), window_px=31)

    @pytest.mark.parametrize("window_type", ["box", "gaussian"])
    def test_both_window_types_agree(self, window_type):
        field = structure_tensor_field(synthetic_sti(1.0), window_type=window_type)
        est = mean_orientation_angle(field, *PHI_FULL)
        assert est.tan_phi == pytest.approx(1.0, rel=0.10)


class TestEq14:
    def test_search_range_excludes_out_of_range_angles(self):
        field = structure_tensor_field(synthetic_sti(3.0))
        # Restrict to a range that cannot contain the true angle.
        est = mean_orientation_angle(field, math.atan(0.1), math.atan(0.5))
        assert est.valid_fraction < 0.05

    def test_peak_and_mean_estimators_agree_on_a_clean_pattern(self):
        field = structure_tensor_field(synthetic_sti(2.0))
        a = mean_orientation_angle(field, *PHI_FULL, estimator="eq14")
        b = mean_orientation_angle(field, *PHI_FULL, estimator="peak")
        assert a.tan_phi == pytest.approx(b.tan_phi, rel=0.05)

    def test_returns_nan_when_nothing_is_in_range(self):
        field = structure_tensor_field(synthetic_sti(3.0))
        est = mean_orientation_angle(field, math.atan(-5.0), math.atan(-4.0))
        assert math.isnan(est.phi)
        assert est.n_used == 0

    def test_histogram_accumulates_over_coherency_levels(self):
        field = structure_tensor_field(synthetic_sti(1.0))
        one = mean_orientation_angle(field, *PHI_FULL, coherency_levels=(0.0,))
        five = mean_orientation_angle(field, *PHI_FULL,
                                      coherency_levels=(0.0, 0.2, 0.4, 0.6, 0.8))
        # Every window is coherent here, so five thresholds count each five times.
        assert five.hist.sum() == pytest.approx(5 * one.hist.sum())

    def test_rejects_inverted_range(self):
        field = structure_tensor_field(synthetic_sti(1.0))
        with pytest.raises(ValueError, match="phi_min < phi_max"):
            mean_orientation_angle(field, 1.0, 0.5)

    def test_rejects_unknown_estimator(self):
        field = structure_tensor_field(synthetic_sti(1.0))
        with pytest.raises(ValueError, match="estimator"):
            mean_orientation_angle(field, *PHI_FULL, estimator="median")


class TestEq16:
    def test_velocity_conversion(self):
        # Sx/St = 0.02 m / 0.0333 s = 0.6 m/s per unit slope.
        assert velocity_from_angle(math.atan(2.0), 0.02, 1 / 30) == pytest.approx(1.2, rel=1e-6)

    def test_zero_angle_is_zero_velocity(self):
        assert velocity_from_angle(0.0, 0.02, 1 / 30) == 0.0

    def test_angle_range_round_trips(self):
        sx, st = 0.025, 1 / 25
        lo, hi = angle_range_from_velocity(-0.5, 4.0, sx, st)
        assert velocity_from_angle(lo, sx, st) == pytest.approx(-0.5)
        assert velocity_from_angle(hi, sx, st) == pytest.approx(4.0)

    def test_angle_range_stays_inside_the_tangent_singularity(self):
        lo, hi = angle_range_from_velocity(-1e9, 1e9, 0.02, 1 / 30)
        assert -math.pi / 2 < lo < hi < math.pi / 2
