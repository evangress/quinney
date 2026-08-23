"""Merchant shear-plane trigonometry — behavior tests.

The three routes to a shear angle only earn their place if they agree when the
chip is steady, so most of these tests are round trips: build a case that obeys
the theory by construction and check every route recovers the same angle.
"""

import numpy as np
import pytest

from quinney.diagnostics.merchant import (
    friction_angle_deg,
    merchant_shear_angle_deg,
    shear_angle_from_thickness_ratio_deg,
)

# Rake angles bracketing the bench case: a slightly negative tool through a
# positive-rake tool. OFHC copper is cut with modest positive rake.
RAKE_ANGLES_DEG = (-5.0, 0.0, 5.0, 10.0, 20.0)


class TestMerchantShearAngle:
    def test_is_45_degrees_for_a_frictionless_zero_rake_tool(self):
        # phi = 45 + alpha/2 - beta/2 collapses to 45 with both angles zero.
        assert merchant_shear_angle_deg(
            rake_angle_deg=0.0, friction_angle_deg=0.0
        ) == pytest.approx(45.0)

    def test_friction_lowers_the_shear_angle(self):
        # More friction on the rake face means a thicker chip and a shallower
        # shear plane. Getting the sign wrong here inverts the entire
        # band-orientation comparison.
        low = merchant_shear_angle_deg(rake_angle_deg=0.0, friction_angle_deg=20.0)
        high = merchant_shear_angle_deg(rake_angle_deg=0.0, friction_angle_deg=40.0)
        assert low > high

    def test_positive_rake_raises_the_shear_angle(self):
        shallow = merchant_shear_angle_deg(rake_angle_deg=0.0, friction_angle_deg=25.0)
        steep = merchant_shear_angle_deg(rake_angle_deg=20.0, friction_angle_deg=25.0)
        assert steep > shallow

    def test_lands_in_the_primary_shear_zone_range_for_realistic_inputs(self):
        # Spec bench case: the primary shear zone runs at roughly 25-45 deg.
        # mu ~ 0.3-0.6 on copper is beta ~ 17-31 deg.
        for beta in (17.0, 25.0, 31.0):
            phi = merchant_shear_angle_deg(rake_angle_deg=5.0, friction_angle_deg=beta)
            assert 25.0 <= phi <= 45.0

    def test_rejects_a_rake_angle_past_vertical(self):
        with pytest.raises(ValueError, match="rake angle"):
            merchant_shear_angle_deg(rake_angle_deg=95.0, friction_angle_deg=20.0)


class TestFrictionAngleFromForces:
    def test_is_the_rake_angle_when_the_thrust_force_vanishes(self):
        # With no thrust force the resultant lies along the cutting direction,
        # so the friction angle is exactly the rake angle.
        assert friction_angle_deg(
            cutting_force_n=800.0, thrust_force_n=0.0, rake_angle_deg=12.0
        ) == pytest.approx(12.0)

    def test_matches_the_classical_force_resolution(self):
        # tan(beta) = F/N with F = Fc sin(a) + Ft cos(a), N = Fc cos(a) - Ft sin(a).
        # Any algebraic slip in the arctan form shows up here.
        cutting_force_n, thrust_force_n, rake_deg = 900.0, 420.0, 8.0
        rake = np.radians(rake_deg)
        friction_force = cutting_force_n * np.sin(rake) + thrust_force_n * np.cos(rake)
        normal_force = cutting_force_n * np.cos(rake) - thrust_force_n * np.sin(rake)
        expected = np.degrees(np.arctan(friction_force / normal_force))

        assert friction_angle_deg(cutting_force_n, thrust_force_n, rake_deg) == pytest.approx(
            expected
        )

    def test_grows_with_the_thrust_to_cutting_force_ratio(self):
        light = friction_angle_deg(1000.0, 200.0, 0.0)
        heavy = friction_angle_deg(1000.0, 600.0, 0.0)
        assert heavy > light

    def test_rejects_a_non_positive_cutting_force(self):
        # A tool that is not cutting has no measurable friction angle; reading
        # one off noise would poison the reference angle downstream.
        with pytest.raises(ValueError, match="cutting force"):
            friction_angle_deg(cutting_force_n=0.0, thrust_force_n=100.0, rake_angle_deg=0.0)

    def test_rejects_forces_that_pull_the_chip_off_the_rake_face(self):
        # N <= 0 means the tool is in tension against the chip, which cannot
        # happen; it means the force signal is garbage, not that beta >= 90.
        with pytest.raises(ValueError, match="rake face"):
            friction_angle_deg(cutting_force_n=100.0, thrust_force_n=1000.0, rake_angle_deg=10.0)


class TestShearAngleFromThicknessRatio:
    def test_is_45_degrees_when_the_chip_is_not_thickened_at_zero_rake(self):
        # r = t_uncut / t_chip = 1 means no thickening at all.
        assert shear_angle_from_thickness_ratio_deg(r=1.0, rake_angle_deg=0.0) == pytest.approx(
            45.0
        )

    def test_a_thicker_chip_means_a_shallower_shear_plane(self):
        thick_chip = shear_angle_from_thickness_ratio_deg(r=0.3, rake_angle_deg=0.0)
        thin_chip = shear_angle_from_thickness_ratio_deg(r=0.8, rake_angle_deg=0.0)
        assert thick_chip < thin_chip

    @pytest.mark.parametrize("rake_angle_deg", RAKE_ANGLES_DEG)
    @pytest.mark.parametrize("shear_angle_deg", (20.0, 30.0, 40.0, 55.0))
    def test_round_trips_a_chip_built_to_a_known_shear_angle(
        self, rake_angle_deg: float, shear_angle_deg: float
    ):
        # Geometry of the shear plane: r = sin(phi) / cos(phi - alpha). Build
        # the thickness ratio a given shear angle implies, then recover it.
        phi = np.radians(shear_angle_deg)
        rake = np.radians(rake_angle_deg)
        ratio = float(np.sin(phi) / np.cos(phi - rake))

        assert shear_angle_from_thickness_ratio_deg(ratio, rake_angle_deg) == pytest.approx(
            shear_angle_deg
        )

    def test_rejects_a_non_positive_thickness_ratio(self):
        with pytest.raises(ValueError, match="thickness ratio"):
            shear_angle_from_thickness_ratio_deg(r=0.0, rake_angle_deg=0.0)


class TestTheTwoRoutesAgreeOnASteadyChip:
    @pytest.mark.parametrize("rake_angle_deg", RAKE_ANGLES_DEG)
    def test_measured_chip_thickness_reproduces_the_force_based_prediction(
        self, rake_angle_deg: float
    ):
        # A chip that obeys Merchant by construction: take the forces, get the
        # friction angle, get the predicted shear angle, build the chip
        # thickness that angle implies, and measure the angle back. The two
        # routes disagreeing on a real run is the diagnostic; on this
        # synthetic steady chip they must not disagree at all.
        cutting_force_n, thrust_force_n = 850.0, 310.0
        beta = friction_angle_deg(cutting_force_n, thrust_force_n, rake_angle_deg)
        predicted = merchant_shear_angle_deg(rake_angle_deg, beta)

        phi = np.radians(predicted)
        ratio = float(np.sin(phi) / np.cos(phi - np.radians(rake_angle_deg)))

        assert shear_angle_from_thickness_ratio_deg(ratio, rake_angle_deg) == pytest.approx(
            predicted
        )
