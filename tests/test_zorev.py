"""Zorev tool-chip friction — behavior tests.

The model's whole content is the transition between two regimes, so the tests
exercise each regime and the boundary between them.
"""

import numpy as np
import pytest

from quinney.materials.zorev import (
    effective_friction_coefficient,
    shear_yield_stress,
    zorev_shear_stress,
)

SLIDING_FRICTION = 0.5
SHEAR_YIELD_PA = 200e6  # tau_max at friction_factor 1.0
# Below this contact pressure the interface slides; above it, it sticks.
TRANSITION_PA = SHEAR_YIELD_PA / SLIDING_FRICTION  # 400 MPa


class TestShearYieldStress:
    def test_is_the_von_mises_shear_yield(self):
        # k = sigma_flow / sqrt(3). Getting this wrong by using sigma/2
        # (Tresca) shifts the whole sticking region.
        assert shear_yield_stress(600e6) == pytest.approx(600e6 / np.sqrt(3))

    def test_is_vectorized_over_contact_points(self):
        result = shear_yield_stress(np.array([300e6, 600e6]))
        assert result == pytest.approx(np.array([300e6, 600e6]) / np.sqrt(3))


class TestZorevShearStress:
    def test_slides_by_coulomb_below_the_transition(self):
        # Out along the rake face the pressure is low and plain Coulomb holds.
        tau = zorev_shear_stress(
            normal_stress_pa=100e6,
            friction_coefficient=SLIDING_FRICTION,
            shear_yield_stress_pa=SHEAR_YIELD_PA,
        )
        assert tau == pytest.approx(50e6)

    def test_sticks_at_the_shear_yield_above_the_transition(self):
        # Near the tool tip the pressure is enormous. Plain Coulomb would
        # predict a shear stress the material cannot carry — it would yield
        # first. That cap is the entire point of Zorev over Coulomb.
        tau = zorev_shear_stress(
            normal_stress_pa=2000e6,
            friction_coefficient=SLIDING_FRICTION,
            shear_yield_stress_pa=SHEAR_YIELD_PA,
        )
        assert tau == pytest.approx(SHEAR_YIELD_PA)

    def test_is_continuous_at_the_transition(self):
        # A jump here would show up as a discontinuity in the force signal and
        # be mistaken for segmentation.
        just_below = zorev_shear_stress(TRANSITION_PA * 0.999, SLIDING_FRICTION, SHEAR_YIELD_PA)
        just_above = zorev_shear_stress(TRANSITION_PA * 1.001, SLIDING_FRICTION, SHEAR_YIELD_PA)
        assert just_below == pytest.approx(just_above, rel=1e-2)
        assert just_below == pytest.approx(SHEAR_YIELD_PA, rel=1e-2)

    def test_sticking_shear_is_independent_of_pressure(self):
        high = zorev_shear_stress(2000e6, SLIDING_FRICTION, SHEAR_YIELD_PA)
        higher = zorev_shear_stress(8000e6, SLIDING_FRICTION, SHEAR_YIELD_PA)
        assert high == pytest.approx(higher)

    def test_friction_factor_scales_the_sticking_limit(self):
        # m < 1 models an interface that yields before the bulk does.
        tau = zorev_shear_stress(2000e6, SLIDING_FRICTION, SHEAR_YIELD_PA, friction_factor=0.8)
        assert tau == pytest.approx(0.8 * SHEAR_YIELD_PA)

    def test_no_contact_carries_no_shear(self):
        # Tensile or zero normal stress means the surfaces are not touching.
        assert zorev_shear_stress(0.0, SLIDING_FRICTION, SHEAR_YIELD_PA) == 0.0
        assert zorev_shear_stress(-50e6, SLIDING_FRICTION, SHEAR_YIELD_PA) == 0.0

    def test_is_vectorized_across_the_rake_face(self):
        # A real rake face spans both regimes at once: sticking at the tip,
        # sliding further out.
        tau = zorev_shear_stress(
            normal_stress_pa=np.array([2000e6, 100e6, 0.0]),
            friction_coefficient=SLIDING_FRICTION,
            shear_yield_stress_pa=SHEAR_YIELD_PA,
        )
        assert tau == pytest.approx([SHEAR_YIELD_PA, 50e6, 0.0])


class TestEffectiveFrictionCoefficient:
    """Kratos applies Coulomb friction through a per-node coefficient.

    Feeding it min(mu, tau_max/sigma_n) reproduces Zorev exactly, which is what
    lets the model ride on machinery that only knows about Coulomb.
    """

    def test_equals_the_sliding_coefficient_below_the_transition(self):
        assert effective_friction_coefficient(
            100e6, SLIDING_FRICTION, SHEAR_YIELD_PA
        ) == pytest.approx(SLIDING_FRICTION)

    def test_falls_below_the_sliding_coefficient_once_sticking(self):
        mu = effective_friction_coefficient(2000e6, SLIDING_FRICTION, SHEAR_YIELD_PA)
        assert mu == pytest.approx(SHEAR_YIELD_PA / 2000e6)
        assert mu < SLIDING_FRICTION

    def test_reproduces_the_shear_stress_it_stands_in_for(self):
        # The defining property: mu_eff * sigma_n must equal the Zorev shear.
        pressures = np.array([50e6, 400e6, 1500e6, 5000e6])
        mu = effective_friction_coefficient(pressures, SLIDING_FRICTION, SHEAR_YIELD_PA)
        expected = zorev_shear_stress(pressures, SLIDING_FRICTION, SHEAR_YIELD_PA)
        assert mu * pressures == pytest.approx(expected)

    def test_no_contact_gives_no_friction(self):
        assert effective_friction_coefficient(0.0, SLIDING_FRICTION, SHEAR_YIELD_PA) == 0.0

    def test_never_exceeds_the_sliding_coefficient(self):
        pressures = np.array([1e3, 1e6, 1e9, 1e12])
        mu = effective_friction_coefficient(pressures, SLIDING_FRICTION, SHEAR_YIELD_PA)
        assert np.all(mu <= SLIDING_FRICTION + 1e-12)
