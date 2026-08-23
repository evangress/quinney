"""Johnson-Cook damage evolution — the phase after initiation.

Initiation says a point has begun to fail. Evolution says how fast it loses
load-carrying capacity on the way to separation, and is regularized by the
point's characteristic length so the energy dissipated per unit crack area does
not change when the mesh does.
"""

import numpy as np
import pytest

from quinney.materials.johnson_cook import (
    degradation_factor,
    evolve_damage,
    fracture_displacement,
)

FRACTURE_ENERGY_J_M2 = 1.0e5
YIELD_STRESS_PA = 200e6
# u_f = 2 G_f / sigma_y = 2e5 / 2e8 = 1e-3 m
FRACTURE_DISPLACEMENT_M = 1.0e-3


class TestFractureDisplacement:
    def test_follows_the_hillerborg_energy_relation(self):
        # u_f = 2 G_f / sigma_y. This is what ties damage evolution to a
        # measurable material property instead of a tuned threshold.
        assert fracture_displacement(FRACTURE_ENERGY_J_M2, YIELD_STRESS_PA) == pytest.approx(
            FRACTURE_DISPLACEMENT_M
        )

    def test_tougher_material_takes_longer_to_separate(self):
        tough = fracture_displacement(2 * FRACTURE_ENERGY_J_M2, YIELD_STRESS_PA)
        brittle = fracture_displacement(FRACTURE_ENERGY_J_M2, YIELD_STRESS_PA)
        assert tough > brittle

    def test_is_vectorized(self):
        result = fracture_displacement(np.array([1e5, 2e5]), np.array([200e6, 200e6]))
        assert result == pytest.approx([1e-3, 2e-3])


class TestEvolveDamage:
    def test_damage_grows_with_plastic_displacement_not_strain(self):
        # The increment is L * delta_eps_p, so a point twice the size reaches
        # the same damage in half the strain. That is the regularization:
        # dissipated energy per crack area stays fixed as the mesh changes.
        coarse = evolve_damage(
            damage=np.array([0.0]),
            delta_plastic_strain=np.array([0.01]),
            characteristic_length_m=np.array([2e-3]),
            fracture_displacement_m=np.array([FRACTURE_DISPLACEMENT_M]),
        )
        fine = evolve_damage(
            damage=np.array([0.0]),
            delta_plastic_strain=np.array([0.01]),
            characteristic_length_m=np.array([1e-3]),
            fracture_displacement_m=np.array([FRACTURE_DISPLACEMENT_M]),
        )
        assert coarse == pytest.approx(2.0 * fine)

    def test_reaches_full_damage_at_the_fracture_displacement(self):
        # L * eps_p = u_f exactly, so D must be 1.
        damage = evolve_damage(
            damage=np.array([0.0]),
            delta_plastic_strain=np.array([1.0]),
            characteristic_length_m=np.array([FRACTURE_DISPLACEMENT_M]),
            fracture_displacement_m=np.array([FRACTURE_DISPLACEMENT_M]),
        )
        assert damage == pytest.approx([1.0])

    def test_accumulates_across_increments(self):
        damage = np.array([0.25])
        damage = evolve_damage(
            damage=damage,
            delta_plastic_strain=np.array([0.25]),
            characteristic_length_m=np.array([FRACTURE_DISPLACEMENT_M]),
            fracture_displacement_m=np.array([FRACTURE_DISPLACEMENT_M]),
        )
        assert damage == pytest.approx([0.50])

    def test_saturates_at_one(self):
        damage = evolve_damage(
            damage=np.array([0.9]),
            delta_plastic_strain=np.array([10.0]),
            characteristic_length_m=np.array([FRACTURE_DISPLACEMENT_M]),
            fracture_displacement_m=np.array([FRACTURE_DISPLACEMENT_M]),
        )
        assert damage == pytest.approx([1.0])

    def test_does_not_heal_on_a_negative_increment(self):
        damage = evolve_damage(
            damage=np.array([0.4]),
            delta_plastic_strain=np.array([-0.3]),
            characteristic_length_m=np.array([FRACTURE_DISPLACEMENT_M]),
            fracture_displacement_m=np.array([FRACTURE_DISPLACEMENT_M]),
        )
        assert damage == pytest.approx([0.4])

    def test_does_not_mutate_its_input(self):
        original = np.array([0.25])
        evolve_damage(
            damage=original,
            delta_plastic_strain=np.array([0.25]),
            characteristic_length_m=np.array([FRACTURE_DISPLACEMENT_M]),
            fracture_displacement_m=np.array([FRACTURE_DISPLACEMENT_M]),
        )
        assert original == pytest.approx([0.25])


class TestDegradationFactor:
    def test_undamaged_material_carries_full_stress(self):
        assert degradation_factor(np.array([0.0])) == pytest.approx([1.0])

    def test_half_damaged_material_carries_half(self):
        # sigma = (1 - D) * sigma_bar — the softening the compiled Kratos law
        # cannot do for us.
        assert degradation_factor(np.array([0.5])) == pytest.approx([0.5])

    def test_fully_damaged_material_retains_a_residual_stiffness(self):
        # Exactly zero would give a massless, stressless point that inverts and
        # destabilizes the explicit step. Separation removes it instead.
        factor = degradation_factor(np.array([1.0]))
        assert 0.0 < factor[0] < 0.01

    def test_is_monotonic_in_damage(self):
        factors = degradation_factor(np.array([0.0, 0.3, 0.7, 1.0]))
        assert np.all(np.diff(factors) < 0.0)
