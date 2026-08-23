"""Johnson-Cook damage law — behavior tests.

Ground truth here is the *structure* of the JC damage model, not a solver:
each factor of the fracture-strain product is exercised independently, so a
sign error or a dropped term fails loudly.
"""

import typing

import numpy as np
import pytest

from quinney.materials.johnson_cook import (
    JohnsonCookDamage,
    accumulate_damage,
    advance,
    fracture_strain,
    triaxiality_from_plane_strain_stress,
)

# A deliberately asymmetric parameter set: every constant is distinct, so a
# test cannot pass by accidentally reading the wrong field.
PARAMS = JohnsonCookDamage(
    d1=0.50,
    d2=4.00,
    d3=-3.00,
    d4=0.020,
    d5=1.00,
    reference_strain_rate_per_s=1.0,
    reference_temperature_k=300.0,
    melt_temperature_k=1300.0,
)

# Conditions that null out every optional factor: strain rate at the reference
# (ln = 0) and temperature at the reference (T* = 0).
BASE = {"plastic_strain_rate_per_s": 1.0, "temperature_k": 300.0}


class TestFractureStrain:
    def test_reduces_to_d1_plus_d2_at_zero_triaxiality_and_reference_state(self):
        # At sigma* = 0, eps_rate = eps_rate_0, T = T_ref the model collapses
        # to D1 + D2. Any stray factor shows up immediately.
        assert fracture_strain(PARAMS, triaxiality=0.0, **BASE) == pytest.approx(4.50)

    def test_tension_embrittles_relative_to_compression(self):
        # D3 < 0 means positive (tensile) triaxiality must LOWER fracture
        # strain. Getting this sign backwards would make the primary shear
        # zone — which is under compression — fail early and fabricate chips.
        tensile = fracture_strain(PARAMS, triaxiality=1.0, **BASE)
        compressive = fracture_strain(PARAMS, triaxiality=-1.0, **BASE)
        assert tensile < compressive

    def test_triaxiality_term_matches_closed_form(self):
        expected = 0.50 + 4.00 * np.exp(-3.00 * 0.75)
        assert fracture_strain(PARAMS, triaxiality=0.75, **BASE) == pytest.approx(expected)

    def test_strain_rate_above_reference_raises_ductility(self):
        # D4 > 0: the rate bracket is 1 + D4*ln(rate ratio).
        fast = fracture_strain(
            PARAMS, triaxiality=0.0, plastic_strain_rate_per_s=np.e, temperature_k=300.0
        )
        assert fast == pytest.approx(4.50 * (1.0 + 0.020))

    def test_strain_rate_below_reference_does_not_reduce_ductility(self):
        # ln(rate ratio) < 0 would shrink — and at extreme ratios invert — the
        # bracket. The rate factor is floored at 1.0 rather than allowed to go
        # negative and produce a nonsensical negative fracture strain.
        slow = fracture_strain(
            PARAMS, triaxiality=0.0, plastic_strain_rate_per_s=1e-9, temperature_k=300.0
        )
        assert slow == pytest.approx(4.50)

    def test_temperature_raises_ductility_via_homologous_term(self):
        # T* = (T - T_ref)/(T_melt - T_ref) = 0.5 at the midpoint; D5 = 1.0.
        hot = fracture_strain(
            PARAMS, triaxiality=0.0, plastic_strain_rate_per_s=1.0, temperature_k=800.0
        )
        assert hot == pytest.approx(4.50 * 1.5)

    def test_temperature_below_reference_is_clamped(self):
        cold = fracture_strain(
            PARAMS, triaxiality=0.0, plastic_strain_rate_per_s=1.0, temperature_k=250.0
        )
        assert cold == pytest.approx(4.50)

    def test_temperature_above_melt_is_clamped(self):
        molten = fracture_strain(
            PARAMS, triaxiality=0.0, plastic_strain_rate_per_s=1.0, temperature_k=5000.0
        )
        assert molten == pytest.approx(4.50 * 2.0)

    def test_result_is_strictly_positive_under_extreme_tension(self):
        # exp(D3 * sigma*) -> 0 for large tensile triaxiality, leaving D1 alone;
        # a fracture strain of zero would divide by zero downstream.
        assert fracture_strain(PARAMS, triaxiality=50.0, **BASE) > 0.0

    def test_is_vectorized_over_material_points(self):
        triaxiality = np.array([-1.0, 0.0, 1.0])
        rate = np.array([1.0, 1.0, 1.0])
        temperature = np.array([300.0, 300.0, 300.0])
        result = fracture_strain(
            PARAMS,
            triaxiality=triaxiality,
            plastic_strain_rate_per_s=rate,
            temperature_k=temperature,
        )
        assert result.shape == (3,)
        assert result[0] > result[1] > result[2]


class TestTriaxialityFromPlaneStrainStress:
    """Triaxiality from the stress components MPM actually exposes.

    MP_PRESSURE is not implemented for MPM elements, and MP_CAUCHY_STRESS_VECTOR
    carries only (sxx, syy, sxy) in plane strain — szz is never reported. Under
    plastic flow the out-of-plane stress is the intermediate principal stress,
    szz = (sxx + syy)/2, which makes mean stress (sxx + syy)/2 as well. That
    identity holds exactly at the plastic limit, and damage only accumulates
    during plastic flow, so it is used unconditionally.
    """

    def test_compression_gives_negative_triaxiality(self):
        # The primary shear zone is compressive; this sign decides whether
        # damage accumulates there at all.
        result = triaxiality_from_plane_strain_stress(
            cauchy_xx=-600e6, cauchy_yy=-600e6, equivalent_stress=300e6
        )
        assert result == pytest.approx(-2.0)

    def test_tension_gives_positive_triaxiality(self):
        result = triaxiality_from_plane_strain_stress(
            cauchy_xx=600e6, cauchy_yy=600e6, equivalent_stress=300e6
        )
        assert result == pytest.approx(2.0)

    def test_mean_stress_is_the_average_of_the_in_plane_components(self):
        result = triaxiality_from_plane_strain_stress(
            cauchy_xx=-900e6, cauchy_yy=-300e6, equivalent_stress=600e6
        )
        assert result == pytest.approx(-1.0)

    def test_pure_shear_has_zero_triaxiality(self):
        result = triaxiality_from_plane_strain_stress(
            cauchy_xx=200e6, cauchy_yy=-200e6, equivalent_stress=400e6
        )
        assert result == pytest.approx(0.0)

    def test_negligible_equivalent_stress_yields_zero_not_infinity(self):
        # Unloaded points carry no deviatoric stress; the ratio must not blow up.
        result = triaxiality_from_plane_strain_stress(
            cauchy_xx=10e6, cauchy_yy=10e6, equivalent_stress=0.0
        )
        assert result == 0.0

    def test_is_vectorized_over_material_points(self):
        result = triaxiality_from_plane_strain_stress(
            cauchy_xx=np.array([-600e6, 600e6]),
            cauchy_yy=np.array([-600e6, 600e6]),
            equivalent_stress=np.array([300e6, 300e6]),
        )
        assert result == pytest.approx([-2.0, 2.0])


class TestAccumulateDamage:
    def test_damage_is_the_ratio_of_strain_increment_to_fracture_strain(self):
        damage = accumulate_damage(
            damage=np.array([0.0]),
            delta_plastic_strain=np.array([0.5]),
            fracture_strain=np.array([2.0]),
        )
        assert damage == pytest.approx([0.25])

    def test_damage_accumulates_across_increments(self):
        damage = np.array([0.25])
        damage = accumulate_damage(
            damage=damage,
            delta_plastic_strain=np.array([0.5]),
            fracture_strain=np.array([2.0]),
        )
        assert damage == pytest.approx([0.50])

    def test_elastic_increment_leaves_damage_unchanged(self):
        damage = accumulate_damage(
            damage=np.array([0.4]),
            delta_plastic_strain=np.array([0.0]),
            fracture_strain=np.array([2.0]),
        )
        assert damage == pytest.approx([0.4])

    def test_damage_never_decreases_when_given_a_negative_increment(self):
        # Plastic strain is monotonic; a negative delta signals noise in the
        # dumped state, and healing is not physical.
        damage = accumulate_damage(
            damage=np.array([0.4]),
            delta_plastic_strain=np.array([-0.3]),
            fracture_strain=np.array([2.0]),
        )
        assert damage == pytest.approx([0.4])

    def test_does_not_mutate_the_damage_it_is_given(self):
        # Replay compares pre- and post-step state; in-place updates would
        # silently destroy the earlier value.
        original = np.array([0.25])
        accumulate_damage(
            damage=original,
            delta_plastic_strain=np.array([0.5]),
            fracture_strain=np.array([2.0]),
        )
        assert original == pytest.approx([0.25])

    def test_damage_is_capped_at_one(self):
        damage = accumulate_damage(
            damage=np.array([0.9]),
            delta_plastic_strain=np.array([10.0]),
            fracture_strain=np.array([2.0]),
        )
        assert damage == pytest.approx([1.0])


class TestAdvance:
    """One damage step, both phases, from raw material-point state.

    Johnson-Cook damage has two stages and conflating them is the classic
    mistake: a point that has *initiated* has not failed, it has begun to fail.
    Separation happens when the evolution variable reaches 1, after dissipating
    the fracture energy over the point's characteristic length.
    """

    STATE: typing.ClassVar[dict] = {
        "cauchy_xx": np.array([100e6, 100e6]),
        "cauchy_yy": np.array([-100e6, -100e6]),
        "equivalent_stress": np.array([100e6, 100e6]),
        "plastic_strain_rate_per_s": np.array([1.0, 1.0]),
        "temperature_k": np.array([300.0, 300.0]),
        "characteristic_length_m": np.array([1e-3, 1e-3]),
        "fracture_displacement_m": np.array([1e-3, 1e-3]),
    }

    def advance_from(self, initiation, damage, delta):
        return advance(
            PARAMS,
            initiation=np.asarray(initiation, dtype=float),
            damage=np.asarray(damage, dtype=float),
            delta_plastic_strain=np.asarray(delta, dtype=float),
            **self.STATE,
        )

    def test_initiation_advances_only_where_plastic_strain_grew(self):
        step = self.advance_from([0.0, 0.0], [0.0, 0.0], [1.0, 0.0])
        assert step.initiation[0] > 0.0
        assert step.initiation[1] == 0.0

    def test_initiation_matches_the_composed_primitives(self):
        triaxiality = triaxiality_from_plane_strain_stress(
            self.STATE["cauchy_xx"], self.STATE["cauchy_yy"], self.STATE["equivalent_stress"]
        )
        expected = accumulate_damage(
            damage=np.array([0.0, 0.0]),
            delta_plastic_strain=np.array([1.0, 0.0]),
            fracture_strain=fracture_strain(
                PARAMS,
                triaxiality=triaxiality,
                plastic_strain_rate_per_s=self.STATE["plastic_strain_rate_per_s"],
                temperature_k=self.STATE["temperature_k"],
            ),
        )
        step = self.advance_from([0.0, 0.0], [0.0, 0.0], [1.0, 0.0])
        assert step.initiation == pytest.approx(expected)

    def test_damage_stays_at_zero_before_initiation_completes(self):
        # The distinction that matters: heavily yielded but not yet initiated
        # material is undamaged and carries full stress.
        step = self.advance_from([0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
        assert step.initiation[0] < 1.0
        assert step.damage == pytest.approx([0.0, 0.0])
        assert not step.separated.any()

    def test_damage_evolves_once_initiation_is_complete(self):
        step = self.advance_from([1.0, 0.0], [0.0, 0.0], [0.5, 0.5])
        assert step.damage[0] > 0.0  # initiated
        assert step.damage[1] == 0.0  # not initiated

    def test_evolution_is_scaled_by_the_characteristic_length(self):
        # Damage per unit strain doubles when the point does — the mesh
        # regularization, seen end to end.
        step = advance(
            PARAMS,
            initiation=np.array([1.0, 1.0]),
            damage=np.array([0.0, 0.0]),
            delta_plastic_strain=np.array([0.1, 0.1]),
            cauchy_xx=np.array([100e6, 100e6]),
            cauchy_yy=np.array([-100e6, -100e6]),
            equivalent_stress=np.array([100e6, 100e6]),
            plastic_strain_rate_per_s=np.array([1.0, 1.0]),
            temperature_k=np.array([300.0, 300.0]),
            characteristic_length_m=np.array([2e-3, 1e-3]),
            fracture_displacement_m=np.array([1e-3, 1e-3]),
        )
        assert step.damage[0] == pytest.approx(2.0 * step.damage[1])

    def test_separates_only_when_evolution_completes(self):
        # A point at full initiation but no evolution has not separated.
        just_initiated = self.advance_from([1.0], [0.0], [0.0])
        assert not just_initiated.separated[0]
        fully_evolved = self.advance_from([1.0], [0.9], [1.0])
        assert fully_evolved.separated[0]

    def test_lowering_the_threshold_separates_earlier(self):
        # The supervisor's adjust_deletion_threshold action.
        strict = advance(
            PARAMS,
            initiation=np.array([1.0]),
            damage=np.array([0.3]),
            delta_plastic_strain=np.array([0.0]),
            cauchy_xx=np.array([100e6]),
            cauchy_yy=np.array([-100e6]),
            equivalent_stress=np.array([100e6]),
            plastic_strain_rate_per_s=np.array([1.0]),
            temperature_k=np.array([300.0]),
            characteristic_length_m=np.array([1e-3]),
            fracture_displacement_m=np.array([1e-3]),
            separation_threshold=0.2,
        )
        assert strict.separated[0]

    def test_does_not_mutate_the_state_it_is_given(self):
        initiation = np.array([0.5, 0.5])
        damage = np.array([0.1, 0.1])
        advance(
            PARAMS,
            initiation=initiation,
            damage=damage,
            delta_plastic_strain=np.array([0.1, 0.1]),
            **self.STATE,
        )
        assert initiation == pytest.approx([0.5, 0.5])
        assert damage == pytest.approx([0.1, 0.1])
