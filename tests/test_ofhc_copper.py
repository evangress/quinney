"""OFHC copper material constants — plausibility guards.

These tests do not certify the published values (that needs the source papers,
see the module docstring). They catch the failure mode that actually bites: a
transposed digit or a dropped minus sign turning the material into something
that fractures nothing, or everything.
"""

import pytest

from quinney.materials.johnson_cook import fracture_strain
from quinney.materials.ofhc_copper import OFHC_COPPER_DAMAGE, kratos_material_variables

# Uniaxial tension is sigma* = 1/3 — the state the published constants were
# fitted against, and the only one with a literature value to compare to.
UNIAXIAL_TENSION = 1.0 / 3.0


def test_tension_embrittles():
    # D3 must be negative or the model predicts that pulling copper apart
    # makes it more ductile, and no chip ever separates.
    assert OFHC_COPPER_DAMAGE.d3 < 0.0


def test_melt_is_above_the_reference_temperature():
    # A non-positive span would divide by zero, or invert, the thermal term.
    assert OFHC_COPPER_DAMAGE.melt_temperature_k > OFHC_COPPER_DAMAGE.reference_temperature_k


def test_melt_temperature_is_copper():
    # Pure copper melts at 1357.8 K. A value near steel's would silently halve
    # every homologous temperature in the shear band.
    assert OFHC_COPPER_DAMAGE.melt_temperature_k == pytest.approx(1356.0, abs=10.0)


def test_reference_strain_rate_is_positive():
    assert OFHC_COPPER_DAMAGE.reference_strain_rate_per_s > 0.0


def test_quasi_static_tensile_ductility_is_copper_like():
    # Annealed OFHC copper is very ductile: true fracture strain around 2 in
    # uniaxial tension at room temperature. An order-of-magnitude miss here
    # means the constants are wrong, whatever the digits look like.
    eps_f = fracture_strain(
        OFHC_COPPER_DAMAGE,
        triaxiality=UNIAXIAL_TENSION,
        plastic_strain_rate_per_s=OFHC_COPPER_DAMAGE.reference_strain_rate_per_s,
        temperature_k=OFHC_COPPER_DAMAGE.reference_temperature_k,
    )
    assert 1.5 < float(eps_f) < 3.5


def test_high_triaxiality_still_leaves_finite_ductility():
    # D1 is the floor the exponential decays to; it must be positive or
    # severely triaxial points fail instantly regardless of loading.
    assert OFHC_COPPER_DAMAGE.d1 > 0.0


class TestKratosMaterialVariables:
    """The solver's material block must be derived from the same constants.

    The Johnson-Cook law inside Kratos and quinney's damage model both use the
    reference and melt temperatures. Typing them in two places lets them drift,
    and a drift would silently change homologous temperature in one but not the
    other — no error, just wrong physics.
    """

    def test_temperatures_match_the_damage_parameters_exactly(self):
        variables = kratos_material_variables()
        assert variables["REFERENCE_TEMPERATURE"] == OFHC_COPPER_DAMAGE.reference_temperature_k
        assert variables["MELD_TEMPERATURE"] == OFHC_COPPER_DAMAGE.melt_temperature_k

    def test_reference_strain_rate_matches_the_damage_parameters(self):
        variables = kratos_material_variables()
        assert variables["REFERENCE_STRAIN_RATE"] == OFHC_COPPER_DAMAGE.reference_strain_rate_per_s

    def test_initial_temperature_defaults_to_the_reference(self):
        # The law's Check() rejects a missing or non-positive TEMPERATURE, and
        # starting anywhere other than the reference silently offsets T*.
        variables = kratos_material_variables()
        assert variables["TEMPERATURE"] == OFHC_COPPER_DAMAGE.reference_temperature_k

    def test_carries_the_johnson_cook_flow_constants(self):
        variables = kratos_material_variables()
        # Lowercase n and m are Kratos' spelling, not a typo.
        for key in (
            "JC_PARAMETER_A",
            "JC_PARAMETER_B",
            "JC_PARAMETER_C",
            "JC_PARAMETER_n",
            "JC_PARAMETER_m",
        ):
            assert key in variables, key
        assert variables["JC_PARAMETER_B"] > variables["JC_PARAMETER_A"] > 0.0

    def test_taylor_quinney_is_a_fraction(self):
        # Above 1.0 would create energy from plastic work.
        assert 0.0 < kratos_material_variables()["TAYLOR_QUINNEY_COEFFICIENT"] <= 1.0

    def test_material_points_per_element_is_overridable(self):
        assert (
            kratos_material_variables(material_points_per_element=6)["MATERIAL_POINTS_PER_ELEMENT"]
            == 6
        )

    def test_yield_can_be_raised_to_make_a_body_effectively_rigid(self):
        # The cutting tool reuses this law rather than adding a second one; a
        # huge JC_PARAMETER_A keeps it elastic under any cutting load.
        rigid = kratos_material_variables(yield_stress_pa=1.0e12, density_kg_m3=15000.0)
        assert rigid["JC_PARAMETER_A"] == 1.0e12
        assert rigid["DENSITY"] == 15000.0
        # Everything else still tracks the canonical copper values.
        assert rigid["MELD_TEMPERATURE"] == OFHC_COPPER_DAMAGE.melt_temperature_k
