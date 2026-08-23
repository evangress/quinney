"""Damage softening applied through per-point material properties.

Kratos' compiled Johnson-Cook law computes stress internally and exposes no
hook to scale it. But it *reads* its constants from the element's Properties
every step, and Properties can be created and mutated at runtime — so
sigma = (1-D)*sigma_bar is applied by degrading the constants the law reads.

Real Kratos elements and real Properties objects throughout.
"""

import KratosMultiphysics as KM
import pytest

from quinney.kratos_adapter.softening import DamageSofteningProcess

PRISTINE_YIELD_PA = 90e6
PRISTINE_HARDENING_PA = 292e6
PRISTINE_MODULUS_PA = 117e9


def make_part(n_elements: int) -> KM.ModelPart:
    model = KM.Model()
    part = model.CreateModelPart("workpiece")
    for node_id in range(1, n_elements + 3):
        part.CreateNewNode(node_id, float(node_id), 0.0, 0.0)
    shared = part.CreateNewProperties(1)
    shared.SetValue(KM.JC_PARAMETER_A, PRISTINE_YIELD_PA)
    shared.SetValue(KM.JC_PARAMETER_B, PRISTINE_HARDENING_PA)
    shared.SetValue(KM.YOUNG_MODULUS, PRISTINE_MODULUS_PA)
    for element_id in range(1, n_elements + 1):
        part.CreateNewElement(
            "Element2D3N", element_id, [element_id, element_id + 1, element_id + 2], shared
        )
    return part


def yield_of(part, element_id):
    return part.GetElement(element_id).Properties.GetValue(KM.JC_PARAMETER_A)


class TestDamageSofteningProcess:
    def test_gives_each_point_its_own_properties(self):
        # Sharing one Properties object would make every point soften together
        # the moment any single point damaged.
        part = make_part(3)
        DamageSofteningProcess(part)
        ids = {part.GetElement(i).Properties.Id for i in (1, 2, 3)}
        assert len(ids) == 3

    def test_undamaged_points_keep_their_pristine_strength(self):
        part = make_part(2)
        process = DamageSofteningProcess(part)
        process.apply({1: 0.0, 2: 0.0})
        assert yield_of(part, 1) == pytest.approx(PRISTINE_YIELD_PA)

    def test_half_damaged_points_carry_half_the_flow_stress(self):
        part = make_part(1)
        process = DamageSofteningProcess(part)
        process.apply({1: 0.5})
        assert yield_of(part, 1) == pytest.approx(0.5 * PRISTINE_YIELD_PA)
        assert part.GetElement(1).Properties.GetValue(KM.JC_PARAMETER_B) == pytest.approx(
            0.5 * PRISTINE_HARDENING_PA
        )

    def test_degradation_is_measured_from_pristine_not_compounded(self):
        # Scaling the current value each call would decay geometrically and
        # reach zero strength long before the point actually separates.
        part = make_part(1)
        process = DamageSofteningProcess(part)
        process.apply({1: 0.5})
        process.apply({1: 0.5})
        process.apply({1: 0.5})
        assert yield_of(part, 1) == pytest.approx(0.5 * PRISTINE_YIELD_PA)

    def test_damage_can_only_advance(self):
        # Damage is irreversible; a lower value arriving later is noise and
        # must not restore strength the point has already lost.
        part = make_part(1)
        process = DamageSofteningProcess(part)
        process.apply({1: 0.6})
        process.apply({1: 0.2})
        assert yield_of(part, 1) == pytest.approx(0.4 * PRISTINE_YIELD_PA)

    def test_fully_damaged_points_keep_a_residual_strength(self):
        # Zero strength gives a stressless point that inverts and wrecks the
        # explicit step before separation can remove it.
        part = make_part(1)
        process = DamageSofteningProcess(part)
        process.apply({1: 1.0})
        assert 0.0 < yield_of(part, 1) < 0.01 * PRISTINE_YIELD_PA

    def test_degrades_stiffness_alongside_strength(self):
        part = make_part(1)
        process = DamageSofteningProcess(part)
        process.apply({1: 0.5})
        assert part.GetElement(1).Properties.GetValue(KM.YOUNG_MODULUS) == pytest.approx(
            0.5 * PRISTINE_MODULUS_PA
        )

    def test_stiffness_degradation_can_be_switched_off(self):
        # Softening stiffness lowers the wave speed and so relaxes the CFL
        # limit, but it also lets damaged points deform without restraint.
        part = make_part(1)
        process = DamageSofteningProcess(part, degrade_stiffness=False)
        process.apply({1: 0.5})
        assert part.GetElement(1).Properties.GetValue(KM.YOUNG_MODULUS) == pytest.approx(
            PRISTINE_MODULUS_PA
        )
        assert yield_of(part, 1) == pytest.approx(0.5 * PRISTINE_YIELD_PA)

    def test_points_absent_from_the_damage_map_are_untouched(self):
        part = make_part(2)
        process = DamageSofteningProcess(part)
        process.apply({1: 0.5})
        assert yield_of(part, 2) == pytest.approx(PRISTINE_YIELD_PA)
