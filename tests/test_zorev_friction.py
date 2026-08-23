"""Zorev friction applied to a Kratos slip boundary.

Kratos applies Coulomb friction through a per-node FRICTION_COEFFICIENT that is
settable at runtime. The process rewrites it each step so the coefficient
carries the Zorev cap, which means the sticking region appears where the
pressure demands it rather than being fixed in advance.

Nodes here are real Kratos nodes on a real ModelPart. Only the contact-pressure
read is injected — the value it needs comes from a running slip boundary that
does not exist until the cutting case is reframed.
"""

import KratosMultiphysics as KM
import numpy as np
import pytest

from quinney.kratos_adapter.zorev_friction import ZorevFrictionProcess

SLIDING_FRICTION = 0.5
FLOW_STRESS_PA = 400e6
# k = 400/sqrt(3) = 230.9 MPa, so sliding holds below 461.9 MPa of pressure.
SHEAR_YIELD_PA = FLOW_STRESS_PA / np.sqrt(3.0)


def make_boundary(n_nodes: int) -> KM.ModelPart:
    model = KM.Model()
    part = model.CreateModelPart("rake_face")
    for node_id in range(1, n_nodes + 1):
        part.CreateNewNode(node_id, float(node_id) * 1e-4, 0.0, 0.0)
    return part


def constant_pressure(*values):
    return lambda _part: np.array(values, dtype=float)


class TestZorevFrictionProcess:
    def test_low_pressure_nodes_keep_the_sliding_coefficient(self):
        part = make_boundary(1)
        process = ZorevFrictionProcess(
            part, SLIDING_FRICTION, FLOW_STRESS_PA, contact_pressure=constant_pressure(100e6)
        )
        process.ExecuteInitializeSolutionStep()
        assert part.GetNode(1).GetValue(KM.FRICTION_COEFFICIENT) == pytest.approx(SLIDING_FRICTION)

    def test_high_pressure_nodes_get_a_reduced_coefficient(self):
        # This is the sticking region: shear is capped, so the coefficient that
        # reproduces it must fall as pressure rises.
        part = make_boundary(1)
        process = ZorevFrictionProcess(
            part, SLIDING_FRICTION, FLOW_STRESS_PA, contact_pressure=constant_pressure(2000e6)
        )
        process.ExecuteInitializeSolutionStep()
        assert part.GetNode(1).GetValue(KM.FRICTION_COEFFICIENT) == pytest.approx(
            SHEAR_YIELD_PA / 2000e6
        )

    def test_both_regimes_coexist_along_one_rake_face(self):
        # A real rake face sticks at the tip and slides further out; the
        # process must produce both from one pressure distribution.
        part = make_boundary(3)
        process = ZorevFrictionProcess(
            part,
            SLIDING_FRICTION,
            FLOW_STRESS_PA,
            contact_pressure=constant_pressure(3000e6, 300e6, 50e6),
        )
        process.ExecuteInitializeSolutionStep()
        coefficients = [part.GetNode(i).GetValue(KM.FRICTION_COEFFICIENT) for i in (1, 2, 3)]
        assert coefficients[0] < SLIDING_FRICTION  # sticking
        assert coefficients[1] == pytest.approx(SLIDING_FRICTION)  # sliding
        assert coefficients[2] == pytest.approx(SLIDING_FRICTION)  # sliding

    def test_separated_nodes_carry_no_friction(self):
        part = make_boundary(1)
        process = ZorevFrictionProcess(
            part, SLIDING_FRICTION, FLOW_STRESS_PA, contact_pressure=constant_pressure(0.0)
        )
        process.ExecuteInitializeSolutionStep()
        assert part.GetNode(1).GetValue(KM.FRICTION_COEFFICIENT) == 0.0

    def test_the_coefficient_tracks_pressure_between_steps(self):
        # Contact pressure swings as the chip forms, so a coefficient computed
        # once at setup would be wrong for most of the run.
        part = make_boundary(1)
        pressures = iter([100e6, 3000e6])
        process = ZorevFrictionProcess(
            part,
            SLIDING_FRICTION,
            FLOW_STRESS_PA,
            contact_pressure=lambda _p: np.array([next(pressures)]),
        )
        process.ExecuteInitializeSolutionStep()
        sliding = part.GetNode(1).GetValue(KM.FRICTION_COEFFICIENT)
        process.ExecuteInitializeSolutionStep()
        sticking = part.GetNode(1).GetValue(KM.FRICTION_COEFFICIENT)
        assert sliding == pytest.approx(SLIDING_FRICTION)
        assert sticking < sliding

    def test_softer_material_sticks_sooner(self):
        # Thermal softening lowers the flow stress along the interface, which
        # lowers the shear cap and widens the sticking region.
        part = make_boundary(1)
        hot = ZorevFrictionProcess(
            part, SLIDING_FRICTION, 200e6, contact_pressure=constant_pressure(600e6)
        )
        hot.ExecuteInitializeSolutionStep()
        hot_coefficient = part.GetNode(1).GetValue(KM.FRICTION_COEFFICIENT)

        cold = ZorevFrictionProcess(
            part, SLIDING_FRICTION, 800e6, contact_pressure=constant_pressure(600e6)
        )
        cold.ExecuteInitializeSolutionStep()
        cold_coefficient = part.GetNode(1).GetValue(KM.FRICTION_COEFFICIENT)
        assert hot_coefficient < cold_coefficient

    def test_marks_friction_active_so_the_solver_applies_it(self):
        # Without this flag Kratos ignores the coefficients entirely.
        part = make_boundary(1)
        ZorevFrictionProcess(
            part, SLIDING_FRICTION, FLOW_STRESS_PA, contact_pressure=constant_pressure(100e6)
        )
        import KratosMultiphysics.MPMApplication as MPM

        assert part.ProcessInfo[MPM.FRICTION_ACTIVE] is True

    def test_rejects_a_pressure_reading_of_the_wrong_length(self):
        part = make_boundary(3)
        process = ZorevFrictionProcess(
            part, SLIDING_FRICTION, FLOW_STRESS_PA, contact_pressure=constant_pressure(100e6)
        )
        with pytest.raises(ValueError):
            process.ExecuteInitializeSolutionStep()
