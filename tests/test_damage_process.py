"""In-loop damage process — against real Kratos objects.

Reading MP_ variables needs genuine material points, which need a full MPM
setup (background grid, body, point generation) that only exists once the
Phase 0 cutting case does. That single read is therefore injected; everything
else here — flagging, erasure, the ledger across steps — runs against a real
Kratos ModelPart with real elements and a real MaterialPointEraseProcess.
"""

import KratosMultiphysics as KM
import numpy as np
import pytest

from quinney.kratos_adapter.damage_process import (
    JohnsonCookDamageProcess,
    MaterialPointState,
    flag_separated,
)
from quinney.materials.johnson_cook import JohnsonCookDamage

PARAMS = JohnsonCookDamage(
    d1=0.5,
    d2=4.0,
    d3=-3.0,
    d4=0.02,
    d5=1.0,
    reference_strain_rate_per_s=1.0,
    reference_temperature_k=300.0,
    melt_temperature_k=1300.0,
)


def make_model_part(n_elements: int) -> KM.ModelPart:
    """A real ModelPart carrying `n_elements` triangles, ids 1..n."""
    model = KM.Model()
    part = model.CreateModelPart("cutting")
    for node_id in range(1, n_elements + 3):
        part.CreateNewNode(node_id, float(node_id), 0.0, 0.0)
    props = part.CreateNewProperties(0)
    for element_id in range(1, n_elements + 1):
        part.CreateNewElement(
            "Element2D3N", element_id, [element_id, element_id + 1, element_id + 2], props
        )
    return part


def state_for(ids, plastic_strain):
    """Material-point state at reference conditions, varying only plastic strain.

    Stress is pure shear (sxx = -syy) so triaxiality is zero and the fracture
    strain reduces to D1 + D2 = 4.5.
    """
    n = len(ids)
    return MaterialPointState(
        ids=list(ids),
        plastic_strain=np.asarray(plastic_strain, dtype=float),
        plastic_strain_rate_per_s=np.ones(n),
        temperature_k=np.full(n, 300.0),
        equivalent_stress=np.full(n, 100e6),
        cauchy_xx=np.full(n, 100e6),
        cauchy_yy=np.full(n, -100e6),
    )


def growing(ids, per_step):
    """A gather that reports plastic strain growing by `per_step` each call.

    Real material points report *cumulative* equivalent plastic strain; the
    process differences it, because MP_DELTA_PLASTIC_STRAIN is not implemented.
    """
    total = [0.0]

    def gather(*_):
        total[0] += per_step
        return state_for(ids, [total[0]] * len(ids))

    return gather


class TestFlagSeparated:
    def test_flags_only_the_separated_points(self):
        part = make_model_part(3)
        flag_separated(part, ids=[1, 2, 3], separated=np.array([False, True, False]))
        assert [part.GetElement(i).Is(KM.TO_ERASE) for i in (1, 2, 3)] == [False, True, False]

    def test_reports_how_many_points_it_flagged(self):
        part = make_model_part(3)
        count = flag_separated(part, ids=[1, 2, 3], separated=np.array([True, True, False]))
        assert count == 2

    def test_flagged_points_are_removed_by_the_erase_process(self):
        # The flag is only half the mechanism; this is the half that actually
        # deletes mass, and deleted_mass_pct is a bounded diagnostic.
        import KratosMultiphysics.MPMApplication as MPM

        part = make_model_part(3)
        flag_separated(part, ids=[1, 2, 3], separated=np.array([True, False, True]))
        MPM.MaterialPointEraseProcess(part).Execute()
        assert [e.Id for e in part.Elements] == [2]

    def test_rejects_mismatched_lengths(self):
        part = make_model_part(3)
        with pytest.raises(ValueError):
            flag_separated(part, ids=[1, 2, 3], separated=np.array([True]))


class TestJohnsonCookDamageProcess:
    def test_separates_points_whose_damage_reaches_the_threshold(self):
        part = make_model_part(2)
        # Fracture strain at these conditions is 4.5; point 1 gets a strain
        # increment well past it, point 2 gets none.
        process = JohnsonCookDamageProcess(
            part, PARAMS, gather=lambda *_: state_for([1, 2], [10.0, 0.0])
        )
        process.ExecuteFinalizeSolutionStep()
        assert part.GetElement(1).Is(KM.TO_ERASE) is True
        assert part.GetElement(2).Is(KM.TO_ERASE) is False

    def test_damage_accumulates_across_steps_until_it_separates(self):
        part = make_model_part(1)
        # 1.5 of strain per step against a fracture strain of 4.5 separates on
        # the third step, not the first — damage must persist between calls.
        process = JohnsonCookDamageProcess(part, PARAMS, gather=growing([1], per_step=1.5))
        process.ExecuteFinalizeSolutionStep()
        assert part.GetElement(1).Is(KM.TO_ERASE) is False
        process.ExecuteFinalizeSolutionStep()
        assert part.GetElement(1).Is(KM.TO_ERASE) is False
        process.ExecuteFinalizeSolutionStep()
        assert part.GetElement(1).Is(KM.TO_ERASE) is True

    def test_exposes_the_damage_field_for_diagnostics(self):
        part = make_model_part(2)
        process = JohnsonCookDamageProcess(
            part, PARAMS, gather=lambda *_: state_for([1, 2], [2.25, 0.0])
        )
        process.ExecuteFinalizeSolutionStep()
        assert process.damage == pytest.approx({1: 0.5, 2: 0.0})

    def test_a_lower_threshold_separates_earlier(self):
        # The supervisor's adjust_deletion_threshold action, end to end.
        part = make_model_part(1)
        process = JohnsonCookDamageProcess(
            part,
            PARAMS,
            separation_threshold=0.3,
            gather=growing([1], per_step=1.5),
        )
        process.ExecuteFinalizeSolutionStep()
        assert part.GetElement(1).Is(KM.TO_ERASE) is True

    def test_damage_stops_growing_when_plastic_strain_plateaus(self):
        # Cumulative strain held constant means zero increment. Without the
        # differencing this would keep re-adding the same strain every step and
        # separate a point that has stopped deforming.
        part = make_model_part(1)
        process = JohnsonCookDamageProcess(part, PARAMS, gather=lambda *_: state_for([1], [2.25]))
        process.ExecuteFinalizeSolutionStep()
        first = process.damage[1]
        process.ExecuteFinalizeSolutionStep()
        process.ExecuteFinalizeSolutionStep()
        assert process.damage[1] == pytest.approx(first)
        assert part.GetElement(1).Is(KM.TO_ERASE) is False
