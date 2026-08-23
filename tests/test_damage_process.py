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

# u_f = 2 G_f / sigma_y = 2e6/2e8 = 1e-2 m against a 1e-3 m point, so each unit
# of post-initiation plastic strain adds 0.1 of damage. Deliberately slow
# enough that initiation and separation are visibly separate events.
FRACTURE_ENERGY_J_M2 = 1.0e6
YIELD_STRESS_PA = 200e6
POINT_VOLUME_M3 = 1e-6  # sqrt -> 1e-3 m characteristic length

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
        volume_m3=np.full(n, POINT_VOLUME_M3),
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
    def test_separates_points_that_initiate_and_then_fully_evolve(self):
        part = make_model_part(2)
        # Fracture strain is 4.5, so 10.0 of strain initiates point 1 and the
        # same increment carries its evolution straight past 1.0.
        process = JohnsonCookDamageProcess(
            part,
            PARAMS,
            FRACTURE_ENERGY_J_M2,
            YIELD_STRESS_PA,
            gather=lambda *_: state_for([1, 2], [10.0, 0.0]),
        )
        process.ExecuteFinalizeSolutionStep()
        assert part.GetElement(1).Is(KM.TO_ERASE) is True
        assert part.GetElement(2).Is(KM.TO_ERASE) is False

    def test_initiation_accumulates_across_steps_before_evolution_starts(self):
        part = make_model_part(1)
        # 1.5 of strain per step against a fracture strain of 4.5 initiates on
        # the third step. Separation needs evolution on top of that, so the
        # point survives all three — initiating is not failing.
        process = JohnsonCookDamageProcess(
            part, PARAMS, FRACTURE_ENERGY_J_M2, YIELD_STRESS_PA, gather=growing([1], per_step=1.5)
        )
        process.ExecuteFinalizeSolutionStep()
        assert process.initiation[1] < 1.0
        process.ExecuteFinalizeSolutionStep()
        assert process.initiation[1] < 1.0
        process.ExecuteFinalizeSolutionStep()
        assert process.initiation[1] == pytest.approx(1.0)
        # Initiated, and now damaging — but nowhere near separated.
        assert 0.0 < process.damage[1] < 1.0
        assert part.GetElement(1).Is(KM.TO_ERASE) is False

    def test_exposes_the_initiation_field_for_diagnostics(self):
        part = make_model_part(2)
        process = JohnsonCookDamageProcess(
            part,
            PARAMS,
            FRACTURE_ENERGY_J_M2,
            YIELD_STRESS_PA,
            gather=lambda *_: state_for([1, 2], [2.25, 0.0]),
        )
        process.ExecuteFinalizeSolutionStep()
        assert process.initiation == pytest.approx({1: 0.5, 2: 0.0})
        # Nothing has initiated, so nothing has damaged.
        assert process.damage == pytest.approx({1: 0.0, 2: 0.0})

    def test_a_lower_threshold_separates_earlier(self):
        # The supervisor's adjust_deletion_threshold action, end to end: the
        # same history under a looser criterion separates while the strict one
        # is still evolving.
        loose_part = make_model_part(1)
        loose = JohnsonCookDamageProcess(
            loose_part,
            PARAMS,
            FRACTURE_ENERGY_J_M2,
            YIELD_STRESS_PA,
            separation_threshold=0.3,
            gather=growing([1], per_step=1.5),
        )
        strict_part = make_model_part(1)
        strict = JohnsonCookDamageProcess(
            strict_part,
            PARAMS,
            FRACTURE_ENERGY_J_M2,
            YIELD_STRESS_PA,
            separation_threshold=1.0,
            gather=growing([1], per_step=1.5),
        )
        for _ in range(4):
            loose.ExecuteFinalizeSolutionStep()
            strict.ExecuteFinalizeSolutionStep()

        assert loose_part.GetElement(1).Is(KM.TO_ERASE) is True
        assert strict_part.GetElement(1).Is(KM.TO_ERASE) is False

    def test_initiation_stops_growing_when_plastic_strain_plateaus(self):
        # Cumulative strain held constant means zero increment. Without the
        # differencing this would keep re-adding the same strain every step and
        # separate a point that has stopped deforming.
        part = make_model_part(1)
        process = JohnsonCookDamageProcess(
            part,
            PARAMS,
            FRACTURE_ENERGY_J_M2,
            YIELD_STRESS_PA,
            gather=lambda *_: state_for([1], [2.25]),
        )
        process.ExecuteFinalizeSolutionStep()
        first = process.initiation[1]
        process.ExecuteFinalizeSolutionStep()
        process.ExecuteFinalizeSolutionStep()
        assert process.initiation[1] == pytest.approx(first)
        assert part.GetElement(1).Is(KM.TO_ERASE) is False


class TestResumingFromACheckpoint:
    """A halt splits one run in two, and the damage ledgers are the join.

    Kratos checkpoints what it owns; damage, initiation and the previous
    plastic strain exist only in this process, so a resume that cannot seed
    them restarts every point undamaged and separates late — with nothing
    raising, because a zero ledger is perfectly well formed.
    """

    def test_damage_carries_on_from_where_the_halt_left_it(self):
        part = make_model_part(1)
        # Resumed mid-evolution: already initiated, 0.4 damaged, and 6.0 of
        # plastic strain behind it. The next reading of 7.0 is therefore an
        # increment of 1.0, worth 0.1 of damage against u_f = 1e-2 m.
        process = JohnsonCookDamageProcess(
            part,
            PARAMS,
            FRACTURE_ENERGY_J_M2,
            YIELD_STRESS_PA,
            gather=lambda *_: state_for([1], [7.0]),
            resume_from={"initiation": {1: 1.0}, "damage": {1: 0.4}, "plastic_strain": {1: 6.0}},
        )
        process.ExecuteFinalizeSolutionStep()

        assert process.damage[1] == pytest.approx(0.5)
        assert part.GetElement(1).Is(KM.TO_ERASE) is False

    def test_a_fresh_run_ignoring_the_seed_would_separate_the_same_point(self):
        # The control: with the plastic-strain baseline empty the same reading
        # is a 7.0 increment, which is what makes that ledger load-bearing
        # rather than decorative. Initiation is seeded so both runs are past
        # initiation.
        part = make_model_part(1)
        process = JohnsonCookDamageProcess(
            part,
            PARAMS,
            FRACTURE_ENERGY_J_M2,
            YIELD_STRESS_PA,
            gather=lambda *_: state_for([1], [7.0]),
            resume_from={"initiation": {1: 1.0}, "damage": {}, "plastic_strain": {}},
        )
        process.ExecuteFinalizeSolutionStep()

        assert process.damage[1] == pytest.approx(0.7)

    def test_the_ledgers_it_hands_out_are_the_ones_it_reads_back(self):
        part = make_model_part(1)
        first = JohnsonCookDamageProcess(
            part, PARAMS, FRACTURE_ENERGY_J_M2, YIELD_STRESS_PA, gather=growing([1], per_step=1.5)
        )
        for _ in range(3):
            first.ExecuteFinalizeSolutionStep()

        resumed = JohnsonCookDamageProcess(
            make_model_part(1),
            PARAMS,
            FRACTURE_ENERGY_J_M2,
            YIELD_STRESS_PA,
            gather=lambda *_: state_for([1], [4.5]),
            resume_from=first.ledgers,
        )
        resumed.ExecuteFinalizeSolutionStep()

        # Same plastic strain as the halt point means no increment, so nothing
        # moved: the resumed process starts exactly where the first stopped.
        assert resumed.damage == pytest.approx(first.damage)
        assert resumed.initiation == pytest.approx(first.initiation)

    def test_refuses_a_checkpoint_that_lost_a_ledger_on_the_way(self):
        # The damaging error is a checkpoint carrying two of the three, not one
        # carrying a fourth: restore_ledgers hands back whatever the sidecar
        # holds and checks nothing, and an absent plastic_strain would arrive
        # as an empty baseline that separates points which should have lived.
        part = make_model_part(1)
        with pytest.raises(ValueError, match="plastic_strain"):
            JohnsonCookDamageProcess(
                part,
                PARAMS,
                FRACTURE_ENERGY_J_M2,
                YIELD_STRESS_PA,
                resume_from={"initiation": {1: 1.0}, "damage": {1: 0.4}},
            )

    def test_refuses_to_resume_from_a_ledger_it_does_not_carry(self):
        part = make_model_part(1)
        with pytest.raises(ValueError, match="softening"):
            JohnsonCookDamageProcess(
                part,
                PARAMS,
                FRACTURE_ENERGY_J_M2,
                YIELD_STRESS_PA,
                resume_from={"softening": {1: 0.5}},
            )
