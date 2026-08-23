"""Telemetry writing — against real Kratos objects, real processes, and real files.

The one read that needs genuine material points (`read_material_point_fields`)
is injected, exactly as `tests/test_damage_process.py` injects the damage
gather: `MP_` variables only exist once a full MPM setup has generated points.
That single read was verified out of band against live material points from
`bench/jc_compression` and recorded in `docs/kratos_api_notes.md`.

Everything above it is exercised for real: a real `ModelPart` supplies the
point count and the step clock, a real `JohnsonCookDamageProcess` supplies the
separations, a real `RigidToolProcess` supplies the force and the velocity it
was measured against, and every assertion about what was written reads the
episode back off disk through `quinney.telemetry`.
"""

import dataclasses

import KratosMultiphysics as KM
import numpy as np
import pytest

from quinney.kratos_adapter.damage_process import (
    JohnsonCookDamageProcess,
    MaterialPointState,
    separated_point_ids,
)
from quinney.kratos_adapter.rigid_tool import RigidToolProcess
from quinney.kratos_adapter.telemetry_writer import (
    EnergyTotals,
    MaterialPointFields,
    TelemetryWriterProcess,
    snapshot_from_fields,
    tool_power_w,
)
from quinney.materials.johnson_cook import (
    JohnsonCookDamage,
    triaxiality_from_plane_strain_stress,
)
from quinney.telemetry import CaseMetadata, final_snapshot, load_episode, load_snapshot

# Damage settings copied from tests/test_damage_process.py: u_f = 2 G_f / sigma_y
# is deliberately large, so initiation and separation are separate events and a
# test can choose which one it wants.
FRACTURE_ENERGY_J_M2 = 1.0e6
YIELD_STRESS_PA = 200e6
POINT_VOLUME_M3 = 1e-6

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

POINT_MASS_KG = 2e-6
KINETIC_ENERGY_J = 7.0
INTERNAL_ENERGY_J = 11.0
CUTTING_FORCE_N = 430.0
THRUST_FORCE_N = -120.0
TIME_STEP_S = 2e-9
CUTTING_SPEED_M_S = 2.0
# The tool advances in -x at the cutting speed, so a resisting cut pushes back
# on it in +x and CUTTING_FORCE_N above is positive.
TOOL_VELOCITY_M_S = (-CUTTING_SPEED_M_S, 0.0)

METADATA = CaseMetadata(
    case_name="unit_cut",
    material="ofhc_copper",
    formulation="mpm",
    cutting_speed_m_s=CUTTING_SPEED_M_S,
    # Same speed, wrong way round: the tool process is the authority on which
    # direction the tool actually goes, and the writer records its vector.
    tool_velocity_m_s=(CUTTING_SPEED_M_S, 0.0),
    rake_angle_deg=5.0,
    depth_of_cut_m=1e-4,
    cell_size_m=2e-5,
    time_step_s=TIME_STEP_S,
    workpiece_bounds_m=(0.0, 1e-3, 0.0, 5e-4),
    tool_tip_m=(8e-4, 4e-4),
    initial_mass_kg=0.0,  # not claimed; the writer measures it
    friction_coefficient=0.3,
    friction_factor=0.6,
    deletion_threshold=1.0,
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
    part.ProcessInfo[KM.DELTA_TIME] = TIME_STEP_S
    return part


def advance_clock(part: KM.ModelPart, step: int) -> None:
    """Move the solver clock the way an analysis does between solution steps."""
    part.ProcessInfo[KM.STEP] = step
    part.ProcessInfo[KM.TIME] = step * TIME_STEP_S


def fields_for(ids, mass_kg=POINT_MASS_KG) -> MaterialPointFields:
    """Point fields whose coordinates and temperature identify the point.

    Point ``i`` sits at (i, 0) and is at 300 + i K, so a record written for one
    point cannot be mistaken for another's.
    """
    n = len(ids)
    return MaterialPointFields(
        ids=list(ids),
        coords_m=np.array([[float(i), 0.0] for i in ids], dtype=float).reshape(-1, 2),
        velocity_m_s=np.tile([-CUTTING_SPEED_M_S, 0.0], (n, 1)).reshape(-1, 2),
        eps_p=np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(n),
        eps_rate_per_s=np.full(n, 1e5),
        temperature_k=np.array([300.0 + i for i in ids], dtype=float),
        mass_kg=np.full(n, mass_kg),
        equivalent_stress_pa=np.full(n, 250e6),
        cauchy_xx_pa=np.full(n, -100e6),
        cauchy_yy_pa=np.full(n, -400e6),
    )


def snapshot_reader(part, step, time_s, initiation_ledger, damage_ledger):
    """Stand-in for the MPM read: synthetic fields for whatever points remain."""
    ids = [element.Id for element in part.Elements]
    return snapshot_from_fields(fields_for(ids), step, time_s, initiation_ledger, damage_ledger)


def energy_reader(part, process_info) -> EnergyTotals:
    return EnergyTotals(kinetic_j=KINETIC_ENERGY_J, internal_j=INTERNAL_ENERGY_J)


def state_for(ids, plastic_strain) -> MaterialPointState:
    """Damage-process state at reference conditions, varying only plastic strain.

    Pure shear (sxx = -syy) makes triaxiality zero, so the fracture strain is
    D1 + D2 = 4.5.
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


def damage_process(part, plastic_strain_by_id) -> JohnsonCookDamageProcess:
    """A real damage process reading a fixed plastic strain for each live point."""

    def gather(*_):
        ids = [element.Id for element in part.Elements]
        return state_for(ids, [plastic_strain_by_id.get(i, 0.0) for i in ids])

    return JohnsonCookDamageProcess(
        part, PARAMS, FRACTURE_ENERGY_J_M2, YIELD_STRESS_PA, gather=gather
    )


def make_rig(part, episode_dir, damage, sample_every, metadata=METADATA):
    """A real tool process and a writer wired to it, as a case would wire them."""
    tool = RigidToolProcess(part, TOOL_VELOCITY_M_S)
    writer = TelemetryWriterProcess(
        part,
        episode_dir,
        metadata,
        damage,
        tool,
        sample_every=sample_every,
        snapshot=snapshot_reader,
        energies=energy_reader,
    )
    return writer, tool


def run_steps(part, damage, writer, tool, steps) -> None:
    """One solution step per entry, in the order a case runs them.

    The tool measures its force first, then damage separates points, then the
    writer records both. The tool's own read needs material points, so its
    measurement is appended directly rather than solved for.
    """
    for step in steps:
        advance_clock(part, step)
        tool.force_history.append(np.array([CUTTING_FORCE_N, THRUST_FORCE_N]))
        damage.ExecuteFinalizeSolutionStep()
        writer.ExecuteFinalizeSolutionStep()


class TestSnapshotAssembly:
    def test_failure_state_follows_the_point_id_not_the_array_position(self):
        # The point set shrinks as points separate, so the nth ledger entry is
        # not the nth live point. Reading damage positionally would hand a
        # diagnostic the wrong point's history and it would look plausible.
        snapshot = snapshot_from_fields(
            fields_for([4, 1, 9]),
            step=10,
            time_s=2e-8,
            initiation_ledger={1: 1.0, 9: 0.5},
            damage_ledger={1: 0.25, 9: 0.0, 7: 0.99},
        )
        assert snapshot.damage == pytest.approx([0.0, 0.25, 0.0])
        assert snapshot.initiation == pytest.approx([0.0, 1.0, 0.5])

    def test_a_point_part_way_to_initiating_is_distinguishable_from_a_pristine_one(self):
        # Both read damage 0.0, and only the initiation column separates them —
        # which is what a supervisor previewing a threshold change needs.
        snapshot = snapshot_from_fields(
            fields_for([1, 2]),
            step=1,
            time_s=0.0,
            initiation_ledger={2: 0.98},
            damage_ledger={},
        )
        assert snapshot.damage == pytest.approx([0.0, 0.0])
        assert snapshot.initiation == pytest.approx([0.0, 0.98])

    def test_stress_survives_in_a_form_triaxiality_can_be_rebuilt_from(self):
        # The whole reason cauchy_xx/yy ride alongside the equivalent stress:
        # a supervisor previewing a deletion-threshold change offline has to
        # get the same triaxiality the solver saw, without re-running it.
        fields = fields_for([1])
        snapshot = snapshot_from_fields(fields, 1, 0.0, {}, {})
        offline = triaxiality_from_plane_strain_stress(
            snapshot.cauchy_xx_pa, snapshot.cauchy_yy_pa, snapshot.equivalent_stress_pa
        )
        in_solver = triaxiality_from_plane_strain_stress(
            fields.cauchy_xx_pa, fields.cauchy_yy_pa, fields.equivalent_stress_pa
        )
        assert offline == pytest.approx(in_solver)


class TestSeparatedPointIds:
    def test_names_the_points_the_damage_law_separated(self):
        assert separated_point_ids([4, 7, 9], np.array([False, True, True])) == (7, 9)

    def test_a_damage_process_reports_the_point_it_just_separated(self):
        part = make_model_part(2)
        # Fracture strain is 4.5, so 10.0 of strain initiates point 1 and
        # carries its evolution straight past separation in one step.
        damage = damage_process(part, {1: 10.0})
        damage.ExecuteFinalizeSolutionStep()
        assert damage.separated_ids == (1,)

    def test_a_run_with_nothing_separating_reports_nothing(self):
        part = make_model_part(2)
        damage = damage_process(part, {})
        damage.ExecuteFinalizeSolutionStep()
        assert damage.separated_ids == ()


class TestSnapshotSampling:
    def test_samples_the_first_step_it_sees_and_then_at_the_sampling_interval(self, tmp_path):
        part = make_model_part(3)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=4)
        run_steps(part, damage, writer, tool, range(1, 10))
        writer.ExecuteFinalize()

        assert load_episode(tmp_path).snapshot_steps == (1, 5, 9)

    def test_the_manifest_index_names_every_snapshot_written(self, tmp_path):
        # append_snapshot deliberately leaves the manifest alone, so a writer
        # that forgets to rewrite it leaves load_episode reading a stale index
        # and every downstream diagnostic analysing the wrong step.
        part = make_model_part(3)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=2)
        run_steps(part, damage, writer, tool, range(1, 8))
        writer.ExecuteFinalize()

        indexed = load_episode(tmp_path).snapshot_steps
        on_disk = sorted(
            int(path.stem.removeprefix("step_")) for path in (tmp_path / "snapshots").glob("*.npz")
        )
        assert list(indexed) == on_disk
        assert final_snapshot(tmp_path).step == max(on_disk)

    def test_a_snapshot_carries_the_state_of_the_points_still_present(self, tmp_path):
        part = make_model_part(3)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=1)
        run_steps(part, damage, writer, tool, [7])

        snapshot = load_snapshot(tmp_path, 7)
        assert list(snapshot.point_ids) == [1, 2, 3]
        assert snapshot.time_s == pytest.approx(7 * TIME_STEP_S)
        assert snapshot.coords_m[:, 0] == pytest.approx([1.0, 2.0, 3.0])

    def test_rejects_a_sampling_interval_below_one_step(self, tmp_path):
        part = make_model_part(1)
        with pytest.raises(ValueError):
            make_rig(part, tmp_path, damage_process(part, {}), sample_every=0)


class TestSurvivingAnUncleanExit:
    """The run being watched may die of the failure it is watched for.

    Everything the supervisor needs to diagnose that death has to already be
    on disk when it happens, not waiting in memory for a shutdown that never
    comes.
    """

    def test_the_episode_reads_back_without_the_run_ever_finishing(self, tmp_path):
        part = make_model_part(3)
        damage = damage_process(part, {2: 10.0})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=2)
        run_steps(part, damage, writer, tool, range(1, 6))
        # No ExecuteFinalize: the solver died here.

        episode = load_episode(tmp_path)
        assert episode.snapshot_steps == (1, 3, 5)
        assert list(episode.series.step) == [1, 2, 3, 4, 5]
        assert list(episode.deletions.point_id) == [2]
        assert final_snapshot(tmp_path).step == 5

    def test_the_force_history_up_to_the_last_sample_survives(self, tmp_path):
        part = make_model_part(2)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=3)
        run_steps(part, damage, writer, tool, range(1, 5))

        # Sampled at 1 and 4, so the flush at step 4 carries all four rows.
        assert load_episode(tmp_path).series.cutting_force_n == pytest.approx(
            np.full(4, CUTTING_FORCE_N)
        )


class TestSeries:
    def test_records_one_row_per_step_with_the_measured_force_and_time_step(self, tmp_path):
        part = make_model_part(3)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=100)
        run_steps(part, damage, writer, tool, range(1, 5))
        writer.ExecuteFinalize()

        series = load_episode(tmp_path).series
        assert list(series.step) == [1, 2, 3, 4]
        assert series.time_s == pytest.approx(np.array([1, 2, 3, 4]) * TIME_STEP_S)
        assert series.cutting_force_n == pytest.approx(np.full(4, CUTTING_FORCE_N))
        assert series.thrust_force_n == pytest.approx(np.full(4, THRUST_FORCE_N))
        assert series.time_step_s == pytest.approx(np.full(4, TIME_STEP_S))

    def test_follows_the_point_count_down_as_points_are_erased(self, tmp_path):
        import KratosMultiphysics.MPMApplication as MPM

        part = make_model_part(4)
        damage = damage_process(part, {2: 10.0})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=100)

        run_steps(part, damage, writer, tool, [1])
        MPM.MaterialPointEraseProcess(part).Execute()
        run_steps(part, damage, writer, tool, [2])
        writer.ExecuteFinalize()

        # Point 2 separates on step 1 but is still live at the end of it: the
        # solver erases at its next search, so the count catches up one step
        # after the mass does.
        series = load_episode(tmp_path).series
        assert list(series.n_points) == [4, 3]
        assert series.deleted_mass_kg == pytest.approx([POINT_MASS_KG, POINT_MASS_KG])

    def test_carries_the_energy_the_solver_reports(self, tmp_path):
        part = make_model_part(2)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=100)
        run_steps(part, damage, writer, tool, [1, 2])
        writer.ExecuteFinalize()

        series = load_episode(tmp_path).series
        assert series.kinetic_energy_j == pytest.approx(np.full(2, KINETIC_ENERGY_J))
        assert series.internal_energy_j == pytest.approx(np.full(2, INTERNAL_ENERGY_J))

    def test_an_episode_with_no_steps_still_loads(self, tmp_path):
        part = make_model_part(2)
        writer, _ = make_rig(part, tmp_path, damage_process(part, {}), sample_every=1)
        writer.ExecuteFinalize()

        episode = load_episode(tmp_path)
        assert len(episode.series.step) == 0
        assert len(episode.deletions.step) == 0
        assert episode.snapshot_steps == ()


class TestStepDiscipline:
    """A skipped step loses whatever separated in it, and nothing downstream
    could tell. Running a process under a `% every_n` guard is the established
    idiom here, so the mistake is an easy one to make.
    """

    def test_refuses_a_step_that_did_not_follow_the_last_one(self, tmp_path):
        part = make_model_part(3)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=1)
        run_steps(part, damage, writer, tool, [1, 2])

        with pytest.raises(ValueError, match="every solution step"):
            run_steps(part, damage, writer, tool, [5])

    def test_starts_wherever_the_run_does(self, tmp_path):
        # A resumed run picks up at the checkpoint's step, not at 1.
        part = make_model_part(3)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=1)
        run_steps(part, damage, writer, tool, [4000, 4001])
        writer.ExecuteFinalize()

        assert list(load_episode(tmp_path).series.step) == [4000, 4001]


class TestToolCoupling:
    """Force and velocity are two halves of one measurement.

    Passed separately, a sign typo in either negates `external_work_j` for the
    whole episode while every other column stays correct — undetectable
    downstream. So the writer takes the tool itself.
    """

    def test_the_recorded_tool_velocity_is_the_one_the_tool_is_driven_at(self, tmp_path):
        part = make_model_part(2)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=1)
        run_steps(part, damage, writer, tool, [1])
        writer.ExecuteFinalize()

        # METADATA claims +x; the tool actually goes -x, and the tool wins.
        assert load_episode(tmp_path).metadata.tool_velocity_m_s == pytest.approx(TOOL_VELOCITY_M_S)

    def test_refuses_metadata_that_describes_a_different_cutting_speed(self, tmp_path):
        part = make_model_part(2)
        wrong = dataclasses.replace(METADATA, cutting_speed_m_s=CUTTING_SPEED_M_S * 2)
        with pytest.raises(ValueError, match="different run"):
            make_rig(part, tmp_path, damage_process(part, {}), sample_every=1, metadata=wrong)

    def test_refuses_to_record_a_step_the_tool_has_not_measured(self, tmp_path):
        # Writer before tool: force_history[-1] would be the previous step's
        # force paired with this step's dt, and on the first step there is no
        # force at all.
        part = make_model_part(2)
        damage = damage_process(part, {})
        writer, _ = make_rig(part, tmp_path, damage, sample_every=1)
        advance_clock(part, 1)

        with pytest.raises(ValueError, match="RigidToolProcess"):
            writer.ExecuteFinalizeSolutionStep()

    def test_refuses_a_stale_force_from_a_step_the_tool_sat_out(self, tmp_path):
        part = make_model_part(2)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=1)
        run_steps(part, damage, writer, tool, [1])

        advance_clock(part, 2)
        with pytest.raises(ValueError, match="has not measured a force"):
            writer.ExecuteFinalizeSolutionStep()


class TestExternalWork:
    """MPM has no work accumulator, so the energy balance's third term is
    integrated here or it does not exist. The risk is entirely in the sign: a
    flipped one inverts the balance and nothing downstream looks wrong.
    """

    def test_a_tool_advancing_into_resisting_material_does_positive_work(self):
        # Driven in -x, pushed back in +x. This is the ordinary state of a cut,
        # and it must come out positive.
        assert tool_power_w((CUTTING_FORCE_N, 0.0), TOOL_VELOCITY_M_S) == pytest.approx(
            CUTTING_FORCE_N * CUTTING_SPEED_M_S
        )

    def test_a_workpiece_pushing_the_tool_along_does_negative_work(self):
        # Force on the tool now points the way the tool is already going, so
        # the workpiece is giving energy back rather than absorbing it.
        assert tool_power_w((-CUTTING_FORCE_N, 0.0), TOOL_VELOCITY_M_S) < 0.0

    def test_a_force_across_the_direction_of_travel_does_no_work(self):
        # Thrust is perpendicular to a tool fed purely in x, so it cannot
        # contribute — the guard against summing force magnitudes.
        assert tool_power_w((0.0, THRUST_FORCE_N), TOOL_VELOCITY_M_S) == pytest.approx(0.0)

    def test_work_accumulates_over_the_run_rather_than_reporting_one_step(self, tmp_path):
        part = make_model_part(2)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=100)
        run_steps(part, damage, writer, tool, [1, 2, 3])
        writer.ExecuteFinalize()

        per_step_j = CUTTING_FORCE_N * CUTTING_SPEED_M_S * TIME_STEP_S
        assert load_episode(tmp_path).series.external_work_j == pytest.approx(
            [per_step_j, 2 * per_step_j, 3 * per_step_j]
        )


class TestInitialMass:
    def test_the_baseline_mass_is_measured_before_anything_can_have_separated(self, tmp_path):
        # A deleted-mass fraction measured against a baseline that has itself
        # lost mass understates the loss, and understating it is the direction
        # that hides a failing run. So the baseline is taken at construction,
        # not from the first sample — here a point is erased before the first
        # sample is ever written.
        import KratosMultiphysics.MPMApplication as MPM

        part = make_model_part(4)
        damage = damage_process(part, {})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=100)

        part.GetElement(1).Set(KM.TO_ERASE, True)
        MPM.MaterialPointEraseProcess(part).Execute()
        run_steps(part, damage, writer, tool, [1])
        writer.ExecuteFinalize()

        assert load_episode(tmp_path).metadata.initial_mass_kg == pytest.approx(4 * POINT_MASS_KG)

    def test_a_baseline_the_caller_got_wrong_is_corrected_and_flagged(self, tmp_path):
        # Not silently overwritten: a caller that computed a mass from geometry
        # and density and disagrees this much has generated a different body
        # from the one the case describes, and that is worth hearing about.
        part = make_model_part(3)
        claimed = dataclasses.replace(METADATA, initial_mass_kg=99.0)
        with pytest.warns(UserWarning, match="not the body the case describes"):
            writer, _ = make_rig(
                part, tmp_path, damage_process(part, {}), sample_every=100, metadata=claimed
            )
        writer.ExecuteFinalize()

        assert load_episode(tmp_path).metadata.initial_mass_kg == pytest.approx(3 * POINT_MASS_KG)

    def test_refuses_to_start_before_the_material_points_exist(self, tmp_path):
        # Initialize() is the natural place to build a process, and MPM
        # generates its points after it. The documented denominator of the
        # deleted-mass fraction would be zero for the whole episode.
        part = make_model_part(0)
        with pytest.raises(ValueError, match="no baseline"):
            make_rig(part, tmp_path, damage_process(part, {}), sample_every=1)


class TestDeletionRecords:
    def test_records_where_and_when_a_separated_point_left(self, tmp_path):
        part = make_model_part(3)
        damage = damage_process(part, {2: 10.0})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=100)
        run_steps(part, damage, writer, tool, [1, 2, 3])
        writer.ExecuteFinalize()

        deletions = load_episode(tmp_path).deletions
        assert list(deletions.point_id) == [2]
        assert list(deletions.step) == [1]
        # Point 2 sits at (2, 0) and is at 302 K; a record built off array
        # position would report point 1's.
        assert deletions.coords_m[0] == pytest.approx([2.0, 0.0])
        assert deletions.temperature_k[0] == pytest.approx(302.0)
        assert deletions.mass_kg[0] == pytest.approx(POINT_MASS_KG)

    def test_a_point_still_awaiting_erasure_is_not_recorded_twice(self, tmp_path):
        # The solver erases at its next element search, so a flagged point is
        # re-reported as separated for as long as it survives. Counting it once
        # per step would inflate deleted mass without any mass being lost.
        part = make_model_part(3)
        damage = damage_process(part, {2: 10.0})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=100)
        run_steps(part, damage, writer, tool, [1, 2, 3, 4])
        writer.ExecuteFinalize()

        assert list(load_episode(tmp_path).deletions.point_id) == [2]

    def test_deleted_mass_accumulates_over_the_run(self, tmp_path):
        import KratosMultiphysics.MPMApplication as MPM

        part = make_model_part(4)
        strains = {2: 10.0, 4: 10.0}
        damage = damage_process(part, strains)
        writer, tool = make_rig(part, tmp_path, damage, sample_every=100)

        run_steps(part, damage, writer, tool, [1])
        MPM.MaterialPointEraseProcess(part).Execute()
        # Point 3 goes on the next step, so the column must step up cumulatively
        # rather than report what was lost within one step.
        strains[3] = 10.0
        run_steps(part, damage, writer, tool, [2])
        writer.ExecuteFinalize()

        assert load_episode(tmp_path).series.deleted_mass_kg == pytest.approx(
            [2 * POINT_MASS_KG, 3 * POINT_MASS_KG]
        )

    def test_a_point_erased_before_the_writer_saw_it_is_an_error(self, tmp_path):
        # Losing a deletion record silently would understate deleted mass, and
        # deleted mass is a bounded diagnostic. The ordering contract — writer
        # after damage, same step — is worth failing over.
        import KratosMultiphysics.MPMApplication as MPM

        part = make_model_part(3)
        damage = damage_process(part, {2: 10.0})
        writer, tool = make_rig(part, tmp_path, damage, sample_every=100)
        advance_clock(part, 1)
        tool.force_history.append(np.array([CUTTING_FORCE_N, THRUST_FORCE_N]))
        damage.ExecuteFinalizeSolutionStep()
        MPM.MaterialPointEraseProcess(part).Execute()

        with pytest.raises(ValueError, match="2"):
            writer.ExecuteFinalizeSolutionStep()
