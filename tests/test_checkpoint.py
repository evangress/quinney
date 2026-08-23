"""Halting the solver: when to stop, and what has to be on disk when it does.

Two halves. The trigger predicates are pure arithmetic over a step counter and
a diagnostic mapping and are exercised here without any solver — that is the
point of keeping them in ``quinney.halt``. The checkpoint writer needs a real
Kratos ModelPart to serialize, so those tests build one.

The physics being asserted is bookkeeping physics: a halt must fire exactly
once per interval, a diagnostic that has gone out of bounds must win over the
clock, and everything the supervisor will read after the solver is gone must be
on disk before the halt is announced.
"""

import json

import KratosMultiphysics as KM
import pytest

from quinney.halt import (
    REASON_DIAGNOSTIC_TRIP,
    REASON_STEP_INTERVAL,
    TRIP_REARM_STEPS,
    DiagnosticTrip,
    should_halt,
)
from quinney.kratos_adapter.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    LEDGERS_FILENAME,
    MANIFEST_FILENAME,
    PARAMETERS_FILENAME,
    HaltProcess,
    SolverHalted,
    write_checkpoint,
)

# Fixture bounds, deliberately not the spec §5 numbers: the canonical drift and
# band-width thresholds belong to the diagnostics that compute those quantities,
# and a trip carries whatever bound it was constructed with.
ENERGY_DRIFT_TRIP = DiagnosticTrip(name="energy_drift_pct", threshold=7.0, direction="above")
BAND_WIDTH_TRIP = DiagnosticTrip(name="band_width_elements", threshold=4.0, direction="below")


class TestStepIntervalTrigger:
    def test_does_not_halt_before_the_interval_has_elapsed(self):
        decision = should_halt(step=99, interval_steps=100, last_halt_step=None)
        assert not decision.halt
        assert decision.reason is None

    def test_halts_once_the_interval_has_elapsed(self):
        decision = should_halt(step=100, interval_steps=100, last_halt_step=None)
        assert decision.halt
        assert decision.reason == REASON_STEP_INTERVAL

    def test_counts_the_interval_from_the_previous_halt_not_from_zero(self):
        assert not should_halt(step=150, interval_steps=100, last_halt_step=100).halt
        assert should_halt(step=200, interval_steps=100, last_halt_step=100).halt

    def test_step_zero_never_halts_on_the_metronome(self):
        # Checkpointing the initial state buys nothing and the supervisor has
        # no history to reason over yet. A *trip* at step 0 is different and
        # does fire: a diagnostic already out of bounds before the first step
        # is a broken setup, not a clock tick.
        assert not should_halt(step=0, interval_steps=100, last_halt_step=None).halt

    def test_no_interval_means_no_clock_driven_halt(self):
        assert not should_halt(step=10_000, interval_steps=None, last_halt_step=None).halt

    def test_rejects_a_non_positive_interval(self):
        with pytest.raises(ValueError, match="interval_steps"):
            should_halt(step=10, interval_steps=0, last_halt_step=None)

    def test_rejects_a_step_that_precedes_the_last_halt(self):
        with pytest.raises(ValueError, match="precedes"):
            should_halt(step=50, interval_steps=100, last_halt_step=100)

    def test_a_run_that_has_never_halted_measures_the_interval_from_zero(self):
        assert should_halt(step=100, interval_steps=100, last_halt_step=None).halt


class TestDiagnosticTrigger:
    def test_halts_when_a_diagnostic_rises_past_its_bound(self):
        decision = should_halt(
            step=7,
            interval_steps=None,
            last_halt_step=None,
            diagnostics={"energy_drift_pct": 8.1},
            trips=(ENERGY_DRIFT_TRIP,),
        )
        assert decision.halt
        assert decision.reason == REASON_DIAGNOSTIC_TRIP
        assert "energy_drift_pct" in decision.detail

    def test_does_not_halt_while_the_diagnostic_is_inside_its_bound(self):
        decision = should_halt(
            step=7,
            interval_steps=None,
            last_halt_step=None,
            diagnostics={"energy_drift_pct": 6.9},
            trips=(ENERGY_DRIFT_TRIP,),
        )
        assert not decision.halt

    def test_halts_when_a_diagnostic_falls_below_its_bound(self):
        # A shear band resolved by too few elements is under-resolved, so the
        # interesting trip is downward.
        decision = should_halt(
            step=7,
            interval_steps=None,
            last_halt_step=None,
            diagnostics={"band_width_elements": 2.0},
            trips=(BAND_WIDTH_TRIP,),
        )
        assert decision.halt

    def test_a_diagnostic_exactly_on_its_bound_does_not_trip(self):
        for trip, value in ((ENERGY_DRIFT_TRIP, 7.0), (BAND_WIDTH_TRIP, 4.0)):
            decision = should_halt(
                step=7,
                interval_steps=None,
                last_halt_step=None,
                diagnostics={trip.name: value},
                trips=(trip,),
            )
            assert not decision.halt, trip.name

    def test_a_diagnostic_trip_outranks_the_step_interval(self):
        decision = should_halt(
            step=100,
            interval_steps=100,
            last_halt_step=None,
            diagnostics={"energy_drift_pct": 12.0},
            trips=(ENERGY_DRIFT_TRIP,),
        )
        assert decision.reason == REASON_DIAGNOSTIC_TRIP

    def test_names_every_diagnostic_that_tripped_not_just_the_first(self):
        decision = should_halt(
            step=7,
            interval_steps=None,
            last_halt_step=None,
            diagnostics={"energy_drift_pct": 12.0, "band_width_elements": 1.0},
            trips=(ENERGY_DRIFT_TRIP, BAND_WIDTH_TRIP),
        )
        assert "energy_drift_pct" in decision.detail
        assert "band_width_elements" in decision.detail

    def test_a_watched_diagnostic_that_was_never_computed_raises(self):
        # Silently not halting because a diagnostic went missing is the failure
        # that loses a run, so it is loud.
        with pytest.raises(KeyError, match="energy_drift_pct"):
            should_halt(
                step=7,
                interval_steps=None,
                last_halt_step=None,
                diagnostics={},
                trips=(ENERGY_DRIFT_TRIP,),
            )

    def test_a_diagnostic_with_no_answer_yet_is_not_a_trip(self):
        # Band width is None until the band localizes, and an unlocalized body
        # is a normal physical state for most of a run. This is the state the
        # diagnostics deliberately report rather than faking a number for.
        decision = should_halt(
            step=7,
            interval_steps=None,
            diagnostics={"band_width_elements": None},
            trips=(BAND_WIDTH_TRIP,),
        )
        assert not decision.halt

    def test_a_diagnostic_with_no_answer_yet_is_not_an_error_either(self):
        # The three states are distinct: absent raises, None passes quietly,
        # NaN raises. Collapsing None into either of the others is what would
        # push a try/except into every caller, or fire the band-width trip on
        # every pre-localization step.
        assert not should_halt(
            step=7,
            interval_steps=None,
            diagnostics={"band_width_elements": None, "energy_drift_pct": 1.0},
            trips=(BAND_WIDTH_TRIP, ENERGY_DRIFT_TRIP),
        ).halt

    def test_an_answerless_diagnostic_does_not_mask_one_that_did_trip(self):
        decision = should_halt(
            step=7,
            interval_steps=None,
            diagnostics={"band_width_elements": None, "energy_drift_pct": 12.0},
            trips=(BAND_WIDTH_TRIP, ENERGY_DRIFT_TRIP),
        )
        assert decision.halt
        assert decision.fired_trips == ("energy_drift_pct",)

    def test_a_non_finite_diagnostic_raises_rather_than_comparing_false(self):
        # NaN compares false against every bound, so a diverging run would sail
        # past its own trip.
        with pytest.raises(ValueError, match="energy_drift_pct"):
            should_halt(
                step=7,
                interval_steps=None,
                last_halt_step=None,
                diagnostics={"energy_drift_pct": float("nan")},
                trips=(ENERGY_DRIFT_TRIP,),
            )

    def test_a_trip_fires_at_step_zero_when_the_case_is_already_out_of_bounds(self):
        decision = should_halt(
            step=0,
            interval_steps=100,
            last_halt_step=None,
            diagnostics={"energy_drift_pct": 12.0},
            trips=(ENERGY_DRIFT_TRIP,),
        )
        assert decision.halt
        assert decision.reason == REASON_DIAGNOSTIC_TRIP

    def test_a_trip_that_cannot_clear_does_not_halt_on_every_step_forever(self):
        # Energy drift is cumulative from run start: it is still over the bound
        # on the first step after the resume, and it cannot un-drift. Without a
        # re-arm window the loop would checkpoint, exit and resume every single
        # step, burning one supervision episode per step and never advancing
        # the physics.
        drift = {"energy_drift_pct": 12.0}
        halt_step = 1200

        first = should_halt(
            step=halt_step,
            interval_steps=None,
            diagnostics=drift,
            trips=(ENERGY_DRIFT_TRIP,),
        )
        assert first.halt
        assert first.fired_trips == ("energy_drift_pct",)

        fired = {"energy_drift_pct": halt_step}
        for step in range(halt_step + 1, halt_step + TRIP_REARM_STEPS):
            assert not should_halt(
                step=step,
                interval_steps=None,
                last_trip_steps=fired,
                diagnostics=drift,
                trips=(ENERGY_DRIFT_TRIP,),
            ).halt, step

    def test_a_trip_still_standing_after_the_window_halts_again(self):
        # The window bounds the damage; it does not silence the diagnostic. A
        # run that is genuinely still diverging gets supervised again.
        decision = should_halt(
            step=1200 + TRIP_REARM_STEPS,
            interval_steps=None,
            last_trip_steps={"energy_drift_pct": 1200},
            diagnostics={"energy_drift_pct": 12.0},
            trips=(ENERGY_DRIFT_TRIP,),
        )
        assert decision.halt
        assert decision.reason == REASON_DIAGNOSTIC_TRIP

    def test_routine_metronome_halts_do_not_make_the_trips_deaf(self):
        # The failure this replaced: with a shared timer, a 20-step metronome
        # and a 50-step trip window, step - last_halt_step is never 50 again
        # for the life of the run, so energy drift could pass 500 percent with
        # every halt still recorded as a routine clock tick.
        interval = 20
        drift = {"energy_drift_pct": 12.0}
        for halt_step in (20, 40, 60, 80, 100):
            decision = should_halt(
                step=halt_step,
                interval_steps=interval,
                last_halt_step=halt_step - interval,
                last_trip_steps={},
                diagnostics=drift,
                trips=(ENERGY_DRIFT_TRIP,),
            )
            assert decision.reason == REASON_DIAGNOSTIC_TRIP, halt_step

    def test_one_trip_firing_does_not_silence_another(self):
        decision = should_halt(
            step=1210,
            interval_steps=None,
            last_trip_steps={"energy_drift_pct": 1200},
            diagnostics={"energy_drift_pct": 12.0, "band_width_elements": 1.0},
            trips=(ENERGY_DRIFT_TRIP, BAND_WIDTH_TRIP),
        )
        assert decision.halt
        assert decision.fired_trips == ("band_width_elements",)

    def test_a_metronome_halt_reports_no_fired_trips_to_disarm(self):
        decision = should_halt(
            step=100,
            interval_steps=100,
            diagnostics={"energy_drift_pct": 1.0},
            trips=(ENERGY_DRIFT_TRIP,),
        )
        assert decision.reason == REASON_STEP_INTERVAL
        assert decision.fired_trips == ()

    def test_a_missing_diagnostic_is_loud_even_inside_the_re_arm_window(self):
        # The window must not become a place where a broken diagnostic hides.
        with pytest.raises(KeyError, match="energy_drift_pct"):
            should_halt(
                step=1201,
                interval_steps=None,
                last_trip_steps={"energy_drift_pct": 1200},
                diagnostics={},
                trips=(ENERGY_DRIFT_TRIP,),
            )

    def test_rejects_an_unknown_trip_direction(self):
        with pytest.raises(ValueError, match="direction"):
            should_halt(
                step=7,
                interval_steps=None,
                last_halt_step=None,
                diagnostics={"x": 1.0},
                trips=(DiagnosticTrip(name="x", threshold=0.0, direction="sideways"),),
            )


def make_part(n_elements: int) -> KM.ModelPart:
    """A real ModelPart the serializer can round-trip."""
    model = KM.Model()
    part = model.CreateModelPart("MPM_Material")
    for node_id in range(1, n_elements + 3):
        part.CreateNewNode(node_id, float(node_id), 0.0, 0.0)
    props = part.CreateNewProperties(1)
    for element_id in range(1, n_elements + 1):
        part.CreateNewElement(
            "Element2D3N", element_id, [element_id, element_id + 1, element_id + 2], props
        )
    part.ProcessInfo[KM.STEP] = 250
    part.ProcessInfo[KM.TIME] = 5.0e-6
    return part


class TestWriteCheckpoint:
    def test_writes_a_restart_file_the_solver_can_be_pointed_at(self, tmp_path):
        part = make_part(4)
        write_checkpoint(part, tmp_path, parameters={"problem_data": {}}, ledgers={})
        restart_files = list((tmp_path / "restart").glob("*.rest"))
        assert [path.name for path in restart_files] == ["MPM_Material_250.rest"]

    def test_manifest_records_where_the_run_was_when_it_stopped(self, tmp_path):
        part = make_part(4)
        write_checkpoint(
            part,
            tmp_path,
            parameters={"problem_data": {}},
            ledgers={},
            reason=REASON_STEP_INTERVAL,
            detail="every 250 steps",
        )
        manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert manifest["step"] == 250
        assert manifest["time_s"] == pytest.approx(5.0e-6)
        assert manifest["reason"] == REASON_STEP_INTERVAL
        assert manifest["restart_load_file_label"] == "250"
        assert manifest["model_part_name"] == "MPM_Material"

    def test_manifest_carries_the_trigger_state_across_the_process_boundary(self, tmp_path):
        # Without it every trip re-arms on every restart, and the
        # halt-every-step loop the re-arm exists to prevent comes back.
        part = make_part(2)
        write_checkpoint(
            part,
            tmp_path,
            parameters={},
            ledgers={},
            trip_steps={"energy_drift_pct": 200},
        )
        manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["trip_steps"] == {"energy_drift_pct": 200}

    def test_stores_the_parameters_the_run_was_using(self, tmp_path):
        # The supervisor sees only files on disk, so the parameters it is about
        # to modify have to be among them.
        part = make_part(2)
        parameters = {"problem_data": {"end_time": 1.0e-5}}
        write_checkpoint(part, tmp_path, parameters=parameters, ledgers={})
        stored = json.loads((tmp_path / PARAMETERS_FILENAME).read_text(encoding="utf-8"))
        assert stored == parameters

    def test_carries_the_per_point_ledgers_kratos_knows_nothing_about(self, tmp_path):
        # Damage has no Kratos variable at all, so a resume that trusted the
        # .rest file alone would restart every point undamaged.
        part = make_part(3)
        write_checkpoint(
            part,
            tmp_path,
            parameters={},
            ledgers={"damage": {1: 0.4, 2: 0.9}, "initiation": {1: 1.0}},
        )
        ledgers = json.loads((tmp_path / LEDGERS_FILENAME).read_text(encoding="utf-8"))
        assert ledgers["damage"] == {"1": 0.4, "2": 0.9}
        assert ledgers["initiation"] == {"1": 1.0}

    def test_refuses_to_overwrite_an_existing_checkpoint(self, tmp_path):
        # Two halts writing to one directory would leave a manifest describing
        # one step and a restart file holding another.
        part = make_part(2)
        write_checkpoint(part, tmp_path, parameters={}, ledgers={})
        with pytest.raises(FileExistsError):
            write_checkpoint(part, tmp_path, parameters={}, ledgers={})


class TestHaltProcess:
    def test_runs_on_without_halting_while_inside_the_interval(self, tmp_path):
        part = make_part(2)
        part.ProcessInfo[KM.STEP] = 3
        process = HaltProcess(part, tmp_path, parameters={}, interval_steps=10, ledgers=dict, resumed_from=None)
        process.ExecuteFinalizeSolutionStep()
        assert not any(tmp_path.iterdir())

    def test_checkpoints_and_asks_the_solver_to_stop_when_the_trigger_fires(self, tmp_path):
        part = make_part(2)
        part.ProcessInfo[KM.STEP] = 10
        process = HaltProcess(part, tmp_path, parameters={}, interval_steps=10, ledgers=dict, resumed_from=None)
        with pytest.raises(SolverHalted) as halted:
            process.ExecuteFinalizeSolutionStep()
        assert halted.value.reason == REASON_STEP_INTERVAL
        assert halted.value.step == 10
        assert (tmp_path / "step_000010" / MANIFEST_FILENAME).exists()

    def test_the_checkpoint_is_complete_before_the_halt_is_announced(self, tmp_path):
        # The solver process dies on this exception; anything written after it
        # is never written.
        part = make_part(2)
        part.ProcessInfo[KM.STEP] = 10
        process = HaltProcess(
            part,
            tmp_path,
            parameters={},
            interval_steps=10,
            ledgers=lambda: {"damage": {1: 0.25}},
            resumed_from=None,
        )
        with pytest.raises(SolverHalted):
            process.ExecuteFinalizeSolutionStep()
        ledgers = json.loads(
            (tmp_path / "step_000010" / LEDGERS_FILENAME).read_text(encoding="utf-8")
        )
        assert ledgers["damage"] == {"1": 0.25}

    def test_a_watched_diagnostic_out_of_bounds_stops_the_run_early(self, tmp_path):
        part = make_part(2)
        part.ProcessInfo[KM.STEP] = 4
        process = HaltProcess(
            part,
            tmp_path,
            parameters={},
            interval_steps=None,
            ledgers=dict,
            resumed_from=None,
            trips=(ENERGY_DRIFT_TRIP,),
            diagnostics=lambda: {"energy_drift_pct": 11.5},
        )
        with pytest.raises(SolverHalted) as halted:
            process.ExecuteFinalizeSolutionStep()
        assert halted.value.reason == REASON_DIAGNOSTIC_TRIP

    def test_forgetting_to_wire_the_ledgers_is_an_error_not_an_empty_checkpoint(self, tmp_path):
        # A checkpoint with ledgers.json == {} is fully valid and resumes every
        # point undamaged. Losing a run to a forgotten keyword is not a trade
        # worth a default.
        part = make_part(2)
        with pytest.raises(TypeError, match="ledgers"):
            HaltProcess(part, tmp_path, parameters={}, interval_steps=10, resumed_from=None)

    def test_forgetting_the_resume_state_is_an_error_not_a_halt_every_step_loop(self, tmp_path):
        # A resumed run built with no manifest restarts its metronome from
        # origin 0 and re-arms every trip: resumed at step 20 with a 20-step
        # interval, it halts at 21, resumes, halts at 22, forever.
        part = make_part(2)
        with pytest.raises(TypeError, match="resumed_from"):
            HaltProcess(part, tmp_path, parameters={}, interval_steps=10, ledgers=dict)

    def test_a_resumed_run_keeps_its_cadence_instead_of_halting_immediately(self, tmp_path):
        part = make_part(2)
        part.ProcessInfo[KM.STEP] = 21
        process = HaltProcess(
            part,
            tmp_path,
            parameters={},
            interval_steps=20,
            ledgers=dict,
            resumed_from={"step": 20, "trip_steps": {}},
        )
        process.ExecuteFinalizeSolutionStep()
        assert not any(tmp_path.iterdir())

    def test_a_resumed_run_keeps_a_standing_trip_disarmed(self, tmp_path):
        # The re-arm window has to survive the process boundary, or every
        # restart re-arms the trip and the loop returns by another door.
        part = make_part(2)
        part.ProcessInfo[KM.STEP] = 1201
        process = HaltProcess(
            part,
            tmp_path,
            parameters={},
            interval_steps=None,
            ledgers=dict,
            resumed_from={"step": 1200, "trip_steps": {"energy_drift_pct": 1200}},
            trips=(ENERGY_DRIFT_TRIP,),
            diagnostics=lambda: {"energy_drift_pct": 99.0},
        )
        process.ExecuteFinalizeSolutionStep()
        assert not any(tmp_path.iterdir())

    def test_a_schema_1_0_manifest_without_trigger_state_still_resumes(self, tmp_path):
        # Minor-version tolerance: the field is simply absent, and re-arming a
        # trip is the safe reading of a checkpoint that predates trips.
        part = make_part(2)
        part.ProcessInfo[KM.STEP] = 1201
        process = HaltProcess(
            part,
            tmp_path,
            parameters={},
            interval_steps=None,
            ledgers=dict,
            resumed_from={"step": 1200},
            trips=(ENERGY_DRIFT_TRIP,),
            diagnostics=lambda: {"energy_drift_pct": 99.0},
        )
        with pytest.raises(SolverHalted):
            process.ExecuteFinalizeSolutionStep()

    def test_a_metronome_halt_leaves_the_trips_armed(self, tmp_path):
        # The regression that made every trip permanently deaf.
        part = make_part(2)
        part.ProcessInfo[KM.STEP] = 20
        drift = {"energy_drift_pct": 1.0}
        process = HaltProcess(
            part,
            tmp_path,
            parameters={},
            interval_steps=20,
            ledgers=dict,
            resumed_from=None,
            trips=(ENERGY_DRIFT_TRIP,),
            diagnostics=lambda: drift,
        )
        with pytest.raises(SolverHalted) as clock_halt:
            process.ExecuteFinalizeSolutionStep()
        assert clock_halt.value.reason == REASON_STEP_INTERVAL

        drift["energy_drift_pct"] = 99.0
        part.ProcessInfo[KM.STEP] = 40
        with pytest.raises(SolverHalted) as trip_halt:
            process.ExecuteFinalizeSolutionStep()
        assert trip_halt.value.reason == REASON_DIAGNOSTIC_TRIP

    def test_the_telemetry_seam_is_offered_the_checkpoint_directory(self, tmp_path):
        # quinney.telemetry owns the episode format; this process only says
        # where and when.
        seen = []
        part = make_part(2)
        part.ProcessInfo[KM.STEP] = 10
        process = HaltProcess(
            part,
            tmp_path,
            parameters={},
            interval_steps=10,
            ledgers=dict,
            resumed_from=None,
            dump_telemetry=seen.append,
        )
        with pytest.raises(SolverHalted):
            process.ExecuteFinalizeSolutionStep()
        assert seen == [tmp_path / "step_000010"]
