"""Resuming a halted run: reading the checkpoint back and changing the plan.

Everything under test here is a transform over plain dicts read off disk, so
none of it needs Kratos — which is the point. The supervisor process that
decides what to change may be a local model, a rule baseline or a human on a
machine with no solver installed, and it has to be able to build the resumed
parameters anyway.

The checkpoints are written by the real writer, against a real Kratos
ModelPart, so what these tests read back is what a halted solver actually
leaves behind.
"""

import json
import subprocess
import sys

import KratosMultiphysics as KM
import numpy as np
import pytest

from quinney.kratos_adapter.checkpoint import (
    MANIFEST_FILENAME,
    write_checkpoint,
)
from quinney.kratos_adapter.damage_process import (
    JohnsonCookDamageProcess,
    MaterialPointState,
)
from quinney.kratos_adapter.restart import (
    inject_parameters,
    read_manifest,
    restore_ledgers,
    resume_from,
)
from quinney.materials.johnson_cook import JohnsonCookDamage

PARAMETERS = {
    "problem_data": {"problem_name": "cut", "start_time": 0.0, "end_time": 1.0e-4},
    "solver_settings": {
        "solver_type": "Dynamic",
        "model_part_name": "MPM_Material",
        "model_import_settings": {"input_type": "mdpa", "input_filename": "body"},
        "grid_model_import_settings": {"input_type": "mdpa", "input_filename": "grid"},
        "time_stepping": {"time_step": 2.0e-8},
    },
}


def make_part(n_elements: int, step: int) -> KM.ModelPart:
    model = KM.Model()
    part = model.CreateModelPart("MPM_Material")
    for node_id in range(1, n_elements + 3):
        part.CreateNewNode(node_id, float(node_id), 0.0, 0.0)
    props = part.CreateNewProperties(1)
    for element_id in range(1, n_elements + 1):
        part.CreateNewElement(
            "Element2D3N", element_id, [element_id, element_id + 1, element_id + 2], props
        )
    part.ProcessInfo[KM.STEP] = step
    part.ProcessInfo[KM.TIME] = step * 2.0e-8
    return part


@pytest.fixture
def checkpoint(tmp_path):
    """A real checkpoint at step 250, carrying two per-point ledgers."""
    write_checkpoint(
        make_part(4, step=250),
        tmp_path,
        parameters=PARAMETERS,
        ledgers={"damage": {1: 0.7, 3: 0.2}, "initiation": {1: 1.0, 3: 1.0}},
        reason="step_interval",
        detail="every 250 steps",
    )
    return tmp_path


class TestReadingTheCheckpointBack:
    def test_reports_where_the_run_stopped(self, checkpoint):
        manifest = read_manifest(checkpoint)
        assert manifest["step"] == 250
        assert manifest["reason"] == "step_interval"

    def test_refuses_a_checkpoint_written_under_an_incompatible_schema(self, checkpoint):
        # A major bump means a field changed meaning; reinterpreting old numbers
        # under new rules is worse than refusing to resume.
        path = checkpoint / MANIFEST_FILENAME
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "99.0"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="schema"):
            read_manifest(checkpoint)

    def test_missing_checkpoint_is_an_error_not_an_empty_resume(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path / "never_written")

    def test_damage_comes_back_keyed_by_the_same_point_ids(self, checkpoint):
        # JSON has no integer keys; a ledger that came back keyed by strings
        # would look empty to every lookup and restart the run undamaged.
        ledgers = restore_ledgers(checkpoint)
        assert ledgers["damage"] == {1: 0.7, 3: 0.2}
        assert ledgers["initiation"] == {1: 1.0, 3: 1.0}


class TestResumeFrom:
    def test_points_the_material_points_at_the_restart_file(self, checkpoint):
        resumed = resume_from(checkpoint)
        import_settings = resumed["solver_settings"]["model_import_settings"]
        assert import_settings["input_type"] == "rest"
        assert import_settings["restart_load_file_label"] == "250"
        assert import_settings["input_filename"] == "MPM_Material"

    def test_the_restart_path_is_absolute_so_the_resume_can_run_from_anywhere(self, checkpoint):
        from pathlib import Path

        path = Path(
            resume_from(checkpoint)["solver_settings"]["model_import_settings"]["input_output_path"]
        )
        assert path.is_absolute()
        assert list(path.glob("*.rest"))

    def test_the_background_grid_is_still_read_from_its_mesh(self, checkpoint):
        # MPM regenerates the grid every step and the solver rejects "rest" for
        # it outright; only the material points carry state across the halt.
        resumed = resume_from(checkpoint)
        assert resumed["solver_settings"]["grid_model_import_settings"]["input_type"] == "mdpa"

    def test_everything_the_supervisor_did_not_touch_is_unchanged(self, checkpoint):
        resumed = resume_from(checkpoint)
        assert resumed["problem_data"] == PARAMETERS["problem_data"]
        assert resumed["solver_settings"]["time_stepping"] == {"time_step": 2.0e-8}

    def test_does_not_mutate_the_stored_parameters(self, checkpoint):
        resume_from(checkpoint)
        stored = json.loads((checkpoint / "parameters.json").read_text(encoding="utf-8"))
        assert stored["solver_settings"]["model_import_settings"]["input_type"] == "mdpa"


class TestInjectParameters:
    def test_applies_a_supervisor_decision_to_the_named_setting(self):
        changed = inject_parameters(PARAMETERS, {"solver_settings.time_stepping.time_step": 1.0e-8})
        assert changed["solver_settings"]["time_stepping"]["time_step"] == 1.0e-8

    def test_leaves_the_original_alone(self):
        inject_parameters(PARAMETERS, {"problem_data.end_time": 5.0e-5})
        assert PARAMETERS["problem_data"]["end_time"] == 1.0e-4

    def test_applies_several_overrides_at_once(self):
        changed = inject_parameters(
            PARAMETERS,
            {"problem_data.end_time": 5.0e-5, "solver_settings.time_stepping.time_step": 5.0e-9},
        )
        assert changed["problem_data"]["end_time"] == 5.0e-5
        assert changed["solver_settings"]["time_stepping"]["time_step"] == 5.0e-9

    def test_no_overrides_is_a_faithful_copy(self):
        assert inject_parameters(PARAMETERS, {}) == PARAMETERS

    def test_an_unknown_setting_raises_rather_than_being_created(self):
        # A misspelled path that quietly added a key would look like a decision
        # the solver accepted and then ignored.
        with pytest.raises(KeyError, match="time_stap"):
            inject_parameters(PARAMETERS, {"solver_settings.time_stapping.time_step": 1.0e-8})

    def test_a_path_that_runs_through_a_leaf_raises(self):
        with pytest.raises(KeyError, match="solver_type"):
            inject_parameters(PARAMETERS, {"solver_settings.solver_type.name": "Dynamic"})

    def test_an_empty_path_raises(self):
        with pytest.raises(ValueError, match="empty"):
            inject_parameters(PARAMETERS, {"": 1})


# Slow enough that initiation and separation are visibly separate events:
# u_f = 2 G_f / sigma_y = 1e-2 m against a 1e-3 m point, so each unit of
# post-initiation plastic strain adds 0.1 of damage.
FRACTURE_ENERGY_J_M2 = 1.0e6
YIELD_STRESS_PA = 200e6
POINT_VOLUME_M3 = 1e-6

DAMAGE_PARAMS = JohnsonCookDamage(
    d1=0.5,
    d2=4.0,
    d3=-3.0,
    d4=0.02,
    d5=1.0,
    reference_strain_rate_per_s=1.0,
    reference_temperature_k=300.0,
    melt_temperature_k=1300.0,
)


def growing_state(ids, per_step):
    """A gather reporting cumulative plastic strain growing by `per_step`.

    Pure shear (sxx = -syy) so triaxiality is zero and the fracture strain
    reduces to D1 + D2.
    """
    total = [0.0]

    def gather(*_):
        total[0] += per_step
        n = len(ids)
        return MaterialPointState(
            ids=list(ids),
            plastic_strain=np.full(n, total[0]),
            plastic_strain_rate_per_s=np.ones(n),
            temperature_k=np.full(n, 300.0),
            equivalent_stress=np.full(n, 100e6),
            cauchy_xx=np.full(n, 100e6),
            cauchy_yy=np.full(n, -100e6),
            volume_m3=np.full(n, POINT_VOLUME_M3),
        )

    return gather


def damage_process(part, gather, resume=None):
    return JohnsonCookDamageProcess(
        part,
        DAMAGE_PARAMS,
        fracture_energy_j_m2=FRACTURE_ENERGY_J_M2,
        yield_stress_pa=YIELD_STRESS_PA,
        gather=gather,
        resume_from=resume,
    )


class TestDamageSurvivesTheHalt:
    """The end-to-end path: a damaged run halts, and the resumed run is damaged.

    Kratos has no damage variable at all, so none of this state is in the
    ``.rest`` file. If the sidecar names and the process's ledger names ever
    drift apart, the resumed run restarts every point undamaged and separates
    late — and nothing about a zero ledger is malformed enough to notice.
    """

    def test_a_resumed_run_carries_the_damage_the_halted_run_accumulated(self, tmp_path):
        part = make_part(3, step=100)
        ids = [1, 2, 3]

        before = damage_process(part, growing_state(ids, per_step=1.0))
        for _ in range(6):
            before.ExecuteFinalizeSolutionStep()
        accumulated = dict(before.damage)
        assert max(accumulated.values()) > 0.0, "the test needs real damage to carry"

        write_checkpoint(part, tmp_path, parameters=PARAMETERS, ledgers=before.ledgers)
        after = damage_process(
            make_part(3, step=100),
            growing_state(ids, per_step=1.0),
            resume=restore_ledgers(tmp_path),
        )

        assert after.damage == accumulated
        assert after.initiation == dict(before.initiation)

    def test_a_resume_that_dropped_the_sidecar_would_start_undamaged(self, tmp_path):
        # The control for the test above: this is exactly what a resume off the
        # .rest file alone produces, and it is why the sidecar exists.
        part = make_part(3, step=100)
        before = damage_process(part, growing_state([1, 2, 3], per_step=1.0))
        for _ in range(6):
            before.ExecuteFinalizeSolutionStep()

        undamaged = damage_process(make_part(3, step=100), growing_state([1, 2, 3], per_step=1.0))
        assert undamaged.damage == {}
        assert before.damage != undamaged.damage

    def test_the_ledger_names_the_writer_stores_are_the_names_the_process_reads(self, tmp_path):
        # These agree today only because two modules chose the same strings.
        part = make_part(2, step=100)
        process = damage_process(part, growing_state([1, 2], per_step=1.0))
        process.ExecuteFinalizeSolutionStep()

        write_checkpoint(part, tmp_path, parameters=PARAMETERS, ledgers=process.ledgers)
        manifest = read_manifest(tmp_path)
        assert manifest["ledger_names"] == sorted(process.ledgers)
        # Unknown names raise rather than being dropped, so this round-trips or
        # it fails loudly.
        damage_process(part, growing_state([1, 2], per_step=1.0), resume=restore_ledgers(tmp_path))


# Run in a fresh interpreter with every KratosMultiphysics import blocked. The
# body is parameterised only by which module it imports, so the same script
# serves as the test and as its own negative control.
_BLOCKED_IMPORT_SCRIPT = """
import sys
from importlib.abc import MetaPathFinder


class Blocked(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "KratosMultiphysics" or fullname.startswith("KratosMultiphysics."):
            raise ImportError("KratosMultiphysics is not installed on this machine")
        return None


sys.meta_path.insert(0, Blocked())
import {module}

assert not [name for name in sys.modules if name.startswith("KratosMultiphysics")], (
    "the blocker let KratosMultiphysics through"
)
print("clean")
"""

# Every public function, called for real. Importing them is not enough: an
# import of Kratos *inside* a function body would sail past an import-only
# check and only fail on the supervisor's machine, at the moment it matters.
_EXERCISE_SCRIPT = _BLOCKED_IMPORT_SCRIPT.replace("import {module}", "") + """
import json
import pathlib
import sys
from quinney.kratos_adapter.restart import (
    inject_parameters,
    read_manifest,
    restore_ledgers,
    resume_from,
)

directory = pathlib.Path(sys.argv[1])
assert read_manifest(directory)["step"] == 250
assert restore_ledgers(directory)["damage"] == {1: 0.7, 3: 0.2}
assert resume_from(directory)["solver_settings"]["model_import_settings"]["input_type"] == "rest"
assert inject_parameters({"a": {"b": 1}}, {"a.b": 2}) == {"a": {"b": 2}}
print("exercised")
"""


def _in_blocked_interpreter(script: str, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script, *arguments], capture_output=True, text=True, check=False
    )


class TestImportableWithoutKratos:
    """The supervisor, the rule baseline and replay all need this module.

    None of them has a Kratos wheel, and no other test here can catch a
    regression because every test module in this suite imports Kratos itself.
    """

    def test_restart_imports_on_a_machine_with_no_solver_installed(self):
        result = _in_blocked_interpreter(
            _BLOCKED_IMPORT_SCRIPT.format(module="quinney.kratos_adapter.restart")
        )
        assert result.returncode == 0, result.stderr
        assert "clean" in result.stdout

    def test_the_blocker_actually_blocks(self):
        # Without this, a mistyped module prefix would leave the test above
        # green forever while checking nothing at all. checkpoint.py imports
        # Kratos unconditionally, so it must fail.
        result = _in_blocked_interpreter(
            _BLOCKED_IMPORT_SCRIPT.format(module="quinney.kratos_adapter.checkpoint")
        )
        assert result.returncode != 0
        assert "KratosMultiphysics is not installed" in result.stderr

    def test_every_public_function_runs_without_a_solver(self, checkpoint):
        # Import-only coverage would miss a Kratos import inside a function
        # body, which fails on the supervisor's machine and nowhere else.
        result = _in_blocked_interpreter(_EXERCISE_SCRIPT, str(checkpoint))
        assert result.returncode == 0, result.stderr
        assert "exercised" in result.stdout
