"""Does a halted-and-resumed run reach the same state as an uninterrupted one?

The acceptance measurement for the halt/checkpoint/restart loop, and the source
of the numbers in docs/kratos_api_notes.md.

    uv run python bench/jc_compression/build_case.py
    uv run python bench/restart_fidelity/run.py --check

Four runs of the small MPM case, every one of them carrying a live
JohnsonCookDamageProcess:

  A  uninterrupted to 2N steps                              -- the reference
  B  halted at N by HaltProcess, resumed to 2N, one process -- the loop
  C  uninterrupted to 2N a second time                      -- the control
  D  same as B but the resume runs in a separate process    -- the architecture

C exists because the answer is meaningless without it. Kratos assembles nodal
sums across OpenMP threads in a nondeterministic order, so two identical runs
of the same case do not agree bit-for-bit either. Comparing B against A alone
would blame the restart for a difference the solver produces on its own.

D exists because the architecture's actual claim is that the solver exits and a
new process resumes from disk alone. B keeps the model in memory between the
halt and the resume, and a resume that quietly depended on warm in-process
state would pass B and fail the real loop.

The damage process is here because without it this measures Kratos' serializer
and nothing else. The damage ledgers are the state Kratos *cannot* carry -- MPM
has no damage variable at all -- and losing them is the failure that would
corrupt every downstream result while looking like physics. So the ledgers are
compared across the halt alongside the material-point fields.

Run with OMP_NUM_THREADS=1 to remove the thread-order term and see the restart
on its own; that is where the bit-exactness claim is measured, and --check
tightens the tolerance to exactly zero there.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication as MPM
import numpy as np
from KratosMultiphysics.MPMApplication.mpm_analysis import MpmAnalysis

from quinney.kratos_adapter.checkpoint import HaltProcess, SolverHalted, checkpoint_dir_for
from quinney.kratos_adapter.damage_process import JohnsonCookDamageProcess
from quinney.kratos_adapter.restart import (
    inject_parameters,
    read_manifest,
    restore_ledgers,
    resume_from,
)
from quinney.materials.ofhc_copper import (
    FRACTURE_ENERGY_J_M2,
    OFHC_COPPER_DAMAGE,
    YIELD_STRESS_PA,
)

CASE = pathlib.Path(__file__).resolve().parents[1] / "jc_compression"

# Short by default so the measurement is cheap enough to run every time. At
# this length no point has initiated (initiation reaches ~2e-3 of the way to
# 1.0), so the damage ledger is all zeros and only the initiation ledger is
# non-trivially exercised. --halt-step/--end-step run it long enough for damage
# proper; the negative control is what proves the comparison has teeth at
# whatever length it is run.
DEFAULT_HALT_STEP = 20
DEFAULT_END_STEP = 40


@dataclass(frozen=True)
class RunLength:
    """Where the run halts and where it is compared, in solver steps."""

    halt_step: int
    end_step: int


# Multithreaded, two uninterrupted runs of this case differ by ~1e-14 relative
# because Kratos reduces nodal sums across threads in a nondeterministic order.
# A resume is only suspect if it exceeds that floor.
MULTITHREADED_TOLERANCE_REL = 1e-13

# At one thread the control run is exactly identical, so the reproducibility
# floor is exactly zero and any restart error at all would show. Allowing 1e-13
# here would let a regression to 1e-15 pass while the documented claim of
# "exactly 0.0" quietly became false.
SINGLE_THREAD_TOLERANCE_REL = 0.0

# Every field docs/kratos_api_notes.md records as readable on a live material
# point, so the measurement cannot quietly skip the ones that matter.
#
# The strain rate earns its place: damage_process reads it every step for the
# Johnson-Cook rate term, and if it were law-internal state the serializer did
# not carry, the first post-resume step would evaluate damage at rate ~0 and
# give a different failure strain at every restart.
#
# The three energies earn theirs: they are the source of energy_drift_pct,
# which is exactly the quantity a DiagnosticTrip will watch. An accumulator
# that reset or jumped across a restart would make the halt trigger fire on
# restart artefacts.
SCALAR_FIELDS = (
    "MP_EQUIVALENT_PLASTIC_STRAIN",
    "MP_EQUIVALENT_PLASTIC_STRAIN_RATE",
    "MP_TEMPERATURE",
    "MP_EQUIVALENT_STRESS",
    "MP_HARDENING_RATIO",
    "MP_KINETIC_ENERGY",
    "MP_STRAIN_ENERGY",
    "MP_POTENTIAL_ENERGY",
    "MP_TOTAL_ENERGY",
    "MP_VOLUME",
    "MP_MASS",
    "MP_DENSITY",
)

# MP_ALMANSI_STRAIN_VECTOR is not in the notes' readability list but reads
# cleanly, and it is the only observable measure of how far each point has
# deformed: MPM 10.4.3 reports no deformation gradient at all.
VECTOR_FIELDS = (
    "MP_COORD",
    "MP_VELOCITY",
    "MP_DISPLACEMENT",
    "MP_CAUCHY_STRESS_VECTOR",
    "MP_ALMANSI_STRAIN_VECTOR",
)

ALL_FIELDS = SCALAR_FIELDS + VECTOR_FIELDS

# Written by JohnsonCookDamageProcess.ledgers, carried in the checkpoint's
# ledgers.json, and read back through its resume_from= argument.
LEDGER_NAMES = ("initiation", "damage", "plastic_strain")


def default_tolerance_rel() -> float:
    """Zero at one thread, the thread-order floor otherwise."""
    if os.environ.get("OMP_NUM_THREADS") == "1":
        return SINGLE_THREAD_TOLERANCE_REL
    return MULTITHREADED_TOLERANCE_REL


def base_parameters(end_step: int) -> dict:
    """The case parameters, ended at ``end_step`` steps of its own time step."""
    parameters = json.loads((CASE / "ProjectParameters.json").read_text(encoding="utf-8"))
    time_step_s = parameters["solver_settings"]["time_stepping"]["time_step"]
    parameters["problem_data"]["end_time"] = end_step * time_step_s
    parameters["problem_data"]["echo_level"] = 0
    parameters["solver_settings"]["echo_level"] = 0
    return parameters


def state_of(model_part: KM.ModelPart, ledgers: dict) -> dict:
    """Every readable material-point field plus the damage ledgers."""
    process_info = model_part.ProcessInfo
    elements = list(model_part.Elements)
    state: dict = {"ids": [element.Id for element in elements]}
    for name in SCALAR_FIELDS:
        variable = getattr(MPM, name)
        state[name] = np.array(
            [
                element.CalculateOnIntegrationPoints(variable, process_info)[0]
                for element in elements
            ],
            dtype=np.float64,
        )
    for name in VECTOR_FIELDS:
        variable = getattr(MPM, name)
        state[name] = np.array(
            [
                list(element.CalculateOnIntegrationPoints(variable, process_info)[0])
                for element in elements
            ],
            dtype=np.float64,
        )
    # Ledgers are keyed by point id, and the id order is the element order, so
    # they line up with the fields above without a second index.
    for name in LEDGER_NAMES:
        ledger = ledgers[name]
        state[f"ledger_{name}"] = np.array(
            [ledger.get(element.Id, 0.0) for element in elements], dtype=np.float64
        )
    return state


ALL_COLUMNS = ALL_FIELDS + tuple(f"ledger_{name}" for name in LEDGER_NAMES)


def save_state(state: dict, path: pathlib.Path) -> None:
    """Store a state snapshot so another process can compare against it."""
    np.savez(path, ids=np.asarray(state["ids"]), **{name: state[name] for name in ALL_COLUMNS})


def load_state(path: pathlib.Path) -> dict:
    loaded = np.load(path)
    return {"ids": loaded["ids"].tolist(), **{name: loaded[name] for name in ALL_COLUMNS}}


class DamagedAnalysis(MpmAnalysis):
    """MpmAnalysis carrying a damage process, and optionally a halt trigger.

    Both are built on the first FinalizeSolutionStep because the material point
    model part does not exist until the solver has generated the points. The
    damage process is seeded there too, which is early enough: it accumulates
    nothing before its first ExecuteFinalizeSolutionStep.
    """

    def __init__(
        self,
        model: KM.Model,
        parameters: dict,
        *,
        checkpoint_root: pathlib.Path | None = None,
        interval_steps: int | None = None,
        resumed_from: dict | None = None,
        resume_ledgers: dict | None = None,
    ) -> None:
        super().__init__(model, KM.Parameters(json.dumps(parameters)))
        self._raw_parameters = parameters
        self._checkpoint_root = checkpoint_root
        self._interval_steps = interval_steps
        self._resumed_from = resumed_from
        self._resume_ledgers = resume_ledgers
        self.damage: JohnsonCookDamageProcess | None = None
        self._halt: HaltProcess | None = None

    def FinalizeSolutionStep(self) -> None:
        super().FinalizeSolutionStep()
        part = self.model["MPM_Material"]
        if self.damage is None:
            self.damage = JohnsonCookDamageProcess(
                part,
                OFHC_COPPER_DAMAGE,
                FRACTURE_ENERGY_J_M2,
                YIELD_STRESS_PA,
                resume_from=self._resume_ledgers,
            )
            if self._checkpoint_root is not None:
                self._halt = HaltProcess(
                    part,
                    self._checkpoint_root,
                    parameters=self._raw_parameters,
                    interval_steps=self._interval_steps,
                    ledgers=lambda: self.damage.ledgers,
                    resumed_from=self._resumed_from,
                )
        self.damage.ExecuteFinalizeSolutionStep()
        if self._halt is not None:
            self._halt.ExecuteFinalizeSolutionStep()

    def final_state(self) -> dict:
        return state_of(self.model["MPM_Material"], self.damage.ledgers)


def run_uninterrupted(end_step: int) -> dict:
    analysis = DamagedAnalysis(KM.Model(), base_parameters(end_step))
    analysis.Run()
    return analysis.final_state()


def run_halted(checkpoint_root: pathlib.Path, length: RunLength) -> pathlib.Path:
    """Run until HaltProcess stops it, returning the checkpoint directory."""
    analysis = DamagedAnalysis(
        KM.Model(),
        base_parameters(length.end_step),
        checkpoint_root=checkpoint_root,
        interval_steps=length.halt_step,
    )
    try:
        analysis.Run()
    except SolverHalted as halted:
        return halted.checkpoint_dir
    raise RuntimeError("the halt trigger never fired")


def run_resumed(
    checkpoint_dir: pathlib.Path,
    checkpoint_root: pathlib.Path,
    length: RunLength,
    *,
    carry_ledgers: bool = True,
) -> dict:
    """Resume the checkpoint, keep accumulating damage, and finish the run.

    ``carry_ledgers=False`` is the negative control: it resumes off the .rest
    file alone, exactly as a caller who forgot the sidecar would. The damage
    ledgers are the state Kratos cannot carry, so this is the run that has to
    come out *different* -- if it does not, the ledger comparison above proves
    nothing.

    A second HaltProcess is attached, so this also exercises halt-resume-halt.
    Every mechanism in the loop keys on ProcessInfo[STEP] surviving the restart:
    if it came back as 1, the first post-resume check would raise "step 1
    precedes last_halt_step 20"; if 0, a later checkpoint directory name would
    collide. Both are asserted rather than assumed.
    """
    manifest = read_manifest(checkpoint_dir)
    parameters = inject_parameters(
        resume_from(checkpoint_dir),
        {"problem_data.end_time": base_parameters(length.end_step)["problem_data"]["end_time"]},
    )
    analysis = DamagedAnalysis(
        KM.Model(),
        parameters,
        checkpoint_root=checkpoint_root,
        interval_steps=length.halt_step,
        resumed_from=manifest,
        resume_ledgers=restore_ledgers(checkpoint_dir) if carry_ledgers else None,
    )
    analysis.Initialize()

    resumed_step = analysis.model["MPM_Material"].ProcessInfo[KM.STEP]
    if resumed_step != length.halt_step:
        raise RuntimeError(
            f"the restart did not carry ProcessInfo[STEP]: resumed at {resumed_step}, "
            f"expected {length.halt_step}"
        )
    print(f"@@@ resumed with ProcessInfo[STEP] == {resumed_step}, as checkpointed", flush=True)

    try:
        analysis.RunSolutionLoop()
    except SolverHalted as second:
        # The metronome fires again at 2N, which is where this run ends anyway.
        if second.step != length.end_step:
            raise RuntimeError(
                f"second halt at {second.step}, expected {length.end_step}"
            ) from second
        print(f"@@@ halt-resume-halt: second halt at step {second.step}", flush=True)
    return analysis.final_state()


def run_cross_process(scratch: pathlib.Path, length: RunLength) -> dict:
    """Halt in one process, resume in another, reading only what is on disk.

    Neither subprocess shares a Kratos Model, a ModelPart, a loaded parameter
    object or a damage ledger with the other or with this one.
    """
    root = scratch / "cross"
    state_path = scratch / "cross_state.npz"
    for stage in ("halt", "resume"):
        result = subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--stage",
                stage,
                "--scratch",
                str(root),
                "--state-path",
                str(state_path),
                "--halt-step",
                str(length.halt_step),
                "--end-step",
                str(length.end_step),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("@@@"):
                print(f"@@@   [{stage} subprocess] {line[4:]}", flush=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"{stage} subprocess failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )
    return load_state(state_path)


def compare(label: str, left: dict, right: dict, tolerance_rel: float | None) -> bool:
    """Print the field-by-field difference; return whether it is within tolerance.

    ``tolerance_rel`` of None reports without judging -- used for the control,
    whose whole job is to establish what the floor is rather than to pass.

    The normalisation is ``max|a - b| / max|a|``: the worst absolute difference
    against the field's own largest magnitude, not a per-point relative error.
    """
    print(f"@@@ --- {label} ---", flush=True)
    same_points = left["ids"] == right["ids"]
    print(f"@@@ point sets identical: {same_points}", flush=True)
    ok = same_points
    for name in ALL_COLUMNS:
        a, b = np.asarray(left[name]), np.asarray(right[name])
        difference = np.abs(a - b)
        scale = max(float(np.abs(a).max()), np.finfo(float).tiny)
        relative = float(difference.max()) / scale
        verdict = ""
        if tolerance_rel is not None and relative > tolerance_rel:
            verdict = "  <-- OVER TOLERANCE"
            ok = False
        print(
            f"@@@ {name:34s} max|d|={difference.max():.3e}"
            f"  rel={relative:.3e}"
            f"  identical={bool((a == b).all())}{verdict}",
            flush=True,
        )
    return ok


def report_damage(label: str, state: dict) -> None:
    """State how much damage there was to carry, so the comparison is not vacuous."""
    initiation = state["ledger_initiation"]
    damage = state["ledger_damage"]
    print(
        f"@@@ {label}: initiation max {initiation.max():.6g}, "
        f"points initiated {int((initiation >= 1.0).sum())}, "
        f"damage max {damage.max():.6g}, points damaged {int((damage > 0.0).sum())}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="restart fidelity measurement")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail with a non-zero exit code if any field exceeds the tolerance",
    )
    parser.add_argument("--tolerance-rel", type=float, default=None)
    parser.add_argument("--halt-step", type=int, default=DEFAULT_HALT_STEP)
    parser.add_argument("--end-step", type=int, default=DEFAULT_END_STEP)
    parser.add_argument("--stage", choices=("halt", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--scratch", help=argparse.SUPPRESS)
    parser.add_argument("--state-path", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    length = RunLength(arguments.halt_step, arguments.end_step)

    # The two cross-process stages re-enter this file as subprocesses.
    if arguments.stage == "halt":
        run_halted(pathlib.Path(arguments.scratch), length)
        return 0
    if arguments.stage == "resume":
        scratch = pathlib.Path(arguments.scratch)
        save_state(
            run_resumed(checkpoint_dir_for(scratch, length.halt_step), scratch, length),
            pathlib.Path(arguments.state_path),
        )
        return 0

    threads = os.environ.get("OMP_NUM_THREADS", "default")
    print(
        f"@@@ OMP_NUM_THREADS={threads}  halt at step {length.halt_step},"
        f" compare at {length.end_step},"
        f"  {len(ALL_COLUMNS)} columns ({len(ALL_FIELDS)} solver fields"
        f" + {len(LEDGER_NAMES)} quinney ledgers)",
        flush=True,
    )

    reference = run_uninterrupted(length.end_step)
    control = run_uninterrupted(length.end_step)
    report_damage("damage carried by the reference run", reference)
    with tempfile.TemporaryDirectory() as scratch:
        root = pathlib.Path(scratch)
        checkpoint_dir = run_halted(root / "in_process", length)
        if checkpoint_dir != checkpoint_dir_for(root / "in_process", length.halt_step):
            raise RuntimeError(f"unexpected checkpoint directory {checkpoint_dir}")
        resumed = run_resumed(checkpoint_dir, root / "in_process", length)
        dropped = run_resumed(checkpoint_dir, root / "dropped", length, carry_ledgers=False)
        cross = run_cross_process(root, length)

    tolerance = (
        arguments.tolerance_rel if arguments.tolerance_rel is not None else default_tolerance_rel()
    )
    compare("control: uninterrupted vs uninterrupted", reference, control, None)
    in_process_ok = compare(
        "test: uninterrupted vs halted-and-resumed (one process)",
        reference,
        resumed,
        tolerance if arguments.check else None,
    )
    cross_ok = compare(
        "test: uninterrupted vs halted-and-resumed (two processes)",
        reference,
        cross,
        tolerance if arguments.check else None,
    )

    # The negative control. It is reported, never judged against the tolerance:
    # its job is to diverge.
    compare(
        "negative control: uninterrupted vs resumed WITHOUT the ledger sidecar",
        reference,
        dropped,
        None,
    )
    ledger_divergence = max(
        float(np.abs(reference[f"ledger_{name}"] - dropped[f"ledger_{name}"]).max())
        for name in LEDGER_NAMES
    )
    print(
        f"@@@ dropping the sidecar changes the ledgers by {ledger_divergence:.6g}"
        " -- this is what the .rest file alone cannot carry",
        flush=True,
    )

    if not arguments.check:
        return 0

    # A ledger comparison over all-zero ledgers would pass no matter what the
    # resume did, so the measurement checks its own teeth before reporting.
    if ledger_divergence <= 0.0:
        print(
            "@@@ VERDICT: FAIL -- dropping the sidecar changed nothing, so the"
            " ledger comparison is vacuous"
        )
        return 1
    passed = in_process_ok and cross_ok
    print(f"@@@ VERDICT: {'PASS' if passed else 'FAIL'} at relative tolerance {tolerance:g}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
