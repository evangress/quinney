"""Stopping the solver and leaving behind everything a supervisor will need.

The architecture is halt-and-restart across a process boundary: the solver
exits, a supervisor process reads files, a new solver process resumes. So the
checkpoint is not a convenience — it is the entire handover, and anything not
written here is gone.

Kratos' own ``RestartUtility`` turns out to cover the material-point physics
completely: a resume off a ``.rest`` file reproduces plastic strain,
temperature, stress, coordinates and velocity bit-for-bit (see
docs/kratos_api_notes.md, "Restart — verified on MPM material points"). What it
cannot cover is the state quinney keeps in Python because Kratos has nowhere to
put it: the damage ledgers, which exist only because MPM has no damage variable
at all. Those go in a JSON sidecar next to the ``.rest`` file, and a resume that
read only the ``.rest`` would silently restart every point undamaged.

The directory layout and the schema version are ``quinney.checkpoint_format``'s,
not this module's, so a reader with no Kratos wheel can still speak the format.

Deliberately absent: the telemetry dump. ``quinney.telemetry`` owns that format
and its episode directory; ``HaltProcess`` only says when and where, through the
``dump_telemetry`` seam.

Also absent: the softening state. ``DamageSofteningProcess``'s pristine-property
record is per-point *and* per-variable, which no ``PointLedger`` can express,
and neither it nor its applied-damage map has an accessor. Nothing here carries
it and nothing should pretend to — **and do not wire a softening resume until
that changes**. The degraded ``Properties`` clones themselves survive the
serializer, so a rebuilt ``DamageSofteningProcess`` records the already-degraded
values as pristine and degrades them again by the same damage: (1-D) squared
instead of (1-D), compounding at every halt. That is worse than resetting to
pristine, because it is monotone and physics-shaped and reads as accelerating
damage rather than as a bug.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import KratosMultiphysics as KM
from KratosMultiphysics.restart_utility import RestartUtility

from quinney.checkpoint_format import (
    CHECKPOINT_SCHEMA_VERSION,
    LEDGERS_FILENAME,
    MANIFEST_FILENAME,
    PARAMETERS_FILENAME,
    RESTART_DIRNAME,
    checkpoint_dir_for,
)
from quinney.halt import TRIP_REARM_STEPS, DiagnosticTrip, should_halt
from quinney.point_ledger import PointLedger

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "LEDGERS_FILENAME",
    "MANIFEST_FILENAME",
    "PARAMETERS_FILENAME",
    "RESTART_DIRNAME",
    "HaltProcess",
    "SolverHalted",
    "checkpoint_dir_for",
    "write_checkpoint",
]

# Binary serialization. The ASCII traces exist for debugging the serializer and
# cost an order of magnitude in file size for nothing we need.
_SERIALIZER_TRACE = "no_trace"


class SolverHalted(Exception):
    """Raised to unwind out of the solution loop so the solver process can exit.

    Not an error. It carries the reason and the checkpoint directory so the
    launcher can report the halt and hand the directory to a supervisor rather
    than treating the non-zero exit as a crash.
    """

    def __init__(self, reason: str, detail: str, step: int, checkpoint_dir: Path) -> None:
        super().__init__(f"halted at step {step}: {reason} ({detail})")
        self.reason = reason
        self.detail = detail
        self.step = step
        self.checkpoint_dir = checkpoint_dir


def write_checkpoint(
    model_part: KM.ModelPart,
    checkpoint_dir: Path | str,
    *,
    parameters: Mapping[str, Any],
    ledgers: Mapping[str, PointLedger],
    trip_steps: Mapping[str, int] = MappingProxyType({}),
    reason: str | None = None,
    detail: str = "",
) -> Path:
    """Serialize the model part and the quinney-side state into one directory.

    ``parameters`` is the ProjectParameters dict the run was launched with. It
    is stored because the supervisor sees only files on disk: it has to be able
    to read what the run was doing before it proposes a change to it.

    ``ledgers`` maps a name to a per-point mapping of id to scalar — typically
    ``JohnsonCookDamageProcess.ledgers``. It has no default: an empty mapping is
    a legitimate value for a run with no damage process, but it is also what a
    caller who simply forgot to wire it would produce, and the resulting
    checkpoint is fully valid and complete-looking while meaning "every point
    undamaged". Passing ``{}`` on purpose costs a caller nothing. JSON turns the
    integer point ids into strings; ``restart.restore_ledgers`` turns them back.

    ``trip_steps`` is the halt-trigger state — which diagnostic trip last fired
    and when — so a resumed run rebuilds its re-arm windows instead of starting
    them over. It defaults to empty because a checkpoint written by anything
    other than ``HaltProcess`` has no trigger state to record; ``HaltProcess``
    always passes its own.

    Refuses to write into a directory that already holds a checkpoint. Two
    halts sharing a directory would leave a manifest describing one step beside
    a restart file holding another, and nothing downstream would notice.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        raise FileExistsError(f"checkpoint already written at {manifest_path}")

    restart_dir = (checkpoint_dir / RESTART_DIRNAME).resolve()
    restart_dir.mkdir(parents=True, exist_ok=True)

    step = int(model_part.ProcessInfo[KM.STEP])
    time_s = float(model_part.ProcessInfo[KM.TIME])

    _save_restart_file(model_part, restart_dir)

    (checkpoint_dir / PARAMETERS_FILENAME).write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    (checkpoint_dir / LEDGERS_FILENAME).write_text(
        json.dumps(_jsonable_ledgers(ledgers), indent=2), encoding="utf-8"
    )

    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "time_s": time_s,
        "reason": reason,
        "detail": detail,
        "model_part_name": model_part.Name,
        "restart_load_file_label": str(step),
        "restart_dirname": RESTART_DIRNAME,
        "n_material_points": model_part.NumberOfElements(),
        "ledger_names": sorted(ledgers),
        # The halt-trigger state, so the re-arm windows survive the process
        # boundary. Without it every trip re-arms on every restart and the
        # halt-every-step loop returns through a different door.
        "trip_steps": {name: int(fired_at) for name, fired_at in trip_steps.items()},
    }
    # Manifest last: its presence is what marks the checkpoint complete.
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return checkpoint_dir


def _save_restart_file(model_part: KM.ModelPart, restart_dir: Path) -> None:
    """Write ``<model_part.Name>_<step>.rest`` into ``restart_dir``.

    ``RestartUtility`` resolves its output path against the *current working
    directory*, so ``input_output_path`` is passed absolute — a resumed solver
    launched from a different directory would otherwise write, or look, in the
    wrong place. ``restart_save_frequency`` is 0 because the schedule is
    ``quinney.halt``'s decision, not the utility's: 0 means save whenever asked.
    """
    settings = KM.Parameters(
        json.dumps(
            {
                "input_filename": model_part.Name,
                "input_output_path": restart_dir.as_posix(),
                "echo_level": 0,
                "serializer_trace": _SERIALIZER_TRACE,
                "restart_control_type": "step",
                "restart_save_frequency": 0.0,
                "save_restart_files_in_folder": True,
                "max_files_to_keep": -1,
            }
        )
    )
    RestartUtility(model_part, settings).SaveRestart()


def _jsonable_ledgers(ledgers: Mapping[str, PointLedger]) -> dict[str, dict[str, float]]:
    """Ledgers with their integer point ids rendered as JSON object keys."""
    return {
        name: {str(point_id): float(value) for point_id, value in ledger.items()}
        for name, ledger in ledgers.items()
    }


class HaltProcess(KM.Process):
    """Checks the halt triggers each step and stops the run when one fires.

    The check itself is ``quinney.halt.should_halt``, which knows nothing about
    Kratos; this process supplies it a step number and whatever diagnostics the
    caller wired in, then does the writing.

    ``ledgers`` and ``diagnostics`` are callables rather than mappings because
    both change every step — the caller passes ``lambda: damage.ledgers`` and
    the values are read at the moment of the halt. ``ledgers`` is required for
    the reason ``write_checkpoint`` states: forgetting it produces a valid
    checkpoint that silently means "every point undamaged". A run with nothing
    to carry passes ``dict``.

    ``dump_telemetry`` is the seam to ``quinney.telemetry``. This module does
    not own the episode format and does not import it; it hands over the
    checkpoint directory once the checkpoint is complete and before the halt is
    announced.

    ``resumed_from`` is the manifest of the checkpoint this run picked up, or
    ``None`` for a fresh run, and it is required for the same reason ``ledgers``
    is: forgetting it is silent and expensive. A resumed run built with
    ``None`` restarts its metronome from origin 0 and re-arms every trip, so a
    run resumed at step 20 with a 20-step interval halts at 21, resumes, halts
    at 22, forever. Reading it out of the manifest rather than taking a bare
    step number means the trigger state travels as one object and cannot be
    half-wired.
    """

    def __init__(
        self,
        model_part: KM.ModelPart,
        checkpoint_root: Path | str,
        *,
        parameters: Mapping[str, Any],
        interval_steps: int | None,
        ledgers: Callable[[], Mapping[str, PointLedger]],
        resumed_from: Mapping[str, Any] | None,
        trips: Sequence[DiagnosticTrip] = (),
        trip_rearm_steps: int = TRIP_REARM_STEPS,
        diagnostics: Callable[[], Mapping[str, float]] = dict,
        dump_telemetry: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__()
        self._model_part = model_part
        self._checkpoint_root = Path(checkpoint_root)
        self._parameters = parameters
        self._interval_steps = interval_steps
        self._trips = tuple(trips)
        # None means this run has not been halted yet. A resumed run takes both
        # windows from the manifest so neither the metronome nor a
        # still-standing trip fires on the first step after the restart.
        # "trip_steps" is absent from a schema 1.0 manifest, which predates
        # trips being usable; re-arming them is the safe reading of that.
        self._last_halt_step = None if resumed_from is None else int(resumed_from["step"])
        self._last_trip_steps: dict[str, int] = (
            {}
            if resumed_from is None
            else {
                str(name): int(fired_at)
                for name, fired_at in resumed_from.get("trip_steps", {}).items()
            }
        )
        self._trip_rearm_steps = trip_rearm_steps
        self._ledgers = ledgers
        self._diagnostics = diagnostics
        self._dump_telemetry = dump_telemetry

    def ExecuteFinalizeSolutionStep(self) -> None:
        step = int(self._model_part.ProcessInfo[KM.STEP])
        decision = should_halt(
            step,
            interval_steps=self._interval_steps,
            last_halt_step=self._last_halt_step,
            last_trip_steps=self._last_trip_steps,
            diagnostics=self._diagnostics(),
            trips=self._trips,
            trip_rearm_steps=self._trip_rearm_steps,
        )
        if not decision.halt:
            return

        checkpoint_dir = checkpoint_dir_for(self._checkpoint_root, step)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Only the trips that actually fired advance their window. A
        # metronome halt must not disarm a trip it had nothing to do with.
        for name in decision.fired_trips:
            self._last_trip_steps[name] = step

        write_checkpoint(
            self._model_part,
            checkpoint_dir,
            parameters=self._parameters,
            ledgers=self._ledgers(),
            trip_steps=self._last_trip_steps,
            reason=decision.reason,
            detail=decision.detail,
        )
        if self._dump_telemetry is not None:
            self._dump_telemetry(checkpoint_dir)

        self._last_halt_step = step
        raise SolverHalted(str(decision.reason), decision.detail, step, checkpoint_dir)
