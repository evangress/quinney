"""Turning a checkpoint plus a supervisor's decision into the next run.

Two transforms, both over plain dicts. ``resume_from`` reads a checkpoint
directory and returns the ProjectParameters that will pick that checkpoint back
up; ``inject_parameters`` applies the supervisor's overrides on top. The split
matters: resuming is mechanical and always the same, changing the plan is the
decision under study, and keeping them apart is what lets a stored episode be
replayed against a different supervisor with the resume held fixed.

This module lives under ``kratos_adapter`` because it knows the solver's
parameter layout, but it deliberately does **not** import Kratos, directly or
transitively — the filenames and schema version it needs come from
``quinney.checkpoint_format`` rather than from ``checkpoint``, which does import
Kratos. The process that decides what to change may be an API model, a local
model, a rule baseline or a human, and none of them should need a solver
installed to build the parameters for one. ``test_restart.py`` asserts the
import stays clean, because nothing else would notice it breaking.

What restarting actually covers is recorded in docs/kratos_api_notes.md: the
``.rest`` file carries the material points bit-for-bit, the background grid is
re-read from its mesh every time, and the damage ledgers come back from the
JSON sidecar because Kratos has no damage variable to have saved them in.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quinney.checkpoint_format import (
    CHECKPOINT_SCHEMA_VERSION,
    LEDGERS_FILENAME,
    MANIFEST_FILENAME,
    PARAMETERS_FILENAME,
    schema_major,
)

# Separates the keys of a dotted override path. Chosen because ProjectParameters
# keys are JSON identifiers and never contain a dot themselves. It is the same
# character as ``checkpoint_format.VERSION_SEPARATOR`` and has nothing to do
# with it.
OVERRIDE_PATH_SEPARATOR = "."


def read_manifest(checkpoint_dir: Path | str) -> dict[str, Any]:
    """The checkpoint's manifest, refusing an incompatible schema major version.

    A minor bump is additive and loads: unknown keys are simply carried. A major
    bump means a field was renamed or had its meaning or units changed, and
    reading old numbers under new rules is worse than refusing to resume.
    """
    path = Path(checkpoint_dir) / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint manifest at {path}")

    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    version = manifest.get("schema_version", "")
    if schema_major(version) != schema_major(CHECKPOINT_SCHEMA_VERSION):
        raise ValueError(
            f"checkpoint schema {version} is incompatible with {CHECKPOINT_SCHEMA_VERSION}"
        )
    return manifest


def restore_ledgers(checkpoint_dir: Path | str) -> dict[str, dict[int, float]]:
    """Per-point ledgers from the sidecar, keyed by integer point id again.

    JSON object keys are strings, and a ledger that came back keyed by strings
    would answer every ``ledger.get(point_id, 0.0)`` with the default — a
    resume that silently restarted every point undamaged.
    """
    path = Path(checkpoint_dir) / LEDGERS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"no ledger sidecar at {path}")

    raw: Mapping[str, Mapping[str, float]] = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: {int(point_id): float(value) for point_id, value in ledger.items()}
        for name, ledger in raw.items()
    }


def resume_from(checkpoint_dir: Path | str) -> dict[str, Any]:
    """ProjectParameters that pick the checkpointed run back up where it stopped.

    Only ``model_import_settings`` changes. The background grid keeps its mesh
    import: MPM rebuilds the grid from the material points every step, carries
    no state on it, and the solver rejects ``"rest"`` for the grid outright.

    Start time is not set either — on a restart the analysis stage takes TIME
    from the restored ProcessInfo, and writing a start time here would be a
    second, silently disagreeing source for it.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest = read_manifest(checkpoint_dir)

    parameters_path = checkpoint_dir / PARAMETERS_FILENAME
    if not parameters_path.exists():
        raise FileNotFoundError(f"no stored parameters at {parameters_path}")
    parameters: dict[str, Any] = json.loads(parameters_path.read_text(encoding="utf-8"))

    restart_dir = (checkpoint_dir / manifest["restart_dirname"]).resolve()
    parameters["solver_settings"]["model_import_settings"] = {
        "input_type": "rest",
        # The MPM solver overwrites this with the model part name on load; it is
        # written correctly anyway so the file on disk is self-explanatory.
        "input_filename": manifest["model_part_name"],
        "input_output_path": restart_dir.as_posix(),
        "restart_load_file_label": manifest["restart_load_file_label"],
    }
    return parameters


def inject_parameters(
    parameters: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy of ``parameters`` with each dotted-path override applied.

    ``{"solver_settings.time_stepping.time_step": 1e-8}`` sets that one leaf and
    touches nothing else. The input is never mutated, so the parameters the
    episode recorded stay exactly as they were recorded.

    A path that does not already exist raises rather than being created. The
    supervisor's authority is bounded by ``envelope/validate.py``, which clamps
    the settings it knows about; a misspelled path that quietly added a new key
    would slip past the clamp and then be ignored by the solver, which looks
    from the outside like a decision that was accepted and had no effect.
    """
    changed = copy.deepcopy(dict(parameters))

    for path, value in overrides.items():
        keys = path.split(OVERRIDE_PATH_SEPARATOR)
        if not path or "" in keys:
            raise ValueError(f"override path {path!r} has an empty key")

        target: Any = changed
        for key in keys[:-1]:
            if not isinstance(target, dict) or key not in target:
                raise KeyError(f"override path {path!r} does not exist: no {key!r}")
            target = target[key]

        leaf = keys[-1]
        if not isinstance(target, dict) or leaf not in target:
            raise KeyError(f"override path {path!r} does not exist: no {leaf!r}")
        target[leaf] = value

    return changed
