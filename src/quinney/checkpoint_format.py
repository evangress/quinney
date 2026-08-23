"""The on-disk shape of a checkpoint, stated without a solver.

A checkpoint is written by a process that has Kratos and read by processes that
may not: the supervisor may be an API model, a local model, a rule baseline or
a human, and `replay.py` re-runs stored episodes on machines with no Kratos
wheel at all. The filenames and the schema version are therefore a *format
contract* rather than adapter detail, and they live here so
``quinney.kratos_adapter.restart`` can import them without dragging
``import KratosMultiphysics`` in behind them.

One checkpoint is one directory::

    <checkpoint_dir>/
      manifest.json     schema version, step, time_s, halt reason, restart
                        label, and the halt-trigger state the next run resumes
                        its re-arm windows from
      parameters.json   the ProjectParameters the run was using
      ledgers.json      per-point scalars keyed by point id
      restart/<model_part_name>_<step>.rest

The manifest is written last by ``write_checkpoint``, so its presence is what
marks a checkpoint complete: a directory holding a half-written ``.rest`` file
has no manifest and is not mistaken for a resumable one.
"""

from __future__ import annotations

from pathlib import Path

# major.minor, same discipline as the telemetry contract: an additive change
# bumps minor and still loads, a rename or a units change bumps major and needs
# a migration note.
#
# 1.1 added the manifest's "ledger_names" and "trip_steps". Both are additive,
# so a 1.0 checkpoint still loads; readers must treat them as absent rather
# than assume them, which for "trip_steps" means a 1.0 resume re-arms every
# trip. That is the safe reading: 1.0 predates trips being usable at all.
CHECKPOINT_SCHEMA_VERSION = "1.1"

# Separates major from minor in the version string above. Distinct from
# restart.OVERRIDE_PATH_SEPARATOR, which happens to be the same character while
# meaning something entirely unrelated.
VERSION_SEPARATOR = "."

MANIFEST_FILENAME = "manifest.json"
PARAMETERS_FILENAME = "parameters.json"
LEDGERS_FILENAME = "ledgers.json"
RESTART_DIRNAME = "restart"

# Zero-padded so a directory listing of a multi-halt run sorts in step order. A
# step wider than the padding still formats consistently, so this bounds nothing.
_CHECKPOINT_STEP_DIGITS = 6


def checkpoint_dir_for(root: Path | str, step: int) -> Path:
    """Directory one halt at ``step`` writes into, under ``root``."""
    return Path(root) / f"step_{step:0{_CHECKPOINT_STEP_DIGITS}d}"


def schema_major(version: str) -> int:
    """Leading integer of a ``major.minor`` checkpoint schema version."""
    try:
        return int(version.split(VERSION_SEPARATOR, 1)[0])
    except (AttributeError, ValueError) as error:
        raise ValueError(f"malformed checkpoint schema version {version!r}") from error
