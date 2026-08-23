"""The versioned on-disk contract between a solver that has exited and a supervisor.

Supervision here is halt-and-restart across a process boundary: the solver
checkpoints, dumps telemetry, and exits, and everything the supervisor knows it
reads back off disk. That makes this module the whole interface. Replay, the
rule baseline, and the model-ladder descent all re-read episodes that no live
solver is attached to any more, some of them written months earlier, so the
layout has to be stable and self-describing rather than convenient.

An episode is one directory::

    <episode_dir>/
      manifest.json                 schema version, case metadata, snapshot index
      series.npz                    homogeneous time series, one row per sampled step
      snapshots/step_000250.npz     ragged per-material-point state, one file per sample

Point state gets a file per sample because erasure changes the point count
between samples, so the samples cannot stack into one rectangular array, and
because a diagnostic usually wants one snapshot rather than the whole run.

Versioning
----------

``SCHEMA_VERSION`` is major.minor, and the promise is that a stored corpus stays
readable: episodes are expensive (a multi-minute solver run each, the early ones
carrying human expert judgement that cannot be regenerated), and the ladder
measurement re-runs that corpus for years. So the contract has to grow without
invalidating what is already written, in *both* directions:

- **A newer episode read by an older reader.** Fields the reader does not know
  about are ignored. Minor bumps are additive by definition, so nothing the
  older reader needs has moved.
- **An older episode read by a newer reader.** Each field records the version it
  was introduced in (see ``INTRODUCED_IN``); a field newer than the episode's declared
  version is not required, and float columns are filled with NaN. NaN means
  *not recorded in this schema version* — never "measured as zero". It is the
  right sentinel because numpy propagates it: a diagnostic computed over a
  column that was never written comes out NaN and says so, where zero would
  quietly produce a plausible wrong number. Integer columns have no NaN, so a
  later-introduced integer column makes the older episode unreadable and says
  so by name; introduce integer columns only when a major bump is acceptable,
  or give them a float twin.

A major bump means a field was renamed, removed, or had its meaning or units
changed. Loading then refuses rather than reinterpreting old numbers under new
rules.

A later reader is written by extending these records — that is what the
versioning above is for, and what this module's tests demonstrate. Such a
module needs ``from __future__ import annotations`` of its own, for the same
reason this one has it, and every field it adds must be annotated in one of the
spellings this module recognises. Anything else is refused by name rather than
quietly demoted to an unvalidated scalar.

Every dataclass checks its own column-length invariant on construction. A
truncated column is the failure that matters most: it does not crash a
diagnostic, it feeds one a plausible number computed over a point set that
never existed.

Deliberately absent: no Kratos import, so a machine with no solver installed
can still replay; no derived quantities, which belong to the diagnostics that
read this; and no sampling policy — when to write a snapshot is the adapter's
decision, not the format's.
"""

# Keep this import. Every field of every record below is classified by its
# annotation *as text* (see _COLUMN_DTYPES): drop it and Python 3.14 hands
# dataclasses evaluated objects instead, and no column is recognised as one.
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import Field, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# major.minor. Major mismatch on load is an error; minor mismatch loads.
#
# 1.1  adds CaseMetadata.tool_velocity_m_s and Snapshot.initiation. A 1.0
#      episode still loads with both NaN-filled, which is the point: it recorded
#      no travel direction and no initiation sum, so geometry derived from the
#      moving tool and any judgement about points that have not yet initiated
#      come out NaN rather than confidently wrong.
# 1.0  first published contract.
SCHEMA_VERSION = "1.1"

MANIFEST_FILENAME = "manifest.json"
SERIES_FILENAME = "series.npz"
SNAPSHOT_DIRNAME = "snapshots"

# Snapshot filenames are step_<n>.npz, zero-padded so a directory listing sorts
# in step order. A step wider than the padding still formats and parses
# consistently, so this bounds nothing.
_SNAPSHOT_STEM_PREFIX = "step_"
_SNAPSHOT_STEP_DIGITS = 6

# Series and Deletion share column names (step, coords_m) and share one .npz,
# so the deletion columns are namespaced.
_DELETION_PREFIX = "deletion_"

# The cut is 2D plane strain, so a position or velocity is a pair.
_PLANE_COMPONENTS = 2

# x0, x1, y0, y1.
_BOUNDS_COMPONENTS = 4

# Keys for the per-field metadata that records when a column appeared and what
# shape it has. They are set at the field with `field(metadata=...)`, so adding
# a column and recording the version it arrived in is one edit in one place,
# not two lists that can drift apart.
INTRODUCED_IN = "introduced_in"
PLANAR = "planar"

# Ready-made metadata for the shape every 2D position and velocity has, so the
# declarations below stay readable. Public because a later version of the
# contract is written by extending these records, and its planar columns should
# not have to spell the metadata dict out by hand.
PLANAR_COLUMN: Mapping[str, Any] = {PLANAR: True}

# The version a field is assumed to date from when it does not say. Separate
# from SCHEMA_VERSION, which moves; this one names the first published contract
# and stays put.
_INITIAL_VERSION = "1.0"

# The contract uses exactly two column dtypes, and the field's own annotation is
# the single declaration of which. Deriving the dtype from it is what stops a
# second list of integer columns from silently disagreeing with the type hints.
_COLUMN_DTYPES: Mapping[str, type] = {
    "NDArray[np.int64]": np.int64,
    "NDArray[np.float64]": np.float64,
}


# The non-column annotations the contract uses. Together with _COLUMN_DTYPES
# this is the *closed* set of ways a field may be declared. An annotation in
# neither set is a typo or an unsupported spelling, never a new kind of field:
# it would be written and read back while escaping length, dtype, and shape
# validation entirely, which is the one failure this contract exists to stop.
# The fixed-arity float tuples the contract uses, and how many floats each
# holds. The arity is already written into the annotation; reading it from there
# keeps the count in one place and lets a tuple field that predates a reader be
# filled to the right length.
_TUPLE_ARITIES: Mapping[str, int] = {
    "tuple[float, float]": _PLANE_COMPONENTS,
    "tuple[float, float, float, float]": _BOUNDS_COMPONENTS,
}

_SCALAR_TYPES: frozenset[str] = frozenset({"int", "float", "str"}) | frozenset(_TUPLE_ARITIES)


def _version_key(version: str) -> tuple[int, int]:
    """(major, minor) of a version string, for comparison and gating."""
    major, _, minor = str(version).partition(".")
    try:
        return int(major), int(minor)
    except ValueError as error:
        raise ValueError(f"malformed schema version {version!r}; expected 'major.minor'") from error


def _introduced_in(spec: Field[Any]) -> str:
    return str(spec.metadata.get(INTRODUCED_IN, _INITIAL_VERSION))


def _column_specs(record: Any) -> list[Field[Any]]:
    """The array-valued fields of a record, in declaration order.

    The first one is the reference column: it defines the row count every other
    column is checked against, which is why ids and step counters are declared
    first on every record here.
    """
    return [spec for spec in fields(record) if spec.type in _COLUMN_DTYPES]


def _scalar_specs(record: Any) -> list[Field[Any]]:
    return [spec for spec in fields(record) if spec.type not in _COLUMN_DTYPES]


def _check_declaration(spec: Field[Any], owner: str) -> None:
    """Reject a field the contract cannot classify, and a malformed version on one.

    Classification is by annotation text, so an alias — ``npt.NDArray[np.float64]``,
    ``NDArray[np.floating]``, a quoted typedef — is not a column as far as this
    module is concerned. Such a field would still be written and read back, but
    silently outside every check, which is exactly the truncated column that
    reaches a diagnostic and produces a plausible number. Refusing by name is
    the only safe reading of an annotation nobody here understands.
    """
    if spec.type not in _COLUMN_DTYPES and spec.type not in _SCALAR_TYPES:
        recognised = ", ".join(sorted(_COLUMN_DTYPES) + sorted(_SCALAR_TYPES))
        raise TypeError(
            f"{owner}.{spec.name} is annotated {spec.type!r}, which this contract does not "
            f"recognise. A field must be spelled exactly one of: {recognised}. An alias would "
            "be stored and read back while escaping length, dtype, and shape validation."
        )
    _version_key(_introduced_in(spec))


def _normalize_columns(instance: object) -> None:
    """Put a record's columns in the contract's dtype and check their shapes.

    Coercion here normalizes *representation*, so every episode stores one
    canonical dtype per column and a caller may hand over a list. It does not
    recover precision: a writer that computed in float32 has already discarded
    mantissa bits, and widening cannot put them back. Feeding this module
    float64 is the writer's job; this only guarantees the file does not vary.

    Raises ``ValueError`` naming the offending field for a scalar where a
    column belongs, a planar column that is not (n, 2), or any column whose
    length disagrees with the reference column.
    """
    owner = type(instance).__name__
    for spec in fields(instance):
        _check_declaration(spec, owner)
    specs = _column_specs(instance)

    for spec in specs:
        value = np.asarray(getattr(instance, spec.name), dtype=_COLUMN_DTYPES[spec.type])
        if value.ndim == 0:
            raise ValueError(
                f"{owner}.{spec.name} must be an array of per-row values, got the scalar {value}"
            )
        if spec.metadata.get(PLANAR, False) and value.shape[1:] != (_PLANE_COMPONENTS,):
            raise ValueError(
                f"{owner}.{spec.name} must have shape (n, {_PLANE_COMPONENTS}), got {value.shape}"
            )
        object.__setattr__(instance, spec.name, value)

    reference = specs[0]
    expected = len(getattr(instance, reference.name))
    for spec in specs[1:]:
        length = len(getattr(instance, spec.name))
        if length != expected:
            raise ValueError(
                f"{owner} columns disagree in length: "
                f"{reference.name} has {expected} rows, {spec.name} has {length}"
            )


@dataclass(frozen=True)
class CaseMetadata:
    """The fixed description of one cutting case.

    Every number a diagnostic needs to turn raw point state into a judgement:
    ``cell_size_m`` is the yardstick a shear band's width is measured in,
    ``depth_of_cut_m`` the one chip thickness is measured against, and the tool
    geometry is what chip and workpiece are separated by.

    ``time_step_s`` is the case's *nominal* step — what the run was configured
    with. The step actually taken varies as the CFL condition tightens, and
    that history is ``Series.time_step_s``, not this.

    ``cutting_speed_m_s`` and ``tool_velocity_m_s`` are the same quantity twice.
    The scalar is the magnitude, kept because the machining literature and the
    Merchant relations are written in terms of it. The vector adds the one thing
    the scalar cannot carry — which way the tool goes — and a consumer that
    needs geometry *over time* wants the vector: the tool moves, so the tip at a
    snapshot is ``tool_tip_m + tool_velocity_m_s * snapshot.time_s``, not
    ``tool_tip_m``. Measuring chip thickness or contact length against the t=0
    tip after the tool has travelled reads the wrong geometry by however far it
    went, and reads it plausibly.

    ``initial_mass_kg`` is recorded rather than derived because the obvious
    substitute is wrong: summing ``Snapshot.mass_kg`` over the first *sampled*
    step gives a baseline that may already be missing points, since nothing
    guarantees the first sample precedes the first separation. A deleted-mass
    fraction measured against a baseline that has itself lost mass understates
    the loss, and understating it is the direction that hides a failing run.
    """

    case_name: str
    material: str
    formulation: str
    cutting_speed_m_s: float  # |tool_velocity_m_s|; the scalar Merchant is written in
    tool_velocity_m_s: tuple[float, float] = field(  # (vx, vy); carries the direction
        metadata={INTRODUCED_IN: "1.1"}
    )
    rake_angle_deg: float
    depth_of_cut_m: float  # t0, the uncut chip thickness
    cell_size_m: float  # background grid cell; the mesh-resolution yardstick
    time_step_s: float  # nominal; see Series.time_step_s for what was used
    workpiece_bounds_m: tuple[float, float, float, float]  # x0, x1, y0, y1
    tool_tip_m: tuple[float, float]
    initial_mass_kg: float  # total workpiece mass at t=0; the denominator of deleted mass fraction
    friction_coefficient: float  # Coulomb mu, dimensionless
    friction_factor: float  # Zorev m, 0..1
    deletion_threshold: float  # accumulated damage at which a point separates

    def __post_init__(self) -> None:
        # Arity comes from each field's own annotation, so a tuple field added
        # later is checked without anyone remembering to add it here.
        owner = type(self).__name__
        for spec in fields(self):
            _check_declaration(spec, owner)
            arity = _TUPLE_ARITIES.get(spec.type)
            if arity is None:
                continue
            given = len(getattr(self, spec.name))
            if given != arity:
                raise ValueError(f"{owner}.{spec.name} needs {arity} values, got {given}")


@dataclass(frozen=True)
class Snapshot:
    """Per-material-point state at one sampled step. All columns share length n_points.

    The two in-plane normal stress components ride alongside
    ``equivalent_stress_pa`` because the offline damage path needs them:
    ``materials.johnson_cook.triaxiality_from_plane_strain_stress`` takes
    (sxx, syy, sigma_eq), and nothing recovers the pair from the equivalent
    stress alone. Without them a supervisor could not preview what moving
    ``deletion_threshold`` would have done — which is the whole reason the
    damage law is a pure, solver-independent module.

    Failure is two-phase, so it takes two columns. ``initiation`` is the
    Johnson-Cook initiation sum omega, 0..1, reaching 1.0 when failure *begins*;
    ``damage`` is the post-initiation degradation D, 0..1, reaching 1.0 at
    separation, and it is meaningful only once omega has got there. Both are on
    disk because the supervisor's central intervention is moving
    ``deletion_threshold``, and D alone cannot answer what that move would do:
    it says nothing about the points that have not initiated yet. A reader
    seeing ``damage == 0.0`` needs ``initiation`` to tell a pristine point from
    one sitting at omega = 0.98, and re-integrating omega from the coarsely
    sampled ``eps_p`` history offline would give a different answer than the run
    did — the exact disagreement the pure damage law exists to rule out.

    There is no szz column because MPM never reports one. In plane strain
    ``MP_CAUCHY_STRESS_VECTOR`` carries only (sxx, syy, sxy), and
    ``MP_PRESSURE`` is registered but not implemented, so the damage law
    reconstructs the mean stress as (sxx + syy)/2 — exact at the plastic limit,
    where szz is the intermediate principal stress. See that function's
    docstring for why the elastic-range discrepancy never reaches the damage
    sum. The shear component sxy is absent for a different reason: no consumer
    reads it yet, and this contract grows a field when there is a reader.
    """

    step: int
    time_s: float
    point_ids: NDArray[np.int64]
    coords_m: NDArray[np.float64] = field(metadata=PLANAR_COLUMN)
    velocity_m_s: NDArray[np.float64] = field(metadata=PLANAR_COLUMN)
    eps_p: NDArray[np.float64]  # equivalent plastic strain, dimensionless
    eps_rate_per_s: NDArray[np.float64]  # equivalent plastic strain rate
    temperature_k: NDArray[np.float64]
    initiation: NDArray[np.float64] = field(  # omega, 0..1; 1.0 = failure has begun
        metadata={INTRODUCED_IN: "1.1"}
    )
    damage: NDArray[np.float64]  # post-initiation D, 0..1; 1.0 = separated. Only after omega = 1
    mass_kg: NDArray[np.float64]
    equivalent_stress_pa: NDArray[np.float64]  # von Mises
    cauchy_xx_pa: NDArray[np.float64]
    cauchy_yy_pa: NDArray[np.float64]

    def __post_init__(self) -> None:
        _normalize_columns(self)


@dataclass(frozen=True)
class Series:
    """One row per recorded step. All columns share length n_steps.

    "Recorded" is not "sampled": the adapter writes a row every solution step,
    because the force signal an FFT reads and the dt trend a collapse is read
    from are meaningless on a coarse grid, while ``Snapshot`` files are written
    on a much longer interval because point state is bulky. So ``step`` here is
    dense and ``Episode.snapshot_steps`` is sparse, and the two are not
    expected to line up.

    ``time_step_s`` is the dt *actually taken* at each sampled step, which is
    what a dt-collapse trend is read from. The case's nominal dt is
    ``CaseMetadata.time_step_s``; the two share a name and are not the same
    quantity.

    ``external_work_j`` is the cumulative work done **on the workpiece by the
    driven tool**, in joules, positive when the tool does work on the
    workpiece — which is the normal state of a tool advancing into material
    that resists it. Negative would mean the workpiece is doing work on the
    tool, as an elastic rebound briefly can. It is cumulative since t=0, like
    ``deleted_mass_kg`` and unlike everything else here.

    Its provenance differs from the two energy columns beside it and that
    matters when reading them together: MPM reports kinetic and internal
    energy per material point, but exposes no work accumulator of any kind, so
    this column is *integrated* at write time from the recorded tool force and
    the prescribed tool velocity rather than read from the solver. See
    ``kratos_adapter.telemetry_writer`` for the integral and its sign.
    """

    step: NDArray[np.int64]
    time_s: NDArray[np.float64]
    cutting_force_n: NDArray[np.float64]  # Fx on the tool
    thrust_force_n: NDArray[np.float64]  # Fy on the tool
    time_step_s: NDArray[np.float64]  # dt actually used at this step, not the nominal one
    kinetic_energy_j: NDArray[np.float64]
    internal_energy_j: NDArray[np.float64]
    external_work_j: NDArray[np.float64]  # cumulative; +ve = tool works on workpiece
    n_points: NDArray[np.int64]  # material points still live at this step
    deleted_mass_kg: NDArray[np.float64]  # cumulative since t=0, not per step

    def __post_init__(self) -> None:
        _normalize_columns(self)


@dataclass(frozen=True)
class Deletion:
    """One record per separated material point — the deletion-health input.

    Where and when points leave is the difference between separation along the
    cut path and mass evaporating out of the bulk, so position and step are
    kept per record rather than counted.
    """

    step: NDArray[np.int64]
    point_id: NDArray[np.int64]
    coords_m: NDArray[np.float64] = field(metadata=PLANAR_COLUMN)  # position at separation
    temperature_k: NDArray[np.float64]
    mass_kg: NDArray[np.float64]

    def __post_init__(self) -> None:
        _normalize_columns(self)


@dataclass(frozen=True)
class Episode:
    """Everything an episode holds except the snapshots, which load on demand.

    ``schema_version`` is the version the episode was *written* under, not this
    module's, so a reader can tell what it is looking at. Nothing marks the
    individual columns that were NaN-filled for predating the reader; a caller
    that needs to know works it out from this version and the field's own
    ``INTRODUCED_IN``, or tests the values for NaN.
    """

    schema_version: str
    metadata: CaseMetadata
    series: Series
    deletions: Deletion
    snapshot_steps: tuple[int, ...]


def _snapshot_path(episode_dir: Path, step: int) -> Path:
    stem = f"{_SNAPSHOT_STEM_PREFIX}{step:0{_SNAPSHOT_STEP_DIGITS}d}"
    return episode_dir / SNAPSHOT_DIRNAME / f"{stem}.npz"


# Prefix for the half-written copy of a file that is rewritten in place. A
# prefix rather than a suffix because np.savez insists on a .npz extension and
# would append a second one.
_PARTIAL_PREFIX = "partial-"


def _replace_atomically(temporary: Path, path: Path) -> None:
    """Swap a fully-written temporary file into place in one step.

    Both files that use this are rewritten repeatedly while a run is in flight
    — the manifest on every sample, the series on every flush — and truncating
    the live file to rewrite it leaves a window in which a solver killed by the
    very failure the supervisor was meant to diagnose leaves an unparseable
    episode behind, with every snapshot beside it intact and unreadable.
    ``os.replace`` is atomic on NTFS and on POSIX, so a reader sees either the
    whole previous file or the whole new one.
    """
    os.replace(temporary, path)


def write_manifest(
    episode_dir: Path,
    metadata: CaseMetadata,
    snapshot_steps: Sequence[int],
) -> None:
    """Write the manifest: schema version, case metadata, and the snapshot index.

    The manifest is the authoritative index. ``append_snapshot`` writes files
    and does not touch it, so a writer that samples as it goes rewrites the
    manifest with the full index at the halt point — and ``final_snapshot``
    refuses to answer if that never happened.
    """
    episode_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "metadata": asdict(metadata),
        "snapshot_steps": [int(step) for step in snapshot_steps],
    }
    path = episode_dir / MANIFEST_FILENAME
    temporary = path.with_name(_PARTIAL_PREFIX + path.name)
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _replace_atomically(temporary, path)


def append_snapshot(episode_dir: Path, snapshot: Snapshot) -> None:
    """Write one sampled step's point state to its own file, named by step."""
    path = _snapshot_path(episode_dir, snapshot.step)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {spec.name: getattr(snapshot, spec.name) for spec in _column_specs(Snapshot)}
    stored.update(
        {spec.name: np.asarray(getattr(snapshot, spec.name)) for spec in _scalar_specs(Snapshot)}
    )
    np.savez(path, **stored)


def write_series(episode_dir: Path, series: Series, deletions: Deletion) -> None:
    """Write the per-step series and the deletion records into one file.

    They share a file because they are written together at the same moment and
    are always read together; the deletion columns carry a prefix because the
    two records share column names.
    """
    episode_dir.mkdir(parents=True, exist_ok=True)
    stored: dict[str, Any] = {spec.name: getattr(series, spec.name) for spec in fields(Series)}
    stored.update(
        {_DELETION_PREFIX + spec.name: getattr(deletions, spec.name) for spec in fields(Deletion)}
    )
    path = episode_dir / SERIES_FILENAME
    temporary = path.with_name(_PARTIAL_PREFIX + path.name)
    np.savez(temporary, **stored)
    _replace_atomically(temporary, path)


def _not_recorded(spec: Field[Any], rows: int, owner: str, version: str) -> Any:
    """The stand-in for a field this episode predates.

    NaN, for anything that can hold it, so a diagnostic reading a column that
    was never written produces NaN rather than a plausible number built on a
    zero that was never measured.
    """
    dtype = _COLUMN_DTYPES.get(spec.type)
    if dtype is np.float64:
        shape = (rows, _PLANE_COMPONENTS) if spec.metadata.get(PLANAR, False) else (rows,)
        return np.full(shape, np.nan)
    if spec.type == "float":
        return float("nan")
    arity = _TUPLE_ARITIES.get(spec.type)
    if arity is not None:
        return (float("nan"),) * arity
    raise ValueError(
        f"{owner}.{spec.name} was introduced in schema version {_introduced_in(spec)} and this "
        f"episode declares {version}, but a {spec.type} field has no missing value to stand in "
        "for it, so the episode cannot be read under the newer contract"
    )


def _read_record(
    source: Mapping[str, Any],
    record: Any,
    version: str,
    where: str,
    prefix: str = "",
) -> dict[str, Any]:
    """Field values for ``record``, resolved against the version the episode declares.

    Entries of ``source`` that the record does not declare are dropped, which
    is what lets an older reader open a newer episode. Fields introduced after
    ``version`` are filled rather than demanded, which is what lets a newer
    reader open an older one. Anything else missing raises, named.
    """
    owner = record.__name__
    values: dict[str, Any] = {}
    absent: list[Field[Any]] = []
    for spec in fields(record):
        stored_name = prefix + spec.name
        if stored_name in source:
            values[spec.name] = source[stored_name]
        else:
            absent.append(spec)

    episode_version = _version_key(version)
    required = [
        prefix + spec.name
        for spec in absent
        if _version_key(_introduced_in(spec)) <= episode_version
    ]
    if required:
        raise ValueError(f"{where} is missing required fields: {', '.join(required)}")

    if absent:
        # Not simply the first declared column: that one may itself be the
        # newly-introduced field being filled, and reaching for it would turn a
        # loadable episode into a bare KeyError.
        present = [spec for spec in _column_specs(record) if spec.name in values]
        rows = len(values[present[0].name]) if present else 0
        for spec in absent:
            values[spec.name] = _not_recorded(spec, rows, owner, version)
    return values


def _metadata_from(raw: Mapping[str, Any], version: str, where: str) -> CaseMetadata:
    values = _read_record(raw, CaseMetadata, version, where)
    # JSON has no tuple, so every tuple-typed field comes back as a list.
    for spec in fields(CaseMetadata):
        if spec.type in _TUPLE_ARITIES:
            values[spec.name] = tuple(values[spec.name])
    return CaseMetadata(**values)


def _read_manifest(episode_dir: Path) -> dict[str, Any]:
    """Parse the manifest and refuse an episode this reader cannot interpret."""
    path = episode_dir / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    missing = [
        key for key in ("schema_version", "metadata", "snapshot_steps") if key not in manifest
    ]
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
    version = str(manifest["schema_version"])
    if _version_key(version)[0] != _version_key(SCHEMA_VERSION)[0]:
        raise ValueError(
            f"episode at {episode_dir} has schema version {version}, which this reader "
            f"({SCHEMA_VERSION}) cannot load: major versions differ, so a field was renamed, "
            "removed, or redefined"
        )
    return manifest


def load_episode(episode_dir: Path) -> Episode:
    """Load an episode's manifest, series, and deletion records.

    Snapshots are not read here — they are the bulk of an episode and a
    diagnostic usually wants one. Raises ``ValueError`` on a major schema
    version mismatch, naming both versions.
    """
    manifest = _read_manifest(episode_dir)
    version = str(manifest["schema_version"])
    series_path = episode_dir / SERIES_FILENAME
    with np.load(series_path) as stored:
        series = Series(**_read_record(stored, Series, version, str(series_path)))
        deletions = Deletion(
            **_read_record(stored, Deletion, version, str(series_path), prefix=_DELETION_PREFIX)
        )
    return Episode(
        schema_version=version,
        metadata=_metadata_from(
            manifest["metadata"], version, f"{episode_dir / MANIFEST_FILENAME} metadata"
        ),
        series=series,
        deletions=deletions,
        snapshot_steps=tuple(int(step) for step in manifest["snapshot_steps"]),
    )


def load_snapshot(episode_dir: Path, step: int) -> Snapshot:
    """Load the point state sampled at one step.

    Goes through the manifest rather than straight to the file: this is the
    primary read path for a diagnostic that wants one snapshot and never calls
    ``load_episode``, and it would otherwise be the one path with no schema
    check on it at all.

    Raises ``FileNotFoundError`` naming the step if that step was never
    sampled — a supervisor asking for a step it invented should hear so rather
    than get the nearest one.
    """
    version = str(_read_manifest(episode_dir)["schema_version"])
    path = _snapshot_path(episode_dir, step)
    if not path.is_file():
        raise FileNotFoundError(f"episode at {episode_dir} has no snapshot for step {step}")
    with np.load(path) as stored:
        values = _read_record(stored, Snapshot, version, str(path))
    values["step"] = int(values["step"])
    values["time_s"] = float(values["time_s"])
    return Snapshot(**values)


def _steps_on_disk(episode_dir: Path) -> tuple[int, ...]:
    """Steps that actually have a snapshot file, read from the filenames."""
    directory = episode_dir / SNAPSHOT_DIRNAME
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            int(path.stem.removeprefix(_SNAPSHOT_STEM_PREFIX))
            for path in directory.glob(f"{_SNAPSHOT_STEM_PREFIX}*.npz")
        )
    )


def final_snapshot(episode_dir: Path) -> Snapshot:
    """Load the latest sampled step — the state the run halted or ended in.

    The manifest stays authoritative, but a manifest that disagrees with the
    snapshots on disk is refused rather than answered from. The case that
    matters is the one this function exists for: the solver dies of the mesh
    collapse the supervisor was meant to diagnose, before the halt path
    rewrites the index, and the last indexed step is hundreds of steps before
    the failure. Returning it silently would hand the supervisor the wrong
    state to reason about.
    """
    indexed = tuple(int(step) for step in _read_manifest(episode_dir)["snapshot_steps"])
    on_disk = _steps_on_disk(episode_dir)
    if not indexed and not on_disk:
        raise ValueError(f"episode at {episode_dir} has no snapshots")

    latest_indexed = max(indexed, default=None)
    latest_on_disk = max(on_disk, default=None)
    if latest_indexed != latest_on_disk:
        raise ValueError(
            f"episode at {episode_dir} disagrees with itself about its latest snapshot: the "
            f"manifest indexes step {latest_indexed} but snapshots/ holds step {latest_on_disk}. "
            "The manifest was not rewritten after the last sample, which is what an unclean "
            "solver exit looks like, so neither step can be trusted as the state the run ended in"
        )
    return load_snapshot(episode_dir, latest_indexed)


def _check_declarations() -> None:
    """Refuse an unclassifiable field or a malformed version at import time.

    A typo like ``INTRODUCED_IN: "1,1"`` would otherwise surface years later,
    the first time somebody loads an episode old enough to exercise it. Episode
    is not checked: it is a container of records, not a record of columns.
    """
    for record in (CaseMetadata, Snapshot, Series, Deletion):
        for spec in fields(record):
            _check_declaration(spec, record.__name__)


_check_declarations()
