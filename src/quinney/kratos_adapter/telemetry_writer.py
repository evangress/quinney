"""Material point state, out of the solver and onto disk as the versioned contract.

This is the only module that crosses the seam the whole architecture rests on:
the supervisor never reaches into live solver state, it reads files. Everything
downstream — the diagnostics, `replay.py`, the model ladder — sees what is
written here and nothing else, so a field this module drops is a field no
supervisor can ever ask about, and a field it fabricates is one that will be
believed.

Four consequences shape the code.

*The manifest is the index, and nothing else maintains it.*
``telemetry.append_snapshot`` deliberately does not touch the manifest, so this
writer rewrites it — and the series beside it — every time it appends a
snapshot, not only at the end. The run this project exists to supervise is one
that may die of the failure it is being watched for, and an episode whose force
history, energy history and deletion records all live in memory until a clean
shutdown is an episode that teaches nothing about the crash. What survives a
kill is therefore everything up to the last sample, not just the snapshot
files.

*Deletion records have to be taken while the point still exists.* A point that
reaches the separation threshold is flagged ``TO_ERASE`` by
``JohnsonCookDamageProcess`` and removed by the solver's own element search on
the next step. Its position, temperature and mass are readable only in the
window between those two events.

*External work has to be integrated here, or it does not exist.* MPM reports
kinetic and internal energy per material point but has no work accumulator of
any kind, and the energy balance an energy-drift diagnostic reads is
meaningless without the work that entered the body. So this writer integrates
it from the force on the driven tool and the velocity that tool is driven at,
cumulatively. At write time rather than in the diagnostics because a stored
episode has to be self-sufficient: replay and the model ladder read the episode
and nothing else, months later, with no solver and no case script to ask.

*The ordering it needs is checked, not documented and hoped for.* Run this
process once per solution step, in ``FinalizeSolutionStep``, **after**
``RigidToolProcess`` and after ``JohnsonCookDamageProcess``. A skipped step
would lose the deletions that happened in it, a stale tool force would pair one
step's impulse with another step's dt, and neither would leave a mark in the
output. Both are refused at the point they happen instead: the step counter
must advance by exactly one, and the tool must have measured a new force since
the last call.

Deliberately absent: any diagnostic. This module measures and writes; deciding
what a number means belongs to `quinney.diagnostics`, which must be able to
reach the same conclusion months later from the files alone.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication as MPM
import numpy as np
from numpy.typing import NDArray

from quinney.kratos_adapter.damage_process import JohnsonCookDamageProcess
from quinney.kratos_adapter.rigid_tool import RigidToolProcess
from quinney.point_ledger import PointLedger, values_for
from quinney.telemetry import (
    CaseMetadata,
    Deletion,
    Series,
    Snapshot,
    append_snapshot,
    write_manifest,
    write_series,
)

# In-plane components of MP_CAUCHY_STRESS_VECTOR, which is (sxx, syy, sxy) in
# plane strain. There is no szz to read; see Snapshot's docstring.
_CAUCHY_XX = 0
_CAUCHY_YY = 1

# CaseMetadata carries the cutting speed as a scalar and the tool velocity as a
# vector, and they are the same quantity twice. Only float noise is tolerable
# between them; a real disagreement means the metadata describes a different
# run from the one the tool is driving.
_SPEED_AGREEMENT_REL_TOL = 1e-9

# A caller that computes the workpiece mass from geometry and density differs
# from the generated point set only by how the body was discretised, which is
# well under a percent. More than that means the points are not the body the
# case describes — botched generation, or the wrong model part.
_MASS_AGREEMENT_REL_TOL = 1e-2


@dataclass(frozen=True)
class MaterialPointFields:
    """The raw per-point read behind one snapshot, before damage is attached.

    Separate from `damage_process.MaterialPointState` on purpose: that one
    carries the volume the fracture-energy regularization needs and skips the
    kinematics, this one carries the kinematics telemetry needs and skips the
    volume. Merging them would make the damage process read coordinates and
    velocities it never looks at, every step.

    ``ids`` is what everything else is keyed by. Position in these arrays is
    meaningless between steps, because erasure changes the point set.
    """

    ids: Sequence[int]
    coords_m: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    eps_p: NDArray[np.float64]
    eps_rate_per_s: NDArray[np.float64]
    temperature_k: NDArray[np.float64]
    mass_kg: NDArray[np.float64]
    equivalent_stress_pa: NDArray[np.float64]
    cauchy_xx_pa: NDArray[np.float64]
    cauchy_yy_pa: NDArray[np.float64]


@dataclass(frozen=True)
class EnergyTotals:
    """Body totals in joules, summed over the material points.

    Kratos reports both per point, so the totals are sums rather than a global
    quantity read from the solver, and both therefore *drop* when a point is
    erased — which is why the deleted-mass columns sit beside them. External
    work has no counterpart to read at all; it is integrated instead, by
    ``tool_power_w``.
    """

    kinetic_j: float
    internal_j: float


def read_material_point_fields(
    model_part: KM.ModelPart,
    process_info: KM.ProcessInfo,
) -> MaterialPointFields:
    """Read every telemetry-relevant variable off the live material points.

    Each variable here was confirmed readable on real material points; the
    registered-but-unimplemented ones (``MP_PRESSURE``,
    ``MP_DELTA_PLASTIC_STRAIN``) are not requested. ``MP_CAUCHY_STRESS_VECTOR``
    supplies the two in-plane normal stresses that let a supervisor rebuild
    triaxiality offline.

    Slicing a Kratos ``array_1d`` yields a ublas vector_slice that pybind
    cannot convert, so vector components are read out one at a time.

    A model part with no material points raises: it means every point has been
    erased, which is total erosion of the body rather than a state to record.
    """
    elements = list(model_part.Elements)
    if not elements:
        raise ValueError(
            f"model part {model_part.Name!r} has no material points left to read: the body has "
            "been entirely erased, so there is no state to record"
        )

    def scalar(variable) -> NDArray[np.float64]:
        return np.array(
            [
                element.CalculateOnIntegrationPoints(variable, process_info)[0]
                for element in elements
            ],
            dtype=np.float64,
        )

    def planar(variable) -> NDArray[np.float64]:
        raw = [
            element.CalculateOnIntegrationPoints(variable, process_info)[0] for element in elements
        ]
        return np.array([[value[0], value[1]] for value in raw], dtype=np.float64)

    stress = np.array(
        [
            element.CalculateOnIntegrationPoints(MPM.MP_CAUCHY_STRESS_VECTOR, process_info)[0]
            for element in elements
        ],
        dtype=np.float64,
    )

    return MaterialPointFields(
        ids=[element.Id for element in elements],
        coords_m=planar(MPM.MP_COORD),
        velocity_m_s=planar(MPM.MP_VELOCITY),
        eps_p=scalar(MPM.MP_EQUIVALENT_PLASTIC_STRAIN),
        eps_rate_per_s=scalar(MPM.MP_EQUIVALENT_PLASTIC_STRAIN_RATE),
        temperature_k=scalar(MPM.MP_TEMPERATURE),
        mass_kg=scalar(MPM.MP_MASS),
        equivalent_stress_pa=scalar(MPM.MP_EQUIVALENT_STRESS),
        cauchy_xx_pa=stress[:, _CAUCHY_XX],
        cauchy_yy_pa=stress[:, _CAUCHY_YY],
    )


def snapshot_from_fields(
    fields: MaterialPointFields,
    step: int,
    time_s: float,
    initiation_ledger: PointLedger,
    damage_ledger: PointLedger,
) -> Snapshot:
    """Attach the two failure ledgers to a point read, by id.

    Failure is the one part of the state the solver does not hold — MPM has no
    damage variable of any kind — so both columns come from the ledgers the
    damage process keeps. They are separate columns because they are separate
    phases: ``initiation`` is the sum omega that reaches 1.0 when failure
    begins, ``damage`` the degradation that runs from there to separation, and
    a point at omega = 0.98 is indistinguishable from a pristine one if only
    the second is recorded.

    Both are looked up per point id, never by array position: the point set
    shrinks as points separate, so the nth ledger entry stops being the nth
    live point the first time anything is erased.
    """
    return Snapshot(
        step=step,
        time_s=time_s,
        point_ids=np.asarray(fields.ids, dtype=np.int64),
        coords_m=fields.coords_m,
        velocity_m_s=fields.velocity_m_s,
        eps_p=fields.eps_p,
        eps_rate_per_s=fields.eps_rate_per_s,
        temperature_k=fields.temperature_k,
        initiation=values_for(initiation_ledger, fields.ids),
        damage=values_for(damage_ledger, fields.ids),
        mass_kg=fields.mass_kg,
        equivalent_stress_pa=fields.equivalent_stress_pa,
        cauchy_xx_pa=fields.cauchy_xx_pa,
        cauchy_yy_pa=fields.cauchy_yy_pa,
    )


def snapshot_from_part(
    model_part: KM.ModelPart,
    step: int,
    time_s: float,
    initiation_ledger: PointLedger,
    damage_ledger: PointLedger,
) -> Snapshot:
    """One sampled step of per-material-point state, read off the model part."""
    fields = read_material_point_fields(model_part, model_part.ProcessInfo)
    return snapshot_from_fields(fields, step, time_s, initiation_ledger, damage_ledger)


def energy_totals_from_part(
    model_part: KM.ModelPart,
    process_info: KM.ProcessInfo,
) -> EnergyTotals:
    """Kinetic and internal energy of the body, in joules.

    ``MP_KINETIC_ENERGY`` and ``MP_STRAIN_ENERGY`` are per-point energies, so
    the body totals are their sums. ``MP_TOTAL_ENERGY`` is their sum too and is
    not read separately — one derived column would only invite the two to
    disagree.

    ``MP_STRAIN_ENERGY`` was measured on live points to be *total* internal
    energy — it tracks the accumulated plastic work, not just the recoverable
    elastic part, which is the opposite of the usual Kratos convention and was
    worth checking rather than assuming. See docs/kratos_api_notes.md, "What
    MP_STRAIN_ENERGY actually measures". The consequence for whoever writes
    the energy-drift diagnostic: plastic dissipation is already inside this
    column, so the balance KE + IE - W_ext can close, and adding a separate
    thermal term for the heat in MP_TEMPERATURE would count it twice.
    """
    elements = list(model_part.Elements)
    kinetic = sum(
        element.CalculateOnIntegrationPoints(MPM.MP_KINETIC_ENERGY, process_info)[0]
        for element in elements
    )
    internal = sum(
        element.CalculateOnIntegrationPoints(MPM.MP_STRAIN_ENERGY, process_info)[0]
        for element in elements
    )
    return EnergyTotals(kinetic_j=float(kinetic), internal_j=float(internal))


def tool_power_w(
    force_on_tool_n: tuple[float, float],
    tool_velocity_m_s: tuple[float, float],
) -> float:
    """Rate at which the driven tool does work on the workpiece, in watts.

    ``force_on_tool_n`` is the force the *workpiece* exerts on the tool — what
    ``rigid_tool.reaction_force`` measures and what ``Series.cutting_force_n``
    and ``thrust_force_n`` record. Newton's third law makes the force on the
    workpiece its negative, and that force acts at the tool's velocity, so the
    power the tool delivers is -F.v.

    The sign, worked through rather than asserted, because a flipped one here
    would invert an energy balance without anything looking wrong: the tool is
    driven at v = (-V, 0) with V > 0, and material resisting the cut pushes
    back on it in +x, so Fx > 0. Then -F.v = -(Fx * -V) = +Fx * V > 0. A tool
    advancing into resisting material does positive work on the workpiece;
    negative means the workpiece is doing work on the tool, as an elastic
    rebound briefly can.
    """
    return -(force_on_tool_n[0] * tool_velocity_m_s[0] + force_on_tool_n[1] * tool_velocity_m_s[1])


@dataclass(frozen=True)
class _SeriesRow:
    """One step's measurements, held as a row until the flush turns it into columns.

    Named fields rather than a tuple: the series has ten columns of the same
    dtype, and a positional layout is one careless insertion away from writing
    the thrust force into the cutting-force column, which nothing downstream
    could detect.
    """

    step: int
    time_s: float
    cutting_force_n: float
    thrust_force_n: float
    time_step_s: float
    kinetic_energy_j: float
    internal_energy_j: float
    external_work_j: float
    n_points: int
    deleted_mass_kg: float


@dataclass(frozen=True)
class _DeletionRow:
    """One separated point, recorded in the step it was still readable in."""

    step: int
    point_id: int
    coords_m: tuple[float, float]
    temperature_k: float
    mass_kg: float


class TelemetryWriterProcess(KM.Process):
    """Writes one episode: a series row every step, snapshots on an interval, deletions as they fall.

    Call it once per solution step, in ``FinalizeSolutionStep``, after both the
    tool process and the damage process; then call ``ExecuteFinalize`` when the
    run stops. Both orderings are enforced rather than trusted — see the module
    docstring. The step clock, the point count and the actual time step all
    come from the model part, so the writer never carries a second copy of the
    solver's state.

    The tool and damage processes arrive as objects rather than as extracted
    values because each supplies things that are only correct *together*: the
    tool's force and the velocity it was measured against (pass them
    separately and one sign typo silently negates the whole work column), and
    the damage ledgers and the ids just separated, which have to be read at the
    same instant to agree.

    Two metadata fields are measured here and replace what the caller
    supplied. ``initial_mass_kg`` because this process is built before the run
    starts, making it the only observer that can see the workpiece before any
    point has separated — a baseline taken later, or summed from the first
    *sampled* snapshot, may already be missing points, and a deleted-mass
    fraction measured against a shrunken baseline understates the loss.
    ``tool_velocity_m_s`` because the tool object is the authority on how the
    tool is actually driven.

    One offset to know when reading the output: ``n_points`` is the live count
    at the end of the step, and a point flagged for separation is still live
    then — the solver erases it at its next element search. So on the step a
    point separates, its mass is already in ``deleted_mass_kg`` while it is
    still counted in ``n_points``, and the count catches up one step later.

    ``snapshot`` and ``energies`` are injected so the one part that needs
    genuine material points can be supplied directly in tests, as
    ``JohnsonCookDamageProcess`` injects its gather.
    """

    def __init__(
        self,
        model_part: KM.ModelPart,
        episode_dir: Path,
        metadata: CaseMetadata,
        damage: JohnsonCookDamageProcess,
        tool: RigidToolProcess,
        sample_every: int,
        snapshot: Callable[
            [KM.ModelPart, int, float, PointLedger, PointLedger], Snapshot
        ] = snapshot_from_part,
        energies: Callable[[KM.ModelPart, KM.ProcessInfo], EnergyTotals] = energy_totals_from_part,
    ) -> None:
        super().__init__()
        if sample_every < 1:
            raise ValueError(f"sample_every must be at least one step, got {sample_every}")

        self._model_part = model_part
        self._episode_dir = Path(episode_dir)
        self._damage = damage
        self._tool = tool
        self._sample_every = sample_every
        self._snapshot = snapshot
        self._energies = energies

        self._snapshot_steps: list[int] = []
        self._last_sampled_step: int | None = None
        self._last_seen_step: int | None = None
        self._forces_recorded = len(tool.force_history)
        self._series_rows: list[_SeriesRow] = []
        self._deletion_rows: list[_DeletionRow] = []
        self._deleted_ids: set[int] = set()
        self._deleted_mass_kg = 0.0
        self._external_work_j = 0.0
        self._metadata = self._measured_metadata(metadata)

    @property
    def snapshot_steps(self) -> tuple[int, ...]:
        """Steps sampled so far — the same index the manifest carries."""
        return tuple(self._snapshot_steps)

    def ExecuteFinalizeSolutionStep(self) -> None:
        info = self._model_part.ProcessInfo
        step = int(info[KM.STEP])
        time_s = float(info[KM.TIME])
        self._check_step_advanced(step)

        separated = self._unrecorded_separations()
        sampling = self._due_to_sample(step)

        energies = self._energies(self._model_part, info)
        forces_n = self._tool_force_n()
        time_step_s = float(info[KM.DELTA_TIME])
        power_w = tool_power_w(forces_n, self._metadata.tool_velocity_m_s)
        self._external_work_j += power_w * time_step_s

        # One read serves both jobs when a step is due a sample and has
        # separations, and neither job runs on an ordinary step.
        snapshot = None
        if sampling or separated:
            snapshot = self._snapshot(
                self._model_part, step, time_s, self._damage.initiation, self._damage.damage
            )
            if separated:
                self._record_deletions(step, snapshot, separated)

        self._series_rows.append(
            _SeriesRow(
                step=step,
                time_s=time_s,
                cutting_force_n=forces_n[0],
                thrust_force_n=forces_n[1],
                time_step_s=time_step_s,
                kinetic_energy_j=energies.kinetic_j,
                internal_energy_j=energies.internal_j,
                external_work_j=self._external_work_j,
                n_points=self._model_part.NumberOfElements(),
                deleted_mass_kg=self._deleted_mass_kg,
            )
        )

        if snapshot is not None and sampling:
            # Snapshot file first, then the series, then the index that claims
            # them: whatever a kill interrupts, the manifest never names a
            # snapshot whose file is not already complete on disk.
            append_snapshot(self._episode_dir, snapshot)
            self._snapshot_steps.append(step)
            self._last_sampled_step = step
            self._flush()

    def ExecuteFinalize(self) -> None:
        """Flush the series, the deletion records, and the final snapshot index."""
        self._flush()

    def _flush(self) -> None:
        write_series(self._episode_dir, self._series(), self._deletions())
        write_manifest(self._episode_dir, self._metadata, self._snapshot_steps)

    def _measured_metadata(self, metadata: CaseMetadata) -> CaseMetadata:
        """The case metadata with the two fields only this process can observe.

        The caller's ``initial_mass_kg`` is not simply discarded: a caller that
        supplied one computed it from the geometry and density it *believes*
        the case has, so a disagreement is evidence that point generation
        produced a different body. That is worth a warning rather than a
        silent overwrite, and worth a warning rather than an exception because
        the measured value is the right one either way.
        """
        measured_kg = self._measure_initial_mass_kg()
        speed_m_s = math.hypot(*self._tool.cutting_velocity)
        if not math.isclose(
            speed_m_s, metadata.cutting_speed_m_s, rel_tol=_SPEED_AGREEMENT_REL_TOL
        ):
            raise ValueError(
                f"case metadata says the tool cuts at {metadata.cutting_speed_m_s} m/s but the "
                f"tool process is driving it at {self._tool.cutting_velocity} m/s, which is "
                f"{speed_m_s} m/s: the episode would describe a different run from the one it "
                "records"
            )
        if metadata.initial_mass_kg > 0.0 and not math.isclose(
            measured_kg, metadata.initial_mass_kg, rel_tol=_MASS_AGREEMENT_REL_TOL
        ):
            warnings.warn(
                f"case metadata claims an initial workpiece mass of "
                f"{metadata.initial_mass_kg} kg but the generated material points weigh "
                f"{measured_kg} kg; recording the measured value. A gap this large means the "
                "points generated are not the body the case describes",
                stacklevel=2,
            )
        return replace(
            metadata,
            initial_mass_kg=measured_kg,
            tool_velocity_m_s=tuple(self._tool.cutting_velocity),
        )

    def _measure_initial_mass_kg(self) -> float:
        """Total material point mass now, before the run can have lost any.

        Read through the same snapshot seam as everything else rather than
        through a second injected callable: it happens once, at construction,
        and one seam is one thing to keep honest in tests.

        A non-positive total raises. The failure it catches is construction
        before the material points exist — the natural place to build a process
        is ``Initialize()``, and MPM generates its points later — which would
        otherwise leave the documented denominator of the deleted-mass fraction
        at zero for the whole episode.
        """
        info = self._model_part.ProcessInfo
        initial = self._snapshot(
            self._model_part,
            int(info[KM.STEP]),
            float(info[KM.TIME]),
            self._damage.initiation,
            self._damage.damage,
        )
        total_kg = float(np.sum(initial.mass_kg))
        if total_kg <= 0.0:
            raise ValueError(
                f"the workpiece weighs {total_kg} kg at construction, so there is no baseline to "
                "measure deleted mass against: build TelemetryWriterProcess after the material "
                "points have been generated, not in Initialize()"
            )
        return total_kg

    def _check_step_advanced(self, step: int) -> None:
        """Refuse a step that is not the one after the last.

        The writer must see every step. A gap means the deletions flagged in
        the skipped step are gone: the damage process reports what separated in
        its *latest* update, and once the solver's element search has erased
        those points nothing reports them again — so the record comes up short
        in exactly the direction that makes an eroding run look healthy. Being
        called under a ``% every_n`` guard is the established idiom for the
        damage process, which makes this an easy mistake to make.
        """
        if self._last_seen_step is not None and step != self._last_seen_step + 1:
            raise ValueError(
                f"TelemetryWriterProcess last saw step {self._last_seen_step} and is now being "
                f"called at step {step}: it must run on every solution step, or the deletions "
                "flagged in the skipped steps are lost with nothing to mark their absence"
            )
        self._last_seen_step = step

    def _tool_force_n(self) -> tuple[float, float]:
        """This step's force on the tool, refusing a stale or absent reading.

        The tool measures an impulse over the step and divides by dt; this
        writer multiplies by dt again to get work. Pairing one step's impulse
        with another step's dt is the failure this guards, and it leaves no
        trace in any other column.
        """
        history = self._tool.force_history
        if len(history) <= self._forces_recorded:
            raise ValueError(
                "the tool process has not measured a force since the last telemetry step: run "
                "RigidToolProcess before TelemetryWriterProcess within the same "
                "FinalizeSolutionStep, or the work integral pairs this step's dt with an "
                "earlier step's force"
            )
        self._forces_recorded = len(history)
        latest = history[-1]
        return float(latest[0]), float(latest[1])

    def _due_to_sample(self, step: int) -> bool:
        """Whether this step gets a snapshot.

        The interval is measured against the last step actually sampled, not
        against ``step % sample_every``, so a run resumed from a checkpoint at
        an arbitrary step still samples on schedule.
        """
        return (
            self._last_sampled_step is None or step - self._last_sampled_step >= self._sample_every
        )

    def _unrecorded_separations(self) -> tuple[int, ...]:
        """Points the damage law has separated that are not yet in the record.

        A flagged point survives until the solver's next element search, so it
        is re-reported as separated for as long as it is there. Recording it
        once per step would inflate deleted mass without any mass being lost.
        """
        return tuple(
            point_id for point_id in self._damage.separated_ids if point_id not in self._deleted_ids
        )

    def _record_deletions(self, step: int, snapshot: Snapshot, ids: Sequence[int]) -> None:
        row_of = {int(point_id): row for row, point_id in enumerate(snapshot.point_ids)}
        for point_id in ids:
            row = row_of.get(int(point_id))
            if row is None:
                raise ValueError(
                    f"material point {point_id} was separated at step {step} but is no longer "
                    "in the model part: run TelemetryWriterProcess in the same solution step as "
                    "JohnsonCookDamageProcess and after it, before the solver's element search "
                    "erases the point"
                )
            self._deletion_rows.append(
                _DeletionRow(
                    step=step,
                    point_id=int(point_id),
                    coords_m=(float(snapshot.coords_m[row, 0]), float(snapshot.coords_m[row, 1])),
                    temperature_k=float(snapshot.temperature_k[row]),
                    mass_kg=float(snapshot.mass_kg[row]),
                )
            )
            self._deleted_ids.add(int(point_id))
            self._deleted_mass_kg += float(snapshot.mass_kg[row])

    def _series(self) -> Series:
        rows = self._series_rows
        return Series(
            step=[row.step for row in rows],
            time_s=[row.time_s for row in rows],
            cutting_force_n=[row.cutting_force_n for row in rows],
            thrust_force_n=[row.thrust_force_n for row in rows],
            time_step_s=[row.time_step_s for row in rows],
            kinetic_energy_j=[row.kinetic_energy_j for row in rows],
            internal_energy_j=[row.internal_energy_j for row in rows],
            external_work_j=[row.external_work_j for row in rows],
            n_points=[row.n_points for row in rows],
            deleted_mass_kg=[row.deleted_mass_kg for row in rows],
        )

    def _deletions(self) -> Deletion:
        rows = self._deletion_rows
        return Deletion(
            step=[row.step for row in rows],
            point_id=[row.point_id for row in rows],
            # reshape so a run that separated nothing still gives (0, 2) rather
            # than an empty 1-D array the planar check would reject.
            coords_m=np.array([row.coords_m for row in rows], dtype=np.float64).reshape(-1, 2),
            temperature_k=[row.temperature_k for row in rows],
            mass_kg=[row.mass_kg for row in rows],
        )
