"""Johnson-Cook damage as an in-loop Kratos process.

Kratos' MPM applications ship Johnson-Cook *plasticity* but no damage model
and no damage variable, so quinney computes damage itself from the material
point state the solver does expose, and separates points that reach the
threshold.

The law lives in ``quinney.materials.johnson_cook`` and is pure. This module is
the only part that touches Kratos: it reads state, calls the law, and flags
points for erasure. Keeping the split sharp is what lets the supervisor preview
a threshold change offline and get the same answer the solver would.

What this gives you: damage initiation and separation at D = 1. What it does
not give you: progressive stress degradation. The compiled JohnsonCookThermal-
Plastic laws compute stress internally with no hook to scale it, so damage here
removes points rather than softening them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication as MPM
import numpy as np
from numpy.typing import NDArray

from quinney.materials.johnson_cook import (
    JohnsonCookDamage,
    advance,
    fracture_displacement,
)
from quinney.point_ledger import PointLedger, updated_ledger, values_for

# The names the per-point ledgers are checkpointed and resumed under. They are
# constants because a checkpoint written today is read back by a restart weeks
# later, and a renamed key would resume every point undamaged without raising.
LEDGER_INITIATION = "initiation"
LEDGER_DAMAGE = "damage"
LEDGER_PLASTIC_STRAIN = "plastic_strain"
LEDGER_NAMES = (LEDGER_INITIATION, LEDGER_DAMAGE, LEDGER_PLASTIC_STRAIN)


@dataclass(frozen=True)
class MaterialPointState:
    """The material-point state one damage increment needs.

    Units follow Kratos: stresses in Pa, temperature in K, strain rate in 1/s.
    ``ids`` identifies the points so damage survives the set shrinking when
    separated points are erased. The plastic strain *increment* is absent by
    necessity: ``MP_DELTA_PLASTIC_STRAIN`` is registered but not implemented
    for MPM elements, so the increment is differenced from ``plastic_strain``
    between steps.
    """

    ids: Sequence[int]
    plastic_strain: NDArray[np.float64]
    plastic_strain_rate_per_s: NDArray[np.float64]
    temperature_k: NDArray[np.float64]
    equivalent_stress: NDArray[np.float64]
    cauchy_xx: NDArray[np.float64]
    cauchy_yy: NDArray[np.float64]
    volume_m3: NDArray[np.float64]


def gather_material_point_state(
    model_part: KM.ModelPart,
    process_info: KM.ProcessInfo,
) -> MaterialPointState:
    """Read the damage-relevant state off every material point.

    Every variable here was confirmed readable on live material points in the
    Phase 0 case. Two that the first draft assumed are NOT implemented for MPM
    elements and raise if requested: ``MP_PRESSURE`` and
    ``MP_DELTA_PLASTIC_STRAIN``. Mean stress is reconstructed from
    ``MP_CAUCHY_STRESS_VECTOR`` instead, and the plastic strain increment is
    differenced between steps.

    ``MP_CAUCHY_STRESS_VECTOR`` is (sxx, syy, sxy) in plane strain — three
    components, with the out-of-plane stress not reported.
    """
    elements = list(model_part.Elements)

    def scalar(variable) -> NDArray[np.float64]:
        return np.array(
            [
                element.CalculateOnIntegrationPoints(variable, process_info)[0]
                for element in elements
            ],
            dtype=np.float64,
        )

    stress = np.array(
        [
            element.CalculateOnIntegrationPoints(MPM.MP_CAUCHY_STRESS_VECTOR, process_info)[0]
            for element in elements
        ],
        dtype=np.float64,
    ).reshape(len(elements), -1)

    return MaterialPointState(
        ids=[element.Id for element in elements],
        plastic_strain=scalar(MPM.MP_EQUIVALENT_PLASTIC_STRAIN),
        plastic_strain_rate_per_s=scalar(MPM.MP_EQUIVALENT_PLASTIC_STRAIN_RATE),
        temperature_k=scalar(MPM.MP_TEMPERATURE),
        equivalent_stress=scalar(MPM.MP_EQUIVALENT_STRESS),
        cauchy_xx=stress[:, 0],
        cauchy_yy=stress[:, 1],
        volume_m3=scalar(MPM.MP_VOLUME),
    )


def separated_point_ids(
    ids: Sequence[int],
    separated: NDArray[np.bool_],
) -> tuple[int, ...]:
    """Ids of the points the damage law separated, in the order given.

    Ids rather than a mask, because the mask is only meaningful against the
    point set it was computed over, and that set shrinks whenever separated
    points are erased.
    """
    separated = np.asarray(separated, dtype=bool)
    if len(ids) != separated.size:
        raise ValueError(
            f"ids and separated disagree in length: {len(ids)} ids, {separated.size} flags"
        )

    return tuple(
        int(point_id)
        for point_id, is_separated in zip(ids, separated.tolist(), strict=True)
        if is_separated
    )


def flag_separated(
    model_part: KM.ModelPart,
    ids: Sequence[int],
    separated: NDArray[np.bool_],
) -> int:
    """Mark separated points TO_ERASE, returning how many were flagged.

    Flagging is deliberately separate from erasing: the supervisor may want to
    see what *would* go before a MaterialPointEraseProcess actually removes it.
    """
    flagged = separated_point_ids(ids, separated)
    for point_id in flagged:
        model_part.GetElement(point_id).Set(KM.TO_ERASE, True)

    return len(flagged)


def _seed_ledgers(resume_from: Mapping[str, PointLedger] | None) -> dict[str, dict[int, float]]:
    """The three ledgers a run starts from, empty unless resuming.

    A resume must bring all three or none. The dangerous case is not a
    checkpoint carrying a fourth ledger, it is one carrying two of the three:
    an absent ``plastic_strain`` would default to empty, and on the first step
    after the halt every point's entire accumulated plastic strain would land
    as a single increment, separating points that should have survived — with
    nothing raising, because an empty ledger is well formed. So a partial
    resume raises here, naming what is missing, and nothing downstream has to
    notice. ``restart.restore_ledgers`` returns whatever names the sidecar
    happens to hold and checks nothing.

    An unknown name raises for the mirror-image reason: the caller believes it
    is restoring something this process does not carry.

    Ledgers come in keyed by point id, as ``point_ledger`` requires — never by
    array position.
    """
    if resume_from is None:
        return {name: {} for name in LEDGER_NAMES}

    unknown = sorted(set(resume_from) - set(LEDGER_NAMES))
    if unknown:
        raise ValueError(
            f"cannot resume from unknown ledgers {', '.join(unknown)}; "
            f"this process carries {', '.join(LEDGER_NAMES)}"
        )
    missing = [name for name in LEDGER_NAMES if name not in resume_from]
    if missing:
        raise ValueError(
            f"cannot resume from a partial checkpoint: it carries "
            f"{', '.join(sorted(resume_from))} but not {', '.join(missing)}. All of "
            f"{', '.join(LEDGER_NAMES)} are needed — resuming without plastic_strain hands the "
            "first step after the halt every point's whole accumulated strain as one increment, "
            "and resuming without initiation or damage restarts damaged points pristine"
        )
    return {
        name: {int(point_id): float(value) for point_id, value in resume_from[name].items()}
        for name in LEDGER_NAMES
    }


class JohnsonCookDamageProcess(KM.Process):
    """Accumulates Johnson-Cook damage each step and separates failed points.

    Damage is held here, keyed by point id, because the solver has nowhere to
    put it. ``damage`` is exposed so the telemetry dump and the diagnostics can
    read the field without recomputing it.

    ``gather`` is injected so the state read — the one part that needs genuine
    material points — can be supplied directly in tests.

    ``resume_from`` re-seeds the ledgers after a halt. They are the only run
    state Kratos cannot checkpoint for it, so a resumed run built without them
    would restart every point undamaged and separate late — silently, since
    nothing about a zero ledger is malformed.
    """

    def __init__(
        self,
        model_part: KM.ModelPart,
        params: JohnsonCookDamage,
        fracture_energy_j_m2: float,
        yield_stress_pa: float,
        separation_threshold: float = 1.0,
        gather: Callable[[KM.ModelPart, KM.ProcessInfo], MaterialPointState] = (
            gather_material_point_state
        ),
        resume_from: Mapping[str, PointLedger] | None = None,
    ) -> None:
        super().__init__()
        self._model_part = model_part
        self._params = params
        self._separation_threshold = separation_threshold
        self._gather = gather
        self._fracture_displacement_m = float(
            fracture_displacement(fracture_energy_j_m2, yield_stress_pa)
        )
        seed = _seed_ledgers(resume_from)
        self._initiation: PointLedger = seed[LEDGER_INITIATION]
        self._damage: PointLedger = seed[LEDGER_DAMAGE]
        self._plastic_strain: PointLedger = seed[LEDGER_PLASTIC_STRAIN]
        self._separated_ids: tuple[int, ...] = ()

    @property
    def initiation(self) -> PointLedger:
        """Johnson-Cook damage sum omega per point id; 1.0 means initiated."""
        return dict(self._initiation)

    @property
    def damage(self) -> PointLedger:
        """Post-initiation damage D per point id, driving softening."""
        return dict(self._damage)

    @property
    def ledgers(self) -> dict[str, PointLedger]:
        """Every per-point ledger, under the names ``resume_from`` reads back.

        The previous plastic strain belongs here as much as the damage does:
        it is the baseline the increment is differenced against, so resuming
        without it hands the first step after the halt the point's whole
        accumulated strain as one increment.
        """
        return {
            LEDGER_INITIATION: dict(self._initiation),
            LEDGER_DAMAGE: dict(self._damage),
            LEDGER_PLASTIC_STRAIN: dict(self._plastic_strain),
        }

    @property
    def separated_ids(self) -> tuple[int, ...]:
        """Point ids separated on the most recent update.

        Read this in the same solution step, before the solver's element search
        erases them: it is the last moment their position, temperature and mass
        can be recorded, and those are what tell separation along the cut path
        from mass evaporating out of the bulk.
        """
        return self._separated_ids

    def ExecuteFinalizeSolutionStep(self) -> None:
        self._separated_ids = ()
        state = self._gather(self._model_part, self._model_part.ProcessInfo)
        if not state.ids:
            return

        # MP_DELTA_PLASTIC_STRAIN is not implemented for MPM elements, so the
        # increment is the growth in equivalent plastic strain since last step.
        # A point seen for the first time reads zero, giving it its full strain
        # as one increment — correct, since it accumulated that strain already.
        previous = values_for(self._plastic_strain, state.ids)
        delta_plastic_strain = state.plastic_strain - previous

        # With THICKNESS = 1 m in plane strain, MP_VOLUME is numerically the
        # in-plane area, so its square root is the in-plane characteristic
        # length the fracture-energy regularization needs.
        characteristic_length_m = np.sqrt(state.volume_m3)

        step = advance(
            self._params,
            initiation=values_for(self._initiation, state.ids),
            damage=values_for(self._damage, state.ids),
            cauchy_xx=state.cauchy_xx,
            cauchy_yy=state.cauchy_yy,
            equivalent_stress=state.equivalent_stress,
            delta_plastic_strain=delta_plastic_strain,
            plastic_strain_rate_per_s=state.plastic_strain_rate_per_s,
            temperature_k=state.temperature_k,
            characteristic_length_m=characteristic_length_m,
            fracture_displacement_m=self._fracture_displacement_m,
            separation_threshold=self._separation_threshold,
        )

        self._initiation = updated_ledger(self._initiation, state.ids, step.initiation)
        self._damage = updated_ledger(self._damage, state.ids, step.damage)
        self._plastic_strain = updated_ledger(self._plastic_strain, state.ids, state.plastic_strain)
        self._separated_ids = separated_point_ids(state.ids, step.separated)
        flag_separated(self._model_part, state.ids, step.separated)
