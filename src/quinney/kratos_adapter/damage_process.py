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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication as MPM
import numpy as np
from numpy.typing import NDArray

from quinney.materials.johnson_cook import JohnsonCookDamage, advance
from quinney.point_ledger import PointLedger, updated_ledger, values_for


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
    separated = np.asarray(separated, dtype=bool)
    if len(ids) != separated.size:
        raise ValueError(
            f"ids and separated disagree in length: {len(ids)} ids, {separated.size} flags"
        )

    for point_id, is_separated in zip(ids, separated.tolist(), strict=True):
        if is_separated:
            model_part.GetElement(point_id).Set(KM.TO_ERASE, True)

    return int(separated.sum())


class JohnsonCookDamageProcess(KM.Process):
    """Accumulates Johnson-Cook damage each step and separates failed points.

    Damage is held here, keyed by point id, because the solver has nowhere to
    put it. ``damage`` is exposed so the telemetry dump and the diagnostics can
    read the field without recomputing it.

    ``gather`` is injected so the state read — the one part that needs genuine
    material points — can be supplied directly in tests.
    """

    def __init__(
        self,
        model_part: KM.ModelPart,
        params: JohnsonCookDamage,
        separation_threshold: float = 1.0,
        gather: Callable[[KM.ModelPart, KM.ProcessInfo], MaterialPointState] = (
            gather_material_point_state
        ),
    ) -> None:
        super().__init__()
        self._model_part = model_part
        self._params = params
        self._separation_threshold = separation_threshold
        self._gather = gather
        self._damage: PointLedger = {}
        self._plastic_strain: PointLedger = {}

    @property
    def damage(self) -> PointLedger:
        """Damage per point id, including points already separated."""
        return dict(self._damage)

    def ExecuteFinalizeSolutionStep(self) -> None:
        state = self._gather(self._model_part, self._model_part.ProcessInfo)
        if not state.ids:
            return

        # MP_DELTA_PLASTIC_STRAIN is not implemented for MPM elements, so the
        # increment is the growth in equivalent plastic strain since last step.
        # A point seen for the first time reads zero, giving it its full strain
        # as one increment — correct, since it accumulated that strain already.
        previous = values_for(self._plastic_strain, state.ids)
        delta_plastic_strain = state.plastic_strain - previous

        step = advance(
            self._params,
            damage=values_for(self._damage, state.ids),
            cauchy_xx=state.cauchy_xx,
            cauchy_yy=state.cauchy_yy,
            equivalent_stress=state.equivalent_stress,
            delta_plastic_strain=delta_plastic_strain,
            plastic_strain_rate_per_s=state.plastic_strain_rate_per_s,
            temperature_k=state.temperature_k,
            separation_threshold=self._separation_threshold,
        )

        self._damage = updated_ledger(self._damage, state.ids, step.damage)
        self._plastic_strain = updated_ledger(self._plastic_strain, state.ids, state.plastic_strain)
        flag_separated(self._model_part, state.ids, step.separated)
