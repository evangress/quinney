"""Progressive damage softening, applied through material properties.

Kratos' compiled ``JohnsonCookThermalPlastic`` laws compute stress internally
and expose no hook to scale it, so sigma = (1-D)*sigma_bar cannot be applied
where the stress is formed. But the law *reads* its constants from the
element's ``Properties`` on every call, and Properties can be created and
mutated from Python at runtime.

So each material point is given its own Properties object at setup, and its
Johnson-Cook constants are scaled by (1-D) as damage grows. The law keeps doing
exactly what it always did and sees a progressively weaker material.

What this is: damage-coupled plasticity, updated once per damage step. What it
is not: a tangent-consistent continuum damage formulation. The stress update
within a step uses the properties as they stood at the start of it, so the
coupling is staggered — acceptable in explicit dynamics, where the step is
already tiny, and worth stating plainly rather than implying otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping

import KratosMultiphysics as KM

from quinney.materials.johnson_cook import degradation_factor

# Constants scaled by (1-D). A and B together set the Johnson-Cook flow stress,
# so scaling both degrades the whole flow curve rather than just its offset.
STRENGTH_VARIABLES = (KM.JC_PARAMETER_A, KM.JC_PARAMETER_B)

# Properties ids for the per-point clones start here, well clear of the ids a
# materials file is likely to have used.
CLONE_ID_BASE = 100_000


class DamageSofteningProcess(KM.Process):
    """Degrades each material point's properties in step with its damage.

    ``degrade_stiffness`` also scales Young's modulus. Softening stiffness is
    the physically complete choice and it *relaxes* the CFL limit, since a
    lower modulus means a slower wave speed. It also lets a badly damaged point
    deform with little restraint, so it can be switched off.
    """

    def __init__(self, model_part: KM.ModelPart, degrade_stiffness: bool = True) -> None:
        super().__init__()
        self._part = model_part
        self._degrade_stiffness = degrade_stiffness
        self._pristine: dict[int, dict] = {}
        self._applied: dict[int, float] = {}
        self._clone_properties()

    def _clone_properties(self) -> None:
        """Give every point its own Properties, recording the pristine values.

        Points generated from one material share a single Properties object, so
        without this the first point to damage would soften all of them.
        """
        for offset, element in enumerate(self._part.Elements):
            source = element.Properties
            clone = self._part.CreateNewProperties(CLONE_ID_BASE + offset)

            recorded = {}
            for variable in (*STRENGTH_VARIABLES, KM.YOUNG_MODULUS):
                value = source.GetValue(variable)
                recorded[variable] = value
                clone.SetValue(variable, value)

            # Carry across everything else the law needs, untouched.
            for variable in (
                KM.JC_PARAMETER_C,
                KM.JC_PARAMETER_n,
                KM.JC_PARAMETER_m,
                KM.DENSITY,
                KM.POISSON_RATIO,
                KM.TEMPERATURE,
                KM.MELD_TEMPERATURE,
                KM.SPECIFIC_HEAT,
            ):
                if source.Has(variable):
                    clone.SetValue(variable, source.GetValue(variable))

            element.Properties = clone
            self._pristine[element.Id] = recorded
            self._applied[element.Id] = 0.0

    def apply(self, damage: Mapping[int, float]) -> None:
        """Scale each named point's constants to match its damage.

        Degradation is always computed from the pristine values, never from the
        current ones: scaling what is already scaled would decay geometrically
        and strip a point of strength long before it separates.

        Damage is irreversible, so a value lower than one already applied is
        ignored rather than allowed to restore lost strength.
        """
        for point_id, value in damage.items():
            pristine = self._pristine.get(point_id)
            if pristine is None:
                continue

            advanced = max(self._applied[point_id], float(value))
            if advanced == self._applied[point_id] and advanced == 0.0:
                continue
            self._applied[point_id] = advanced

            factor = float(degradation_factor(advanced))
            properties = self._part.GetElement(point_id).Properties
            for variable in STRENGTH_VARIABLES:
                properties.SetValue(variable, factor * pristine[variable])
            if self._degrade_stiffness:
                properties.SetValue(KM.YOUNG_MODULUS, factor * pristine[KM.YOUNG_MODULUS])
