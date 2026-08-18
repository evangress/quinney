"""A cutting tool driven at constant velocity, and the force it feels.

Orthogonal cutting assumes a rigid tool advancing at a fixed cutting speed. MPM
has no rigid-body constraint, so the tool is a body of material points whose
velocity is restored to the prescribed value every step.

The correction is not a hack to be apologised for — it *is* the measurement.
The impulse required to restore the prescribed velocity equals the impulse the
workpiece delivered to the tool, so cutting and thrust force come out of
driving the tool, with nothing extra to instrument.
"""

from __future__ import annotations

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication as MPM
import numpy as np
from numpy.typing import ArrayLike, NDArray


def reaction_force(
    mass: ArrayLike,
    velocity: ArrayLike,
    target_velocity: tuple[float, float],
    time_step_s: float,
) -> NDArray[np.float64]:
    """Force the workpiece exerted on the tool over one step, as (Fx, Fy) in N.

    ``velocity`` is (n_points, 2) as measured *before* the correction. The sign
    convention is the force on the tool: a tool driven in -x that the cut slows
    down reports a positive Fx.
    """
    if time_step_s <= 0.0:
        raise ValueError(f"time step must be positive, got {time_step_s}")

    mass = np.asarray(mass, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    target = np.asarray(target_velocity, dtype=np.float64)

    momentum_deficit = (mass[:, None] * (target - velocity)).sum(axis=0)
    return -momentum_deficit / time_step_s


class RigidToolProcess(KM.Process):
    """Holds the tool at its cutting velocity and records the force each step.

    ``force_history`` accumulates (Fx, Fy) per step in newtons — the raw signal
    behind ``cutting_force_mean``, ``force_oscillation_amplitude`` and the
    segmentation FFT of spec §5.
    """

    def __init__(
        self,
        tool_model_part: KM.ModelPart,
        cutting_velocity: tuple[float, float],
        time_step_s: float,
    ) -> None:
        super().__init__()
        self._part = tool_model_part
        self._velocity = cutting_velocity
        self._time_step_s = time_step_s
        self.force_history: list[NDArray[np.float64]] = []

    def ExecuteFinalizeSolutionStep(self) -> None:
        info = self._part.ProcessInfo
        elements = list(self._part.Elements)
        if not elements:
            return

        mass = np.array(
            [e.CalculateOnIntegrationPoints(MPM.MP_MASS, info)[0] for e in elements],
            dtype=np.float64,
        )
        # Slicing a Kratos array_1d yields a ublas vector_slice that pybind
        # cannot convert, so components are read out individually.
        velocities = [e.CalculateOnIntegrationPoints(MPM.MP_VELOCITY, info)[0] for e in elements]
        measured = np.array([[v[0], v[1]] for v in velocities], dtype=np.float64)

        self.force_history.append(reaction_force(mass, measured, self._velocity, self._time_step_s))

        restored = [self._velocity[0], self._velocity[1], 0.0]
        for element in elements:
            element.SetValuesOnIntegrationPoints(MPM.MP_VELOCITY, [restored], info)
