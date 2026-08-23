"""Zorev friction driven onto a Kratos slip boundary.

Kratos MPM applies friction at *slip boundaries* — grid conditions carrying
normals, marked SLIP, with a per-node ``FRICTION_COEFFICIENT`` and a tangential
penalty. It knows only Coulomb. It does not apply friction between two
deformable bodies: single-grid MPM contact is inherently no-slip, with no
coefficient to tune.

Since Zorev is Coulomb with a shear cap, this process rewrites the per-node
coefficient every step to ``min(mu, m*k/sigma_n)``. The solver keeps doing
plain Coulomb and the sticking region emerges wherever contact pressure demands
it — no change to Kratos, and no fixed sticking length assumed in advance.

Consequence for case setup: the tool must be a slip boundary, not a body of
material points. See docs/formulation_spike.md.
"""

from __future__ import annotations

from collections.abc import Callable

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication as MPM
import numpy as np
from numpy.typing import NDArray

from quinney.materials.zorev import effective_friction_coefficient, shear_yield_stress


def nodal_contact_pressure(model_part: KM.ModelPart) -> NDArray[np.float64]:
    """Contact pressure at each boundary node, from the interface contact force.

    !!! UNVERIFIED !!!
    ``MPC_CONTACT_FORCE`` is populated by Kratos on slip conditions during a
    run. This reader has never been exercised against one, because the cutting
    case still models the tool as a body rather than a boundary. Confirm the
    magnitude and sign against a running case before trusting a number that
    came through it; the unit tests inject pressures instead.
    """
    nodes = list(model_part.Nodes)
    forces = np.array([node.GetValue(MPM.MPC_CONTACT_FORCE) for node in nodes], dtype=np.float64)
    return np.linalg.norm(forces, axis=1)


class ZorevFrictionProcess(KM.Process):
    """Rewrites the boundary's friction coefficients to carry the Zorev cap.

    ``flow_stress_pa`` sets the shear yield, k = sigma_flow/sqrt(3). Passing the
    *current* flow stress rather than the room-temperature value is what lets
    thermal softening widen the sticking region as the interface heats — the
    coupling that matters for chip morphology.

    ``contact_pressure`` is injected so the one read that needs a live slip
    boundary can be supplied directly in tests.
    """

    def __init__(
        self,
        slip_model_part: KM.ModelPart,
        sliding_friction_coefficient: float,
        flow_stress_pa: float,
        friction_factor: float = 1.0,
        contact_pressure: Callable[[KM.ModelPart], NDArray[np.float64]] = (nodal_contact_pressure),
    ) -> None:
        super().__init__()
        self._part = slip_model_part
        self._sliding_friction = sliding_friction_coefficient
        self._shear_yield_pa = float(shear_yield_stress(flow_stress_pa))
        self._friction_factor = friction_factor
        self._contact_pressure = contact_pressure

        # Without this flag Kratos ignores the per-node coefficients entirely.
        self._part.ProcessInfo[MPM.FRICTION_ACTIVE] = True

    def ExecuteInitializeSolutionStep(self) -> None:
        nodes = list(self._part.Nodes)
        if not nodes:
            return

        pressure = np.asarray(self._contact_pressure(self._part), dtype=np.float64)
        if pressure.size != len(nodes):
            raise ValueError(
                f"contact pressure disagrees with the boundary: "
                f"{pressure.size} values, {len(nodes)} nodes"
            )

        coefficients = effective_friction_coefficient(
            normal_stress_pa=pressure,
            friction_coefficient=self._sliding_friction,
            shear_yield_stress_pa=self._shear_yield_pa,
            friction_factor=self._friction_factor,
        )

        for node, coefficient in zip(nodes, coefficients.tolist(), strict=True):
            node.SetValue(KM.FRICTION_COEFFICIENT, coefficient)
