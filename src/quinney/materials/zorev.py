"""Zorev tool-chip friction — pure, vectorized, solver-independent.

Spec §2 requires a sticking/sliding treatment rather than pure Coulomb, and the
reason is physical. Contact pressure on the rake face is enormous near the tool
tip and falls off along it. Plain Coulomb, tau = mu * sigma_n, predicts a shear
stress near the tip that the workpiece material simply cannot carry — it yields
in shear first. Zorev caps the shear at the material's shear yield:

    tau = min(mu * sigma_n, m * k)      k = sigma_flow / sqrt(3)

giving a sticking region at the tip where shear is pressure-independent, and a
sliding region further out where Coulomb holds. The friction factor m (0..1]
allows an interface that yields before the bulk does.

The cap matters for more than realism: it bounds the frictional heat generated
at the interface, and interface heating feeds the thermal softening that drives
shear localization. An uncapped Coulomb law overheats the tool-chip contact and
distorts the chip morphology that spec §5 is trying to classify.

Every function is pure: array in, array out, no state, no I/O.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# von Mises relates shear yield to tensile flow stress by 1/sqrt(3). Tresca
# would give 1/2 — about 15% higher, which shifts the sticking region and with
# it the tool-chip contact length.
VON_MISES_SHEAR_FACTOR = 1.0 / np.sqrt(3.0)

# Contact pressure at or below this (Pa) means the surfaces are not touching,
# so no shear is transmitted and the friction coefficient is undefined.
MIN_CONTACT_PRESSURE_PA = 1e-9


def shear_yield_stress(flow_stress_pa: ArrayLike) -> NDArray[np.float64]:
    """Shear yield k from tensile flow stress, by von Mises.

    Flow stress is the Johnson-Cook value at the *current* state, so k tracks
    strain hardening, rate hardening and thermal softening along the interface
    rather than being a fixed constant.
    """
    return np.asarray(flow_stress_pa, dtype=np.float64) * VON_MISES_SHEAR_FACTOR


def zorev_shear_stress(
    normal_stress_pa: ArrayLike,
    friction_coefficient: float,
    shear_yield_stress_pa: ArrayLike,
    friction_factor: float = 1.0,
) -> NDArray[np.float64]:
    """Interface shear stress, sticking near the tip and sliding further out.

    ``normal_stress_pa`` is the contact pressure, positive in compression.
    Zero or tensile values mean no contact and carry no shear.
    """
    normal_stress_pa = np.asarray(normal_stress_pa, dtype=np.float64)
    shear_yield_stress_pa = np.asarray(shear_yield_stress_pa, dtype=np.float64)

    sliding = friction_coefficient * normal_stress_pa
    sticking = friction_factor * shear_yield_stress_pa

    in_contact = normal_stress_pa > MIN_CONTACT_PRESSURE_PA
    return np.where(in_contact, np.minimum(sliding, sticking), 0.0)


def effective_friction_coefficient(
    normal_stress_pa: ArrayLike,
    friction_coefficient: float,
    shear_yield_stress_pa: ArrayLike,
    friction_factor: float = 1.0,
) -> NDArray[np.float64]:
    """The Coulomb coefficient that reproduces the Zorev shear stress.

    Kratos applies friction through a per-node ``FRICTION_COEFFICIENT`` and
    knows only about Coulomb. Since Zorev *is* Coulomb with a cap, feeding it

        mu_eff = min(mu, m * k / sigma_n)

    gives mu_eff * sigma_n = min(mu * sigma_n, m * k) exactly — the sticking
    region emerges as a locally reduced coefficient, with no change needed to
    the solver.
    """
    normal_stress_pa = np.asarray(normal_stress_pa, dtype=np.float64)

    in_contact = normal_stress_pa > MIN_CONTACT_PRESSURE_PA
    safe_pressure = np.where(in_contact, normal_stress_pa, 1.0)
    shear = zorev_shear_stress(
        normal_stress_pa, friction_coefficient, shear_yield_stress_pa, friction_factor
    )
    return np.where(in_contact, shear / safe_pressure, 0.0)
