"""Johnson-Cook damage model — pure, vectorized, solver-independent.

The model is the standard three-factor form (Johnson & Cook, 1985):

    eps_f = [D1 + D2 * exp(D3 * sigma*)] * [1 + D4 * ln(eps_rate*)] * [1 + D5 * T*]

    sigma*      stress triaxiality, mean stress / equivalent stress (dimensionless)
    eps_rate*   equivalent plastic strain rate / reference rate      (dimensionless)
    T*          homologous temperature, (T - T_ref)/(T_melt - T_ref) (0..1)

Damage accumulates as D = sum(delta eps_p / eps_f) and the material point is
considered separated at D = 1.

This module owns the damage law for the whole project. The in-loop Kratos
process and the offline replay path both call these functions, so a
supervisor's what-if analysis and the running simulation can never disagree
about what a threshold change would do.

Deliberately absent: progressive stress degradation. Kratos' compiled
JohnsonCookThermalPlastic laws compute stress internally and expose no hook to
scale it, so damage here drives *separation* (point removal), not softening.
Anything claiming otherwise would be fiction.

Every function is pure: array in, array out, no state, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Fracture strain divides the plastic strain increment, so it may never reach
# zero. Extreme tensile triaxiality drives the exponential term to zero,
# leaving D1 alone — and a material with D1 = 0 would otherwise divide by zero.
MIN_FRACTURE_STRAIN = 1e-12

# Equivalent stress below this (Pa) means the point is effectively unloaded, so
# triaxiality is undefined rather than infinite.
MIN_EQUIVALENT_STRESS_PA = 1e-9


@dataclass(frozen=True)
class JohnsonCookDamage:
    """Johnson-Cook damage constants for one material.

    D1..D5 are dimensionless. The reference strain rate is the rate at which
    the rate term vanishes; the reference and melt temperatures bound the
    homologous temperature.
    """

    d1: float
    d2: float
    d3: float
    d4: float
    d5: float
    reference_strain_rate_per_s: float
    reference_temperature_k: float
    melt_temperature_k: float


def triaxiality_from_plane_strain_stress(
    cauchy_xx: ArrayLike,
    cauchy_yy: ArrayLike,
    equivalent_stress: ArrayLike,
) -> NDArray[np.float64]:
    """Stress triaxiality from the components MPM exposes.

    ``MP_PRESSURE`` is registered but *not implemented* for MPM elements, and
    ``MP_CAUCHY_STRESS_VECTOR`` carries only (sxx, syy, sxy) in plane strain —
    the out-of-plane stress is never reported. Under plastic flow szz is the
    intermediate principal stress, szz = (sxx + syy)/2, which makes the mean
    stress (sxx + syy)/2 as well.

    That identity is exact at the plastic limit and about 10% off in the
    elastic range (where it would be (1+nu)(sxx+syy)/3). Damage only
    accumulates while plastic strain grows, so the elastic discrepancy never
    reaches the damage sum.

    Returns zero where the point carries no deviatoric load, and the sign is
    used directly: compression is negative, as measured against a running case.
    """
    cauchy_xx = np.asarray(cauchy_xx, dtype=np.float64)
    cauchy_yy = np.asarray(cauchy_yy, dtype=np.float64)
    equivalent_stress = np.asarray(equivalent_stress, dtype=np.float64)

    mean_stress = 0.5 * (cauchy_xx + cauchy_yy)
    loaded = equivalent_stress > MIN_EQUIVALENT_STRESS_PA
    return np.where(loaded, mean_stress / np.where(loaded, equivalent_stress, 1.0), 0.0)


def fracture_strain(
    params: JohnsonCookDamage,
    triaxiality: ArrayLike,
    plastic_strain_rate_per_s: ArrayLike,
    temperature_k: ArrayLike,
) -> NDArray[np.float64]:
    """Equivalent plastic strain at failure for the given state.

    The rate factor is floored at 1.0: below the reference rate the logarithm
    turns negative and, at small enough ratios, would drive the bracket
    negative and hand back a nonsensical negative fracture strain. Homologous
    temperature is clamped to 0..1 for the same reason at both ends.
    """
    triaxiality = np.asarray(triaxiality, dtype=np.float64)
    plastic_strain_rate_per_s = np.asarray(plastic_strain_rate_per_s, dtype=np.float64)
    temperature_k = np.asarray(temperature_k, dtype=np.float64)

    stress_factor = params.d1 + params.d2 * np.exp(params.d3 * triaxiality)

    rate_ratio = np.maximum(plastic_strain_rate_per_s / params.reference_strain_rate_per_s, 1.0)
    rate_factor = 1.0 + params.d4 * np.log(rate_ratio)

    temperature_span = params.melt_temperature_k - params.reference_temperature_k
    homologous = np.clip(
        (temperature_k - params.reference_temperature_k) / temperature_span, 0.0, 1.0
    )
    thermal_factor = 1.0 + params.d5 * homologous

    return np.maximum(stress_factor * rate_factor * thermal_factor, MIN_FRACTURE_STRAIN)


def accumulate_damage(
    damage: ArrayLike,
    delta_plastic_strain: ArrayLike,
    fracture_strain: ArrayLike,
) -> NDArray[np.float64]:
    """Advance damage by one increment, returning a new array.

    Negative strain increments are ignored: plastic strain is monotonic, so a
    negative delta is noise in the dumped state, and damage does not heal.
    Damage saturates at 1.0 — a point at 1.0 is already separated, and letting
    it climb would distort any statistic taken over the field.
    """
    damage = np.asarray(damage, dtype=np.float64)
    delta_plastic_strain = np.asarray(delta_plastic_strain, dtype=np.float64)
    fracture_strain = np.asarray(fracture_strain, dtype=np.float64)

    increment = np.maximum(delta_plastic_strain, 0.0) / fracture_strain
    return np.minimum(damage + increment, 1.0)


@dataclass(frozen=True)
class DamageStep:
    """Result of one damage increment over a set of material points.

    ``initiation`` is the Johnson-Cook damage sum omega: at 1.0 the point has
    *begun* to fail. ``damage`` is the post-initiation variable D that drives
    softening and reaches 1.0 at separation. Keeping them apart is the whole
    point — a point that has initiated still carries full stress.
    """

    initiation: NDArray[np.float64]
    damage: NDArray[np.float64]
    separated: NDArray[np.bool_]


def advance(
    params: JohnsonCookDamage,
    initiation: ArrayLike,
    damage: ArrayLike,
    cauchy_xx: ArrayLike,
    cauchy_yy: ArrayLike,
    equivalent_stress: ArrayLike,
    delta_plastic_strain: ArrayLike,
    plastic_strain_rate_per_s: ArrayLike,
    temperature_k: ArrayLike,
    characteristic_length_m: ArrayLike,
    fracture_displacement_m: ArrayLike,
    separation_threshold: float = 1.0,
) -> DamageStep:
    """Advance both damage phases one increment from raw material-point state.

    Initiation accumulates everywhere plastic strain grows. Evolution advances
    only where initiation has completed, so material that has yielded heavily
    but not yet initiated stays undamaged and at full strength.

    ``separation_threshold`` is the knob the supervisor tunes (§6
    ``adjust_deletion_threshold``). Because this function is pure, calling it
    with a candidate threshold and *not* applying the result is exactly how the
    supervisor previews a change before proposing it.
    """
    triaxiality = triaxiality_from_plane_strain_stress(cauchy_xx, cauchy_yy, equivalent_stress)
    advanced_initiation = accumulate_damage(
        damage=initiation,
        delta_plastic_strain=delta_plastic_strain,
        fracture_strain=fracture_strain(
            params,
            triaxiality=triaxiality,
            plastic_strain_rate_per_s=plastic_strain_rate_per_s,
            temperature_k=temperature_k,
        ),
    )

    evolved = evolve_damage(
        damage=damage,
        delta_plastic_strain=delta_plastic_strain,
        characteristic_length_m=characteristic_length_m,
        fracture_displacement_m=fracture_displacement_m,
    )
    initiated = advanced_initiation >= 1.0
    advanced_damage = np.where(initiated, evolved, np.asarray(damage, dtype=np.float64))

    return DamageStep(
        initiation=advanced_initiation,
        damage=advanced_damage,
        separated=advanced_damage >= separation_threshold,
    )


# A fully damaged point keeps a sliver of strength rather than none. Exactly
# zero gives a stressless point that inverts and destabilizes the explicit
# step; separation removes it instead of leaving it to misbehave.
RESIDUAL_STRENGTH_FRACTION = 1e-3


def fracture_displacement(
    fracture_energy_j_m2: ArrayLike,
    yield_stress_pa: ArrayLike,
) -> NDArray[np.float64]:
    """Plastic displacement at separation, u_f = 2 G_f / sigma_y (Hillerborg).

    This is what makes damage evolution mesh-objective. Expressing failure as a
    displacement over a characteristic length, rather than a strain, fixes the
    energy dissipated per unit crack area — so refining the mesh no longer
    changes how much energy it costs to separate the chip.

    Spec §1 calls the deletion threshold "notoriously arbitrary and
    mesh-dependent". This is the part that makes it neither.
    """
    fracture_energy_j_m2 = np.asarray(fracture_energy_j_m2, dtype=np.float64)
    yield_stress_pa = np.asarray(yield_stress_pa, dtype=np.float64)
    return 2.0 * fracture_energy_j_m2 / yield_stress_pa


def evolve_damage(
    damage: ArrayLike,
    delta_plastic_strain: ArrayLike,
    characteristic_length_m: ArrayLike,
    fracture_displacement_m: ArrayLike,
) -> NDArray[np.float64]:
    """Advance post-initiation damage D toward separation at D = 1.

    The increment is the plastic *displacement* L * delta_eps_p over the
    fracture displacement, so a point twice the size damages twice as fast per
    unit strain. Call this only for points that have already initiated;
    before initiation the material is undamaged however much it has yielded.
    """
    damage = np.asarray(damage, dtype=np.float64)
    delta_plastic_strain = np.asarray(delta_plastic_strain, dtype=np.float64)
    characteristic_length_m = np.asarray(characteristic_length_m, dtype=np.float64)
    fracture_displacement_m = np.asarray(fracture_displacement_m, dtype=np.float64)

    plastic_displacement = characteristic_length_m * np.maximum(delta_plastic_strain, 0.0)
    return np.minimum(damage + plastic_displacement / fracture_displacement_m, 1.0)


def degradation_factor(damage: ArrayLike) -> NDArray[np.float64]:
    """Stress-carrying fraction (1 - D), floored at a residual.

    Applied as sigma = (1 - D) * sigma_bar. Kratos' compiled Johnson-Cook law
    computes stress internally with no hook to scale it, so quinney applies
    this by degrading the material properties the law reads — see
    ``kratos_adapter.softening``.
    """
    damage = np.asarray(damage, dtype=np.float64)
    return np.maximum(1.0 - damage, RESIDUAL_STRENGTH_FRACTION)
