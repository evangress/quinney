"""OFHC copper — Johnson-Cook constants.

Spec §2 selects OFHC copper because its Johnson-Cook parameters are well
established and there is abundant orthogonal-cutting validation data.

Sources:
  Johnson & Cook (1983), "A constitutive model and data for metals subjected
  to large strains, high strain rates and high temperatures" — flow stress.
  Johnson & Cook (1985), "Fracture characteristics of three metals subjected
  to various strains, strain rates, temperatures and pressures" — damage.

!!! VERIFY BEFORE TRUSTING ANY RESULT !!!
These values are transcribed from the widely reproduced literature set and
have NOT been checked against the original papers in this environment. D3 in
particular is a sensitive fit parameter: an error there changes chip
morphology without changing anything that looks obviously wrong. Check each
constant against the source, then delete this warning.
"""

from __future__ import annotations

from quinney.materials.johnson_cook import JohnsonCookDamage

# Melting point of pure copper (K). Bounds the homologous temperature, so the
# thermal softening that drives shear localization scales against it.
MELT_TEMPERATURE_K = 1356.0

# The temperature the constants were fitted at (K) — room temperature.
REFERENCE_TEMPERATURE_K = 294.0

# Quasi-static reference rate (1/s) at which the JC rate term vanishes.
# Cutting's primary shear zone runs 10^4-10^6 /s, so the rate term is always
# active there; this only sets where the logarithm is zero.
REFERENCE_STRAIN_RATE_PER_S = 1.0

OFHC_COPPER_DAMAGE = JohnsonCookDamage(
    d1=0.54,
    d2=4.89,
    d3=-3.03,
    d4=0.014,
    d5=1.12,
    reference_strain_rate_per_s=REFERENCE_STRAIN_RATE_PER_S,
    reference_temperature_k=REFERENCE_TEMPERATURE_K,
    melt_temperature_k=MELT_TEMPERATURE_K,
)


# --- Johnson-Cook flow stress (Johnson & Cook, 1983) ---
# sigma = (A + B*eps^n) * (1 + C*ln(eps_rate*)) * (1 - T*^m)
YIELD_STRESS_PA = 90.0e6  # A
HARDENING_MODULUS_PA = 292.0e6  # B
HARDENING_EXPONENT = 0.31  # n
STRAIN_RATE_COEFFICIENT = 0.025  # C
THERMAL_SOFTENING_EXPONENT = 1.09  # m

# --- physical constants ---
DENSITY_KG_M3 = 8960.0
YOUNGS_MODULUS_PA = 117.0e9
POISSON_RATIO = 0.34
SPECIFIC_HEAT_J_KG_K = 385.0

# Fraction of plastic work converted to heat. This is the coupling that drives
# thermal softening, hence shear localization, hence serrated chips — the whole
# phenomenon spec §2 exists to reproduce.
TAYLOR_QUINNEY_COEFFICIENT = 0.9


def kratos_material_variables(
    material_points_per_element: int = 3,
    yield_stress_pa: float = YIELD_STRESS_PA,
    density_kg_m3: float = DENSITY_KG_M3,
) -> dict[str, float | int]:
    """The Variables block for a Kratos JohnsonCookThermalPlastic material.

    Derived from the constants above rather than re-typed, so the law running
    inside the solver and quinney's damage model cannot disagree about the
    reference or melt temperature. A drift there would change homologous
    temperature in one and not the other, with nothing failing loudly.

    ``yield_stress_pa`` and ``density_kg_m3`` are overridable so a cutting tool
    can reuse this law with a yield stress it will never reach, staying elastic
    without needing a second constitutive law in the model.

    Note ``TEMPERATURE`` is separate from ``REFERENCE_TEMPERATURE``: the law's
    Check() demands the former as a positive starting value and aborts across
    every OpenMP thread without it.
    """
    return {
        "DENSITY": density_kg_m3,
        "YOUNG_MODULUS": YOUNGS_MODULUS_PA,
        "POISSON_RATIO": POISSON_RATIO,
        "JC_PARAMETER_A": yield_stress_pa,
        "JC_PARAMETER_B": HARDENING_MODULUS_PA,
        "JC_PARAMETER_n": HARDENING_EXPONENT,
        "JC_PARAMETER_C": STRAIN_RATE_COEFFICIENT,
        "JC_PARAMETER_m": THERMAL_SOFTENING_EXPONENT,
        "TEMPERATURE": OFHC_COPPER_DAMAGE.reference_temperature_k,
        "REFERENCE_TEMPERATURE": OFHC_COPPER_DAMAGE.reference_temperature_k,
        "MELD_TEMPERATURE": OFHC_COPPER_DAMAGE.melt_temperature_k,
        "REFERENCE_STRAIN_RATE": OFHC_COPPER_DAMAGE.reference_strain_rate_per_s,
        "TAYLOR_QUINNEY_COEFFICIENT": TAYLOR_QUINNEY_COEFFICIENT,
        "SPECIFIC_HEAT": SPECIFIC_HEAT_J_KG_K,
        "MATERIAL_POINTS_PER_ELEMENT": material_points_per_element,
        "THICKNESS": 1.0,
    }


# Fracture energy per unit crack area (J/m^2), for damage evolution.
#
# !!! VERIFY !!!  This is the least certain constant in the file. Ductile
# copper tears rather than cleaves, so the work of fracture is large and widely
# scattered in the literature; an order-of-magnitude estimate from
# G_f ~ K_IC^2 / E with K_IC ~ 50 MPa*sqrt(m) gives ~2e4 J/m^2, and fully
# ductile values run higher. It sets how fast damage evolves after initiation,
# so it directly controls chip separation. Treat every result that depends on
# it as provisional until it is sourced.
FRACTURE_ENERGY_J_M2 = 1.0e5
