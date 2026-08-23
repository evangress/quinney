"""Merchant's shear-plane geometry — the reference angle a band is judged against.

`bands.band_orientation_deg` measures where the high-strain region actually
lies. On its own that is a number with nothing to compare it to. Merchant's
minimum-work solution supplies the comparison:

    phi = 45 + alpha/2 - beta/2      alpha rake angle, beta friction angle

Both inputs are derivable from telemetry the run already produces — beta from
the measured cutting and thrust forces, alpha from the case metadata — so the
reference angle costs no extra instrumentation, and a band sitting 20 degrees
off it is a fact about the simulation rather than a fact about the analyst's
expectations.

Chip thickness gives a second, independent route to the same angle. The two
agreeing means the chip is in steady state and the shear-plane idealization
holds; the two disagreeing is itself the diagnostic, and it is the reason both
routes live here rather than one.

Deliberately absent: any of the corrected shear-angle theories (Lee-Shaffer,
Oxley). Merchant is known to under-predict phi for real metals by 5-15
degrees. It is kept because it is the *stated* reference in spec section 5 and
because a reference the supervisor can reason about beats a more accurate one
it cannot. Anything better belongs in the prompt as a stated bias, not hidden
inside a fitted constant here.

Every function is pure trigonometry: floats in, float out, no state, no I/O.
"""

from __future__ import annotations

import numpy as np

# A rake angle at or past vertical is not a tool geometry, it is a typo. At
# exactly +/-90 the rake face is parallel to the cutting direction and the
# force resolution below degenerates: the friction and normal components swap
# roles, and the shear-plane relations stop describing anything.
MAX_RAKE_ANGLE_DEG = 90.0


def _validate_rake_angle_deg(rake_angle_deg: float) -> None:
    """Raise unless the rake angle is a tool geometry. Positive is away from the work."""
    if not abs(rake_angle_deg) < MAX_RAKE_ANGLE_DEG:
        raise ValueError(f"rake angle must lie strictly within +/-90 deg, got {rake_angle_deg}")


def merchant_shear_angle_deg(rake_angle_deg: float, friction_angle_deg: float) -> float:
    """Predicted shear-plane angle from the cutting direction, in degrees.

    Merchant's result from minimizing cutting work: phi = 45 + alpha/2 -
    beta/2. More friction shears the material on a shallower plane and thickens
    the chip; more positive rake steepens it.

    The result is returned unclamped. With beta - alpha past 90 degrees the
    formula gives a non-positive angle, which is not a shear plane — it means
    the measured force ratio has left the range where a single-shear-plane
    model says anything, and hiding that behind a clamp would let the
    supervisor compare a real band against a fabricated reference.
    """
    _validate_rake_angle_deg(rake_angle_deg)
    return 45.0 + 0.5 * rake_angle_deg - 0.5 * friction_angle_deg


def friction_angle_deg(
    cutting_force_n: float,
    thrust_force_n: float,
    rake_angle_deg: float,
) -> float:
    """Rake-face friction angle beta from the measured force pair, in degrees.

    Resolving the tool force onto the rake face gives the friction and normal
    components

        F = Fc sin(alpha) + Ft cos(alpha)
        N = Fc cos(alpha) - Ft sin(alpha)

    and beta = arctan(F/N), which collapses to beta = alpha + arctan(Ft/Fc).
    This is what makes the Merchant reference derivable from telemetry alone:
    the force signal is already recorded for the force diagnostics.
    """
    _validate_rake_angle_deg(rake_angle_deg)
    rake = float(np.radians(rake_angle_deg))
    if cutting_force_n <= 0.0:
        raise ValueError(
            f"cutting force must be positive to read a force ratio, got {cutting_force_n}"
        )

    friction_force_n = cutting_force_n * np.sin(rake) + thrust_force_n * np.cos(rake)
    normal_force_n = cutting_force_n * np.cos(rake) - thrust_force_n * np.sin(rake)
    if normal_force_n <= 0.0:
        raise ValueError(
            "resolved force pulls the chip off the rake face "
            f"(N = {normal_force_n:.4g} N); the force signal is not a cutting signal"
        )

    return float(np.degrees(np.arctan(friction_force_n / normal_force_n)))


def shear_angle_from_thickness_ratio_deg(r: float, rake_angle_deg: float) -> float:
    """Shear-plane angle implied by the measured chip thickness ratio, in degrees.

    ``r`` is the **cutting ratio, uncut thickness over chip thickness**
    (``t_uncut / t_chip``), 0..1 for a chip that thickens, which is the usual
    case. The inverse convention (chip over uncut, always >= 1) is at least as
    common in the literature and this function cannot detect the mistake: an
    inverted r = 2.5 returns a perfectly plausible 68.2 deg where the truth is
    21.8 deg. Check which way round the telemetry reports it.

    Shear-plane geometry gives

        tan(phi) = r cos(alpha) / (1 - r sin(alpha))

    This route uses only the chip's shape, so it is independent of the force
    route through `friction_angle_deg`. Comparing the two is a steady-state
    check: a segmenting chip has no single thickness and the two answers part
    company.

    ``arctan2`` handles the denominator passing through zero, which happens at
    positive rake for a chip thinner than the uncut layer, and returns the
    obtuse angle rather than a sign flip.
    """
    _validate_rake_angle_deg(rake_angle_deg)
    rake = float(np.radians(rake_angle_deg))
    if r <= 0.0:
        raise ValueError(f"chip thickness ratio must be positive, got {r}")

    return float(np.degrees(np.arctan2(r * np.cos(rake), 1.0 - r * np.sin(rake))))
