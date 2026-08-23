"""Chip morphology — thickness, tool-chip contact, and the regime classification.

Morphology is the outcome variable. Spec section 5 lists it as such and spec
section 6 makes ``classify_chip_regime`` an observation action scored against
ground truth, on the grounds that whether a model can name the regime is the
cleanest measure of whether it understands the state at all. Everything here
therefore has to be inspectable: the regime edges live in one table with a
cited reason apiece, and the classifier hands back the numbers that decided the
call so a supervisor can cite them as ``evidence``.

The geometry is rake-face geometry *at the snapshot's own time*. Both halves of
that have been got wrong here before, and each is load-bearing:

* **Perpendicular to the face, not to the screen.** Chip thickness is measured
  out from the rake face and contact length along it. Those coincide with
  horizontal and vertical extent only at zero rake, which is the case the bench
  runs, so a vertical-extent estimator survives every check until the first
  raked case.
* **From where the tool is now, not where it started.** ``CaseMetadata`` holds
  one ``tool_tip_m``, written once, and the tool is the thing that moves. The
  tip at a snapshot is ``tool_tip_m + tool_velocity_m_s * time_s``. Measuring
  against the t=0 tip after 500 um of travel does not merely add an offset: it
  adds the travel to every thickness, drops the contact length to zero because
  nothing is within a cell of a plane that has moved on, and inverts the test
  that keeps tool material out of the chip cloud, because the tool body ends up
  on the side the chip is supposed to be.

Chip material is identified by displacement above the original workpiece
surface, as bench/orthogonal_cutting/analyze.py prototyped, and then filtered
to the workpiece side of the current rake face -- the tool occupies the other
half-plane, and telemetry carrying tool points would otherwise report the
tool's height as chip thickness. Every spatial extent is a percentile rather
than a maximum, because MPM ejects isolated points from a free surface
routinely and one of them must not set the outcome variable.

What the regime edges rest on: a continuous chip gives a near-flat cutting
force; a segmented chip gives a strongly *periodic* force, because the force
collapses each time a segment shears and rebuilds as the next forms; a
discontinuous chip gives large *irregular* drops as fragments separate. So
amplitude alone cannot separate the three -- periodicity is what distinguishes
segmented from discontinuous at equal amplitude.

Periodicity alone is not enough either, and this is the confusion the project
exists to resolve rather than to produce. MPM material points cross background
cells at v / h, and that ringing is *regular*, so it scores a high periodicity
and would be claimed as segmentation with the grid-crossing frequency reported
as the segmentation frequency. The defence is that the two live in different
places on the frequency axis, and where they live is computable from the case
metadata alone: a segmentation pitch is of the order of the uncut chip
thickness, while cell crossing is set by the mesh. So the classifier takes the
metadata and requires a periodicity to sit in a plausible band before it will
call it segmentation.

Deliberately not here: the shear angle, in either of its forms
(``diagnostics.merchant`` owns the Merchant relation and the thickness-ratio
route to phi, and duplicating that trigonometry here would give the supervisor
two answers to the same question); anything that reads a solver. Every function
is pure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from quinney.diagnostics.bands import MIN_RESOLVED_BAND_ELEMENTS
from quinney.diagnostics.forces import (
    MIN_OSCILLATION_SAMPLES,
    force_dominant_freq_hz,
    force_engagement_fraction,
    force_oscillation_ratio,
    force_periodicity,
    steady_window,
)
from quinney.telemetry import CaseMetadata, Series, Snapshot

# Height above the original free surface, in background cells, above which a
# point counts as displaced chip material. Grid interpolation and elastic wave
# motion lift undisturbed free-surface points by a fraction of a cell, and a
# tenth of a cell sits above that jitter and far below any real displacement.
CHIP_LIFT_TOLERANCE_CELLS = 0.1

# Distance from the rake face, in background cells, within which chip material
# counts as touching the tool. MPM has no contact algorithm at all -- chip and
# tool interact through the shared background grid, which is what made the
# formulation attractive in spec section 3 -- so a gap smaller than one cell is
# not representable, and a tighter tolerance would report zero contact for a
# chip fully loaded against the rake face.
CONTACT_TOLERANCE_CELLS = 1.0

# Points needed above the original surface before a chip is said to exist.
# Fewer than this cannot be told apart from debris ejected off the free
# surface; with a 50 um cell and a 100 um depth of cut a formed chip carries
# tens of points. bench/orthogonal_cutting/analyze.py used the same floor.
MIN_CHIP_POINTS = 5

# Percentile at which every spatial extent is taken instead of the maximum, so
# that one ejected point cannot set a dimension. The extent is divided back by
# this fraction because both extents are measured from a face the material lies
# against: the distribution runs from zero to the true extent, so the raw 95th
# percentile of a uniformly filled chip is 95% of its thickness, and the
# rescaling removes that bias rather than leaving it in the outcome variable.
EXTENT_PERCENTILE_PCT = 95.0


class ChipRegime(StrEnum):
    """The chip morphology regimes spec section 5 asks the supervisor to name."""

    CONTINUOUS = "continuous"
    SEGMENTED = "segmented"
    DISCONTINUOUS = "discontinuous"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class RegimeEdge:
    """One threshold in the regime classification, with the reason it sits there."""

    value: float
    reason: str


# The regime edges, in one place. Each is defined here once and nowhere else;
# the module constants below are read back out of this table rather than
# retyped, so there is no way for a threshold and its citation to drift apart.
REGIME_EDGES: Mapping[str, RegimeEdge] = MappingProxyType(
    {
        "segment_significance_ratio": RegimeEdge(
            0.05,
            "Oscillation amplitude, as a fraction of the mean cutting force, below "
            "which the force counts as flat and the chip as continuous. Measured "
            "continuous-chip force records in ductile FCC metals fluctuate by under "
            "about 10% peak to peak, and the residual there is machine dynamics and "
            "instrument noise rather than chip formation. Note what that maps to in "
            "this estimator, which is a 5th-to-95th percentile half-spread: for a "
            "*periodic* fluctuation 10% peak to peak is 0.05, while for a *noise-like* "
            "one, whose true peak-to-peak runs to about 6 sigma over a long record, the "
            "same 10% is nearer 0.03. The edge is set for the periodic case because the "
            "alternative to continuous is segmented, which is periodic by definition; "
            "the effect on a noisy record is to be mildly readier to call it "
            "continuous, which is the right direction, since a record with no "
            "periodicity is not a segmented chip whatever its amplitude. This is also "
            "the floor under segment_frequency_hz: below it there is no segmentation "
            "frequency to report, only the largest bin of noise.",
        ),
        "segmented_periodicity_min": RegimeEdge(
            0.50,
            "Normalized autocorrelation at the repeat lag above which the force counts "
            "as periodic, and so the chip as segmented rather than fragmenting. For a "
            "periodic signal buried in noise this quantity is SNR / (1 + SNR), so 0.50 "
            "is exactly the point at which the repeating component carries as much "
            "power as everything else put together: above it the force mainly repeats, "
            "below it the repeat is the minority of what is happening. The measured "
            "estimator sits about 0.03 above that identity here, from maximising over "
            "many lags. The edge is out of chance's reach at the window lengths this "
            "module accepts -- pure noise stays under 0.42 over 400 trials at "
            "forces.MIN_OSCILLATION_SAMPLES, which is what sets that floor.",
        ),
        "discontinuous_oscillation_ratio_min": RegimeEdge(
            0.50,
            "Oscillation amplitude, as a fraction of the mean cutting force, above "
            "which an aperiodic force record counts as fragments separating rather "
            "than as a noisy cut. A discontinuous chip stops being one chip: the force "
            "unloads towards zero between fragments. In this estimator a force that "
            "unloads fully gives a ratio of about 1.0 -- 0.988 for a sine reaching zero "
            "and 1.0 for a square wave -- so 0.50 is a force swinging between half and "
            "one and a half times its own mean, a 100% peak-to-peak modulation. An "
            "attached chip whose thickness merely varies cannot modulate the cutting "
            "force that far without separating. Between this and "
            "segment_significance_ratio, with no periodicity, the record is genuinely "
            "ambiguous and the answer is INDETERMINATE.",
        ),
        "min_engaged_force_fraction": RegimeEdge(
            0.10,
            "Steady mean force as a fraction of the largest force in the same window, "
            "below which the window is taken to contain no cutting. The tool "
            "approaches through air at zero force before it cuts, so a window that "
            "predates first contact averages near zero against whatever noise is "
            "present; once cutting, the mean stays within a small factor of the peak "
            "even for a fully unloading discontinuous chip, whose mean is about half "
            "of it. Stated as a fraction rather than in newtons because absolute force "
            "scales with depth of cut and cut width.",
        ),
        "max_segment_pitch_t0": RegimeEdge(
            5.0,
            "Longest chip-segment pitch, in multiples of the uncut chip thickness, "
            "that still counts as segmentation -- it sets the *low* edge of the "
            "plausible frequency band, since pitch and frequency are reciprocal "
            "through f = v / pitch. Serrated-chip segment spacing is of the order of "
            "the uncut chip thickness, typically a fraction of it to a small multiple. "
            "A force undulation whose period corresponds to the tool advancing five "
            "uncut thicknesses is something slower than chip formation -- built-up "
            "edge cycling, or the machine -- and calling it segmentation would put a "
            "number on the outcome variable that no chip produced.",
        ),
        "min_resolved_segment_cells": RegimeEdge(
            MIN_RESOLVED_BAND_ELEMENTS,
            "Shortest chip-segment pitch, in background cells, that the mesh can "
            "resolve -- it sets the *high* edge of the plausible frequency band. This "
            "is spec section 5's rule that a feature under about three elements across "
            "is mesh-resolution-limited rather than physics, applied to a segment pitch "
            "instead of a band width; the value is imported from "
            "bands.MIN_RESOLVED_BAND_ELEMENTS rather than retyped, because it is the "
            "same rule. It is also what separates segmentation from the artefact that "
            "most looks like it: material points cross background cells at v / h, which "
            "is regular enough to score a high periodicity, and this edge puts that "
            "frequency a clear factor of three outside the band.",
        ),
    }
)

SEGMENT_SIGNIFICANCE_RATIO = REGIME_EDGES["segment_significance_ratio"].value
SEGMENTED_PERIODICITY_MIN = REGIME_EDGES["segmented_periodicity_min"].value
DISCONTINUOUS_OSCILLATION_RATIO_MIN = REGIME_EDGES["discontinuous_oscillation_ratio_min"].value
MIN_ENGAGED_FORCE_FRACTION = REGIME_EDGES["min_engaged_force_fraction"].value
MAX_SEGMENT_PITCH_T0 = REGIME_EDGES["max_segment_pitch_t0"].value
MIN_RESOLVED_SEGMENT_CELLS = REGIME_EDGES["min_resolved_segment_cells"].value


@dataclass(frozen=True)
class RegimeClassification:
    """A regime call together with every number that produced it.

    Spec section 6 requires ``evidence`` on every action, and a bare enum
    cannot be audited: a correct call from a wrong reading is indistinguishable
    from a correct call from a correct one. A field is NaN or None when the
    record did not support computing it, which is itself the reason for an
    ``INDETERMINATE`` call.

    ``dominant_freq_hz`` is the strongest oscillation whatever it turned out to
    be; ``segment_frequency_hz`` is that same number only when it was accepted
    as segmentation. The two differing is the evidence that a periodicity was
    found and rejected as an artefact, which is a different state from no
    periodicity at all and has to be distinguishable in the record.
    """

    regime: ChipRegime
    oscillation_ratio: float
    periodicity: float
    engagement_fraction: float
    dominant_freq_hz: float | None
    segment_frequency_hz: float | None
    reason: str


def plausible_segment_frequency_band_hz(metadata: CaseMetadata) -> tuple[float, float]:
    """Frequencies at which this case could physically segment, in hertz.

    A segment forms as the tool advances by one segment pitch, so pitch and
    frequency are related by f = v / pitch, and both edges of the band are
    pitches: the longest that is still chip formation rather than something
    slower, and the shortest the mesh can resolve.

    The band exists to separate segmentation from cell-crossing ringing, which
    sits at v / h and is regular enough to be mistaken for it. If the mesh is
    coarse enough relative to the depth of cut that the band comes out empty,
    that is a true statement about the case -- it cannot resolve segmentation
    at all -- and every periodicity it produces will be rejected.
    """
    speed_m_s = metadata.cutting_speed_m_s
    lowest_hz = speed_m_s / (MAX_SEGMENT_PITCH_T0 * metadata.depth_of_cut_m)
    highest_hz = speed_m_s / (MIN_RESOLVED_SEGMENT_CELLS * metadata.cell_size_m)
    return lowest_hz, highest_hz


def _face_axes(rake_angle_deg: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit vectors up the rake face and out from it, from the tool tip.

    Positive rake leans the face over the uncut material, making the tool wedge
    sharper, so the along-face axis tilts towards the workpiece as it rises.
    The offset axis is the outward normal, pointing away from the tool body, so
    chip material has a non-negative offset and the tool's own half-plane a
    negative one.
    """
    alpha_rad = math.radians(rake_angle_deg)
    along = np.array([-math.sin(alpha_rad), math.cos(alpha_rad)])
    return along, np.array([-math.cos(alpha_rad), -math.sin(alpha_rad)])


def tool_tip_at_m(metadata: CaseMetadata, time_s: float) -> NDArray[np.float64]:
    """Where the tool tip is at a given time, in metres.

    ``CaseMetadata.tool_tip_m`` is the tip at t=0 and the tool is the thing
    that moves, so every geometric diagnostic has to advance it. Raises
    ``ValueError`` when the travel direction was never recorded: the contract
    NaN-fills ``tool_velocity_m_s`` for an episode written before schema 1.1,
    and measuring against a stationary tip would be wrong by however far the
    tool has actually gone -- 500 um on the bench case, against a chip 250 um
    thick.
    """
    velocity_m_s = np.asarray(metadata.tool_velocity_m_s, dtype=np.float64)
    if not np.all(np.isfinite(velocity_m_s)):
        raise ValueError(
            "CaseMetadata.tool_velocity_m_s is not finite: this episode predates the "
            "schema version that recorded the tool's travel direction, so where the "
            "tip is at this snapshot is unknown, and measuring against the t=0 tip "
            "would be wrong by the whole travel distance"
        )
    return np.asarray(metadata.tool_tip_m, dtype=np.float64) + velocity_m_s * time_s


def _face_coordinates(
    snapshot: Snapshot, metadata: CaseMetadata
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Every point as (distance along the rake face from the tip, offset out from it).

    Measured from the tip's position at this snapshot's own time, not at t=0.
    """
    along_axis, offset_axis = _face_axes(metadata.rake_angle_deg)
    relative_m = snapshot.coords_m - tool_tip_at_m(metadata, snapshot.time_s)
    return relative_m @ along_axis, relative_m @ offset_axis


def _chip_mask(snapshot: Snapshot, metadata: CaseMetadata) -> NDArray[np.bool_]:
    """Points that are displaced chip material rather than workpiece, tool, or noise."""
    along_m, offset_m = _face_coordinates(snapshot, metadata)
    surface_y_m = metadata.workpiece_bounds_m[3]
    lifted_m = surface_y_m + CHIP_LIFT_TOLERANCE_CELLS * metadata.cell_size_m
    return (snapshot.coords_m[:, 1] > lifted_m) & (along_m > 0.0) & (offset_m >= 0.0)


def _has_chip(snapshot: Snapshot, metadata: CaseMetadata) -> bool:
    return int(np.count_nonzero(_chip_mask(snapshot, metadata))) >= MIN_CHIP_POINTS


def _robust_extent_m(distances_m: NDArray[np.float64]) -> float:
    """Extent of a distribution that starts at zero, immune to a few stray points."""
    fraction = EXTENT_PERCENTILE_PCT / 100.0
    return float(np.percentile(distances_m, EXTENT_PERCENTILE_PCT)) / fraction


def chip_thickness_m(snapshot: Snapshot, metadata: CaseMetadata) -> float:
    """Thickness of the formed chip, measured perpendicular to the rake face, in metres.

    Chip material is what has been displaced above the original workpiece
    surface and lies on the workpiece side of the rake face *as the face stands
    at this snapshot's time*. Its thickness is the extent of that cloud out
    from the face, which is the direction chip thickness is defined in because
    the chip flows along the face.

    Raises ``ValueError`` when fewer than ``MIN_CHIP_POINTS`` points have been
    displaced: a run with no chip has no chip thickness, and returning zero
    would make the thickness ratio infinite rather than absent.

    Two biases are known and are not corrected here; see
    ``chip_thickness_ratio``, which is what the assembly layer consumes.
    """
    chip = _chip_mask(snapshot, metadata)
    n_chip = int(np.count_nonzero(chip))
    if n_chip < MIN_CHIP_POINTS:
        raise ValueError(
            f"no chip has formed: {n_chip} points lie above the original workpiece "
            f"surface, under the {MIN_CHIP_POINTS} needed to tell a chip from ejected debris"
        )
    _along_m, offset_m = _face_coordinates(snapshot, metadata)
    return _robust_extent_m(offset_m[chip])


def chip_thickness_ratio(snapshot: Snapshot, metadata: CaseMetadata) -> float:
    """Cutting ratio r = t0 / tc — uncut thickness over chip thickness (0..1 for a real chip).

    A chip is always thicker than the layer it came from, because the material
    is compressed along the cutting direction as it shears, so r below 1 is the
    physical case and r at or above 1 means either no chip has formed or the
    thickness measurement has picked up something that is not chip.

    This is the number the assembly layer feeds to
    ``merchant.shear_angle_from_thickness_ratio_deg``, so its two biases matter
    downstream and are stated here rather than buried:

    * A chip that has curled away from the rake face is measured across its
      curl, so r reads low and the shear angle with it. Not removed here;
      restricting the offset percentile to points within the tool-chip contact
      band would remove most of it, since the curl develops beyond contact.
    * Material piled ahead of the tool that is above the original surface but
      has not yet sheared into chip is counted as chip and thickens it. Not
      removed here; this telemetry does carry what would separate the two --
      ``Snapshot.velocity_m_s``, since chip material flows up the face while
      pile-up moves with the tool, and ``Snapshot.eps_p``, since chip material
      has passed through the shear zone and pile-up has not.

    Both are left for a pass that can calibrate them against a real run rather
    than against synthetic clouds, because either filter can remove real chip
    if its threshold is set from a fixture.
    """
    return metadata.depth_of_cut_m / chip_thickness_m(snapshot, metadata)


def tool_chip_contact_length_m(snapshot: Snapshot, metadata: CaseMetadata) -> float:
    """Length of rake face the chip is in contact with, measured from the tip, in metres.

    Contact length sets the area over which friction does work, and so the
    frictional heat that feeds thermal softening at the interface. Material
    below the tool tip is the machined surface, not chip flowing over the rake
    face, and is excluded.

    Returns 0.0 in two cases, both of which are answers rather than missing
    measurements. Nothing within ``CONTACT_TOLERANCE_CELLS`` of the face is a
    chip that has left the tool. And no chip at all is a tool that has not made
    one yet: the bench case stands the tool off by exactly one cell, so before
    it touches anything the undisturbed workpiece face sits inside the contact
    tolerance and its full height above the tip would otherwise be reported as
    tool-chip contact.
    """
    if not _has_chip(snapshot, metadata):
        return 0.0
    along_m, offset_m = _face_coordinates(snapshot, metadata)
    tolerance_m = CONTACT_TOLERANCE_CELLS * metadata.cell_size_m
    touching = (along_m >= 0.0) & (offset_m >= 0.0) & (offset_m <= tolerance_m)
    if not np.any(touching):
        return 0.0
    return _robust_extent_m(along_m[touching])


def segment_frequency_hz(
    series: Series, metadata: CaseMetadata, window: slice | None = None
) -> float | None:
    """Frequency at which the chip segments, in hertz, or None if it does not segment.

    This is exactly ``classify_chip_regime(...).segment_frequency_hz``, and it
    delegates rather than re-deriving so that the two can never disagree --
    which they did in an earlier form, this function reporting a confident
    frequency for a record the classifier was calling ``DISCONTINUOUS``.

    ``None`` covers every way there is no segmentation frequency: a flat force,
    an aperiodic one, a periodicity at a frequency no chip could produce, a
    tool that is not engaged, and a window too short to tell. Which one it was
    is in ``classify_chip_regime(...).reason``.
    """
    return classify_chip_regime(series, metadata, window).segment_frequency_hz


def classify_chip_regime(
    series: Series, metadata: CaseMetadata, window: slice | None = None
) -> RegimeClassification:
    """Name the chip morphology regime from the cutting force record.

    Amplitude first, then periodicity, then plausibility. An oscillation that
    does not clear ``SEGMENT_SIGNIFICANCE_RATIO`` is a flat force and so a
    continuous chip. Above it, a record that repeats *at a frequency this case
    could segment at* is a segmented chip whatever its amplitude, because a
    force that collapses on a clock is one chip shearing repeatedly. A record
    that repeats outside that band is not called segmentation at all -- cell
    crossing at v / h is regular, and reporting it as a segmentation frequency
    would be the instrument producing the confusion it exists to resolve. A
    record that does not repeat is a discontinuous chip once its amplitude
    reaches ``DISCONTINUOUS_OSCILLATION_RATIO_MIN``, which takes the near-total
    unloading that separating fragments produce.

    ``INDETERMINATE`` is a real answer and is returned rather than guessed
    past: when the window contains no cutting, when it is too short to tell
    periodic from irregular, when the periodicity is at an implausible
    frequency, and when the record oscillates too much to be flat but is
    neither periodic nor severe enough to be fragmenting.
    """
    resolved = window if window is not None else steady_window(series)
    engagement = force_engagement_fraction(series, resolved)
    if engagement < MIN_ENGAGED_FORCE_FRACTION:
        return RegimeClassification(
            regime=ChipRegime.INDETERMINATE,
            oscillation_ratio=float("nan"),
            periodicity=float("nan"),
            engagement_fraction=engagement,
            dominant_freq_hz=None,
            segment_frequency_hz=None,
            reason=(
                f"steady mean force is {engagement:.2f} of the window's peak, under the "
                f"{MIN_ENGAGED_FORCE_FRACTION:.2f} engagement floor: the window contains "
                "no cutting"
            ),
        )

    ratio = force_oscillation_ratio(series, resolved)
    n_steady = len(series.step[resolved])
    if n_steady < MIN_OSCILLATION_SAMPLES:
        return RegimeClassification(
            regime=ChipRegime.INDETERMINATE,
            oscillation_ratio=ratio,
            periodicity=float("nan"),
            engagement_fraction=engagement,
            dominant_freq_hz=None,
            segment_frequency_hz=None,
            reason=(
                f"{n_steady} steady samples is under the {MIN_OSCILLATION_SAMPLES} needed to "
                "tell a periodic force from an irregular one, and amplitude alone cannot"
            ),
        )

    # Both are computed before any branch, so that a window an FFT cannot
    # honestly be taken over raises whatever the answer would have been. Fail
    # loud must not depend on which way the classification came out.
    periodicity = force_periodicity(series, resolved)
    dominant_hz = force_dominant_freq_hz(series, resolved)
    lowest_hz, highest_hz = plausible_segment_frequency_band_hz(metadata)

    common = {
        "oscillation_ratio": ratio,
        "periodicity": periodicity,
        "engagement_fraction": engagement,
        "dominant_freq_hz": dominant_hz,
        "segment_frequency_hz": None,
    }

    if ratio <= SEGMENT_SIGNIFICANCE_RATIO:
        return RegimeClassification(
            regime=ChipRegime.CONTINUOUS,
            **common,
            reason=(
                f"oscillation is {ratio:.3f} of the mean force, at or under the "
                f"{SEGMENT_SIGNIFICANCE_RATIO:.2f} significance floor: the force is flat"
            ),
        )

    if periodicity >= SEGMENTED_PERIODICITY_MIN:
        if lowest_hz <= dominant_hz <= highest_hz:
            return RegimeClassification(
                regime=ChipRegime.SEGMENTED,
                **{**common, "segment_frequency_hz": dominant_hz},
                reason=(
                    f"oscillation is {ratio:.3f} of the mean force and its periodicity "
                    f"{periodicity:.2f} is at or over {SEGMENTED_PERIODICITY_MIN:.2f}: the "
                    f"force repeats, at {dominant_hz / 1e3:.0f} kHz, inside the "
                    f"{lowest_hz / 1e3:.0f}-{highest_hz / 1e3:.0f} kHz band this case "
                    "could segment at"
                ),
            )
        return RegimeClassification(
            regime=ChipRegime.INDETERMINATE,
            **common,
            reason=(
                f"the force repeats (periodicity {periodicity:.2f}) but at "
                f"{dominant_hz / 1e3:.0f} kHz, outside the "
                f"{lowest_hz / 1e3:.0f}-{highest_hz / 1e3:.0f} kHz band this case could "
                "segment at: too regular to be fragmentation, and with cell crossing at "
                f"{metadata.cutting_speed_m_s / metadata.cell_size_m / 1e3:.0f} kHz this "
                "is more likely numerical than chip formation"
            ),
        )

    if ratio >= DISCONTINUOUS_OSCILLATION_RATIO_MIN:
        return RegimeClassification(
            regime=ChipRegime.DISCONTINUOUS,
            **common,
            reason=(
                f"oscillation is {ratio:.3f} of the mean force, at or over "
                f"{DISCONTINUOUS_OSCILLATION_RATIO_MIN:.2f}, with periodicity "
                f"{periodicity:.2f} under {SEGMENTED_PERIODICITY_MIN:.2f}: large "
                "irregular unloading, not a periodic collapse"
            ),
        )

    return RegimeClassification(
        regime=ChipRegime.INDETERMINATE,
        **common,
        reason=(
            f"oscillation {ratio:.3f} is over the {SEGMENT_SIGNIFICANCE_RATIO:.2f} "
            f"significance floor but under the {DISCONTINUOUS_OSCILLATION_RATIO_MIN:.2f} "
            f"fragmentation edge, and periodicity {periodicity:.2f} is under "
            f"{SEGMENTED_PERIODICITY_MIN:.2f}: neither flat, periodic, nor separating"
        ),
    )
