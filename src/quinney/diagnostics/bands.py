"""Shear-band estimators — is the localization physical, or is it mesh collapse?

This is the diagnostic group the project exists for (spec section 1). Physical
shear localization and numerical degeneration both show up as a region of
enormous plastic strain, and no single number separates them. Several weakly
correlated ones do:

* **coherence** — is the high-strain set spatially contiguous, or sprinkled
  through the body? A real band is a connected sheet; numerical erosion is
  diffuse.
* **orientation** — does the set have a direction at all, and does that
  direction match the Merchant shear angle (`diagnostics.merchant`)? Noise has
  a principal axis too, so this one is allowed to answer "no direction".
* **width** — a band thinner than `MIN_RESOLVED_BAND_ELEMENTS` background
  cells is reporting the mesh, not the material.
* **temperature alignment and gradient** — adiabatic heating must corroborate
  the band. High strain with no heat on it is not doing plastic work.

**Coordinate contract.** `coords_m` is an (n, 2) array of material-point
positions in metres, column 0 along the cutting direction and column 1
perpendicular to it in the cutting plane. Every angle here is measured
counter-clockwise from that cutting direction. Callers assembling telemetry
must honour this; a swapped pair silently reports the complement of every band
angle.

Every estimator takes plain arrays: a point cloud, a boolean membership mask,
and a length scale. Nothing about them is specific to plastic strain, and that
is deliberate — the same coherence question is asked of *deleted* material
points (clustered along a separation surface is physical chip separation,
diffuse through the body is numerical erosion), and one estimator answering
both keeps the two answers comparable.

**Contamination is structural, not occasional.** `high_strain_mask` selects a
fixed decile, so whenever the primary shear zone occupies less than a tenth of
the body the rest of the mask is by construction the next-most-strained
material wherever it happens to be — the rake-face secondary zone, the free
surface, isolated degenerate points. Every estimator here is therefore built to
survive a contaminated mask rather than to assume a clean one: `band_coherence`
is a ratio that contamination dilutes rather than distorts, and orientation and
width are fitted only to the mask's dominant contiguous, collinear part (see
`_band_inliers`) with a quantile extent rather than a second moment.

Deliberately absent: any verdict. Nothing here returns "physical" or
"numerical". Weighing these signals against each other is the supervisor's job
and the thing being measured; a threshold rule baked in at this level would
quietly answer the question the experiment is asking. Also absent: segmenting a
mask into several bands and reporting each. A mask holding two comparable
structures gets `None` from the orientation and width estimators, not an
average of them — averaging two separated clusters returns the direction of the
*separation between* them, which is unrelated to either one's orientation and
is the worst failure this module can produce, because it is confident and
actionable and wrong.

Absent band, three different answers, because the three have different honest
ones. Coherence returns 0.0 — an empty or near-empty set is genuinely not
clustered, and that is the state a deletion mask sits in for most of a healthy
run. Orientation and width return `None` — before localization there is no band
to orient or measure, which is a normal physical state and not malformed input.
Malformed input (mismatched lengths, non-positive cell size) still raises.

Every function is pure: arrays in, number out, no state, no I/O, no RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

# The band candidate set is the top decile of equivalent plastic strain. The
# primary shear zone plus the secondary zone on the rake face is a small
# fraction of a workpiece that is mostly undeformed bulk; a looser cut drags in
# the uniformly strained chip body and blurs the band it is trying to isolate,
# while a tighter one leaves too few points to fit an axis to.
HIGH_STRAIN_PERCENTILE = 90.0

# Spec section 5: a band under about three background cells wide is
# mesh-resolution-limited, not physics. Defined here and imported everywhere
# else, including tests.
MIN_RESOLVED_BAND_ELEMENTS = 3.0

# Coherence neighbourhood radius, in background cells. On a lattice at the cell
# spacing this is exactly the eight-point Moore neighbourhood (faces at 1.0,
# diagonals at 1.41, the next ring at 2.0). Fixing it in *physical* units
# rather than in points keeps the score comparable between runs that carry
# different numbers of material points per cell.
NEIGHBOURHOOD_CELLS = 1.5

# Fewer members than this cannot be said to have a shape: two points always lie
# on a perfect line and would report a spurious orientation with a perfect
# aspect ratio.
MIN_BAND_POINTS = 3

# Orientation is reported only when the minor principal moment is below this
# fraction of the major one. sqrt(0.25) = 0.5, so the cloud must be at least
# twice as long as it is wide — the weakest elongation that still names a
# direction rather than the direction of noise.
MAX_ISOTROPIC_EIGENVALUE_RATIO = 0.25

# Band width is the 5th-to-95th percentile span of the perpendicular offset,
# divided by 0.90 because that span covers 90% of a uniformly filled band.
# Exact for a uniform slab and, unlike a second moment, unmoved by strays: the
# second moment averages *squared* offsets and so has a breakdown point of
# zero, where this one tolerates a twentieth of the mask per side.
WIDTH_QUANTILE_LOW_PCT = 5.0
WIDTH_QUANTILE_HIGH_PCT = 95.0
# Derived, never re-typed: moving either percentile without rescaling this
# would mis-size every width, and no assertion at a 10% tolerance would notice
# the 6% shift that changing 95.0 to 90.0 produces.
UNIFORM_SLAB_QUANTILE_SPAN = (WIDTH_QUANTILE_HIGH_PCT - WIDTH_QUANTILE_LOW_PCT) / 100.0

# Two masked points belong to the same structure if they lie within this
# multiple of the mask's own median nearest-neighbour spacing. Four spacings
# bridges an eroded stretch a couple of cells long without reaching across the
# empty bulk to a separate cluster.
#
# Measuring it in the mask's own spacings makes it independent of point
# *count*, which is the wrong invariance for the question it decides: whether
# two structures are separate is a question in metres, and this radius works
# out at 2 cells for 2x2 material points per cell but 8 cells for one point per
# two cells. `cell_size_m`, which this module already takes elsewhere, would
# have been the density-free choice. At the bench case's density the margin is
# comfortable — the separations that matter are 10 cells and up — so the value
# stands and the coupling is recorded rather than hidden.
BAND_LINK_SPACINGS = 4.0

# A masked point counts as part of the band if it lies within this multiple of
# the band's own width of the fitted axis: a corridor one full width either
# side, which keeps a ragged edge and a collinear continuation without reaching
# a structure lying clear of the band.
#
# Was 2.0. Measured over a sweep of elongated contaminants, halving it cuts the
# fits that come back more than 10 degrees wrong from 35 to 15 and leaves every
# clean case's error unchanged to two decimal places — a +/-2w corridor around
# a seed at the anisotropy gate is as wide as that seed is long, and admits
# whatever crosses it.
#
# It multiplies `_perpendicular_extent_m`, which reads up to 45% short for a
# band sampled by two or three point rows, so the corridor is tightest exactly
# where the band is thinnest and a refusal costs most. The `max(width, spacing)`
# floor in `_band_inliers` covers the single-row case only; between two and four
# rows the corridor is narrower than this factor alone implies.
INLIER_HALF_WIDTHS = 1.0

# The fitted band must account for at least this fraction of the masked points.
# Below three quarters the mask holds a second structure of comparable weight,
# and naming one axis would misrepresent it.
MIN_ON_AXIS_FRACTION = 0.75

# Neighbours used for the temperature gradient. On a regular 2D point lattice
# the four nearest are the face neighbours, so the difference set always spans
# both coordinate directions; a single-nearest-neighbour estimate can pick the
# along-band pair every time and report no gradient across a band edge that is
# actually steep.
GRADIENT_NEIGHBOURS = 4


def _validated_coords(coords_m: ArrayLike) -> NDArray[np.float64]:
    """Point positions as an (n, 2) float array, x along the cutting direction."""
    coords = np.asarray(coords_m, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords_m must have shape (n, 2), got {coords.shape}")
    return coords


def _validated_mask(mask: ArrayLike, n_points: int) -> NDArray[np.bool_]:
    """Membership mask as a boolean array aligned with the point cloud."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n_points,):
        raise ValueError(f"mask must have shape ({n_points},), got {mask.shape}")
    return mask


def _validated_cell_size_m(cell_size_m: float) -> float:
    """Background grid cell size, which every length here is reported against."""
    if not cell_size_m > 0.0:
        raise ValueError(f"cell_size_m must be positive, got {cell_size_m}")
    return float(cell_size_m)


def _principal_moments(
    points_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues (descending, in m^2) and eigenvectors of the second-moment tensor.

    Population moments, not sample moments: these are the physical second
    moments of the point set about its centroid. Eigenvalues are clamped at
    zero because a set lying on a perfect line has a minor moment of exactly
    zero, which LAPACK may return as a tiny negative number: nothing downstream
    takes a square root of it any more, but a negative second moment is not a
    second moment, and it would surface as an anisotropy ratio below zero.
    """
    centred = points_m - points_m.mean(axis=0)
    covariance = centred.T @ centred / len(points_m)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return np.maximum(eigenvalues[order], 0.0), eigenvectors[:, order]


def _perpendicular_extent_m(
    points_m: NDArray[np.float64],
    minor_axis: NDArray[np.float64],
    origin_m: NDArray[np.float64],
) -> float:
    """Width of the uniform slab whose 5-95 percentile span matches these points."""
    perpendicular = (points_m - origin_m) @ minor_axis
    span = float(
        np.percentile(perpendicular, WIDTH_QUANTILE_HIGH_PCT)
        - np.percentile(perpendicular, WIDTH_QUANTILE_LOW_PCT)
    )
    return span / UNIFORM_SLAB_QUANTILE_SPAN


class BandRefusal(StrEnum):
    """Why no single band could be fitted to a mask.

    A bare ``None`` from `band_orientation_deg` is ambiguous in a way that
    matters: it is also the answer for a diffuse, isotropic high-strain cloud,
    which is the *mesh-collapse signature itself*. So a refusal that the
    supervisor cannot attribute does not fail to a neutral state — it fails
    toward the numerical verdict, the false-positive direction for the one
    question this project exists to answer. These say which it was.

    A `StrEnum` so the value serializes into the episode record as a plain
    string and stays readable a year later.
    """

    TOO_FEW_POINTS = "too_few_points"
    """Fewer than `MIN_BAND_POINTS` masked. Nothing has localized yet."""

    COINCIDENT_POINTS = "coincident_points"
    """Most masked points share a position. Degenerate telemetry, not a band."""

    SEED_NOT_ORIENTED = "seed_not_oriented"
    """The dominant contiguous piece is round. This is the diffuse,
    scattered-localization case — the one that reads as mesh collapse."""

    SECOND_STRUCTURE = "second_structure"
    """The mask holds two structures neither of which lies on the other's axis.
    Both may be perfectly real physics; there is simply no single angle."""

    BAND_NOT_ORIENTED = "band_not_oriented"
    """The corridor admitted enough off-axis material that the fitted set is no
    longer elongated, so its minor axis — and any width taken along it — would
    be arbitrary."""


@dataclass(frozen=True)
class _BandCandidate:
    """Outcome of fitting one band to a mask: the fitted set, or why not."""

    inliers: NDArray[np.bool_] | None
    on_axis_fraction: float
    refusal: BandRefusal | None


def _band_inliers(coords_m: NDArray[np.float64], mask: NDArray[np.bool_]) -> _BandCandidate:
    """The masked points that form one band, or the reason they do not.

    Four gates, each closing a distinct way a raw principal-axis fit gets a
    confident wrong answer on a mask that holds more than the band:

    1. **Seed on the dominant contiguous piece.** Masked points within
       `BAND_LINK_SPACINGS` median spacings of each other are one structure;
       the largest such structure seeds the fit, so a separated cluster cannot
       drag the axis toward the direction of its own separation.
    2. **Require the seed to have a direction.** A round seed's principal axis
       is noise, and a corridor built along it sweeps in whatever happens to lie
       that way — worth a 90-degree error in the sweep that found it.
    3. **Require the seed to explain the rest.** At least `MIN_ON_AXIS_FRACTION`
       of the masked points must lie within `INLIER_HALF_WIDTHS` band widths of
       the seed's axis. Points outside are dropped from the fit — which is what
       makes it survive the structurally guaranteed strays — but if too many lie
       outside, the mask holds a second structure and no single axis describes
       it.
    4. **Require the fitted set to still be a band.** The corridor admits
       collinear material, and enough of it off-axis leaves an aggregate that is
       round or whose principal axes have swapped. Gating here rather than in
       the callers is what keeps orientation and width agreeing: without it,
       orientation refuses on a swapped set while width happily reports the
       band's *length* over 0.90 as its thickness.

    Steps 1 and 2 are connectivity tests, 3 and 4 shape tests, and both kinds
    are needed: a band eroded into two collinear halves is one band and passes
    step 3 on either half, while two clusters offset from each other fail it
    however well connected each is internally.
    """
    empty = np.zeros(len(coords_m), dtype=bool)

    if int(mask.sum()) < MIN_BAND_POINTS:
        return _BandCandidate(None, 0.0, BandRefusal.TOO_FEW_POINTS)
    points = coords_m[mask]

    tree = KDTree(points)
    spacing_m = float(np.median(tree.query(points, k=2)[0][:, 1]))
    if spacing_m <= 0.0:
        # More than half the masked points are coincident. That is degenerate
        # telemetry, not a band, and there is no spacing to build a graph on.
        return _BandCandidate(None, 0.0, BandRefusal.COINCIDENT_POINTS)

    pairs = tree.query_pairs(r=BAND_LINK_SPACINGS * spacing_m, output_type="ndarray")
    if len(pairs) == 0:
        # Every masked point is isolated: a maximally diffuse set.
        return _BandCandidate(None, 0.0, BandRefusal.SEED_NOT_ORIENTED)
    adjacency = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(len(points), len(points)),
    )
    _, labels = connected_components(adjacency, directed=False)
    seed = labels == int(np.bincount(labels).argmax())
    if int(seed.sum()) < MIN_BAND_POINTS:
        return _BandCandidate(None, 0.0, BandRefusal.SEED_NOT_ORIENTED)

    eigenvalues, eigenvectors = _principal_moments(points[seed])
    if eigenvalues[0] <= 0.0 or eigenvalues[1] / eigenvalues[0] > MAX_ISOTROPIC_EIGENVALUE_RATIO:
        return _BandCandidate(None, 0.0, BandRefusal.SEED_NOT_ORIENTED)

    centre_m = points[seed].mean(axis=0)
    width_m = _perpendicular_extent_m(points[seed], eigenvectors[:, 1], centre_m)
    # A seed one point-row thick has zero quantile span; its own point spacing
    # is the narrowest corridor that can still admit the band's ragged edge.
    corridor_m = INLIER_HALF_WIDTHS * max(width_m, spacing_m)

    on_axis = np.abs((points - centre_m) @ eigenvectors[:, 1]) <= corridor_m
    on_axis_fraction = float(on_axis.mean())
    if on_axis_fraction < MIN_ON_AXIS_FRACTION or int(on_axis.sum()) < MIN_BAND_POINTS:
        return _BandCandidate(None, on_axis_fraction, BandRefusal.SECOND_STRUCTURE)

    inliers = empty.copy()
    inliers[np.flatnonzero(mask)[on_axis]] = True

    fitted, _ = _principal_moments(coords_m[inliers])
    if fitted[0] <= 0.0 or fitted[1] / fitted[0] > MAX_ISOTROPIC_EIGENVALUE_RATIO:
        return _BandCandidate(None, on_axis_fraction, BandRefusal.BAND_NOT_ORIENTED)

    return _BandCandidate(inliers, on_axis_fraction, None)


def _fitted_axis(points_m: NDArray[np.float64]) -> tuple[float, float, float]:
    """Axis angle (0..180 deg), slab width (m) and anisotropy of a fitted band."""
    eigenvalues, eigenvectors = _principal_moments(points_m)
    axis = eigenvectors[:, 0]
    return (
        float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0),
        _perpendicular_extent_m(points_m, eigenvectors[:, 1], points_m.mean(axis=0)),
        float(eigenvalues[1] / eigenvalues[0]) if eigenvalues[0] > 0.0 else 0.0,
    )


def high_strain_mask(
    eps_p: ArrayLike,
    percentile: float = HIGH_STRAIN_PERCENTILE,
) -> NDArray[np.bool_]:
    """Band candidate set: points above the given percentile of plastic strain.

    The comparison is strict. A body that has not localized at all has a flat
    strain field whose percentile is that same flat value, and ``>=`` would
    select every point in it and hand the estimators below a body-wide "band".
    ``>`` selects nothing, which is the truth.
    """
    eps_p = np.asarray(eps_p, dtype=np.float64)
    if eps_p.ndim != 1:
        raise ValueError(f"eps_p must be one-dimensional, got shape {eps_p.shape}")
    if eps_p.size == 0:
        raise ValueError("eps_p is empty; there is no strain field to threshold")

    return eps_p > float(np.percentile(eps_p, percentile))


def band_coherence(coords_m: ArrayLike, mask: ArrayLike, cell_size_m: float) -> float:
    """Spatial clustering of the masked set, 0..1.

    A normalized join count: the fraction of a masked point's neighbours (within
    `NEIGHBOURHOOD_CELLS` background cells) that are themselves masked,
    rescaled against the fraction expected if the same number of members were
    scattered at random over the same point cloud —

        (observed - expected) / (1 - expected)

    Chosen over a nearest-neighbour distance ratio (Clark-Evans) and over
    Moran's I because both of those read the *parent* point distribution as
    well as the mask: material points sit on a near-lattice, which floors every
    nearest-neighbour distance and drags the score toward a value set by point
    density rather than by clustering. Measuring the mask against a random
    relabelling of the same cloud cancels the cloud out, so the score depends
    on neither point count nor mask size, and a run at 2x2 points per cell is
    comparable with one at 3x3.

    What it *does* depend on, strongly, is band thickness measured in cells,
    because the neighbourhood radius is fixed in cells. A contiguous band scores
    0.34 at one cell thick, 0.62 at two, 0.73 at three and 0.84 at six; a
    scattered set of the same size scores 0.01. Three consequences worth
    stating plainly:

    * A threshold calibrated on a thick band misreads a *thin* one as
      incoherent, when thinness is the separate signal `band_width_elements`
      already reports.
    * On contiguous masks this score and the width are therefore substantially
      redundant. The supervisor is offered these as several weakly correlated
      signals, and this particular pair is the least independent of them.
    * `cell_size_m` may itself be supervisor-controllable, so the score is not
      comparable across a mesh change within a run.

    Negative values — a set that avoids itself — are clipped to 0, since
    dispersal is not a degree of clustering. The score is also a ranking rather
    than an absolute fraction: a set thinner than the neighbourhood radius has
    most of its neighbourhood off the band whatever it does, so a one-point-wide
    line of deleted points scores about 0.16 against 0.00 for the same points
    sprinkled through the body. The separation from the diffuse case survives;
    the numeric level does not transfer between a strain band and a deletion
    trail, and the two should be read against their own histories.

    Returns 0.0 when the mask holds fewer than `MIN_BAND_POINTS` members or
    when it holds every point: neither case localizes anything.
    """
    coords = _validated_coords(coords_m)
    mask = _validated_mask(mask, len(coords))
    cell_size_m = _validated_cell_size_m(cell_size_m)

    n_points = len(coords)
    n_members = int(mask.sum())
    if n_members < MIN_BAND_POINTS or n_members == n_points:
        return 0.0

    radius_m = NEIGHBOURHOOD_CELLS * cell_size_m
    members = coords[mask]
    # Each point is in its own tree at distance zero, so both counts are one
    # too high; the offsets do not cancel in the ratio, hence the explicit -1.
    all_neighbours = KDTree(coords).query_ball_point(members, r=radius_m, return_length=True) - 1
    member_neighbours = (
        KDTree(members).query_ball_point(members, r=radius_m, return_length=True) - 1
    )

    total_neighbours = int(all_neighbours.sum())
    if total_neighbours == 0:
        return 0.0

    observed = float(member_neighbours.sum()) / total_neighbours
    expected = (n_members - 1) / (n_points - 1)
    return float(np.clip((observed - expected) / (1.0 - expected), 0.0, 1.0))


def band_orientation_deg(coords_m: ArrayLike, mask: ArrayLike) -> float | None:
    """Principal axis of the band, measured from the cutting direction.

    Reported in 0..180 because a principal axis has no sign: the eigenvector's
    direction is arbitrary, so a band at 155 degrees and one at -25 degrees are
    the same band. Compare against the Merchant angle with
    `axis_angle_difference_deg`, not by subtraction.

    Fitted to `_band_inliers` rather than to the raw mask, so the structurally
    guaranteed strays (see the module docstring) do not tilt it. Returns
    ``None`` when there is no single band to orient: fewer than
    `MIN_BAND_POINTS` points, a cloud too round to have a direction (minor
    principal moment above `MAX_ISOTROPIC_EIGENVALUE_RATIO` of the major one),
    or a mask holding two structures neither of which the other's axis
    explains.

    A diffuse high-strain field still has a principal axis, and reporting the
    orientation of noise as though it were a shear plane is the failure this
    module is built to avoid — the supervisor would see it disagree with the
    Merchant reference and conclude that correct physics was a numerical
    artefact. ``None`` is honest and can be routed around; a confident wrong
    angle cannot.

    Known residual: a contaminant of comparable mass *fused* to the band is one
    connected region, and it biases the reported axis by up to 27 degrees for a
    round contaminant and 74 degrees for an elongated one crossing the band.
    Read `BandFit.anisotropy` alongside this angle — it is the measured
    discriminator for that family (see `band_fit`). Refusals cover the
    *separated* case, where the unguarded error reaches 65 degrees and the
    reported axis converges on the direction of the separation rather than on
    any band.

    Use `band_fit` when the refusal reason or the fit quality matters; this is
    the bare angle.
    """
    coords = _validated_coords(coords_m)
    mask = _validated_mask(mask, len(coords))

    candidate = _band_inliers(coords, mask)
    if candidate.inliers is None:
        return None
    return _fitted_axis(coords[candidate.inliers])[0]


def band_width_elements(
    coords_m: ArrayLike,
    mask: ArrayLike,
    cell_size_m: float,
) -> float | None:
    """Band thickness perpendicular to its axis, in background cells.

    The 5th-to-95th percentile span of the perpendicular offset over 0.90 —
    the width of the uniform slab that would produce the same span. A second
    moment would be exact too, and simpler, but it averages *squared* offsets,
    so a single stray masked point takes a 4.0-cell band to 5.1 and three take
    it to 6.7; worse, two strays take a genuinely mesh-limited 2.0-cell band to
    6.5 and an unresolved band reads as twice the resolution limit. The mask
    policy makes those strays structural rather than occasional, so the
    estimator has to tolerate them. This one moves by 1% for the same strays.

    Measured over the same `_band_inliers` set that `band_orientation_deg`
    fits, so the two always describe the same object, and returns ``None`` in
    exactly the cases that does: no band, or a mask holding more than one
    structure. Before localization there is no band to measure and no width
    exists; returning 0.0 there would read as "thinner than the mesh can
    resolve", the specific misdiagnosis this module exists to prevent.

    Two degenerate sets are worth naming, because both would otherwise look
    like ordinary small numbers. A band sampled by a *single* row of points has
    no perpendicular span and reports exactly 0.0 cells: true, in that such a
    set is unresolved by any measure, but it is a floor rather than a
    measurement and it is the one returned width that is not a length. A mask
    whose points are mostly *coincident* has no spacing to work from at all and
    returns ``None`` rather than 0.0, since coincident material points are
    degenerate telemetry rather than an infinitely thin band.

    The sampling bias is not one-signed, and the sign depends on how many point
    rows span the thickness. Measured against a slab sampled by k equally spaced
    rows, the estimator reads short for k below 11 — 44% short at k = 2, 7% at
    k = 6 — and *over*-reports by up to 5.3% for k between 11 and 19, because
    the 5th and 95th percentiles land inside the outermost rows rather than on
    them. At the bench case's 2x2 points per cell, k = 11..19 rows is a band
    5.5 to 9.5 cells thick, far above `MIN_RESOLVED_BAND_ELEMENTS`, while a band
    near the 3-cell gate spans about 6 rows and reads short. So the bias is
    toward under-reporting *in the region the gate decides* — a marginal band
    reads as mesh-limited rather than the reverse — but that is a property of
    the sampling density, not of the estimator.
    """
    coords = _validated_coords(coords_m)
    mask = _validated_mask(mask, len(coords))
    cell_size_m = _validated_cell_size_m(cell_size_m)

    candidate = _band_inliers(coords, mask)
    if candidate.inliers is None:
        return None
    return _fitted_axis(coords[candidate.inliers])[1] / cell_size_m


@dataclass(frozen=True)
class BandFit:
    """Everything the band fit knows: the numbers, and how much to trust them.

    `orientation_deg`, `width_elements` and `anisotropy` are all ``None``
    together, and exactly then `refusal` says which gate stopped the fit.
    """

    orientation_deg: float | None
    """Band axis from the cutting direction, 0..180 deg."""

    width_elements: float | None
    """Band thickness in background cells."""

    anisotropy: float | None
    """Minor over major principal moment of the fitted set, 0..
    `MAX_ISOTROPIC_EIGENVALUE_RATIO`. **Lower is more band-like**: a crisp band
    sits at 0.02, and this is the one measured discriminator for the residual
    contamination the gates cannot catch."""

    on_axis_fraction: float
    """Share of the masked points the fitted band accounts for, 0..1. Read the
    warning in `band_fit` before using this as a confidence score."""

    refusal: BandRefusal | None
    """Which gate refused, or ``None`` when a band was fitted."""


def band_fit(coords_m: ArrayLike, mask: ArrayLike, cell_size_m: float) -> BandFit:
    """Fit one band to a mask and report how much to trust the result.

    `band_orientation_deg` and `band_width_elements` are the bare numbers; this
    adds the two things a supervisor needs to weigh them.

    **`anisotropy` is the confidence signal.** Over a sweep of 562 accepted
    fits on contaminated masks, every fit that came back more than 10 degrees
    wrong had an anisotropy between 0.112 and 0.250 (median 0.151), while the
    fits within 10 degrees ranged 0.020 to 0.235 with a median of 0.020. A set
    that is barely elongated enough to clear the gate is a set whose axis is
    being argued over by more than one structure; a crisp band is two orders of
    magnitude clear of it. Discount a fit whose anisotropy is near
    `MAX_ISOTROPIC_EIGENVALUE_RATIO`.

    **`on_axis_fraction` is not a confidence signal, despite looking like one.**
    Measured over the same sweep it reads exactly 1.000 for every one of the 56
    worst fits, and *below* 1.0 for good ones — 0.90 for a clean band with a
    small satellite correctly excluded. The reason is structural: the fits that
    survive every gate are the ones where a fused contaminant sits inside the
    corridor, so nothing is excluded and the fraction saturates. It falls only
    when the fit succeeded in throwing something away. It is reported because it
    says what the fit covered, not how good it was.

    `refusal` disambiguates the ``None`` cases, which matters because a diffuse
    isotropic cloud (`SEED_NOT_ORIENTED`) is the mesh-collapse signature while
    two clean crossing bands (`SECOND_STRUCTURE`) are physics the estimator
    simply cannot reduce to one angle. Reading both as a bare "no band" biases
    the supervisor toward the numerical verdict.
    """
    coords = _validated_coords(coords_m)
    mask = _validated_mask(mask, len(coords))
    cell_size_m = _validated_cell_size_m(cell_size_m)

    candidate = _band_inliers(coords, mask)
    if candidate.inliers is None:
        return BandFit(
            orientation_deg=None,
            width_elements=None,
            anisotropy=None,
            on_axis_fraction=candidate.on_axis_fraction,
            refusal=candidate.refusal,
        )

    orientation_deg, width_m, anisotropy = _fitted_axis(coords[candidate.inliers])
    return BandFit(
        orientation_deg=orientation_deg,
        width_elements=width_m / cell_size_m,
        anisotropy=anisotropy,
        on_axis_fraction=candidate.on_axis_fraction,
        refusal=None,
    )


def temperature_band_alignment(
    coords_m: ArrayLike,
    mask: ArrayLike,
    temperature_k: ArrayLike,
) -> float:
    """Does the hot region coincide with the band? 0..1.

    Point-biserial correlation between band membership and temperature — the
    Pearson correlation of a 0/1 variable with a continuous one. Chosen over a
    hot-minus-cold mean difference because the correlation is dimensionless and
    so comparable across runs at different cutting speeds, where the absolute
    temperature rise differs by hundreds of kelvin.

    This is the corroborating signal spec section 1 names: adiabatic heating
    along a real shear band puts the heat exactly where the strain is. High
    strain with no heat on it is not doing plastic work, which points at
    numerical degeneration.

    A negative correlation is reported as 0.0. A band colder than the bulk
    corroborates nothing, and neither does an unrelated temperature field;
    collapsing both to zero keeps the number on 0..1 where the supervisor can
    read it as a degree of corroboration. 0.0 is also the answer when the body
    is isothermal or the mask is empty or full — in each case one of the two
    variables has no variance and the correlation is undefined.

    ``coords_m`` is not used: a point-biserial correlation needs no positions.
    It is taken so that every band estimator has the same leading arguments and
    the caller passes one triple, and it is length-checked against the mask so
    a mismatched telemetry dump still fails loudly here.
    """
    coords = _validated_coords(coords_m)
    mask = _validated_mask(mask, len(coords))
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    if temperature_k.shape != (len(coords),):
        raise ValueError(
            f"temperature_k must have shape ({len(coords)},), got {temperature_k.shape}"
        )

    membership = mask.astype(np.float64)
    membership_deviation = membership - membership.mean()
    temperature_deviation = temperature_k - temperature_k.mean()

    spread = np.sqrt(
        float(membership_deviation @ membership_deviation)
        * float(temperature_deviation @ temperature_deviation)
    )
    if spread == 0.0:
        return 0.0

    correlation = float(membership_deviation @ temperature_deviation) / spread
    return max(correlation, 0.0)


def temp_gradient_max_k_per_m(coords_m: ArrayLike, temperature_k: ArrayLike) -> float:
    """Steepest temperature difference quotient in the field, K/m.

    Over every point and each of its `GRADIENT_NEIGHBOURS` nearest neighbours:
    max |T_i - T_j| / |x_i - x_j|. A one-sided finite difference on the point
    cloud, assuming no grid — MPM points move off any lattice they started on.

    Taking four neighbours rather than one matters on a cloud that is still
    close to a lattice, where the single nearest neighbour is a tie between the
    along-band and across-band directions and the tie-break can hide the
    gradient that matters. The estimate is still a lower bound on the true
    gradient magnitude, since it projects onto whatever directions the
    neighbours happen to lie in.

    Coincident points are skipped rather than reported as an infinite gradient;
    two material points at the same position carry no distance to divide by.
    """
    coords = _validated_coords(coords_m)
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    if temperature_k.shape != (len(coords),):
        raise ValueError(
            f"temperature_k must have shape ({len(coords)},), got {temperature_k.shape}"
        )
    if len(coords) < 2:
        raise ValueError(f"need at least two points to difference, got {len(coords)}")

    # Column 0 of the query is the point itself; ask for one extra and drop it.
    neighbours = min(GRADIENT_NEIGHBOURS + 1, len(coords))
    distances_m, indices = KDTree(coords).query(coords, k=neighbours)
    distances_m, indices = distances_m[:, 1:], indices[:, 1:]

    differences_k = np.abs(temperature_k[:, None] - temperature_k[indices])
    separated = distances_m > 0.0
    gradients = np.where(separated, differences_k / np.where(separated, distances_m, 1.0), 0.0)
    return float(gradients.max())


def axis_angle_difference_deg(angle_a_deg: float, angle_b_deg: float) -> float:
    """Separation between two undirected axes, 0..90 degrees.

    Band orientation is an axis, not a direction, so it wraps at 180: a band at
    179 degrees is two degrees from one at 1 degree, not 178. This is the
    comparison to use between `band_orientation_deg` and any of the shear
    angles in `diagnostics.merchant`; plain subtraction gets the wrap wrong in
    exactly the cases where the band is nearly aligned with the cutting
    direction.
    """
    separation = abs(angle_a_deg - angle_b_deg) % 180.0
    return float(min(separation, 180.0 - separation))
