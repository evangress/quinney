"""Shear-band estimators — behavior tests against synthetic ground truth.

Every point cloud here is built by hand, so the right answer is known before
the estimator runs: a band laid down at 35 degrees must come back at 35
degrees, a band seven point-rows thick must come back seven rows thick, and a
scattered cloud must refuse to name an orientation at all. That refusal is the
point of the module — telling a physical shear band from numerical mesh
collapse is exactly the case where a confident wrong number is worse than no
number.

Clouds are built with ``np.arange``/``np.meshgrid`` or a seeded generator.
Nothing here is unseeded.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quinney.diagnostics.bands import (
    HIGH_STRAIN_PERCENTILE,
    MIN_ON_AXIS_FRACTION,
    MIN_RESOLVED_BAND_ELEMENTS,
    axis_angle_difference_deg,
    band_coherence,
    band_orientation_deg,
    band_width_elements,
    high_strain_mask,
    temp_gradient_max_k_per_m,
    temperature_band_alignment,
)

# Bench case geometry: a 50 um background grid cell with the MPM default of
# 2x2 material points per cell, so points sit 25 um apart.
CELL_SIZE_M = 50e-6
POINTS_PER_CELL_1D = 2
SPACING_M = CELL_SIZE_M / POINTS_PER_CELL_1D

# Enough lattice to hold a band several cells wide with bulk material around
# it: 40 x 40 points is 20 x 20 cells, or 1 mm square.
LATTICE_N = 40

SEED = 20260819

# Reference and band temperatures for the bench case: adiabatic heating along
# the primary shear zone lifts OFHC copper order 100 K above ambient.
AMBIENT_K = 294.0
BAND_RISE_K = 150.0

# Angles spanning the primary shear zone (25-45 deg) plus two outside it, so
# the estimator is not tuned to one quadrant.
BAND_ANGLES_DEG = (0.0, 25.0, 35.0, 45.0, 60.0, 120.0, 155.0)

# Recovering a band axis to better than a degree matters because the
# supervisor compares it against the Merchant angle, and the disagreements
# that mean something are 5-10 degrees.
ORIENTATION_TOLERANCE_DEG = 1.0

# Coherence is only useful if it is comparable between runs, so the same band
# seen at a different point density or against a bigger workpiece must give
# very nearly the same score. Measured spread across those cases is 0.03.
COHERENCE_INVARIANCE_TOLERANCE = 0.05

# Width is recovered to within 3% for a band that cuts across the point rows.
# A band lying along them is sampled by only a handful of rows, and the 5-95
# percentile span then falls on the outermost row centres rather than the slab
# faces, which reads about 7% short.
WIDTH_TOLERANCE = 0.10

# A contiguous band three cells thick measures 0.73; the level is set by
# thickness in cells, so this floor belongs to the 3-cell cases that assert it.
COHERENT_BAND_FLOOR = 0.6

# A band one cell thick is fully contiguous but scores far lower, because most
# of a 1.5-cell neighbourhood necessarily lies off it. Anything above the
# scattered ceiling is the real claim.
THIN_BAND_COHERENCE_RANGE = (0.25, 0.5)

# A spatially random mask scores 0 up to finite-sample noise.
SCATTERED_COHERENCE_CEILING = 0.1

# Separation the band-vs-scatter comparison must clear.
BAND_TO_SCATTER_RATIO = 4.0

# Corroboration levels: an exactly-hot band correlates at 1.0, a noisy one
# still well above a half, and a band whose heat is somewhere else at nothing.
NOISY_ALIGNMENT_FLOOR = 0.7
UNALIGNED_CEILING = 0.2

# The gradient estimator projects onto whatever directions the neighbours lie
# in, so a ramp oblique to the lattice reads slightly low and never high.
OBLIQUE_GRADIENT_FLOOR = 0.8


def lattice(n_x: int = LATTICE_N, n_y: int = LATTICE_N, spacing_m: float = SPACING_M):
    """A regular n_x by n_y point cloud, the stand-in for MPM material points."""
    x = np.arange(n_x, dtype=np.float64) * spacing_m
    y = np.arange(n_y, dtype=np.float64) * spacing_m
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def band_axes(coords_m: NDArray[np.float64], angle_deg: float):
    """Coordinates along and perpendicular to a band axis through the centroid."""
    angle = np.radians(angle_deg)
    centred = coords_m - coords_m.mean(axis=0)
    along = centred @ np.array([np.cos(angle), np.sin(angle)])
    perpendicular = centred @ np.array([-np.sin(angle), np.cos(angle)])
    return along, perpendicular


def band_mask(
    coords_m: NDArray[np.float64],
    angle_deg: float,
    width_m: float,
    length_m: float | None = None,
) -> NDArray[np.bool_]:
    """Points inside a strip of the given width at the given angle.

    ``length_m`` truncates the strip so it does not run into the domain
    corners; a strip clipped by the domain is still a strip, but a strip fully
    inside it has an exactly known principal axis.
    """
    along, perpendicular = band_axes(coords_m, angle_deg)
    inside = np.abs(perpendicular) <= width_m / 2.0
    if length_m is not None:
        inside &= np.abs(along) <= length_m / 2.0
    return inside


def scattered_mask(coords_m: NDArray[np.float64], fraction: float = 0.1) -> NDArray[np.bool_]:
    """A spatially random subset of the same size a band would have."""
    rng = np.random.default_rng(SEED)
    mask = np.zeros(len(coords_m), dtype=bool)
    chosen = rng.choice(len(coords_m), size=int(fraction * len(coords_m)), replace=False)
    mask[chosen] = True
    return mask


def stray_points(coords_m: NDArray[np.float64], angle_deg: float, count: int) -> NDArray[np.bool_]:
    """The `count` points furthest from a band axis — the worst-placed strays.

    `high_strain_mask` takes a fixed decile, so whenever the band is smaller
    than that decile the remainder of the mask is the next-most-strained
    material wherever it lies. These stand in for it at its most damaging.
    """
    _, perpendicular = band_axes(coords_m, angle_deg)
    mask = np.zeros(len(coords_m), dtype=bool)
    mask[np.argsort(np.abs(perpendicular))[::-1][:count]] = True
    return mask


def two_cluster_mask(coords_m: NDArray[np.float64]):
    """A primary band plus a separate cluster well clear of it.

    A 14 x 2 cell band at 35 degrees and a 2-cell-radius cluster whose centroid
    sits 14 cells along the cutting direction and 10 cells across it. Fitted as
    one set, the between-cluster separation contributes a rank-one term growing
    as the square of the distance, which dominates both clusters' own spread:
    the aggregate axis comes out at 148 degrees against a true band axis of 35,
    and the eigenvalue ratio is 0.15, comfortably inside the anisotropy gate.
    """
    band = band_mask(coords_m, angle_deg=35.0, width_m=2 * CELL_SIZE_M, length_m=14 * CELL_SIZE_M)
    centred = coords_m - coords_m.mean(axis=0)
    offset_m = np.array([14.0, -10.0]) * CELL_SIZE_M
    cluster = np.hypot(centred[:, 0] - offset_m[0], centred[:, 1] - offset_m[1]) <= 2 * CELL_SIZE_M
    return band, cluster


class TestHighStrainMask:
    def test_selects_the_top_tail_of_the_plastic_strain_field(self):
        eps_p = np.linspace(0.0, 2.0, 1000)
        mask = high_strain_mask(eps_p)
        assert mask.sum() == pytest.approx(
            len(eps_p) * (100.0 - HIGH_STRAIN_PERCENTILE) / 100.0, rel=0.05
        )

    def test_selects_the_core_of_a_localized_band_and_not_the_bulk(self):
        # Strain peaking on the band and decaying away from it, over a
        # uniformly worked bulk — the shape a real primary shear zone has.
        coords_m = lattice()
        _, perpendicular = band_axes(coords_m, angle_deg=35.0)
        half_width_m = 1.5 * CELL_SIZE_M
        eps_p = 0.3 + 3.0 * np.exp(-((perpendicular / half_width_m) ** 2))

        mask = high_strain_mask(eps_p)
        assert mask.sum() > 0
        assert np.all(np.abs(perpendicular[mask]) < half_width_m)

    def test_selects_nothing_when_the_field_is_uniform(self):
        # Nothing has localized yet. A percentile cut with >= would select the
        # entire body here and report a body-wide "band"; > selects nothing,
        # which is the truth.
        assert high_strain_mask(np.full(500, 0.4)).sum() == 0

    def test_honours_an_explicit_percentile(self):
        eps_p = np.linspace(0.0, 1.0, 1000)
        assert (
            high_strain_mask(eps_p, percentile=50.0).sum()
            > high_strain_mask(eps_p, percentile=99.0).sum()
        )


class TestBandOrientation:
    @pytest.mark.parametrize("angle_deg", BAND_ANGLES_DEG)
    def test_recovers_the_angle_of_a_band_laid_down_at_a_known_angle(self, angle_deg: float):
        coords_m = lattice()
        mask = band_mask(
            coords_m,
            angle_deg=angle_deg,
            width_m=3 * CELL_SIZE_M,
            length_m=14 * CELL_SIZE_M,
        )
        recovered = band_orientation_deg(coords_m, mask)

        assert recovered is not None
        assert axis_angle_difference_deg(recovered, angle_deg) < ORIENTATION_TOLERANCE_DEG

    def test_reports_the_axis_not_the_direction(self):
        # Principal axes have no sign, so a band at 155 deg is the same band as
        # one at -25 deg and must report inside 0..180 either way.
        coords_m = lattice()
        recovered = band_orientation_deg(
            coords_m, band_mask(coords_m, -25.0, 3 * CELL_SIZE_M, 14 * CELL_SIZE_M)
        )
        assert recovered is not None
        assert 0.0 <= recovered < 180.0
        assert axis_angle_difference_deg(recovered, 155.0) < ORIENTATION_TOLERANCE_DEG

    def test_refuses_to_orient_a_scattered_cloud(self):
        # The whole reason this returns None: a diffuse high-strain field has
        # a principal axis, but it is the axis of noise, and handing the
        # supervisor a confident number for it is the failure mode that
        # matters most in this project.
        coords_m = lattice()
        assert band_orientation_deg(coords_m, scattered_mask(coords_m)) is None

    def test_refuses_to_orient_a_round_blob(self):
        # A compact hot spot is coherent but has no direction. Coherence and
        # orientation are independent signals, and this is the case that
        # separates them.
        coords_m = lattice()
        centred = coords_m - coords_m.mean(axis=0)
        blob = np.hypot(centred[:, 0], centred[:, 1]) <= 4 * CELL_SIZE_M
        assert band_orientation_deg(coords_m, blob) is None

    def test_refuses_to_orient_a_pair_of_points(self):
        # Two points always lie on a perfect line. Reporting that line as a
        # band orientation would be pure fabrication.
        coords_m = lattice()
        mask = np.zeros(len(coords_m), dtype=bool)
        mask[[0, 5]] = True
        assert band_orientation_deg(coords_m, mask) is None

    def test_refuses_to_orient_an_empty_mask(self):
        coords_m = lattice()
        assert band_orientation_deg(coords_m, np.zeros(len(coords_m), dtype=bool)) is None


class TestAMaskHoldingMoreThanOneStructure:
    """The worst failure this module can produce is a confident wrong angle.

    Two separated clusters do not average to a compromise orientation: the
    reported axis converges on the direction of the separation between them,
    which is unrelated to either one. A supervisor comparing that to a Merchant
    reference near 30 degrees sees a 67-degree disagreement and concludes the
    localization is not on the shear plane and must therefore be numerical —
    correct physics, confidently misread as a numerical artefact.
    """

    def test_refuses_to_orient_two_separated_clusters(self):
        # Fitted as one set this reports 148.0 deg against a true 35.0, at an
        # eigenvalue ratio of 0.15 that sails through the anisotropy gate.
        coords_m = lattice(2 * LATTICE_N, 2 * LATTICE_N, CELL_SIZE_M / 2)
        band, cluster = two_cluster_mask(coords_m)
        assert band.sum() > 0 and cluster.sum() > 0

        assert band_orientation_deg(coords_m, band | cluster) is None

    def test_refuses_to_measure_the_width_of_two_separated_clusters(self):
        # Orientation and width describe the same object or neither does.
        coords_m = lattice(2 * LATTICE_N, 2 * LATTICE_N, CELL_SIZE_M / 2)
        band, cluster = two_cluster_mask(coords_m)

        assert band_width_elements(coords_m, band | cluster, CELL_SIZE_M) is None

    def test_orients_the_band_alone_from_the_same_construction(self):
        # The refusal above must come from the second cluster, not from the
        # band being unfittable.
        coords_m = lattice(2 * LATTICE_N, 2 * LATTICE_N, CELL_SIZE_M / 2)
        band, _ = two_cluster_mask(coords_m)

        recovered = band_orientation_deg(coords_m, band)
        assert recovered is not None
        assert axis_angle_difference_deg(recovered, 35.0) < ORIENTATION_TOLERANCE_DEG

    @pytest.mark.parametrize("hole_cells", (1, 2, 3, 4))
    def test_still_orients_a_band_eroded_into_collinear_pieces(self, hole_cells: int):
        # A band with its middle eroded away is broken but still collinear, and
        # a connectivity test alone would refuse it. It is one band, and the
        # axis fitted to it is right, so it must still report.
        coords_m = lattice(2 * LATTICE_N, 2 * LATTICE_N, CELL_SIZE_M / 2)
        band, _ = two_cluster_mask(coords_m)
        along, _ = band_axes(coords_m, 35.0)
        eroded = band & ~(np.abs(along) < hole_cells * CELL_SIZE_M / 2)

        recovered = band_orientation_deg(coords_m, eroded)
        assert recovered is not None
        assert axis_angle_difference_deg(recovered, 35.0) < ORIENTATION_TOLERANCE_DEG

    def test_ignores_a_small_satellite_rather_than_refusing(self):
        # A satellite too small to be a second structure is dropped from the
        # fit instead. Refusing here would throw away a good band over a
        # handful of points, and including it would tilt the axis.
        coords_m = lattice(2 * LATTICE_N, 2 * LATTICE_N, CELL_SIZE_M / 2)
        band, _ = two_cluster_mask(coords_m)
        centred = coords_m - coords_m.mean(axis=0)
        offset_m = np.array([9.0, 0.0]) * CELL_SIZE_M
        satellite = (
            np.hypot(centred[:, 0] - offset_m[0], centred[:, 1] - offset_m[1]) <= CELL_SIZE_M
        )
        # "Small" means below the share that makes the mask two structures.
        satellite_share = satellite.sum() / (band | satellite).sum()
        assert 0.0 < satellite_share < 1.0 - MIN_ON_AXIS_FRACTION

        recovered = band_orientation_deg(coords_m, band | satellite)
        assert recovered is not None
        assert axis_angle_difference_deg(recovered, 35.0) < ORIENTATION_TOLERANCE_DEG


class TestBandCoherence:
    @pytest.mark.parametrize("angle_deg", BAND_ANGLES_DEG)
    def test_a_contiguous_band_scores_high(self, angle_deg: float):
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=angle_deg, width_m=3 * CELL_SIZE_M)
        assert band_coherence(coords_m, mask, CELL_SIZE_M) > COHERENT_BAND_FLOOR

    def test_a_scattered_cloud_of_the_same_size_scores_near_zero(self):
        # Same point count, same domain, no structure. If this did not
        # separate cleanly from the band case the estimator would be measuring
        # mask size rather than clustering.
        coords_m = lattice()
        assert (
            band_coherence(coords_m, scattered_mask(coords_m), CELL_SIZE_M)
            < SCATTERED_COHERENCE_CEILING
        )

    def test_separates_a_band_from_scatter_of_identical_size(self):
        coords_m = lattice()
        band = band_mask(coords_m, angle_deg=35.0, width_m=3 * CELL_SIZE_M)
        scatter = np.zeros(len(coords_m), dtype=bool)
        rng = np.random.default_rng(SEED)
        scatter[rng.choice(len(coords_m), size=int(band.sum()), replace=False)] = True

        assert band_coherence(coords_m, band, CELL_SIZE_M) > (
            BAND_TO_SCATTER_RATIO * band_coherence(coords_m, scatter, CELL_SIZE_M)
        )

    def test_is_independent_of_point_density(self):
        # Same physical band, twice the material points per cell. If the score
        # moved with resolution it could not be compared between runs, which
        # is the only thing it is for.
        coarse = lattice(spacing_m=CELL_SIZE_M / 2)
        fine = lattice(n_x=2 * LATTICE_N, n_y=2 * LATTICE_N, spacing_m=CELL_SIZE_M / 4)

        coarse_score = band_coherence(coarse, band_mask(coarse, 35.0, 3 * CELL_SIZE_M), CELL_SIZE_M)
        fine_score = band_coherence(fine, band_mask(fine, 35.0, 3 * CELL_SIZE_M), CELL_SIZE_M)
        assert abs(coarse_score - fine_score) < COHERENCE_INVARIANCE_TOLERANCE

    def test_is_independent_of_how_much_bulk_material_surrounds_the_band(self):
        # Doubling the workpiece halves the mask fraction without touching the
        # band. A raw neighbour fraction would be unchanged, but so would a
        # raw one for scatter; normalizing against the mask fraction is what
        # keeps the two comparable.
        small = lattice()
        large = lattice(n_x=2 * LATTICE_N, n_y=2 * LATTICE_N)

        small_score = band_coherence(small, band_mask(small, 35.0, 3 * CELL_SIZE_M), CELL_SIZE_M)
        large_score = band_coherence(large, band_mask(large, 35.0, 3 * CELL_SIZE_M), CELL_SIZE_M)
        assert abs(small_score - large_score) < COHERENCE_INVARIANCE_TOLERANCE

    def test_is_bounded_to_the_unit_interval(self):
        coords_m = lattice()
        for mask in (
            band_mask(coords_m, 35.0, CELL_SIZE_M),
            scattered_mask(coords_m),
            np.ones(len(coords_m), dtype=bool),
        ):
            assert 0.0 <= band_coherence(coords_m, mask, CELL_SIZE_M) <= 1.0

    def test_a_one_cell_band_scores_far_lower_than_a_thick_one(self):
        # The score's dominant dependence is band thickness measured in cells,
        # because the neighbourhood radius is fixed in cells. A perfectly
        # contiguous one-cell band cannot reach the level a three-cell band
        # does, so a threshold calibrated on thick bands would call a real thin
        # band incoherent — thinness is what band_width_elements reports, and
        # this is where the two signals stop being independent.
        coords_m = lattice()
        thin = band_coherence(coords_m, band_mask(coords_m, 35.0, CELL_SIZE_M), CELL_SIZE_M)
        thick = band_coherence(coords_m, band_mask(coords_m, 35.0, 3 * CELL_SIZE_M), CELL_SIZE_M)

        low, high = THIN_BAND_COHERENCE_RANGE
        assert low < thin < high
        assert thin < thick
        assert thin > band_coherence(coords_m, scattered_mask(coords_m), CELL_SIZE_M)

    def test_rejects_a_non_positive_cell_size(self):
        # A negative cell size gives a negative radius, no neighbours at all,
        # and an arbitrary score rather than an error.
        coords_m = lattice()
        mask = band_mask(coords_m, 35.0, 3 * CELL_SIZE_M)
        for bad_cell_size_m in (0.0, -CELL_SIZE_M):
            with pytest.raises(ValueError, match="cell_size_m"):
                band_coherence(coords_m, mask, bad_cell_size_m)

    def test_an_empty_mask_is_not_coherent(self):
        # Nothing has localized, and nothing is the honest score. This is the
        # state a deletion mask sits in for most of a healthy run.
        coords_m = lattice()
        assert band_coherence(coords_m, np.zeros(len(coords_m), dtype=bool), CELL_SIZE_M) == 0.0

    def test_a_mask_over_the_whole_body_localizes_nothing(self):
        coords_m = lattice()
        assert band_coherence(coords_m, np.ones(len(coords_m), dtype=bool), CELL_SIZE_M) == 0.0

    def test_works_on_a_deletion_style_mask_of_a_few_points(self):
        # The sibling use: are deleted points strung along a separation line
        # (physical) or sprinkled through the body (numerical erosion)?
        coords_m = lattice()
        along_a_line = band_mask(coords_m, angle_deg=35.0, width_m=SPACING_M)
        sprinkled = scattered_mask(coords_m, fraction=float(along_a_line.mean()))

        assert band_coherence(coords_m, along_a_line, CELL_SIZE_M) > band_coherence(
            coords_m, sprinkled, CELL_SIZE_M
        )


class TestBandWidth:
    def test_measures_a_band_lying_along_the_lattice_in_cells(self):
        # A band aligned with the lattice selects whole point-rows, so the
        # thickness it samples is exactly (rows x 25 um) — a ground truth read
        # off the construction rather than assumed.
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=0.0, width_m=6 * SPACING_M)
        rows = len(np.unique(coords_m[mask][:, 1]))
        assert rows == 6

        expected_cells = rows * SPACING_M / CELL_SIZE_M
        measured = band_width_elements(coords_m, mask, CELL_SIZE_M)
        assert measured is not None
        assert measured == pytest.approx(expected_cells, rel=WIDTH_TOLERANCE)

    @pytest.mark.parametrize("angle_deg", BAND_ANGLES_DEG)
    def test_measures_the_same_width_whatever_the_band_angle(self, angle_deg: float):
        # Width is perpendicular to the band axis, not to the lattice.
        coords_m = lattice()
        width_m = 4 * CELL_SIZE_M
        mask = band_mask(coords_m, angle_deg=angle_deg, width_m=width_m, length_m=14 * CELL_SIZE_M)

        measured = band_width_elements(coords_m, mask, CELL_SIZE_M)
        assert measured is not None
        assert measured == pytest.approx(width_m / CELL_SIZE_M, rel=WIDTH_TOLERANCE)

    def test_a_one_cell_band_is_below_the_resolution_limit(self):
        # Spec section 5: a band thinner than three background cells is
        # mesh-resolution-limited, not physics. This is the case that must not
        # be read as a real shear band.
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=35.0, width_m=CELL_SIZE_M)
        measured = band_width_elements(coords_m, mask, CELL_SIZE_M)
        assert measured is not None
        assert measured < MIN_RESOLVED_BAND_ELEMENTS

    def test_a_well_resolved_band_is_above_the_resolution_limit(self):
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=35.0, width_m=5 * CELL_SIZE_M)
        measured = band_width_elements(coords_m, mask, CELL_SIZE_M)
        assert measured is not None
        assert measured > MIN_RESOLVED_BAND_ELEMENTS

    @pytest.mark.parametrize("n_strays", (1, 3, 5))
    def test_stray_masked_points_do_not_inflate_the_width(self, n_strays: int):
        # A second moment averages *squared* offsets, so its breakdown point is
        # zero: one stray took this 4-cell band to 5.07 cells and three took it
        # to 6.66. The percentile span does not move.
        coords_m = lattice()
        band = band_mask(coords_m, 35.0, 4 * CELL_SIZE_M, 14 * CELL_SIZE_M)
        clean = band_width_elements(coords_m, band, CELL_SIZE_M)
        contaminated = band_width_elements(
            coords_m, band | stray_points(coords_m, 35.0, n_strays), CELL_SIZE_M
        )

        assert clean is not None and contaminated is not None
        assert contaminated == pytest.approx(clean, rel=0.05)

    def test_a_mesh_limited_band_stays_below_the_limit_when_strays_are_added(self):
        # The failure that matters: with a second moment, two strays took a
        # genuinely unresolved 2-cell band to 6.55 cells, so material the mesh
        # cannot resolve reported as twice the resolution limit and the
        # supervisor would have read mesh collapse as a real shear band.
        coords_m = lattice()
        band = band_mask(coords_m, 35.0, 2 * CELL_SIZE_M, 14 * CELL_SIZE_M)
        measured = band_width_elements(
            coords_m, band | stray_points(coords_m, 35.0, 2), CELL_SIZE_M
        )

        assert measured is not None
        assert measured < MIN_RESOLVED_BAND_ELEMENTS

    def test_a_single_row_of_points_has_no_thickness_at_all(self):
        # 0.0 is a floor, not a measurement — the one returned width that is
        # not a length. It reads as unresolved, which is the right reading.
        coords_m = lattice()
        mask = coords_m[:, 1] == coords_m[0, 1]
        assert mask.sum() == LATTICE_N

        assert band_width_elements(coords_m, mask, CELL_SIZE_M) == 0.0

    def test_declines_to_measure_coincident_points(self):
        # Degenerate telemetry, not an infinitely thin band: there is no
        # spacing to measure anything against.
        coords_m = np.zeros((10, 2))
        assert band_width_elements(coords_m, np.ones(10, dtype=bool), CELL_SIZE_M) is None

    def test_declines_to_measure_a_band_that_is_not_there(self):
        # Nothing has localized yet — the normal state for most of a run's
        # early history, and for t=0 where the strain field is identically
        # zero. That is not malformed input. Silently returning 0.0 would read
        # as "thinner than the mesh can resolve", the exact misdiagnosis this
        # module exists to prevent.
        coords_m = lattice()
        assert (
            band_width_elements(coords_m, np.zeros(len(coords_m), dtype=bool), CELL_SIZE_M) is None
        )

    def test_declines_a_band_of_two_points(self):
        coords_m = lattice()
        mask = np.zeros(len(coords_m), dtype=bool)
        mask[[0, 5]] = True
        assert band_width_elements(coords_m, mask, CELL_SIZE_M) is None

    def test_still_raises_on_malformed_input(self):
        # The fail-loud rule is for corrupt telemetry, not for an empty mask.
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=35.0, width_m=3 * CELL_SIZE_M)
        with pytest.raises(ValueError, match="cell_size_m"):
            band_width_elements(coords_m, mask, -CELL_SIZE_M)
        with pytest.raises(ValueError, match="mask"):
            band_width_elements(coords_m, mask[:-1], CELL_SIZE_M)


class TestTemperatureBandAlignment:
    def test_is_one_when_only_the_band_is_hot(self):
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=35.0, width_m=3 * CELL_SIZE_M)
        temperature_k = np.where(mask, AMBIENT_K + BAND_RISE_K, AMBIENT_K)

        assert temperature_band_alignment(coords_m, mask, temperature_k) == pytest.approx(1.0)

    def test_is_high_when_the_band_is_hot_with_a_noisy_background(self):
        # Adiabatic heating along the band, over a background scattered by 30%
        # of the band's own temperature rise. The correlation degrades to 0.78
        # rather than collapsing, which is what makes it usable on a real
        # temperature field instead of a clean one.
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=35.0, width_m=3 * CELL_SIZE_M)
        rng = np.random.default_rng(SEED)
        temperature_k = AMBIENT_K + rng.normal(0.0, 0.3 * BAND_RISE_K, len(coords_m))
        temperature_k[mask] += BAND_RISE_K

        assert temperature_band_alignment(coords_m, mask, temperature_k) > NOISY_ALIGNMENT_FLOOR

    def test_is_near_zero_when_the_hot_region_sits_somewhere_else(self):
        # High strain with no heating on it is the signature of a band that is
        # not doing plastic work: mesh collapse, not localization.
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=35.0, width_m=3 * CELL_SIZE_M)
        elsewhere = band_mask(coords_m, angle_deg=125.0, width_m=3 * CELL_SIZE_M)
        temperature_k = np.where(elsewhere, AMBIENT_K + BAND_RISE_K, AMBIENT_K)

        assert temperature_band_alignment(coords_m, mask, temperature_k) < UNALIGNED_CEILING

    def test_a_cold_band_corroborates_nothing(self):
        # An anticorrelated field is no more corroboration than an unrelated
        # one, and the score is defined on 0..1.
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=35.0, width_m=3 * CELL_SIZE_M)
        temperature_k = np.where(mask, AMBIENT_K, AMBIENT_K + BAND_RISE_K)

        assert temperature_band_alignment(coords_m, mask, temperature_k) == 0.0

    def test_an_isothermal_body_corroborates_nothing(self):
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=35.0, width_m=3 * CELL_SIZE_M)
        temperature_k = np.full(len(coords_m), AMBIENT_K)

        assert temperature_band_alignment(coords_m, mask, temperature_k) == 0.0

    def test_an_empty_mask_corroborates_nothing(self):
        coords_m = lattice()
        temperature_k = np.full(len(coords_m), AMBIENT_K)
        empty = np.zeros(len(coords_m), dtype=bool)

        assert temperature_band_alignment(coords_m, empty, temperature_k) == 0.0

    def test_rejects_a_temperature_field_of_the_wrong_length(self):
        coords_m = lattice()
        mask = band_mask(coords_m, angle_deg=35.0, width_m=3 * CELL_SIZE_M)
        with pytest.raises(ValueError):
            temperature_band_alignment(coords_m, mask, np.full(len(coords_m) - 1, AMBIENT_K))


class TestTemperatureGradient:
    def test_recovers_a_known_linear_ramp(self):
        # T = G x on a regular lattice: the steepest neighbour difference is
        # exactly G. A neighbourhood that only ever looked at one neighbour
        # could pick the along-row pair and report zero here.
        coords_m = lattice()
        gradient_k_per_m = 2.0e6
        temperature_k = AMBIENT_K + gradient_k_per_m * coords_m[:, 0]

        assert temp_gradient_max_k_per_m(coords_m, temperature_k) == pytest.approx(
            gradient_k_per_m, rel=1e-6
        )

    def test_recovers_a_ramp_running_across_the_lattice_rows(self):
        # 150 K over three cells is the order of an adiabatic band edge.
        coords_m = lattice()
        gradient_k_per_m = BAND_RISE_K / (3 * CELL_SIZE_M)
        along, _ = band_axes(coords_m, angle_deg=35.0)
        temperature_k = AMBIENT_K + gradient_k_per_m * along

        # Lattice neighbours are not aligned with the ramp, so the recovered
        # value is the projection onto the closest neighbour direction.
        measured = temp_gradient_max_k_per_m(coords_m, temperature_k)
        assert (
            OBLIQUE_GRADIENT_FLOOR * gradient_k_per_m <= measured <= gradient_k_per_m * (1.0 + 1e-9)
        )

    def test_is_zero_on_an_isothermal_body(self):
        coords_m = lattice()
        assert temp_gradient_max_k_per_m(coords_m, np.full(len(coords_m), AMBIENT_K)) == 0.0

    def test_finds_a_single_hot_point_against_a_cold_body(self):
        # The steepest gradient in the field, not the average one: a lone
        # overheated point is a numerical symptom worth surfacing.
        coords_m = lattice()
        temperature_k = np.full(len(coords_m), AMBIENT_K)
        temperature_k[len(coords_m) // 2] += BAND_RISE_K

        assert temp_gradient_max_k_per_m(coords_m, temperature_k) == pytest.approx(
            BAND_RISE_K / SPACING_M
        )

    def test_rejects_a_cloud_too_small_to_difference(self):
        with pytest.raises(ValueError):
            temp_gradient_max_k_per_m(np.zeros((1, 2)), np.array([AMBIENT_K]))


class TestAxisAngleDifference:
    def test_wraps_around_180_degrees(self):
        # A band at 179 deg and one at 1 deg are two degrees apart, not 178.
        assert axis_angle_difference_deg(179.0, 1.0) == pytest.approx(2.0)

    def test_is_symmetric(self):
        assert axis_angle_difference_deg(30.0, 100.0) == axis_angle_difference_deg(100.0, 30.0)

    def test_never_exceeds_a_right_angle(self):
        # Two axes can be at most perpendicular.
        angles = np.linspace(-360.0, 360.0, 97)
        for a in angles:
            for b in angles:
                assert 0.0 <= axis_angle_difference_deg(float(a), float(b)) <= 90.0 + 1e-9
