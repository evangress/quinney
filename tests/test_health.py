"""Deletion and numerical health — behavior tests against synthetic episodes.

Every record here is built by hand so the right answer exists before the
estimator runs: a deletion set laid along a line must score higher than the
same number of points sprinkled through the body, an energy record given a 5%
residual must come back 5%, and a dt record given a chosen slope must come back
with that slope and that sign.

The question the module under test serves is spec section 1's: deletion
clustered along the hot shear band is physical chip separation, deletion
smeared through the bulk is numerical erosion. Both look like element
degeneration from a distance, so the tests below always pair a physical case
with the numerical one it has to be told apart from.

Nothing here is unseeded, and no test touches the filesystem.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quinney.diagnostics.health import (
    MAX_DELETED_MASS_PCT,
    MAX_ENERGY_DRIFT_PCT,
    aspect_ratio_p99,
    deleted_mass_pct,
    deletion_rate_per_s,
    deletion_spatial_coherence,
    deletion_temp_correlation,
    dt_slope_per_step,
    energy_drift_pct,
    hourglass_ie_ratio_pct,
    jacobian_min,
    plastic_work_delta_last_remesh,
)
from quinney.telemetry import Deletion, Series, Snapshot

# Bench case geometry: a 50 um background grid cell with the MPM default of
# 2x2 material points per cell, so points sit 25 um apart.
CELL_SIZE_M = 50e-6
SPACING_M = CELL_SIZE_M / 2

# 40 x 40 points is 20 x 20 cells, or a 1 mm square of workpiece — enough bulk
# around a separation line for "clustered" and "diffuse" to differ.
LATTICE_N = 40

# Reference temperature of the bench case, and the order-100 K rise adiabatic
# heating puts into the primary shear zone.
AMBIENT_K = 294.0
BAND_RISE_K = 150.0

SEED = 20260819

# A nanosecond sample interval over 200 samples: short enough that a window is
# a meaningful fraction of the record.
SAMPLE_DT_S = 1e-9
N_SAMPLES = 200


def lattice(n: int = LATTICE_N, spacing_m: float = SPACING_M) -> NDArray[np.float64]:
    """An n x n grid of material points, one row per point, columns (x, y)."""
    axis = np.arange(n, dtype=np.float64) * spacing_m
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def line_indices(coords_m: NDArray[np.float64], count: int) -> NDArray[np.int64]:
    """Indices of the ``count`` points nearest a 35-degree line through the body.

    This is the separation path: physical deletion strings out along it.
    """
    angle_rad = np.deg2rad(35.0)
    normal = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    centred = coords_m - coords_m.mean(axis=0)
    return np.argsort(np.abs(centred @ normal))[:count]


def scattered_indices(n_points: int, count: int) -> NDArray[np.int64]:
    """Indices of ``count`` points chosen without structure — numerical erosion."""
    return np.random.default_rng(SEED).choice(n_points, size=count, replace=False)


def snapshot_of(
    coords_m: NDArray[np.float64],
    temperature_k: NDArray[np.float64] | None = None,
    step: int = N_SAMPLES - 1,
) -> Snapshot:
    """A snapshot of the given points, with every field health does not read zeroed."""
    n = len(coords_m)
    zeros = np.zeros(n)
    if temperature_k is None:
        temperature_k = np.full(n, AMBIENT_K)
    return Snapshot(
        step=step,
        time_s=step * SAMPLE_DT_S,
        point_ids=np.arange(n),
        coords_m=coords_m,
        velocity_m_s=np.zeros((n, 2)),
        eps_p=zeros,
        eps_rate_per_s=zeros,
        temperature_k=temperature_k,
        # Pristine, and self-consistent: post-initiation damage D cannot
        # accumulate until the initiation sum omega has reached 1.0, so a zeroed
        # damage column needs a zeroed omega beside it to be a state the physics
        # can actually produce. Neither is read by a health diagnostic.
        initiation=zeros,
        damage=zeros,
        mass_kg=np.full(n, 1e-9),
        equivalent_stress_pa=zeros,
        cauchy_xx_pa=zeros,
        cauchy_yy_pa=zeros,
    )


def deletions_of(
    steps: NDArray[np.int64],
    coords_m: NDArray[np.float64],
    temperature_k: NDArray[np.float64] | float = AMBIENT_K,
    mass_kg: float = 1e-9,
) -> Deletion:
    """A deletion record for points that left at the given steps and places."""
    n = len(coords_m)
    return Deletion(
        step=np.asarray(steps, dtype=np.int64),
        point_id=np.arange(n),
        coords_m=coords_m,
        temperature_k=np.broadcast_to(np.asarray(temperature_k, dtype=float), (n,)).copy(),
        mass_kg=np.full(n, mass_kg),
    )


def no_deletions() -> Deletion:
    """The record of a run in which nothing has separated yet."""
    return Deletion(
        step=np.empty(0, dtype=np.int64),
        point_id=np.empty(0, dtype=np.int64),
        coords_m=np.empty((0, 2)),
        temperature_k=np.empty(0),
        mass_kg=np.empty(0),
    )


def series_of(
    *,
    n: int = N_SAMPLES,
    steps: NDArray[np.int64] | None = None,
    time_step_s: NDArray[np.float64] | float = SAMPLE_DT_S,
    kinetic_energy_j: NDArray[np.float64] | float = 0.0,
    internal_energy_j: NDArray[np.float64] | float = 0.0,
    external_work_j: NDArray[np.float64] | float = 0.0,
    deleted_mass_kg: NDArray[np.float64] | float = 0.0,
) -> Series:
    """A series whose columns default to zero, so each test names only what it drives.

    ``steps`` overrides the default one-row-per-step sampling, which is how a
    test varies the adapter's sampling interval without changing the physics.
    """
    steps = np.arange(n, dtype=np.int64) if steps is None else np.asarray(steps, dtype=np.int64)
    n = len(steps)
    fill = np.zeros(n)

    def column(value: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()

    return Series(
        step=steps,
        time_s=steps * SAMPLE_DT_S,
        cutting_force_n=fill,
        thrust_force_n=fill,
        time_step_s=column(time_step_s),
        kinetic_energy_j=column(kinetic_energy_j),
        internal_energy_j=column(internal_energy_j),
        external_work_j=column(external_work_j),
        n_points=np.full(n, LATTICE_N**2, dtype=np.int64),
        deleted_mass_kg=column(deleted_mass_kg),
    )


class TestDeletedMass:
    def test_reports_the_cumulative_loss_as_a_percentage_of_the_starting_body(self):
        series = series_of(deleted_mass_kg=np.linspace(0.0, 0.015, N_SAMPLES))
        assert deleted_mass_pct(series, initial_mass_kg=1.0) == pytest.approx(1.5)

    def test_a_run_that_has_lost_nothing_reports_zero(self):
        assert deleted_mass_pct(series_of(), initial_mass_kg=1.0) == 0.0

    def test_the_bound_separates_an_acceptable_run_from_one_that_has_eroded(self):
        acceptable = series_of(deleted_mass_kg=np.linspace(0.0, 0.015, N_SAMPLES))
        eroded = series_of(deleted_mass_kg=np.linspace(0.0, 0.04, N_SAMPLES))
        assert deleted_mass_pct(acceptable, initial_mass_kg=1.0) < MAX_DELETED_MASS_PCT
        assert deleted_mass_pct(eroded, initial_mass_kg=1.0) > MAX_DELETED_MASS_PCT

    def test_a_body_with_no_mass_is_malformed_rather_than_infinitely_eroded(self):
        with pytest.raises(ValueError, match="initial mass"):
            deleted_mass_pct(series_of(), initial_mass_kg=0.0)


class TestDeletionRate:
    def test_counts_separations_per_second_over_the_window(self):
        # 100 points leave, one every other sampled step, across the whole
        # record. The record spans 199 ns, so a 100 ns window holds 50 of them.
        steps = np.arange(0, 2 * 100, 2, dtype=np.int64)
        deletions = deletions_of(steps, lattice()[: len(steps)])
        window_s = 100 * SAMPLE_DT_S
        assert deletion_rate_per_s(deletions, series_of(), window_s) == pytest.approx(
            50 / window_s, rel=0.05
        )

    def test_separations_older_than_the_window_do_not_count(self):
        early = np.arange(0, 40, dtype=np.int64)
        deletions = deletions_of(early, lattice()[: len(early)])
        assert deletion_rate_per_s(deletions, series_of(), 50 * SAMPLE_DT_S) == 0.0

    def test_a_run_that_has_separated_nothing_has_no_rate(self):
        assert deletion_rate_per_s(no_deletions(), series_of(), 50 * SAMPLE_DT_S) == 0.0

    def test_a_window_longer_than_the_record_is_refused(self):
        with pytest.raises(ValueError, match="window"):
            deletion_rate_per_s(no_deletions(), series_of(), 1.0)

    def test_a_window_of_no_duration_is_refused(self):
        with pytest.raises(ValueError, match="window"):
            deletion_rate_per_s(no_deletions(), series_of(), 0.0)


class TestDeletionSpatialCoherence:
    def test_separation_along_a_line_scores_above_erosion_of_the_same_size(self):
        coords_m = lattice()
        count = 60
        along = line_indices(coords_m, count)
        sprinkled = scattered_indices(len(coords_m), count)
        steps = np.full(count, N_SAMPLES - 1, dtype=np.int64)

        physical = deletion_spatial_coherence(
            deletions_of(steps, coords_m[along]),
            snapshot_of(np.delete(coords_m, along, axis=0)),
            CELL_SIZE_M,
        )
        numerical = deletion_spatial_coherence(
            deletions_of(steps, coords_m[sprinkled]),
            snapshot_of(np.delete(coords_m, sprinkled, axis=0)),
            CELL_SIZE_M,
        )
        # A deletion trail is one point wide, so it never reaches the score a
        # dense plastic-strain band does. What has to hold is the ordering and
        # a wide margin, not a particular magnitude — the estimator normalizes
        # against the parent cloud, so its scale moves with mask sparsity.
        assert 0.0 <= numerical < physical <= 1.0
        assert physical > 10.0 * numerical
        # A floor as well as a margin: when scatter scores exactly 0.0 the
        # margin above collapses into the ordering assertion, and a refactor
        # that degraded the parent cloud without destroying it — windowing the
        # mask but not the stacked coordinates, say — would drop the trail to a
        # few thousandths and still pass both. This is not a magnitude
        # assertion; a trail an order of magnitude below its measured 0.208
        # is no longer telling separation from erosion.
        assert physical > 0.1

    def test_a_run_that_has_separated_nothing_is_not_coherent(self):
        # Nothing has localized, and nothing is the honest score. This is the
        # state a healthy run sits in before the chip starts to form.
        empty = deletion_spatial_coherence(no_deletions(), snapshot_of(lattice()), CELL_SIZE_M)
        assert empty == 0.0

    def test_the_window_hides_old_scatter_behind_recent_separation(self):
        # Early erosion, then the chip starts separating along a line. Scored
        # over the whole run the two mix; scored over the recent window the
        # line is what the supervisor sees.
        coords_m = lattice()
        along = line_indices(coords_m, 60)
        sprinkled = scattered_indices(len(coords_m), 60)
        sprinkled = sprinkled[~np.isin(sprinkled, along)]

        steps = np.concatenate(
            [
                np.zeros(len(sprinkled), dtype=np.int64),
                np.full(len(along), N_SAMPLES - 1, dtype=np.int64),
            ]
        )
        deleted = np.concatenate([sprinkled, along])
        deletions = deletions_of(steps, coords_m[deleted])
        surviving = snapshot_of(np.delete(coords_m, deleted, axis=0))

        windowed = deletion_spatial_coherence(deletions, surviving, CELL_SIZE_M, window_steps=10)
        whole_run = deletion_spatial_coherence(deletions, surviving, CELL_SIZE_M)
        assert windowed > whole_run

    def test_separations_after_the_snapshot_are_not_gone_yet(self):
        # A point that leaves at a later step is still in the body at this
        # snapshot; counting it would place it twice.
        coords_m = lattice()
        along = line_indices(coords_m, 20)
        future_steps = np.full(len(along), N_SAMPLES + 50, dtype=np.int64)
        score = deletion_spatial_coherence(
            deletions_of(future_steps, coords_m[along]),
            snapshot_of(coords_m, step=N_SAMPLES - 1),
            CELL_SIZE_M,
        )
        assert score == 0.0

    def test_a_window_of_no_steps_is_refused(self):
        with pytest.raises(ValueError, match="window"):
            deletion_spatial_coherence(
                no_deletions(), snapshot_of(lattice()), CELL_SIZE_M, window_steps=0
            )


class TestDeletionTemperatureCorrelation:
    def test_points_leaving_from_the_hot_band_correlate_positively(self):
        coords_m = lattice()
        hot = line_indices(coords_m, 60)
        deletions = deletions_of(
            np.full(len(hot), N_SAMPLES - 1, dtype=np.int64),
            coords_m[hot],
            temperature_k=AMBIENT_K + BAND_RISE_K,
        )
        body = snapshot_of(np.delete(coords_m, hot, axis=0))
        assert deletion_temp_correlation(deletions, body) > 0.9

    def test_points_leaving_at_bulk_temperature_do_not_correlate(self):
        # Numerical erosion takes material wherever the integration is unhappy,
        # which has nothing to do with where the heat is.
        rng = np.random.default_rng(SEED)
        coords_m = lattice()
        picked = scattered_indices(len(coords_m), 60)
        temperature_k = AMBIENT_K + rng.normal(0.0, 10.0, len(coords_m))
        deletions = deletions_of(
            np.full(len(picked), N_SAMPLES - 1, dtype=np.int64),
            coords_m[picked],
            temperature_k=temperature_k[picked],
        )
        body = snapshot_of(
            np.delete(coords_m, picked, axis=0),
            temperature_k=np.delete(temperature_k, picked),
        )
        assert abs(deletion_temp_correlation(deletions, body)) < 0.2

    def test_points_leaving_colder_than_the_body_correlate_negatively(self):
        coords_m = lattice()
        picked = scattered_indices(len(coords_m), 60)
        deletions = deletions_of(
            np.full(len(picked), N_SAMPLES - 1, dtype=np.int64),
            coords_m[picked],
            temperature_k=AMBIENT_K - BAND_RISE_K,
        )
        body = snapshot_of(np.delete(coords_m, picked, axis=0))
        assert deletion_temp_correlation(deletions, body) < -0.9

    def test_a_run_that_has_separated_nothing_has_nothing_to_correlate(self):
        assert deletion_temp_correlation(no_deletions(), snapshot_of(lattice())) == 0.0

    def test_an_isothermal_body_has_nothing_to_correlate(self):
        coords_m = lattice()
        picked = scattered_indices(len(coords_m), 60)
        deletions = deletions_of(
            np.full(len(picked), N_SAMPLES - 1, dtype=np.int64), coords_m[picked]
        )
        body = snapshot_of(np.delete(coords_m, picked, axis=0))
        assert deletion_temp_correlation(deletions, body) == 0.0

    def test_separations_after_the_snapshot_are_not_gone_yet(self):
        coords_m = lattice()
        hot = line_indices(coords_m, 60)
        deletions = deletions_of(
            np.full(len(hot), N_SAMPLES + 50, dtype=np.int64),
            coords_m[hot],
            temperature_k=AMBIENT_K + BAND_RISE_K,
        )
        assert deletion_temp_correlation(deletions, snapshot_of(coords_m)) == 0.0


class TestEnergyDrift:
    def test_a_conserved_run_has_no_drift(self):
        work_j = np.linspace(0.0, 1.0, N_SAMPLES)
        series = series_of(internal_energy_j=work_j, external_work_j=work_j)
        assert energy_drift_pct(series) == 0.0

    def test_a_residual_at_the_bound_reports_the_bound(self):
        work_j = np.linspace(0.0, 1.0, N_SAMPLES)
        residual_j = np.linspace(0.0, MAX_ENERGY_DRIFT_PCT / 100.0, N_SAMPLES)
        series = series_of(internal_energy_j=work_j + residual_j, external_work_j=work_j)
        assert energy_drift_pct(series) == pytest.approx(MAX_ENERGY_DRIFT_PCT)

    def test_energy_appearing_from_nowhere_is_positive_and_energy_lost_is_negative(self):
        work_j = np.linspace(0.0, 1.0, N_SAMPLES)
        excess_j = np.linspace(0.0, 0.02, N_SAMPLES)
        created = series_of(
            kinetic_energy_j=excess_j, internal_energy_j=work_j, external_work_j=work_j
        )
        lost = series_of(
            kinetic_energy_j=-excess_j, internal_energy_j=work_j, external_work_j=work_j
        )
        assert energy_drift_pct(created) > 0.0
        assert energy_drift_pct(lost) < 0.0

    def test_the_window_measures_only_the_recent_residual(self):
        # All the drift happens in the first half. A supervisor asking about
        # the steady window should hear that the run is currently clean.
        work_j = np.linspace(0.0, 1.0, N_SAMPLES)
        settled_j = MAX_ENERGY_DRIFT_PCT / 100.0
        residual_j = np.concatenate(
            [np.linspace(0.0, settled_j, N_SAMPLES // 2), np.full(N_SAMPLES // 2, settled_j)]
        )
        series = series_of(internal_energy_j=work_j + residual_j, external_work_j=work_j)
        assert energy_drift_pct(series) == pytest.approx(MAX_ENERGY_DRIFT_PCT)
        assert energy_drift_pct(series, window_steps=50) == pytest.approx(0.0)

    def test_the_same_drift_reads_the_same_however_often_the_adapter_sampled(self):
        # The adapter's sampling policy is not physics. A residual climbing at
        # a fixed rate per step has to report the same percentage whether every
        # step was stored or every hundredth — episodes recorded a year apart
        # under different policies have to stay comparable, and a run whose
        # true drift straddles the bound would otherwise pass or fail on how
        # often someone chose to write a row.
        drift_per_step_j = 1e-5
        total_steps = 20_000
        window_steps = 200

        def sampled_every(stride: int) -> Series:
            steps = np.arange(0, total_steps + 1, stride, dtype=np.int64)
            work_j = np.linspace(0.0, 1.0, len(steps))
            return series_of(
                steps=steps,
                internal_energy_j=work_j + drift_per_step_j * steps,
                external_work_j=work_j,
            )

        dense = energy_drift_pct(sampled_every(1), window_steps=window_steps)
        sparse = energy_drift_pct(sampled_every(100), window_steps=window_steps)
        assert dense == pytest.approx(sparse, rel=1e-9)
        assert dense == pytest.approx(100.0 * drift_per_step_j * window_steps)

    def test_a_window_longer_than_the_record_is_truncated_to_the_record(self):
        # Not extrapolated: drift over steps that were never simulated is
        # fiction, so an over-long window reads exactly like the whole record.
        work_j = np.linspace(0.0, 1.0, N_SAMPLES)
        residual_j = np.linspace(0.0, MAX_ENERGY_DRIFT_PCT / 100.0, N_SAMPLES)
        series = series_of(internal_energy_j=work_j + residual_j, external_work_j=work_j)
        assert energy_drift_pct(series, window_steps=10 * N_SAMPLES) == pytest.approx(
            energy_drift_pct(series)
        )

    def test_a_run_that_has_done_no_work_has_no_drift_to_measure(self):
        with pytest.raises(ValueError, match="work"):
            energy_drift_pct(series_of())


class TestTimeStepSlope:
    # A collapsing step is the trend spec section 0 calls out: dt falling
    # steadily over thousands of steps means the discretization is degenerating
    # under the solver, whatever the instantaneous CFL number says.
    COLLAPSE_PER_STEP_S = -2e-13

    def test_a_steady_step_has_no_slope(self):
        assert dt_slope_per_step(series_of()) == pytest.approx(0.0)

    def test_a_collapsing_step_reports_its_rate_with_a_negative_sign(self):
        steps = np.arange(N_SAMPLES, dtype=float)
        series = series_of(time_step_s=SAMPLE_DT_S + self.COLLAPSE_PER_STEP_S * steps)
        assert dt_slope_per_step(series) == pytest.approx(self.COLLAPSE_PER_STEP_S)

    def test_a_recovering_step_reports_a_positive_slope(self):
        steps = np.arange(N_SAMPLES, dtype=float)
        series = series_of(time_step_s=SAMPLE_DT_S - self.COLLAPSE_PER_STEP_S * steps)
        assert dt_slope_per_step(series) > 0.0

    def test_the_window_sees_a_collapse_the_whole_run_averages_away(self):
        # Flat for most of the run, then falling. Over the whole record the
        # slope is diluted; over the recent window it is the collapse rate.
        half = N_SAMPLES // 2
        time_step_s = np.concatenate(
            [
                np.full(half, SAMPLE_DT_S),
                SAMPLE_DT_S + self.COLLAPSE_PER_STEP_S * np.arange(1, half + 1),
            ]
        )
        series = series_of(time_step_s=time_step_s)
        assert dt_slope_per_step(series, window_steps=half - 1) == pytest.approx(
            self.COLLAPSE_PER_STEP_S, rel=0.05
        )
        assert dt_slope_per_step(series) > self.COLLAPSE_PER_STEP_S

    def test_a_single_sample_cannot_carry_a_trend(self):
        with pytest.raises(ValueError, match="at least 2 sampled steps"):
            dt_slope_per_step(series_of(n=1))

    def test_a_window_of_no_steps_is_refused(self):
        with pytest.raises(ValueError, match="window"):
            dt_slope_per_step(series_of(), window_steps=0)


class TestLagrangianOnlyFields:
    # MPM has no element Jacobian to degenerate, no hourglass modes, and no
    # remesh. These stay in the contract for a future Lagrangian arm; a number
    # here would be invented, and an invented number is worse than none.
    def test_the_mesh_quantities_mpm_cannot_measure_report_nothing(self):
        series = series_of()
        snapshot = snapshot_of(lattice())
        assert hourglass_ie_ratio_pct(series) is None
        assert jacobian_min(snapshot) is None
        assert aspect_ratio_p99(snapshot) is None
        assert plastic_work_delta_last_remesh(series) is None


class TestMalformedRecords:
    # A supervisor reads these messages. Every one has to name the quantity
    # that could not be computed and the record that defeated it — a bare
    # IndexError says neither.
    def test_a_run_that_halted_before_its_first_sample_names_the_diagnostic(self):
        empty = series_of(steps=np.empty(0, dtype=np.int64))
        for compute in (
            lambda: deleted_mass_pct(empty, initial_mass_kg=1.0),
            lambda: deletion_rate_per_s(no_deletions(), empty, SAMPLE_DT_S),
            lambda: energy_drift_pct(empty),
            lambda: dt_slope_per_step(empty),
        ):
            with pytest.raises(ValueError, match="series holds 0"):
                compute()

    def test_a_series_whose_steps_run_backwards_is_refused_not_sign_flipped(self):
        # A writer that resumed from a checkpoint and re-based its step counter
        # emits exactly this. Every windowed quantity divides by a difference of
        # steps, so a backwards record does not merely lose precision: it
        # inverts the sign, and a run leaking energy would report as creating
        # it — the wrong side of any bound tested for excess, so the check that
        # should have fired never does.
        rebased_steps = np.array([10, -5, 0], dtype=np.int64)
        work_j = np.array([0.0, 0.5, 1.0])
        # Raw residual rises by +0.01 J across the two in-window rows; the
        # unchecked record span of -10 steps would report that as -2.0%.
        rebased = series_of(
            steps=rebased_steps,
            internal_energy_j=work_j + np.array([0.0, 0.0, 0.01]),
            external_work_j=work_j,
        )
        with pytest.raises(ValueError, match="steps increase"):
            energy_drift_pct(rebased)
        with pytest.raises(ValueError, match="steps increase"):
            deletion_rate_per_s(no_deletions(), rebased, SAMPLE_DT_S)

    def test_a_deletion_recorded_outside_the_series_is_refused_not_clamped(self):
        # Interpolation would silently pin it to an endpoint and count it in a
        # window it may not belong to, inflating the rate with a fabricated time.
        stray = deletions_of(np.array([N_SAMPLES + 500], dtype=np.int64), lattice()[:1])
        with pytest.raises(ValueError, match="outside the series' step range"):
            deletion_rate_per_s(stray, series_of(), 50 * SAMPLE_DT_S)
