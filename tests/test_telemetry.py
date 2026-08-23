"""The telemetry contract as downstream consumers will find it on disk.

The supervisor never touches live solver state. It reads an episode directory
written by a process that has already exited, so everything a diagnostic needs
has to survive that round trip unchanged: shear-band coherence reads
``Snapshot.coords_m`` and ``eps_p``, chip morphology reads a snapshot against
``CaseMetadata``, force spectra read ``Series``, deletion health reads
``Deletion``.

Two failure modes matter more than the rest. A truncated array reaching a
diagnostic produces a plausible number from a point set that never existed, so
length disagreement must raise at construction. A schema that quietly changed
meaning makes episodes stored a year apart incomparable, so a major-version
mismatch must refuse to load. These tests pin both down.
"""

# Keep this import. The future-contract records below are classified by their
# annotation text, exactly as quinney.telemetry classifies its own — see
# _COLUMN_DTYPES there.
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pytest
from numpy.typing import NDArray

from quinney import telemetry
from quinney.materials.johnson_cook import triaxiality_from_plane_strain_stress
from quinney.telemetry import (
    INTRODUCED_IN,
    PLANAR_COLUMN,
    SCHEMA_VERSION,
    CaseMetadata,
    Deletion,
    Episode,
    Series,
    Snapshot,
    append_snapshot,
    final_snapshot,
    load_episode,
    load_snapshot,
    write_manifest,
    write_series,
)


def a_case() -> CaseMetadata:
    return CaseMetadata(
        case_name="baseline_2m_s",
        material="ofhc_copper",
        formulation="mpm",
        cutting_speed_m_s=2.0,
        tool_velocity_m_s=(-2.0, 0.0),
        rake_angle_deg=5.0,
        depth_of_cut_m=1.0e-4,
        cell_size_m=1.25e-5,
        time_step_s=3.7e-9,
        workpiece_bounds_m=(0.0, 2.0e-3, -5.0e-4, 0.0),
        tool_tip_m=(1.0e-3, -1.0e-4),
        initial_mass_kg=8.96e-3,
        friction_coefficient=0.35,
        friction_factor=0.7,
        deletion_threshold=1.0,
    )


def a_snapshot(step: int = 250, n_points: int = 3) -> Snapshot:
    index = np.arange(n_points, dtype=np.float64)
    return Snapshot(
        step=step,
        time_s=step * 3.7e-9,
        point_ids=np.arange(n_points, dtype=np.int64) + 100,
        coords_m=np.column_stack([index * 1e-5, index * 2e-5]),
        velocity_m_s=np.column_stack([index * -2.0, index * 0.5]),
        eps_p=index * 0.25,
        eps_rate_per_s=index * 1.0e5,
        temperature_k=300.0 + index * 120.0,
        initiation=index * 0.3,
        damage=index * 0.1,
        mass_kg=np.full(n_points, 1.4e-9),
        equivalent_stress_pa=300.0e6 + index * 1.0e6,
        cauchy_xx_pa=-600.0e6 - index * 2.0e6,
        cauchy_yy_pa=-400.0e6 - index * 1.0e6,
    )


def a_series(n_steps: int = 4) -> Series:
    index = np.arange(n_steps, dtype=np.float64)
    return Series(
        step=np.arange(n_steps, dtype=np.int64) * 250,
        time_s=index * 9.25e-7,
        cutting_force_n=200.0 + index * 3.0,
        thrust_force_n=80.0 - index * 1.5,
        time_step_s=np.full(n_steps, 3.7e-9),
        kinetic_energy_j=index * 1.0e-3,
        internal_energy_j=index * 5.0e-3,
        external_work_j=index * 6.0e-3,
        n_points=np.full(n_steps, 4096, dtype=np.int64),
        deleted_mass_kg=index * 2.0e-10,
    )


def a_deletion(n_records: int = 2) -> Deletion:
    index = np.arange(n_records, dtype=np.float64)
    return Deletion(
        step=np.array([250, 500], dtype=np.int64)[:n_records],
        point_id=np.array([1001, 1002], dtype=np.int64)[:n_records],
        coords_m=np.column_stack([index * 1e-4, index * -1e-4]),
        temperature_k=900.0 + index * 50.0,
        mass_kg=np.full(n_records, 1.4e-9),
    )


def no_deletions() -> Deletion:
    return Deletion(
        step=np.zeros(0, dtype=np.int64),
        point_id=np.zeros(0, dtype=np.int64),
        coords_m=np.zeros((0, 2)),
        temperature_k=np.zeros(0),
        mass_kg=np.zeros(0),
    )


def write_whole_episode(episode_dir, snapshots=(), deletions=None) -> None:
    write_manifest(episode_dir, a_case(), [snapshot.step for snapshot in snapshots])
    write_series(episode_dir, a_series(), a_deletion() if deletions is None else deletions)
    for snapshot in snapshots:
        append_snapshot(episode_dir, snapshot)


class TestRoundTrip:
    def test_case_metadata_survives_the_manifest_unchanged(self, tmp_path):
        write_whole_episode(tmp_path)
        assert load_episode(tmp_path).metadata == a_case()

    def test_geometry_tuples_come_back_as_tuples(self, tmp_path):
        # JSON has no tuple, and a diagnostic unpacking x0, x1, y0, y1 must not
        # care that the value made a trip through a list.
        write_whole_episode(tmp_path)
        metadata = load_episode(tmp_path).metadata
        assert metadata.workpiece_bounds_m == (0.0, 2.0e-3, -5.0e-4, 0.0)
        assert metadata.tool_tip_m == (1.0e-3, -1.0e-4)

    def test_series_columns_survive_bit_for_bit(self, tmp_path):
        # Force spectra are computed from these samples; a rounded write would
        # move the serration frequency.
        write_whole_episode(tmp_path)
        loaded = load_episode(tmp_path).series
        expected = a_series()
        for name in (
            "step",
            "time_s",
            "cutting_force_n",
            "thrust_force_n",
            "time_step_s",
            "kinetic_energy_j",
            "internal_energy_j",
            "external_work_j",
            "n_points",
            "deleted_mass_kg",
        ):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(expected, name))

    def test_step_counters_stay_integers(self, tmp_path):
        write_whole_episode(tmp_path)
        loaded = load_episode(tmp_path)
        assert loaded.series.step.dtype == np.int64
        assert loaded.series.n_points.dtype == np.int64

    def test_deletion_records_survive(self, tmp_path):
        write_whole_episode(tmp_path)
        loaded = load_episode(tmp_path).deletions
        expected = a_deletion()
        np.testing.assert_array_equal(loaded.point_id, expected.point_id)
        np.testing.assert_array_equal(loaded.coords_m, expected.coords_m)
        np.testing.assert_array_equal(loaded.temperature_k, expected.temperature_k)

    def test_a_run_with_no_deletions_yet_round_trips(self, tmp_path):
        # Early in a cut nothing has separated; an empty record set is normal
        # and must not need a special case downstream.
        write_whole_episode(tmp_path, deletions=no_deletions())
        loaded = load_episode(tmp_path).deletions
        assert loaded.point_id.shape == (0,)
        assert loaded.coords_m.shape == (0, 2)

    def test_snapshot_fields_survive_bit_for_bit(self, tmp_path):
        write_whole_episode(tmp_path, snapshots=[a_snapshot()])
        loaded = load_snapshot(tmp_path, 250)
        expected = a_snapshot()
        assert loaded.step == expected.step
        assert loaded.time_s == expected.time_s
        for name in (
            "point_ids",
            "coords_m",
            "velocity_m_s",
            "eps_p",
            "eps_rate_per_s",
            "temperature_k",
            "initiation",
            "damage",
            "mass_kg",
            "equivalent_stress_pa",
            "cauchy_xx_pa",
            "cauchy_yy_pa",
        ):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(expected, name))

    def test_snapshots_may_shrink_between_samples(self, tmp_path):
        # Erasure changes the point count, which is the whole reason each
        # sample gets its own file instead of one stacked array.
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250, 5), a_snapshot(500, 3)])
        assert load_snapshot(tmp_path, 250).eps_p.shape == (5,)
        assert load_snapshot(tmp_path, 500).eps_p.shape == (3,)

    def test_episode_indexes_its_snapshots(self, tmp_path):
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250), a_snapshot(500)])
        assert load_episode(tmp_path).snapshot_steps == (250, 500)

    def test_episode_records_the_schema_version_it_was_written_under(self, tmp_path):
        write_whole_episode(tmp_path)
        assert load_episode(tmp_path).schema_version == SCHEMA_VERSION


class TestFinalSnapshot:
    def test_returns_the_latest_sampled_step(self, tmp_path):
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250), a_snapshot(1000)])
        assert final_snapshot(tmp_path).step == 1000

    def test_refuses_an_episode_that_sampled_nothing(self, tmp_path):
        write_whole_episode(tmp_path)
        with pytest.raises(ValueError, match="no snapshots"):
            final_snapshot(tmp_path)


class TestMissingSnapshot:
    def test_names_the_step_that_was_never_written(self, tmp_path):
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250)])
        with pytest.raises(FileNotFoundError, match="750"):
            load_snapshot(tmp_path, 750)


class TestLengthInvariants:
    def test_a_truncated_snapshot_array_names_both_fields(self):
        # Accepting this hands a diagnostic a plausible number computed from a
        # point set that never existed.
        with pytest.raises(ValueError, match="point_ids.*eps_p"):
            Snapshot(
                step=1,
                time_s=0.0,
                point_ids=np.array([1, 2, 3], dtype=np.int64),
                coords_m=np.zeros((3, 2)),
                velocity_m_s=np.zeros((3, 2)),
                eps_p=np.zeros(2),
                eps_rate_per_s=np.zeros(3),
                temperature_k=np.zeros(3),
                initiation=np.zeros(3),
                damage=np.zeros(3),
                mass_kg=np.zeros(3),
                equivalent_stress_pa=np.zeros(3),
                cauchy_xx_pa=np.zeros(3),
                cauchy_yy_pa=np.zeros(3),
            )

    def test_a_short_series_column_raises(self):
        full = a_series()
        with pytest.raises(ValueError, match="cutting_force_n"):
            Series(
                step=full.step,
                time_s=full.time_s,
                cutting_force_n=full.cutting_force_n[:-1],
                thrust_force_n=full.thrust_force_n,
                time_step_s=full.time_step_s,
                kinetic_energy_j=full.kinetic_energy_j,
                internal_energy_j=full.internal_energy_j,
                external_work_j=full.external_work_j,
                n_points=full.n_points,
                deleted_mass_kg=full.deleted_mass_kg,
            )

    def test_a_short_deletion_column_raises(self):
        full = a_deletion()
        with pytest.raises(ValueError, match="temperature_k"):
            Deletion(
                step=full.step,
                point_id=full.point_id,
                coords_m=full.coords_m,
                temperature_k=full.temperature_k[:-1],
                mass_kg=full.mass_kg,
            )

    def test_flat_coordinates_are_rejected(self):
        # (2n,) and (n, 2) hold the same numbers; only one of them means
        # anything to a band-orientation fit.
        with pytest.raises(ValueError, match=r"coords_m.*\(n, 2\)"):
            Deletion(
                step=np.zeros(2, dtype=np.int64),
                point_id=np.zeros(2, dtype=np.int64),
                coords_m=np.zeros(4),
                temperature_k=np.zeros(2),
                mass_kg=np.zeros(2),
            )

    def test_workpiece_bounds_must_have_four_edges(self):
        with pytest.raises(ValueError, match="workpiece_bounds_m"):
            CaseMetadata(**{**vars(a_case()), "workpiece_bounds_m": (0.0, 1.0)})

    def test_tool_tip_must_be_a_point_in_the_plane(self):
        with pytest.raises(ValueError, match="tool_tip_m"):
            CaseMetadata(**{**vars(a_case()), "tool_tip_m": (0.0, 1.0, 2.0)})


def rewritten_manifest(episode_dir) -> dict:
    return json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))


def save_manifest(episode_dir, manifest) -> None:
    (episode_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestSchemaVersioning:
    def test_a_major_version_bump_refuses_to_load_and_names_both_versions(self, tmp_path):
        write_whole_episode(tmp_path)
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "2.0"
        save_manifest(tmp_path, manifest)
        with pytest.raises(ValueError, match="2.0") as raised:
            load_episode(tmp_path)
        assert SCHEMA_VERSION in str(raised.value)

    def test_a_minor_version_bump_still_loads(self, tmp_path):
        # Minor changes are additive by contract, so an older reader keeps
        # working against a newer episode.
        write_whole_episode(tmp_path)
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "1.2"
        save_manifest(tmp_path, manifest)
        assert load_episode(tmp_path).schema_version == "1.2"

    def test_an_unknown_metadata_field_from_a_newer_writer_is_ignored(self, tmp_path):
        # This is what makes "additive minor change" true rather than aspirational.
        write_whole_episode(tmp_path)
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "1.2"
        manifest["metadata"]["tool_edge_radius_m"] = 5.0e-6
        save_manifest(tmp_path, manifest)
        assert load_episode(tmp_path).metadata == a_case()

    def test_a_missing_metadata_field_names_what_is_missing(self, tmp_path):
        write_whole_episode(tmp_path)
        manifest = rewritten_manifest(tmp_path)
        del manifest["metadata"]["rake_angle_deg"]
        save_manifest(tmp_path, manifest)
        with pytest.raises(ValueError, match="rake_angle_deg"):
            load_episode(tmp_path)

    def test_a_missing_series_column_names_what_is_missing(self, tmp_path):
        write_whole_episode(tmp_path)
        columns = dict(np.load(tmp_path / "series.npz"))
        del columns["thrust_force_n"]
        np.savez(tmp_path / "series.npz", **columns)
        with pytest.raises(ValueError, match="thrust_force_n"):
            load_episode(tmp_path)


class TestEpisodeShape:
    def test_an_episode_carries_everything_a_supervisor_reads(self, tmp_path):
        write_whole_episode(tmp_path, snapshots=[a_snapshot()])
        episode = load_episode(tmp_path)
        assert isinstance(episode, Episode)
        assert episode.metadata.formulation == "mpm"
        assert episode.series.cutting_force_n.size == 4
        assert episode.deletions.step.size == 2
        assert isinstance(episode.snapshot_steps, tuple)


class TestOfflineDamageRecomputation:
    """The contract has to be able to feed the damage law, not merely store numbers."""

    def test_a_stored_snapshot_drives_the_triaxiality_the_damage_law_needs(self, tmp_path):
        # Previewing a deletion-threshold change means re-running Johnson-Cook
        # damage over telemetry alone, with no solver attached. If triaxiality
        # cannot be rebuilt from what was written, the supervisor's what-if and
        # the running simulation can disagree about what the change would do.
        compressed = Snapshot(
            step=500,
            time_s=1.85e-6,
            point_ids=np.array([1, 2], dtype=np.int64),
            coords_m=np.zeros((2, 2)),
            velocity_m_s=np.zeros((2, 2)),
            eps_p=np.array([0.4, 0.8]),
            eps_rate_per_s=np.array([1.0e5, 2.0e5]),
            temperature_k=np.array([600.0, 900.0]),
            initiation=np.array([1.0, 1.0]),
            damage=np.array([0.2, 0.5]),
            mass_kg=np.full(2, 1.4e-9),
            equivalent_stress_pa=np.array([250.0e6, 400.0e6]),
            cauchy_xx_pa=np.array([-600.0e6, -300.0e6]),
            cauchy_yy_pa=np.array([-400.0e6, -100.0e6]),
        )
        write_manifest(tmp_path, a_case(), [compressed.step])
        write_series(tmp_path, a_series(), no_deletions())
        append_snapshot(tmp_path, compressed)

        loaded = final_snapshot(tmp_path)
        triaxiality = triaxiality_from_plane_strain_stress(
            loaded.cauchy_xx_pa, loaded.cauchy_yy_pa, loaded.equivalent_stress_pa
        )
        # Mean stress is (sxx + syy)/2: -500 MPa over 250 MPa, then -200 MPa
        # over 400 MPa. Both negative, because a cut is compressive.
        assert triaxiality == pytest.approx([-2.0, -0.5])

    def test_the_stress_components_a_snapshot_carries_are_the_ones_that_law_takes(self, tmp_path):
        # Guards against the two columns drifting apart from the signature
        # they exist to satisfy: the law takes (sxx, syy, sigma_eq), so a
        # snapshot must expose exactly those three, per point.
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250, 4)])
        loaded = load_snapshot(tmp_path, 250)
        assert loaded.cauchy_xx_pa.shape == loaded.equivalent_stress_pa.shape
        assert loaded.cauchy_yy_pa.shape == loaded.point_ids.shape
        assert (
            triaxiality_from_plane_strain_stress(
                loaded.cauchy_xx_pa, loaded.cauchy_yy_pa, loaded.equivalent_stress_pa
            ).shape
            == loaded.point_ids.shape
        )


# The dataclasses a later minor version of the contract might carry. They stand
# in for next month's telemetry.py: same fields, plus one that did not exist
# when the episodes on disk were written.


@dataclass(frozen=True)
class SnapshotWithShear(Snapshot):
    sxy_pa: NDArray[np.float64] = field(metadata={INTRODUCED_IN: "1.2"})


@dataclass(frozen=True)
class SnapshotWithTenthMinorShear(Snapshot):
    sxy_pa: NDArray[np.float64] = field(metadata={INTRODUCED_IN: "1.10"})


@dataclass(frozen=True)
class SeriesWithContactCount(Series):
    contact_count: NDArray[np.int64] = field(metadata={INTRODUCED_IN: "1.2"})


@dataclass(frozen=True)
class CaseWithEdgeRadius(CaseMetadata):
    tool_edge_radius_m: float = field(metadata={INTRODUCED_IN: "1.2"})


class TestGrowingTheContract:
    """Adding a field must not orphan the episodes already on disk.

    An episode is a multi-minute solver run, and the early ones carry expert
    judgement that cannot be regenerated. The ladder measurement re-runs that
    corpus for years, so the first added diagnostic field must not make the
    whole corpus unreadable.
    """

    def test_an_episode_written_before_a_column_existed_loads_with_it_nan_filled(
        self, tmp_path, monkeypatch
    ):
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250, 4)])
        monkeypatch.setattr(telemetry, "Snapshot", SnapshotWithShear)

        loaded = load_snapshot(tmp_path, 250)

        assert loaded.sxy_pa.shape == (4,)
        assert np.isnan(loaded.sxy_pa).all()
        # NaN means "not recorded under this schema version". A diagnostic that
        # reaches for it gets NaN out, not a plausible number built on a zero
        # nobody measured.
        assert np.isnan(loaded.sxy_pa.mean())
        # Everything that did exist still round-trips untouched.
        np.testing.assert_array_equal(loaded.eps_p, a_snapshot(250, 4).eps_p)

    def test_a_planar_column_introduced_later_is_nan_filled_in_the_right_shape(
        self, tmp_path, monkeypatch
    ):
        @dataclass(frozen=True)
        class DeletionWithVelocity(Deletion):
            velocity_m_s: NDArray[np.float64] = field(
                metadata={**PLANAR_COLUMN, INTRODUCED_IN: "1.2"}
            )

        write_whole_episode(tmp_path)
        monkeypatch.setattr(telemetry, "Deletion", DeletionWithVelocity)

        loaded = load_episode(tmp_path).deletions

        assert loaded.velocity_m_s.shape == (2, 2)
        assert np.isnan(loaded.velocity_m_s).all()

    def test_a_scalar_introduced_later_is_nan_filled(self, tmp_path, monkeypatch):
        write_whole_episode(tmp_path)
        monkeypatch.setattr(telemetry, "CaseMetadata", CaseWithEdgeRadius)

        assert np.isnan(load_episode(tmp_path).metadata.tool_edge_radius_m)

    def test_the_tenth_minor_version_is_newer_than_the_ninth(self, tmp_path, monkeypatch):
        # Compared as text, "1.10" sorts before "1.9", which would make a
        # column from the tenth revision look required by an episode from the
        # ninth and reject a corpus that is perfectly readable.
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250, 4)])
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "1.9"
        save_manifest(tmp_path, manifest)
        monkeypatch.setattr(telemetry, "Snapshot", SnapshotWithTenthMinorShear)

        assert np.isnan(load_snapshot(tmp_path, 250).sxy_pa).all()

    def test_a_column_the_episode_claims_to_carry_is_still_required(self, tmp_path, monkeypatch):
        # The permissive path must not swallow genuine corruption: at 1.2 the
        # column should be there, so its absence is a broken file.
        write_whole_episode(tmp_path, snapshots=[a_snapshot()])
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "1.2"
        save_manifest(tmp_path, manifest)
        monkeypatch.setattr(telemetry, "Snapshot", SnapshotWithShear)

        with pytest.raises(ValueError, match="sxy_pa"):
            load_snapshot(tmp_path, 250)

    def test_an_integer_column_introduced_later_refuses_by_name(self, tmp_path, monkeypatch):
        # Integers have no NaN, so there is no honest stand-in. Saying so beats
        # inventing a zero count of contacts that were never counted.
        write_whole_episode(tmp_path)
        monkeypatch.setattr(telemetry, "Series", SeriesWithContactCount)

        with pytest.raises(ValueError, match="contact_count"):
            load_episode(tmp_path)

    def test_an_extra_series_column_from_a_newer_writer_is_ignored(self, tmp_path):
        # The other direction: today's reader opening tomorrow's episode.
        write_whole_episode(tmp_path)
        columns = dict(np.load(tmp_path / "series.npz"))
        columns["contact_pressure_pa"] = np.zeros(4)
        np.savez(tmp_path / "series.npz", **columns)
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "1.2"
        save_manifest(tmp_path, manifest)

        loaded = load_episode(tmp_path)

        assert not hasattr(loaded.series, "contact_pressure_pa")
        np.testing.assert_array_equal(loaded.series.cutting_force_n, a_series().cutting_force_n)


class TestLazySnapshotLoadIsGated:
    def test_a_snapshot_from_an_incompatible_major_version_is_refused(self, tmp_path):
        # A diagnostic that lazily loads one snapshot never calls load_episode,
        # so without a gate here the contract's primary read path has none.
        write_whole_episode(tmp_path, snapshots=[a_snapshot()])
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "2.0"
        save_manifest(tmp_path, manifest)

        with pytest.raises(ValueError, match="2.0"):
            load_snapshot(tmp_path, 250)


class TestManifestReconciliation:
    def test_a_manifest_that_lags_the_snapshots_on_disk_refuses_to_answer(self, tmp_path):
        # This is the run that matters: the solver died of the mesh collapse the
        # supervisor exists to diagnose, before the halt path rewrote the index.
        # Handing back the state 250 steps before the failure would be worse
        # than refusing, because nothing about it looks wrong.
        write_manifest(tmp_path, a_case(), [250])
        write_series(tmp_path, a_series(), a_deletion())
        append_snapshot(tmp_path, a_snapshot(250))
        append_snapshot(tmp_path, a_snapshot(500))

        with pytest.raises(ValueError, match="500") as raised:
            final_snapshot(tmp_path)
        assert "250" in str(raised.value)

    def test_an_agreeing_manifest_still_answers(self, tmp_path):
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250), a_snapshot(500)])
        assert final_snapshot(tmp_path).step == 500


class TestScalarWhereAColumnBelongs:
    def test_naming_the_field_rather_than_failing_inside_len(self):
        # One point's worth of state passed unwrapped used to surface as
        # "len() of unsized object" with no clue which field was at fault.
        with pytest.raises(ValueError, match="temperature_k"):
            Deletion(
                step=np.array([1], dtype=np.int64),
                point_id=np.array([9], dtype=np.int64),
                coords_m=np.zeros((1, 2)),
                temperature_k=900.0,
                mass_kg=np.full(1, 1.4e-9),
            )


@dataclass(frozen=True)
class DeletionLedByANewColumn:
    """A later Deletion whose added column is declared first, not appended.

    Nothing obliges a new column to go last, and a record's first column is
    what the loader sizes its NaN fills from.
    """

    arrival_time_s: NDArray[np.float64] = field(metadata={INTRODUCED_IN: "1.2"})
    step: NDArray[np.int64]
    point_id: NDArray[np.int64]
    coords_m: NDArray[np.float64] = field(metadata=PLANAR_COLUMN)
    temperature_k: NDArray[np.float64]
    mass_kg: NDArray[np.float64]


class TestUnclassifiableAnnotations:
    """An annotation this contract cannot read is refused, not demoted."""

    def test_an_aliased_array_annotation_is_refused_by_name(self):
        # npt.NDArray[np.float64] is the same type, spelled differently. Reading
        # it as a scalar would write it, read it back, and skip every length,
        # dtype, and shape check on the way — a truncated column reaching a band
        # diagnostic with nothing raising.
        @dataclass(frozen=True)
        class SnapshotWithAliasedStress(Snapshot):
            sxy_pa: npt.NDArray[np.float64]

        with pytest.raises(TypeError, match="sxy_pa"):
            SnapshotWithAliasedStress(**vars(a_snapshot(250, 3)), sxy_pa=np.zeros(3))

    def test_a_malformed_introduced_in_version_is_refused(self):
        # A comma for a dot would otherwise lie dormant until someone loads an
        # episode old enough for the version to be compared.
        @dataclass(frozen=True)
        class SnapshotWithTypo(Snapshot):
            sxy_pa: NDArray[np.float64] = field(metadata={INTRODUCED_IN: "1,1"})

        with pytest.raises(ValueError, match="1,1"):
            SnapshotWithTypo(**vars(a_snapshot(250, 3)), sxy_pa=np.zeros(3))


class TestNewColumnDeclaredFirst:
    def test_the_row_count_comes_from_a_column_that_is_actually_there(self, tmp_path, monkeypatch):
        # Sizing the fill from the first *declared* column reaches for the very
        # column being filled, and the load dies with a bare KeyError naming
        # nothing.
        write_whole_episode(tmp_path)
        monkeypatch.setattr(telemetry, "Deletion", DeletionLedByANewColumn)

        loaded = load_episode(tmp_path).deletions

        assert loaded.arrival_time_s.shape == (2,)
        assert np.isnan(loaded.arrival_time_s).all()
        np.testing.assert_array_equal(loaded.point_id, a_deletion().point_id)


class TestToolVelocity:
    """The travel direction, and what a 1.0 episode does without it."""

    def test_the_velocity_vector_survives_the_manifest(self, tmp_path):
        write_whole_episode(tmp_path)
        assert load_episode(tmp_path).metadata.tool_velocity_m_s == (-2.0, 0.0)

    def test_an_episode_can_place_the_tool_tip_at_the_time_of_a_snapshot(self, tmp_path):
        # The tool moves. Chip thickness and contact length measured against the
        # t=0 tip read the wrong geometry by however far it has travelled, and
        # read it plausibly. The vector is what makes tip(t) computable at all.
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250)])
        episode = load_episode(tmp_path)
        snapshot = load_snapshot(tmp_path, 250)

        vx, vy = episode.metadata.tool_velocity_m_s
        x0, y0 = episode.metadata.tool_tip_m
        tip = (x0 + vx * snapshot.time_s, y0 + vy * snapshot.time_s)

        assert tip[0] < x0  # this case cuts in -x
        assert tip[1] == pytest.approx(y0)

    def test_the_scalar_speed_is_the_magnitude_of_the_vector(self, tmp_path):
        # Two spellings of one quantity. A case where they disagree is a case
        # whose geometry and whose Merchant relations describe different cuts.
        metadata = load_episode(write_whole_episode(tmp_path) or tmp_path).metadata
        vx, vy = metadata.tool_velocity_m_s
        assert np.hypot(vx, vy) == pytest.approx(metadata.cutting_speed_m_s)

    def test_a_one_point_zero_episode_still_loads_with_no_direction_recorded(self, tmp_path):
        # A real episode on disk from before the field existed. It must not
        # become unreadable, and it must not acquire a direction nobody wrote.
        write_whole_episode(tmp_path)
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "1.0"
        del manifest["metadata"]["tool_velocity_m_s"]
        save_manifest(tmp_path, manifest)

        metadata = load_episode(tmp_path).metadata

        assert np.isnan(metadata.tool_velocity_m_s).all()
        assert metadata.cutting_speed_m_s == 2.0  # what 1.0 did record is intact

    def test_geometry_derived_from_a_missing_direction_comes_out_nan(self, tmp_path):
        # The whole reason the fill is NaN and not zero: a consumer that forgets
        # to check gets NaN, not a tool that plausibly never moved.
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250)])
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "1.0"
        del manifest["metadata"]["tool_velocity_m_s"]
        save_manifest(tmp_path, manifest)

        metadata = load_episode(tmp_path).metadata
        vx, _ = metadata.tool_velocity_m_s
        assert np.isnan(metadata.tool_tip_m[0] + vx * load_snapshot(tmp_path, 250).time_s)

    def test_a_one_point_zero_episode_keeps_its_own_version(self, tmp_path):
        write_whole_episode(tmp_path)
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "1.0"
        del manifest["metadata"]["tool_velocity_m_s"]
        save_manifest(tmp_path, manifest)
        assert load_episode(tmp_path).schema_version == "1.0"

    def test_a_one_point_one_episode_must_carry_the_direction(self, tmp_path):
        # At 1.1 the field is not optional, so its absence is a broken file
        # rather than an old one.
        write_whole_episode(tmp_path)
        manifest = rewritten_manifest(tmp_path)
        del manifest["metadata"]["tool_velocity_m_s"]
        save_manifest(tmp_path, manifest)
        with pytest.raises(ValueError, match="tool_velocity_m_s"):
            load_episode(tmp_path)

    def test_a_direction_that_is_not_a_point_in_the_plane_is_rejected(self):
        with pytest.raises(ValueError, match="tool_velocity_m_s"):
            CaseMetadata(**{**vars(a_case()), "tool_velocity_m_s": (1.0, 0.0, 0.0)})


def strip_snapshot_column(episode_dir, step: int, name: str) -> None:
    """Rewrite a snapshot file as an older writer would have left it."""
    path = episode_dir / "snapshots" / f"step_{step:06d}.npz"
    columns = dict(np.load(path))
    del columns[name]
    np.savez(path, **columns)


class TestInitiationAndDamage:
    """Failure is two-phase, so one number cannot describe it."""

    def test_both_phases_survive_the_round_trip_separately(self, tmp_path):
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250, 3)])
        loaded = load_snapshot(tmp_path, 250)
        np.testing.assert_array_equal(loaded.initiation, a_snapshot(250, 3).initiation)
        np.testing.assert_array_equal(loaded.damage, a_snapshot(250, 3).damage)

    def test_a_pristine_point_is_distinguishable_from_one_about_to_initiate(self, tmp_path):
        # Both carry damage 0.0, because D only starts once omega reaches 1.0.
        # Without omega on disk the supervisor cannot see that lowering
        # deletion_threshold would take the second point and not the first.
        about_to_go = Snapshot(
            step=250,
            time_s=9.25e-7,
            point_ids=np.array([1, 2], dtype=np.int64),
            coords_m=np.zeros((2, 2)),
            velocity_m_s=np.zeros((2, 2)),
            eps_p=np.array([0.01, 0.95]),
            eps_rate_per_s=np.array([1.0e3, 1.0e5]),
            temperature_k=np.array([300.0, 850.0]),
            initiation=np.array([0.0, 0.98]),
            damage=np.zeros(2),
            mass_kg=np.full(2, 1.4e-9),
            equivalent_stress_pa=np.array([250.0e6, 400.0e6]),
            cauchy_xx_pa=np.array([-600.0e6, -300.0e6]),
            cauchy_yy_pa=np.array([-400.0e6, -100.0e6]),
        )
        write_manifest(tmp_path, a_case(), [about_to_go.step])
        write_series(tmp_path, a_series(), no_deletions())
        append_snapshot(tmp_path, about_to_go)

        loaded = final_snapshot(tmp_path)

        assert (loaded.damage == 0.0).all()
        assert loaded.initiation[0] == pytest.approx(0.0)
        assert loaded.initiation[1] == pytest.approx(0.98)

    def test_a_one_point_zero_snapshot_loads_with_initiation_unrecorded(self, tmp_path):
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250, 4)])
        strip_snapshot_column(tmp_path, 250, "initiation")
        manifest = rewritten_manifest(tmp_path)
        manifest["schema_version"] = "1.0"
        save_manifest(tmp_path, manifest)

        loaded = load_snapshot(tmp_path, 250)

        # NaN, never 0.0: a real 0.0 would mean pristine, and claiming every
        # point in an old episode is pristine is the wrong answer confidently.
        assert np.isnan(loaded.initiation).all()
        assert loaded.initiation.shape == (4,)
        np.testing.assert_array_equal(loaded.damage, a_snapshot(250, 4).damage)

    def test_a_one_point_one_snapshot_must_carry_initiation(self, tmp_path):
        write_whole_episode(tmp_path, snapshots=[a_snapshot(250)])
        strip_snapshot_column(tmp_path, 250, "initiation")
        with pytest.raises(ValueError, match="initiation"):
            load_snapshot(tmp_path, 250)
