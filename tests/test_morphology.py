"""Chip morphology as the outcome variable — thickness, contact, and regime.

Chip regime classification is scored against ground truth, so these tests are
where the thresholds are pinned. Every regime edge is exercised from both
sides *near where it sits*, not at 0 and 1: a fixture pair that only
establishes ordering leaves the threshold free to be moved anywhere between
them without a test noticing.

Geometry is built by hand rather than taken from a solver run, and it is built
against a *moving* tool. The tool travels 500 um in the bench case, so a
fixture that lays the chip against the t=0 rake face bakes in "the tool has not
moved" and validates the code against its own assumption. Every geometric
fixture here therefore takes a time, places the face where the tool actually is
at that time, and the tests assert the measurement is unchanged by travel.

The other thing a hand-built cloud is for: laying a slab of chosen thickness
along a face of chosen rake angle is the only way to show the estimator
measures perpendicular to the rake face rather than vertically. The two agree
at zero rake and disagree everywhere else, which is exactly how a wrong
estimator survives a single-case check.
"""

from __future__ import annotations

import numpy as np
import pytest

from quinney.diagnostics.bands import MIN_RESOLVED_BAND_ELEMENTS
from quinney.diagnostics.morphology import (
    CHIP_LIFT_TOLERANCE_CELLS,
    CONTACT_TOLERANCE_CELLS,
    DISCONTINUOUS_OSCILLATION_RATIO_MIN,
    MAX_SEGMENT_PITCH_T0,
    MIN_ENGAGED_FORCE_FRACTION,
    MIN_RESOLVED_SEGMENT_CELLS,
    REGIME_EDGES,
    SEGMENT_SIGNIFICANCE_RATIO,
    SEGMENTED_PERIODICITY_MIN,
    ChipRegime,
    chip_thickness_m,
    chip_thickness_ratio,
    classify_chip_regime,
    plausible_segment_frequency_band_hz,
    segment_frequency_hz,
    tool_chip_contact_length_m,
)
from quinney.telemetry import CaseMetadata, Snapshot
from test_forces import (
    SAMPLE_INTERVAL_S,
    a_series,
    duty_cycled,
    sine_wave,
    square_wave,
    tone_in_noise,
)

# The bench orthogonal-cutting case, in the geometry build_case.py lays out:
# the tool starts *outside* the workpiece, stood off by one cell, and drives
# into it at 20 m/s.
CELL_M = 50.0e-6
DEPTH_OF_CUT_M = 100.0e-6
SURFACE_Y_M = 0.30e-3
WORKPIECE_X1_M = 1.0e-3
TOOL_GAP_M = 50.0e-6
CUTTING_SPEED_M_S = 20.0
TOOL_TIP_X0_M = WORKPIECE_X1_M + TOOL_GAP_M
TOOL_TIP_Y_M = SURFACE_Y_M - DEPTH_OF_CUT_M

# The bench cut is 450 um long; a snapshot this late is the one the t=0 tip
# would be most wrong about.
FULL_CUT_TIME_S = 500.0e-6 / CUTTING_SPEED_M_S


def a_case(
    rake_angle_deg: float = 0.0, cutting_speed_m_s: float = CUTTING_SPEED_M_S
) -> CaseMetadata:
    """The bench case: zero rake, 100 um cut, 50 um cells, tool driving in -x."""
    return CaseMetadata(
        case_name="synthetic",
        material="ofhc_copper",
        formulation="mpm",
        cutting_speed_m_s=cutting_speed_m_s,
        tool_velocity_m_s=(-cutting_speed_m_s, 0.0),
        rake_angle_deg=rake_angle_deg,
        depth_of_cut_m=DEPTH_OF_CUT_M,
        cell_size_m=CELL_M,
        time_step_s=8.0e-9,
        workpiece_bounds_m=(0.0, WORKPIECE_X1_M, 0.0, SURFACE_Y_M),
        tool_tip_m=(TOOL_TIP_X0_M, TOOL_TIP_Y_M),
        # The 1.0 x 0.3 mm workpiece at OFHC copper's 8960 kg/m3, per metre of
        # cut width: a zero here is degenerate and other diagnostics reject it.
        initial_mass_kg=8960.0 * WORKPIECE_X1_M * SURFACE_Y_M,
        friction_coefficient=0.35,
        friction_factor=0.7,
        deletion_threshold=1.0,
    )


def a_snapshot(coords_m: np.ndarray, time_s: float = 0.0) -> Snapshot:
    """A snapshot whose only interesting content is where its points are.

    ``initiation`` ramps but never completes and ``damage`` is therefore zero
    throughout: damage is post-initiation, so a point cannot carry any before
    its initiation sum reaches 1.0, and a fixture that violated that would be
    describing a state the material law cannot produce.
    """
    coords_m = np.asarray(coords_m, dtype=float).reshape(-1, 2)
    n = len(coords_m)
    return Snapshot(
        step=250,
        time_s=time_s,
        point_ids=np.arange(n, dtype=np.int64),
        coords_m=coords_m,
        velocity_m_s=np.zeros((n, 2)),
        eps_p=np.zeros(n),
        eps_rate_per_s=np.zeros(n),
        temperature_k=np.full(n, 300.0),
        initiation=np.linspace(0.0, 0.5, n),
        damage=np.zeros(n),
        mass_kg=np.full(n, 1.4e-9),
        equivalent_stress_pa=np.full(n, 300.0e6),
        cauchy_xx_pa=np.full(n, -600.0e6),
        cauchy_yy_pa=np.full(n, -400.0e6),
    )


def tip_at(time_s: float) -> np.ndarray:
    """Where the fixture's tool tip is at a given time."""
    return np.array([TOOL_TIP_X0_M - CUTTING_SPEED_M_S * time_s, TOOL_TIP_Y_M])


def chip_slab(
    thickness_m: float,
    height_m: float,
    rake_angle_deg: float = 0.0,
    spacing_m: float = CELL_M / 4.0,
    time_s: float = 0.0,
) -> np.ndarray:
    """Points filling a slab of the given thickness lying along the rake face.

    Built in rake-face coordinates — along the face and perpendicular to it —
    around the tip's position at ``time_s``, so its thickness is exact whatever
    the rake angle and however far the tool has travelled.
    """
    alpha = np.radians(rake_angle_deg)
    along_axis = np.array([-np.sin(alpha), np.cos(alpha)])
    offset_axis = np.array([-np.cos(alpha), -np.sin(alpha)])

    along = np.arange(0.0, height_m + 0.5 * spacing_m, spacing_m)
    offset = np.arange(0.0, thickness_m + 0.5 * spacing_m, spacing_m)
    grid_along, grid_offset = np.meshgrid(along, offset, indexing="ij")

    return (
        tip_at(time_s)
        + grid_along.reshape(-1, 1) * along_axis
        + grid_offset.reshape(-1, 1) * offset_axis
    )


def tool_body(time_s: float = 0.0, height_m: float = 200.0e-6) -> np.ndarray:
    """Material points of the tool itself, behind its rake face and above its tip."""
    tip = tip_at(time_s)
    x = tip[0] + np.arange(0.0, 200.0e-6, CELL_M / 4.0)
    y = tip[1] + np.arange(0.0, height_m, CELL_M / 4.0)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def uncut_surface(n: int = 40, time_s: float = 0.0) -> np.ndarray:
    """Workpiece points sitting exactly on the original free surface."""
    x = np.linspace(0.0, tip_at(time_s)[0] - CELL_M, n)
    return np.column_stack([x, np.full(n, SURFACE_Y_M)])


class TestGeometryFollowsTheTool:
    """CaseMetadata holds the t=0 tip, and the tool is the thing that moves."""

    def test_thickness_is_unchanged_by_how_far_the_tool_has_travelled(self):
        case = a_case()
        early = a_snapshot(chip_slab(250.0e-6, 600.0e-6, time_s=0.0), time_s=0.0)
        late = a_snapshot(
            chip_slab(250.0e-6, 600.0e-6, time_s=FULL_CUT_TIME_S), time_s=FULL_CUT_TIME_S
        )

        assert chip_thickness_m(late, case) == pytest.approx(chip_thickness_m(early, case))
        assert chip_thickness_m(late, case) == pytest.approx(250.0e-6, rel=0.02)

    def test_a_stationary_tip_would_have_added_the_travel_to_the_chip(self):
        # The failure this replaces, stated as the number it produced: measured
        # against the t=0 face, a 250 um chip 500 um downstream reads 750 um
        # and its ratio 0.13 instead of 0.4.
        case = a_case()
        late = a_snapshot(
            chip_slab(250.0e-6, 600.0e-6, time_s=FULL_CUT_TIME_S), time_s=FULL_CUT_TIME_S
        )
        travel_m = CUTTING_SPEED_M_S * FULL_CUT_TIME_S

        assert chip_thickness_m(late, case) < 0.5 * (250.0e-6 + travel_m)
        assert chip_thickness_ratio(late, case) == pytest.approx(0.4, rel=0.02)

    def test_contact_length_survives_the_tool_advancing(self):
        # Against a stationary face, nothing is ever within one cell of a plane
        # 500 um away, so contact reads 0.0 for the whole run.
        case = a_case()
        late = a_snapshot(
            chip_slab(0.5 * CELL_M, 300.0e-6, time_s=FULL_CUT_TIME_S), time_s=FULL_CUT_TIME_S
        )

        assert tool_chip_contact_length_m(late, case) == pytest.approx(300.0e-6, rel=0.05)

    def test_tool_material_is_kept_out_of_the_chip_after_the_tool_moves(self):
        # The tool sits behind its own rake face, so it has negative offset and
        # is excluded — but only if the face is where the tool actually is. Put
        # the face back at t=0 and the whole 200 um tool lands on the chip side.
        case = a_case()
        cloud = np.vstack(
            [
                chip_slab(250.0e-6, 600.0e-6, time_s=FULL_CUT_TIME_S),
                tool_body(time_s=FULL_CUT_TIME_S),
            ]
        )
        snapshot = a_snapshot(cloud, time_s=FULL_CUT_TIME_S)

        assert chip_thickness_m(snapshot, case) == pytest.approx(250.0e-6, rel=0.02)

    def test_an_episode_that_never_recorded_the_travel_direction_is_refused(self):
        # Schema 1.0 episodes NaN-fill tool_velocity_m_s. Measuring against the
        # t=0 tip would be wrong by the whole travel distance, so the caller is
        # told rather than handed geometry from the wrong place.
        case = a_case()
        unrecorded = CaseMetadata(
            **{
                **{f.name: getattr(case, f.name) for f in case.__dataclass_fields__.values()},
                "tool_velocity_m_s": (float("nan"), float("nan")),
            }
        )
        snapshot = a_snapshot(chip_slab(250.0e-6, 600.0e-6), time_s=FULL_CUT_TIME_S)

        with pytest.raises(ValueError, match="tool_velocity_m_s"):
            chip_thickness_m(snapshot, unrecorded)


class TestChipThickness:
    def test_a_slab_of_known_thickness_recovers_its_thickness_ratio(self):
        # t0 = 100 um against a 250 um chip is r = 0.4, a plausible OFHC value.
        snapshot = a_snapshot(chip_slab(thickness_m=250.0e-6, height_m=600.0e-6))
        case = a_case()

        assert chip_thickness_m(snapshot, case) == pytest.approx(250.0e-6, rel=0.02)
        assert chip_thickness_ratio(snapshot, case) == pytest.approx(0.4, rel=0.02)

    def test_a_real_chip_is_thicker_than_the_cut_so_the_ratio_is_below_one(self):
        snapshot = a_snapshot(chip_slab(thickness_m=250.0e-6, height_m=600.0e-6))

        assert chip_thickness_ratio(snapshot, a_case()) < 1.0

    def test_thickness_is_measured_perpendicular_to_a_raked_face(self):
        # At 20 degrees rake the slab's vertical extent and its thickness
        # differ by 1/cos(20) = 6%, so a vertical-extent estimator fails here
        # and passes the zero-rake case above.
        case = a_case(rake_angle_deg=20.0)
        snapshot = a_snapshot(chip_slab(250.0e-6, 600.0e-6, rake_angle_deg=20.0))

        assert chip_thickness_m(snapshot, case) == pytest.approx(250.0e-6, rel=0.02)

    def test_undisturbed_material_is_not_a_chip(self):
        # Nothing has been lifted above the original surface: there is no chip
        # to measure, and reporting one would invent the outcome variable.
        snapshot = a_snapshot(uncut_surface())

        with pytest.raises(ValueError, match="chip"):
            chip_thickness_m(snapshot, a_case())

    def test_a_handful_of_ejected_points_is_not_a_chip(self):
        stray = np.column_stack(
            [
                np.full(3, TOOL_TIP_X0_M - CELL_M),
                SURFACE_Y_M + np.array([1.0, 2.0, 3.0]) * CELL_M,
            ]
        )
        snapshot = a_snapshot(np.vstack([uncut_surface(), stray]))

        with pytest.raises(ValueError, match="chip"):
            chip_thickness_m(snapshot, a_case())

    def test_surface_jitter_below_the_lift_tolerance_is_not_a_chip(self):
        # Grid interpolation lifts free-surface points by a fraction of a cell.
        jitter = uncut_surface(200)
        jitter[:, 1] += 0.5 * CHIP_LIFT_TOLERANCE_CELLS * CELL_M
        snapshot = a_snapshot(jitter)

        with pytest.raises(ValueError, match="chip"):
            chip_thickness_m(snapshot, a_case())

    def test_a_single_ejected_point_does_not_set_the_thickness(self):
        far = np.array([[TOOL_TIP_X0_M - 900.0e-6, SURFACE_Y_M + 400.0e-6]])
        snapshot = a_snapshot(np.vstack([chip_slab(250.0e-6, 600.0e-6), far]))

        assert chip_thickness_m(snapshot, a_case()) == pytest.approx(250.0e-6, rel=0.02)


class TestContactLength:
    def test_contact_length_is_the_extent_of_chip_along_the_rake_face(self):
        # A slab thinner than the contact tolerance hugs the face for its
        # whole height, so contact length is that height.
        snapshot = a_snapshot(chip_slab(thickness_m=0.5 * CELL_M, height_m=300.0e-6))

        assert tool_chip_contact_length_m(snapshot, a_case()) == pytest.approx(300.0e-6, rel=0.05)

    def test_a_raked_face_measures_contact_along_the_face_not_vertically(self):
        case = a_case(rake_angle_deg=20.0)
        snapshot = a_snapshot(chip_slab(0.5 * CELL_M, 300.0e-6, rake_angle_deg=20.0))

        assert tool_chip_contact_length_m(snapshot, case) == pytest.approx(300.0e-6, rel=0.05)

    def test_a_chip_that_has_curled_off_the_face_is_not_in_contact(self):
        # Every point sits further from the face than the contact tolerance.
        detached = chip_slab(thickness_m=0.5 * CELL_M, height_m=300.0e-6)
        detached[:, 0] -= 3.0 * CONTACT_TOLERANCE_CELLS * CELL_M
        snapshot = a_snapshot(detached)

        assert tool_chip_contact_length_m(snapshot, a_case()) == pytest.approx(0.0)

    def test_material_below_the_tool_tip_is_the_machined_surface_not_contact(self):
        below = np.column_stack(
            [
                np.full(40, TOOL_TIP_X0_M - 0.2 * CELL_M),
                np.linspace(0.0, TOOL_TIP_Y_M - CELL_M, 40),
            ]
        )
        snapshot = a_snapshot(np.vstack([below, chip_slab(0.5 * CELL_M, 300.0e-6)]))

        assert tool_chip_contact_length_m(snapshot, a_case()) == pytest.approx(300.0e-6, rel=0.05)

    def test_a_tool_that_has_not_touched_anything_is_not_in_contact(self):
        # The bench stands the tool off by exactly one cell, which is the
        # contact tolerance, so the undisturbed workpiece face sits inside it
        # before the tool has cut anything. Without a chip there is no
        # tool-chip contact length, whatever is near the face.
        approaching = np.column_stack(
            [
                np.full(60, TOOL_TIP_X0_M - CELL_M),
                np.linspace(0.0, SURFACE_Y_M, 60),
            ]
        )
        snapshot = a_snapshot(approaching)

        assert tool_chip_contact_length_m(snapshot, a_case()) == pytest.approx(0.0)


class TestSegmentFrequency:
    def test_a_flat_force_signal_has_no_segment_frequency(self):
        # A continuous chip does not segment. Reporting the largest FFT bin of
        # a flat record as a segmentation frequency is the failure to avoid.
        assert segment_frequency_hz(a_series(np.full(512, 1000.0)), a_case()) is None

    def test_an_oscillation_below_the_significance_floor_has_no_segment_frequency(self):
        # 30 N on a 1 kN mean is an oscillation ratio of 0.030, under the 0.05
        # floor. Stated absolutely rather than as a multiple of the constant:
        # a fixture scaled by the edge it is testing moves with it and can
        # never pin where it sits.
        series = a_series(square_wave(512, 8, mean_n=1000.0, amplitude_n=30.0))

        assert segment_frequency_hz(series, a_case()) is None

    def test_an_oscillation_above_the_significance_floor_reports_its_frequency(self):
        # 70 N on a 1 kN mean is a ratio of 0.070, over the 0.05 floor.
        n, cycles = 512, 8
        series = a_series(square_wave(n, cycles, 1000.0, 70.0))
        # A uniform wave completes the same cycles per second whatever window
        # it is measured over, so this holds without restating the discard.
        expected_hz = cycles / (n * SAMPLE_INTERVAL_S)

        assert segment_frequency_hz(series, a_case()) == pytest.approx(expected_hz, rel=1e-9)

    def test_a_record_where_the_tool_never_engages_has_no_segment_frequency(self):
        assert segment_frequency_hz(a_series(np.zeros(512)), a_case()) is None

    def test_an_aperiodic_record_has_no_segment_frequency(self):
        # This function and classify_chip_regime are one answer, not two: it
        # must not report a confident frequency for a record the classifier is
        # calling DISCONTINUOUS.
        series = a_series(_irregular_drops())
        result = classify_chip_regime(series, a_case())

        assert result.regime is ChipRegime.DISCONTINUOUS
        assert segment_frequency_hz(series, a_case()) is None
        assert segment_frequency_hz(series, a_case()) == result.segment_frequency_hz


class TestRegimeEdges:
    def test_every_edge_carries_a_cited_reason(self):
        assert REGIME_EDGES
        for name, edge in REGIME_EDGES.items():
            assert edge.reason.strip(), f"{name} has no cited reason"
            assert np.isfinite(edge.value)

    def test_the_named_constants_are_the_table(self):
        # One canonical definition per edge: the constants must be the table's
        # values, not a second copy of them.
        assert SEGMENT_SIGNIFICANCE_RATIO == REGIME_EDGES["segment_significance_ratio"].value
        assert SEGMENTED_PERIODICITY_MIN == REGIME_EDGES["segmented_periodicity_min"].value
        assert (
            DISCONTINUOUS_OSCILLATION_RATIO_MIN
            == REGIME_EDGES["discontinuous_oscillation_ratio_min"].value
        )
        assert MIN_ENGAGED_FORCE_FRACTION == REGIME_EDGES["min_engaged_force_fraction"].value
        assert MAX_SEGMENT_PITCH_T0 == REGIME_EDGES["max_segment_pitch_t0"].value
        assert MIN_RESOLVED_SEGMENT_CELLS == REGIME_EDGES["min_resolved_segment_cells"].value

    def test_the_resolution_edge_is_the_spec_rule_not_a_second_copy_of_it(self):
        # "Under about three elements across is mesh-limited, not physics" is
        # one spec section 5 rule; bands.py owns the number.
        assert MIN_RESOLVED_SEGMENT_CELLS == MIN_RESOLVED_BAND_ELEMENTS


class TestPlausibleFrequencyBand:
    def test_the_band_brackets_a_segment_pitch_of_the_order_of_the_cut(self):
        lowest_hz, highest_hz = plausible_segment_frequency_band_hz(a_case())
        one_pitch_per_uncut_thickness_hz = CUTTING_SPEED_M_S / DEPTH_OF_CUT_M

        assert lowest_hz < one_pitch_per_uncut_thickness_hz / 2.0 < highest_hz

    def test_cell_crossing_sits_outside_the_band(self):
        # v / h is the frequency at which material points cross background
        # cells. It is regular, so nothing else in this module can reject it.
        _lowest_hz, highest_hz = plausible_segment_frequency_band_hz(a_case())
        cell_crossing_hz = CUTTING_SPEED_M_S / CELL_M

        assert cell_crossing_hz > highest_hz
        assert cell_crossing_hz / highest_hz == pytest.approx(MIN_RESOLVED_SEGMENT_CELLS)


class TestChipRegimeClassification:
    """Each edge from both sides. A classifier that always commits measures nothing."""

    def test_a_flat_force_record_is_a_continuous_chip(self):
        result = classify_chip_regime(a_series(np.full(512, 1000.0)), a_case())

        assert result.regime is ChipRegime.CONTINUOUS
        assert result.segment_frequency_hz is None

    def test_an_oscillation_just_below_the_significance_edge_is_continuous(self):
        # A 41 N sine on a 1 kN mean measures 0.040, just under the 0.05 edge.
        # Absolute, not a multiple of the constant: a fixture that scales with
        # the edge cannot say where the edge is.
        result = classify_chip_regime(a_series(sine_wave(512, 8, 1000.0, 41.0)), a_case())

        assert result.oscillation_ratio == pytest.approx(0.040, abs=0.002)
        assert result.regime is ChipRegime.CONTINUOUS

    def test_a_periodic_oscillation_just_above_the_significance_edge_is_segmented(self):
        # 61 N measures 0.060, just over. Nothing else about the two records
        # differs: both are pure tones at the same in-band frequency.
        result = classify_chip_regime(a_series(sine_wave(512, 8, 1000.0, 61.0)), a_case())

        assert result.oscillation_ratio == pytest.approx(0.060, abs=0.002)
        assert result.regime is ChipRegime.SEGMENTED
        assert result.segment_frequency_hz is not None

    def test_the_same_amplitude_broadband_is_not_segmented(self):
        # Amplitude alone cannot separate segmentation from noise: what makes
        # a chip segmented is that the force is periodic.
        rng = np.random.default_rng(20260819)
        noisy = a_series(rng.normal(1000.0, 120.0, 512))
        periodic = a_series(sine_wave(512, 8, 1000.0, 197.0))

        assert classify_chip_regime(periodic, a_case()).regime is ChipRegime.SEGMENTED
        assert classify_chip_regime(noisy, a_case()).regime is not ChipRegime.SEGMENTED

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_large_irregular_force_drops_are_a_discontinuous_chip(self, seed):
        # Parametrized over seeds because a single irregular record that lands
        # on the right side of the edge proves nothing about where the edge is.
        result = classify_chip_regime(a_series(_irregular_drops(seed)), a_case())

        assert result.regime is ChipRegime.DISCONTINUOUS
        assert result.oscillation_ratio >= DISCONTINUOUS_OSCILLATION_RATIO_MIN
        assert result.periodicity < SEGMENTED_PERIODICITY_MIN

    def test_equally_large_but_periodic_drops_are_a_segmented_chip(self):
        # A chip that collapses on a clock is still one attached chip; what
        # makes a chip discontinuous is that the fragments arrive irregularly.
        series = a_series(square_wave(512, 8, mean_n=1000.0, amplitude_n=1000.0))
        result = classify_chip_regime(series, a_case())

        assert result.regime is ChipRegime.SEGMENTED
        assert result.oscillation_ratio >= DISCONTINUOUS_OSCILLATION_RATIO_MIN

    def test_a_moderate_aperiodic_wobble_is_indeterminate(self):
        # Neither flat, nor periodic, nor severe: the honest answer is that
        # this record does not say which regime it is.
        rng = np.random.default_rng(20260819)
        result = classify_chip_regime(a_series(rng.normal(1000.0, 120.0, 512)), a_case())

        assert result.regime is ChipRegime.INDETERMINATE

    def test_a_record_too_short_to_resolve_a_spectrum_is_indeterminate(self):
        # Too few samples to tell periodic from irregular, and the classifier
        # must say so rather than guess from amplitude alone.
        series = a_series(_irregular_drops()[:40])

        assert classify_chip_regime(series, a_case()).regime is ChipRegime.INDETERMINATE

    def test_the_deciding_values_come_back_with_the_regime(self):
        # spec section 6 requires `evidence`: the supervisor has to be able to
        # cite the numbers that produced the call.
        result = classify_chip_regime(a_series(sine_wave(512, 8, 1000.0, 200.0)), a_case())

        assert result.oscillation_ratio == pytest.approx(0.198, abs=0.02)
        assert result.periodicity > SEGMENTED_PERIODICITY_MIN
        assert result.engagement_fraction > 0.0
        assert "periodicity" in result.reason


class TestThePeriodicityEdgeSitsWhereItSays:
    """Fixtures 0.03 either side of 0.50, so the edge cannot be moved unnoticed."""

    # SNR 0.8 puts a tone in noise right at the edge; these two seeds land at
    # periodicity 0.485 and 0.515, bracketing 0.50 tightly. Both have an
    # oscillation ratio between the significance floor and the fragmentation
    # edge, so the only thing separating their regimes is the periodicity.
    JUST_BELOW_SEED = 1
    JUST_ABOVE_SEED = 26

    def test_a_repeat_just_under_the_edge_is_not_called_segmentation(self):
        result = classify_chip_regime(a_series(tone_in_noise(0.8, self.JUST_BELOW_SEED)), a_case())

        assert result.periodicity == pytest.approx(0.485, abs=0.005)
        assert result.periodicity < SEGMENTED_PERIODICITY_MIN
        assert result.regime is ChipRegime.INDETERMINATE

    def test_a_repeat_just_over_the_edge_is_called_segmentation(self):
        result = classify_chip_regime(a_series(tone_in_noise(0.8, self.JUST_ABOVE_SEED)), a_case())

        assert result.periodicity == pytest.approx(0.515, abs=0.005)
        assert result.periodicity >= SEGMENTED_PERIODICITY_MIN
        assert result.regime is ChipRegime.SEGMENTED


class TestTheEngagementEdgeSitsWhereItSays:
    """A duty-cycled force gives an engagement fraction of exactly 1 / duty."""

    def test_a_window_just_under_the_engagement_edge_is_indeterminate(self):
        result = classify_chip_regime(a_series(duty_cycled(12)), a_case())

        assert result.engagement_fraction == pytest.approx(1.0 / 12.0, rel=0.02)
        assert result.regime is ChipRegime.INDETERMINATE
        assert "engagement floor" in result.reason

    def test_a_window_just_over_the_engagement_edge_is_judged_on_its_force(self):
        result = classify_chip_regime(a_series(duty_cycled(7)), a_case())

        assert result.engagement_fraction == pytest.approx(1.0 / 7.0, rel=0.02)
        assert "engagement floor" not in result.reason


class TestPeriodicNumericalArtefacts:
    """A regular artefact is the thing most likely to be claimed as segmentation."""

    def test_a_repeat_at_the_cell_crossing_frequency_is_not_called_segmentation(self):
        # v / h ringing is regular, so periodicity alone would call it a
        # segmented chip and report the grid frequency as the segmentation
        # frequency — the instrument producing the confusion it exists to
        # resolve. 20 cycles in the steady window is 391 kHz against a band
        # topping out at 133 kHz.
        n = 512
        series = a_series(sine_wave(n, 40, 1000.0, 200.0))
        result = classify_chip_regime(series, a_case())

        assert result.periodicity > SEGMENTED_PERIODICITY_MIN
        assert result.regime is ChipRegime.INDETERMINATE
        assert result.segment_frequency_hz is None
        assert result.dominant_freq_hz is not None
        assert "cell crossing" in result.reason

    def test_a_repeat_too_slow_to_be_a_segment_pitch_is_not_called_segmentation(self):
        # The same 78 kHz signal against a case cutting three times faster:
        # its segment pitch would have to be over five uncut thicknesses.
        fast = a_case(cutting_speed_m_s=3.0 * CUTTING_SPEED_M_S)
        lowest_hz, _highest_hz = plausible_segment_frequency_band_hz(fast)
        series = a_series(sine_wave(512, 8, 1000.0, 200.0))
        result = classify_chip_regime(series, fast)

        assert result.dominant_freq_hz < lowest_hz
        assert result.periodicity > SEGMENTED_PERIODICITY_MIN
        assert result.regime is ChipRegime.INDETERMINATE

    def test_the_same_repeat_inside_the_band_is_segmentation(self):
        # The control for both tests above: only the band moved.
        series = a_series(sine_wave(512, 8, 1000.0, 200.0))
        result = classify_chip_regime(series, a_case())

        assert result.regime is ChipRegime.SEGMENTED
        assert result.segment_frequency_hz == result.dominant_freq_hz


class TestFailLoudDoesNotDependOnTheAnswer:
    def test_unevenly_sampled_time_is_refused_whatever_the_regime_would_be(self):
        # An FFT over unevenly spaced samples is invalid regardless of which
        # branch the classification would have taken, so the check cannot live
        # on the segmented branch alone.
        n = 512
        time_s = np.arange(n, dtype=float) * SAMPLE_INTERVAL_S
        # Inside the steady window, not at its edge: a rate change in the
        # discarded approach is not this diagnostic's problem.
        time_s[3 * n // 4 :] += 40.0 * SAMPLE_INTERVAL_S
        rng = np.random.default_rng(20260819)

        for record in (np.full(n, 1000.0), rng.normal(1000.0, 120.0, n), _irregular_drops()):
            with pytest.raises(ValueError, match="spacing"):
                classify_chip_regime(a_series(record, time_s=time_s), a_case())


def _irregular_drops(seed: int = 4242) -> np.ndarray:
    """A cutting force interrupted by fragments separating at irregular intervals."""
    rng = np.random.default_rng(seed)
    record: list[float] = []
    cutting = True
    while len(record) < 512:
        run = int(rng.integers(4, 22))
        record += [1000.0 if cutting else 40.0] * run
        cutting = not cutting
    return np.array(record[:512])
