"""Force statistics as a supervisor will read them off a halted run.

Every test here builds a signal whose answer is known by construction — a
square wave whose frequency was chosen, a flat record, a ramp with a tone
buried in it. That is the only way to show an estimator is right rather than
merely plausible: a real cutting record has no ground truth to check against,
which is precisely why these diagnostics exist.

Four failure modes are worth more than the rest, and each has its own test.
Averaging a force record that still contains the tool's approach reports a mean
of two different physical regimes. Taking the sample spacing from the step
index rather than from the time column reports a frequency that is wrong by
whatever the sampling stride was. Reporting the lowest FFT bin of a drifting
record as a real oscillation turns a trend into a segmentation frequency. And
letting a single force spike set any denominator lets one sample decide what
the whole record was.
"""

from __future__ import annotations

import numpy as np
import pytest

from quinney.diagnostics.forces import (
    MIN_OSCILLATION_SAMPLES,
    STEADY_STATE_DISCARD_FRACTION,
    cutting_force_mean_n,
    force_dominant_freq_hz,
    force_engagement_fraction,
    force_oscillation_amplitude_n,
    force_oscillation_ratio,
    force_periodicity,
    steady_window,
    thrust_force_mean_n,
)
from quinney.telemetry import Series

# Telemetry sampling stride. The bench case steps the solver at 8 ns; the
# series is sampled far less often than that. 200 ns is chosen so that a
# plausible chip-segmentation frequency for the bench geometry lands in
# mid-band, which is what the morphology tests need to exercise.
SAMPLE_INTERVAL_S = 2.0e-7


def a_series(
    cutting_force_n: np.ndarray,
    thrust_force_n: np.ndarray | None = None,
    sample_interval_s: float = SAMPLE_INTERVAL_S,
    time_s: np.ndarray | None = None,
) -> Series:
    """A Series carrying the given force record and nothing else of interest."""
    cutting_force_n = np.asarray(cutting_force_n, dtype=float)
    n = len(cutting_force_n)
    if thrust_force_n is None:
        thrust_force_n = 0.4 * cutting_force_n
    if time_s is None:
        time_s = np.arange(n, dtype=float) * sample_interval_s
    return Series(
        step=np.arange(n, dtype=np.int64),
        time_s=time_s,
        cutting_force_n=cutting_force_n,
        thrust_force_n=np.asarray(thrust_force_n, dtype=float),
        time_step_s=np.full(n, 8.0e-9),
        kinetic_energy_j=np.zeros(n),
        internal_energy_j=np.zeros(n),
        external_work_j=np.zeros(n),
        n_points=np.full(n, 1200, dtype=np.int64),
        deleted_mass_kg=np.zeros(n),
    )


def square_wave(n: int, cycles: int, mean_n: float, amplitude_n: float) -> np.ndarray:
    """A square wave with exactly ``cycles`` whole periods across n samples."""
    phase = (np.arange(n) * cycles / n) % 1.0
    return mean_n + np.where(phase < 0.5, amplitude_n, -amplitude_n)


def sine_wave(n: int, cycles: int, mean_n: float, amplitude_n: float) -> np.ndarray:
    return mean_n + amplitude_n * np.sin(2.0 * np.pi * cycles * np.arange(n) / n)


def tone_in_noise(snr: float, seed: int, n: int = 512, cycles: int = 8) -> np.ndarray:
    """A tone of the given signal-to-noise power ratio, on a 1 kN mean.

    The tone carries power A^2/2, so noise of variance A^2/(2 * snr) gives the
    stated ratio. This is the fixture the periodicity edge is derived against,
    since a tone in noise has a known periodicity of snr / (1 + snr).
    """
    amplitude_n = 200.0
    noise = np.random.default_rng(seed).normal(0.0, amplitude_n / np.sqrt(2.0 * snr), n)
    return sine_wave(n, cycles, 1000.0, amplitude_n) + noise


def duty_cycled(duty: int, n: int = 512, force_n: float = 1000.0) -> np.ndarray:
    """Force present on one sample in ``duty``, zero on the rest.

    The steady mean is then ``force_n / duty`` while the robust peak stays at
    ``force_n``, so the engagement fraction is 1 / duty by construction — which
    is how the engagement edge gets exercised at its edge rather than at 0 and 1.
    """
    record = np.zeros(n)
    record[::duty] = force_n
    return record


class TestSteadyWindow:
    """The tool approaches before it cuts, so the record has two regimes in it."""

    def test_the_approach_is_excluded_from_the_mean(self):
        approach = np.zeros(256)
        cutting = np.full(256, 1000.0)
        series = a_series(np.concatenate([approach, cutting]))

        assert cutting_force_mean_n(series) == pytest.approx(1000.0)

    def test_averaging_the_whole_record_would_have_halved_the_force(self):
        # The point of the window, stated as the number it avoids.
        record = np.concatenate([np.zeros(256), np.full(256, 1000.0)])
        series = a_series(record)

        assert record.mean() == pytest.approx(500.0)
        assert cutting_force_mean_n(series) == pytest.approx(1000.0)

    def test_the_window_starts_at_the_discard_fraction(self):
        series = a_series(np.zeros(400))

        assert steady_window(series) == slice(int(400 * STEADY_STATE_DISCARD_FRACTION), 400)

    def test_a_larger_discard_keeps_a_later_window(self):
        series = a_series(np.zeros(400))

        assert steady_window(series, discard_fraction=0.75) == slice(300, 400)

    def test_thrust_force_uses_the_same_window(self):
        thrust = np.concatenate([np.zeros(256), np.full(256, 250.0)])
        series = a_series(np.concatenate([np.zeros(256), np.full(256, 1000.0)]), thrust)

        assert thrust_force_mean_n(series) == pytest.approx(250.0)

    def test_a_record_with_one_sample_has_no_steady_window(self):
        with pytest.raises(ValueError, match="steady window"):
            steady_window(a_series(np.array([1000.0])))

    def test_discarding_the_entire_record_is_refused(self):
        with pytest.raises(ValueError, match="discard_fraction"):
            steady_window(a_series(np.zeros(400)), discard_fraction=1.0)


class TestOscillationAmplitude:
    """Amplitude is the excursion about the steady mean, robust to one spike."""

    def test_a_flat_record_has_no_oscillation(self):
        series = a_series(np.full(256, 1000.0))

        assert force_oscillation_amplitude_n(series) == pytest.approx(0.0)
        assert force_oscillation_ratio(series) == pytest.approx(0.0)

    def test_a_square_wave_reports_its_own_amplitude(self):
        series = a_series(square_wave(512, cycles=8, mean_n=1000.0, amplitude_n=200.0))

        assert force_oscillation_amplitude_n(series) == pytest.approx(200.0, rel=1e-6)
        assert force_oscillation_ratio(series) == pytest.approx(0.2, rel=1e-6)

    def test_a_single_spike_does_not_set_the_amplitude(self):
        # One re-contact spike or one deleted point must not make a flat record
        # look segmented.
        record = np.full(512, 1000.0)
        record[300] = 50000.0
        series = a_series(record)

        assert force_oscillation_amplitude_n(series) < 1.0

    def test_the_ratio_is_independent_of_the_force_scale(self):
        # Regime edges are quoted as ratios precisely because absolute force
        # scales with depth of cut and cut width.
        small = a_series(square_wave(512, cycles=8, mean_n=100.0, amplitude_n=20.0))
        large = a_series(square_wave(512, cycles=8, mean_n=100000.0, amplitude_n=20000.0))

        assert force_oscillation_ratio(small) == pytest.approx(force_oscillation_ratio(large))


class TestDominantFrequency:
    def test_a_square_wave_recovers_the_frequency_it_was_built_with(self):
        n, cycles = 512, 8
        series = a_series(square_wave(n, cycles, mean_n=1000.0, amplitude_n=200.0))
        expected_hz = cycles / (n * SAMPLE_INTERVAL_S)

        found = force_dominant_freq_hz(series, window=slice(0, n))

        assert found == pytest.approx(expected_hz, rel=1e-9)

    def test_the_frequency_comes_from_the_time_column_not_the_sample_index(self):
        # Same samples, sampled twice as fast: the frequency must double. A
        # diagnostic that assumed unit spacing would return the same number.
        n, cycles = 512, 8
        record = sine_wave(n, cycles, mean_n=1000.0, amplitude_n=200.0)
        slow = a_series(record, sample_interval_s=SAMPLE_INTERVAL_S)
        fast = a_series(record, sample_interval_s=SAMPLE_INTERVAL_S / 2.0)

        assert force_dominant_freq_hz(fast, window=slice(0, n)) == pytest.approx(
            2.0 * force_dominant_freq_hz(slow, window=slice(0, n))
        )

    def test_the_spacing_is_the_sampling_stride_not_the_solver_step(self):
        # Series.time_step_s is the dt actually taken at one solver step and
        # CaseMetadata.time_step_s the case's nominal dt; neither is the gap
        # between two sampled rows. Here the solver steps at 8 ns while rows
        # are 200 ns apart, so reading the wrong field is wrong by 25x.
        n, cycles = 512, 8
        series = a_series(sine_wave(n, cycles, 1000.0, 200.0))
        assert series.time_step_s[0] != SAMPLE_INTERVAL_S

        found = force_dominant_freq_hz(series, window=slice(0, n))

        assert found == pytest.approx(cycles / (n * SAMPLE_INTERVAL_S), rel=1e-9)

    def test_a_drifting_record_does_not_report_its_drift_as_an_oscillation(self):
        # A ramp puts all its power in the lowest bins. The tone buried in it
        # is the real periodicity and must be what comes back.
        n, cycles = 512, 12
        ramp = np.linspace(0.0, 900.0, n)
        series = a_series(ramp + sine_wave(n, cycles, mean_n=1000.0, amplitude_n=60.0))
        expected_hz = cycles / (n * SAMPLE_INTERVAL_S)

        assert force_dominant_freq_hz(series, window=slice(0, n)) == pytest.approx(
            expected_hz, rel=1e-9
        )

    def test_a_record_too_short_to_resolve_a_peak_is_refused(self):
        n = MIN_OSCILLATION_SAMPLES - 1
        series = a_series(sine_wave(n, 4, 1000.0, 200.0))

        with pytest.raises(ValueError, match="spectrum"):
            force_dominant_freq_hz(series, window=slice(0, n))

    def test_unevenly_sampled_time_is_refused_rather_than_averaged(self):
        # An FFT over samples that are not equally spaced returns a frequency
        # for a signal nobody recorded.
        n = 512
        time_s = np.arange(n, dtype=float) * SAMPLE_INTERVAL_S
        time_s[n // 2 :] += 40.0 * SAMPLE_INTERVAL_S
        series = a_series(sine_wave(n, 8, 1000.0, 200.0), time_s=time_s)

        with pytest.raises(ValueError, match="spacing"):
            force_dominant_freq_hz(series, window=slice(0, n))

    def test_time_running_backwards_is_refused(self):
        n = 64
        series = a_series(sine_wave(n, 8, 1000.0, 200.0), time_s=np.zeros(n))

        with pytest.raises(ValueError, match="increas"):
            force_dominant_freq_hz(series, window=slice(0, n))


class TestPeriodicity:
    """Does the force repeat — the difference between segmented and fragmented."""

    def test_a_pure_tone_repeats_almost_exactly(self):
        n = 512
        series = a_series(sine_wave(n, 8, 1000.0, 200.0))

        assert force_periodicity(series, window=slice(0, n)) > 0.95

    def test_a_flat_record_has_nothing_to_repeat(self):
        series = a_series(np.full(512, 1000.0))

        assert force_periodicity(series, window=slice(0, 512)) == pytest.approx(0.0)

    def test_broadband_noise_does_not_repeat(self):
        n = 512
        noise = np.random.default_rng(20260819).normal(1000.0, 120.0, n)
        series = a_series(noise)

        assert force_periodicity(series, window=slice(0, n)) < 0.25

    @pytest.mark.parametrize("snr", [1.0, 3.0])
    def test_periodicity_follows_the_snr_identity_the_edge_is_derived_from(self, snr):
        # rho = SNR / (1 + SNR) is what makes the segmented edge interpretable
        # rather than fitted, so the estimator has to track it at more than one
        # point. Two SNRs constrain the functional form; a seed sweep pins the
        # mean tightly enough that a bias of the size that would matter at a
        # 0.50 edge cannot hide inside the tolerance.
        n = 512
        measured = np.array(
            [
                force_periodicity(a_series(tone_in_noise(snr, seed, n)), slice(0, n))
                for seed in range(60)
            ]
        )
        identity = snr / (1.0 + snr)
        bias = float(measured.mean()) - identity

        assert float(measured.mean()) == pytest.approx(identity, abs=0.05)
        # Maximising over many lags can only push the estimate up, and the push
        # has to stay small enough not to move the edge.
        assert -0.01 <= bias <= 0.05

    def test_the_bias_shrinks_as_the_repeat_strengthens(self):
        n = 512
        weak, strong = (
            np.mean(
                [
                    force_periodicity(a_series(tone_in_noise(snr, s, n)), slice(0, n))
                    for s in range(60)
                ]
            )
            - snr / (1.0 + snr)
            for snr in (1.0, 3.0)
        )

        assert weak > strong

    def test_alternating_sign_ringing_is_not_a_periodicity(self):
        # Two samples per period is Nyquist, where numerical ringing and a real
        # period are the same sequence. Every multiple of that lag correlates
        # just as strongly, so a search bounded only on the long side would
        # find it at lag four and report a perfect repeat.
        n = 512
        ringing = 1000.0 + 200.0 * (-1.0) ** np.arange(n)
        series = a_series(ringing)

        assert force_periodicity(series, window=slice(0, n)) == pytest.approx(0.0)

    def test_a_single_slow_cycle_is_a_trend_not_a_periodicity(self):
        # One cycle across the window never repeats inside it, so it is drift
        # and must not be reported as a period.
        n = 512
        series = a_series(sine_wave(n, 1, 1000.0, 200.0))

        assert force_periodicity(series, window=slice(0, n)) < 0.15

    def test_a_record_too_short_for_a_periodicity_is_refused(self):
        n = MIN_OSCILLATION_SAMPLES - 1
        series = a_series(sine_wave(n, 4, 1000.0, 200.0))

        with pytest.raises(ValueError, match="periodicity"):
            force_periodicity(series, window=slice(0, n))


class TestColumnsTheEpisodeNeverRecorded:
    """NaN in a column means "not recorded at this schema version", never zero."""

    def test_an_unrecorded_force_column_is_refused_rather_than_averaged(self):
        # The contract NaN-fills a column the episode predates. np.mean would
        # return NaN as though it were a force; the caller has to be told.
        record = np.full(512, 1000.0)
        record[400] = np.nan
        series = a_series(record)

        with pytest.raises(ValueError, match="never recorded"):
            cutting_force_mean_n(series)

    def test_an_unrecorded_column_is_refused_by_the_spectrum_too(self):
        record = np.full(512, 1000.0)
        record[400] = np.nan
        series = a_series(record)

        with pytest.raises(ValueError, match="never recorded"):
            force_dominant_freq_hz(series, window=slice(0, 512))

    def test_a_column_this_module_does_not_read_is_not_policed(self):
        # Only the columns actually consumed are checked. A NaN in an unrelated
        # column must not make the force statistics unavailable.
        series = a_series(np.full(512, 1000.0))
        series.internal_energy_j[:] = np.nan

        assert cutting_force_mean_n(series) == pytest.approx(1000.0)


class TestEngagement:
    """Is there a steady cutting force to divide by at all."""

    def test_a_fully_engaged_cut_reports_a_high_fraction(self):
        series = a_series(np.concatenate([np.zeros(256), np.full(256, 1000.0)]))

        assert force_engagement_fraction(series) == pytest.approx(1.0)

    def test_a_window_that_missed_the_cut_reports_a_low_fraction(self):
        # Cutting happened early and the steady window sees only the retract.
        series = a_series(np.concatenate([np.full(256, 1000.0), np.zeros(256)]))

        assert force_engagement_fraction(series) == pytest.approx(0.0)

    def test_a_record_with_no_force_at_all_reports_zero(self):
        assert force_engagement_fraction(a_series(np.zeros(256))) == pytest.approx(0.0)

    def test_a_single_spike_does_not_disengage_a_flat_cut(self):
        # A maximum-based denominator divides the engagement of a perfectly
        # steady 1 kN cut by the size of one 50 kN artefact and calls the cut
        # disengaged. It is the same spike this module's amplitude is a
        # percentile to survive.
        record = np.full(512, 1000.0)
        record[300] = 50000.0

        assert force_engagement_fraction(a_series(record)) > 0.9

    def test_a_first_contact_transient_does_not_disengage_the_later_cut(self):
        # The tool arrives at 20 m/s, so the impact overshoot lives in the part
        # steady_window exists to discard. A denominator taken over the whole
        # record would let the discarded transient set every later statistic.
        impact = np.concatenate([np.zeros(200), np.full(56, 5000.0)])
        series = a_series(np.concatenate([impact, np.full(256, 1000.0)]))

        assert force_engagement_fraction(series) == pytest.approx(1.0)

    @pytest.mark.parametrize(("duty", "expected"), [(12, 1.0 / 12.0), (7, 1.0 / 7.0)])
    def test_the_engagement_fraction_is_pinned_either_side_of_the_edge(self, duty, expected):
        # 1/12 = 0.083 sits below the 0.10 edge and 1/7 = 0.143 above it, so
        # the edge is exercised where it lives rather than at 0 and 1.
        assert force_engagement_fraction(a_series(duty_cycled(duty))) == pytest.approx(
            expected, rel=0.02
        )
