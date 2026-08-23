"""Cutting and thrust force statistics — the diagnostics comparable to experiment.

Force is the one quantity a cutting simulation produces that a lab also
produces, so spec section 5 lists it as directly comparable. What it takes to
make the comparison honest is a window. The tool starts clear of the workpiece
and approaches before it cuts, so the front of every force record is a
zero-force approach followed by the transient of the first chip forming. A mean
taken over the whole record averages three different physical situations and is
not the steady cutting force of anything.

Periodicity is the second half of the job. A continuous chip gives a near-flat
force; a segmented chip gives a periodic force at the segmentation frequency,
because the force collapses each time a segment shears and rebuilds as the next
one forms; a discontinuous chip gives swings just as large with no period at
all. Telling those apart is a question about whether the record *repeats*, not
about how big the wiggle is, so this module reports the amplitude, the dominant
frequency and the periodicity separately, and leaves the judgement to
``morphology.classify_chip_regime``.

Four details are load-bearing rather than incidental:

* Sample spacing comes from the series' own ``time_s`` column. Telemetry is
  sampled every N solver steps and the step size is itself a supervised
  parameter, so assuming uniform spacing from the row index would report a
  frequency scaled by whatever the stride happened to be. Note which field
  that is: neither ``Series.time_step_s`` (the dt actually taken at one step)
  nor ``CaseMetadata.time_step_s`` (the case's nominal dt) is the spacing
  between two *sampled* rows, and substituting either would be wrong by the
  sampling stride.
* Every column is checked for NaN before it is used. Under the contract's
  schema-version rules a float column the episode predates is NaN-filled, and
  NaN means "not recorded at this schema version", never "measured as zero".
  A mean would swallow that and return NaN as though it were a number, so
  these functions raise instead.
* The steady signal is linearly detrended before transforming. A residual ramp
  puts its power in the lowest bins, and reporting that as a segmentation
  frequency is exactly the mistake this module exists to prevent.
* Amplitude is a percentile spread, not a peak-to-peak. Explicit dynamics
  produces isolated force spikes on re-contact and on point deletion, and one
  spike must not make a flat record look segmented.
* Whether the force *repeats* is measured in the lag domain, not read off the
  spectrum. A signal that is merely low-frequency -- fragments separating at
  irregular but slow intervals -- puts all its power in a handful of low bins
  and so looks concentrated to any spectral measure, which is the exact
  confusion the regime classification has to avoid. The normalized
  autocorrelation does not make that mistake: it asks whether the signal
  matches a shifted copy of itself, which an irregular signal does not.

Deliberately not here: any classification (``morphology`` owns the regime
edges), the shear-angle trigonometry (``merchant`` owns the Merchant relation
and the friction angle), and any I/O. Every function is pure: a ``Series`` in,
a number out.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quinney.telemetry import Series

# Fraction of the record dropped from the front before any statistic is taken.
# In the bench case the tool stands off 50 um and then cuts 450 um, so approach
# alone is 10% of the record, and the first chip is still forming for several
# multiples of the 100 um depth of cut after first contact. Half the record is
# what bench/orthogonal_cutting/analyze.py discarded to get the force numbers
# already on record, so keeping the same fraction keeps the new numbers
# comparable with those; it also leaves the spectrum half the samples, which is
# ample at the sampling rates used here.
STEADY_STATE_DISCARD_FRACTION = 0.5

# A mean and a spread need at least two samples. Below this there is no window.
MIN_STEADY_SAMPLES = 2

# Steady samples required before any statement about oscillation — a frequency
# or a periodicity — can be made. The floor is set by chance: taking the best
# of many lags, pure noise reaches a normalized autocorrelation of 0.58 at 32
# samples but stays under 0.42 over 400 trials at 64, so at 64 the 0.50
# periodicity a segmented chip has to clear
# (morphology.SEGMENTED_PERIODICITY_MIN) is out of noise's reach, and below it
# noise would clear that edge by arithmetic alone.
MIN_OSCILLATION_SAMPLES = 64

# Cycles a periodicity must complete inside the window before it can be called
# one. A peak in bin 1 or 2 is one or two cycles across the whole window, which
# is drift wearing a frequency's clothes; three is the smallest count from
# which a repeating period can be claimed at all. It bounds the transform's bin
# search and the autocorrelation's lag search alike, so the two agree on what
# counts as a period.
MIN_RESOLVED_CYCLES = 3

# Percentile pair defining the oscillation amplitude: the 5th and 95th. Chosen
# so a single spike, which is at most a fraction of a percent of the samples in
# any window used here, cannot set the amplitude, while a real oscillation --
# present in every cycle -- is captured in full. The upper member is also the
# robust stand-in for "the largest force present" in the engagement fraction,
# for the same reason and so that one tail definition serves the module.
OSCILLATION_PERCENTILE_PCT = 5.0

# Samples a period must span before the lag search will call it one. Two
# samples per period is Nyquist, where a genuine period and alternating-sign
# numerical ringing are the same sequence; four is the conventional minimum at
# which a waveform is a waveform. It bounds the short side of the lag search,
# which would otherwise be set by where the autocorrelation happens to change
# sign -- a data-dependent criterion that lets ringing in at lag 1.
MIN_SAMPLES_PER_PERIOD = 4

# Fraction of the best correlation within which a shorter lag counts as
# equally good. Every multiple of a period correlates as strongly as the period
# itself, so the *smallest* strongly-correlating lag is the fundamental, and it
# is the fundamental that has to be resolvable. 0.9 is a margin wide enough to
# survive the noise on a real correlation estimate and narrow enough not to
# mistake a weaker sub-peak for the fundamental.
FUNDAMENTAL_LAG_TOLERANCE = 0.9

# Tolerated departure of any one sample interval from the window's mean
# interval, as a fraction (0..1). Spacing error maps one-for-one onto frequency
# error, so 5% drift already means a reported frequency wrong by 5%; past that
# the mean spacing is a fiction and the caller is told rather than handed a
# number for a signal nobody recorded.
MAX_SAMPLE_SPACING_DRIFT = 0.05


def steady_window(
    series: Series,
    discard_fraction: float = STEADY_STATE_DISCARD_FRACTION,
) -> slice:
    """Rows of ``series`` over which force statistics are meaningful.

    The front of the record is the tool approaching through air and the first
    chip forming; neither is steady cutting. Everything in this module is
    computed over the returned slice, never over the whole record.

    Raises ``ValueError`` if ``discard_fraction`` is not in [0, 1) or if what
    remains is shorter than ``MIN_STEADY_SAMPLES``.
    """
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError(f"discard_fraction must lie in [0, 1), got {discard_fraction}")

    n_steps = len(series.step)
    start = int(n_steps * discard_fraction)
    if n_steps - start < MIN_STEADY_SAMPLES:
        raise ValueError(
            f"no steady window: {n_steps} sampled steps less a {discard_fraction:.2f} "
            f"discard leaves {n_steps - start}, under the {MIN_STEADY_SAMPLES} needed "
            "for a mean and a spread"
        )
    return slice(start, n_steps)


def _resolved(series: Series, window: slice | None) -> slice:
    return steady_window(series) if window is None else window


def _column(series: Series, name: str, window: slice) -> NDArray[np.float64]:
    """One float column over the window, refusing values the episode never recorded.

    The telemetry contract NaN-fills a column introduced after the episode's
    declared schema version, and that NaN means *not recorded*, not zero. It
    would pass silently through a mean and come back out as a force, so it is
    caught here — once, where every statistic in this module reads its data.
    """
    values: NDArray[np.float64] = getattr(series, name)[window]
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Series.{name} is not finite over the requested window: the episode's "
            "schema version predates this column, so it was never recorded, and a "
            "force statistic computed over it would be a number for a measurement "
            "that does not exist"
        )
    return values


def cutting_force_mean_n(series: Series, window: slice | None = None) -> float:
    """Mean force along the cutting direction over the steady window, in newtons."""
    return float(np.mean(_column(series, "cutting_force_n", _resolved(series, window))))


def thrust_force_mean_n(series: Series, window: slice | None = None) -> float:
    """Mean force normal to the cut surface over the steady window, in newtons."""
    return float(np.mean(_column(series, "thrust_force_n", _resolved(series, window))))


def force_oscillation_amplitude_n(series: Series, window: slice | None = None) -> float:
    """Amplitude of the cutting-force excursion about the steady mean, in newtons.

    Half the spread between the 5th and 95th percentiles, which for a square
    wave is its amplitude exactly and for a sine is within 2% of it, while an
    isolated spike moves it by nothing.
    """
    steady = _column(series, "cutting_force_n", _resolved(series, window))
    low, high = np.percentile(
        steady, [OSCILLATION_PERCENTILE_PCT, 100.0 - OSCILLATION_PERCENTILE_PCT]
    )
    return float(0.5 * (high - low))


def force_oscillation_ratio(series: Series, window: slice | None = None) -> float:
    """Oscillation amplitude as a fraction of the steady mean force magnitude (0..1+).

    Regime edges are quoted in this dimensionless form because absolute force
    scales with depth of cut and cut width, so an edge in newtons would have to
    be re-derived for every case. Returns NaN when the steady mean is zero,
    which means the window contains no cutting rather than a flat cut --
    ``force_engagement_fraction`` is the test for that.
    """
    mean_n = abs(cutting_force_mean_n(series, window))
    if mean_n == 0.0:
        return float("nan")
    return force_oscillation_amplitude_n(series, window) / mean_n


def force_engagement_fraction(series: Series, window: slice | None = None) -> float:
    """Steady mean force as a fraction of the largest force in the same window (0..1).

    The guard that has to pass before anything is divided by the mean force.
    While the tool is approaching, the force is zero or grid noise about zero,
    so the mean is a small fraction of the largest force present; once cutting,
    the mean is comparable to that peak even for a fully unloading
    discontinuous chip, whose mean is about half of it.

    Both terms are taken over the *steady window*, and the peak is a percentile
    rather than a maximum. Two reasons, and both are failures this had in an
    earlier form. A maximum lets one re-contact spike -- the very artefact
    ``force_oscillation_amplitude_n`` is a percentile to survive -- divide the
    engagement of a perfectly flat cut by that spike's size. And a peak taken
    over the whole record would be set by the first-contact transient, which
    lives in the part ``steady_window`` exists to discard, so a 5x impact on a
    steady 1 kN cut would put every later statistic below the engagement floor.

    Returns 0.0 when the window carries no force at all, which is a window that
    predates first contact entirely.
    """
    steady = np.abs(_column(series, "cutting_force_n", _resolved(series, window)))
    peak_n = float(np.percentile(steady, 100.0 - OSCILLATION_PERCENTILE_PCT))
    if peak_n == 0.0:
        return 0.0
    return abs(cutting_force_mean_n(series, window)) / peak_n


def _sample_interval_s(time_s: NDArray[np.float64]) -> float:
    """Mean sample spacing of a window, refusing spacings an FFT cannot represent."""
    intervals = np.diff(time_s)
    if np.any(intervals <= 0.0):
        raise ValueError("series time_s must be strictly increasing to take a spectrum")

    mean_interval_s = float(np.mean(intervals))
    drift = float(np.max(np.abs(intervals - mean_interval_s)) / mean_interval_s)
    if drift > MAX_SAMPLE_SPACING_DRIFT:
        raise ValueError(
            f"sample spacing varies by {drift:.1%} across the window, over the "
            f"{MAX_SAMPLE_SPACING_DRIFT:.0%} an evenly spaced transform tolerates; "
            "the record spans a change of sampling rate"
        )
    return mean_interval_s


def _detrended(values: NDArray[np.float64], time_s: NDArray[np.float64]) -> NDArray[np.float64]:
    """The signal with its least-squares straight line removed.

    Removing the mean alone would leave a ramp, and a ramp is all low-frequency
    power. Removing the line leaves only what actually oscillates.
    """
    centred_time_s = time_s - time_s.mean()
    spread = float(centred_time_s @ centred_time_s)
    residual = values - values.mean()
    if spread == 0.0:
        return residual
    slope = float(centred_time_s @ residual) / spread
    return residual - slope * centred_time_s


def _spectrum(
    series: Series, window: slice | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Frequencies in hertz and spectral power of the detrended steady cutting force.

    Raises ``ValueError`` if the window is shorter than
    ``MIN_OSCILLATION_SAMPLES`` or unevenly sampled.
    """
    resolved = _resolved(series, window)
    force_n = _column(series, "cutting_force_n", resolved)
    time_s = _column(series, "time_s", resolved)

    n_samples = len(force_n)
    if n_samples < MIN_OSCILLATION_SAMPLES:
        raise ValueError(
            f"a spectrum needs at least {MIN_OSCILLATION_SAMPLES} steady samples to "
            f"separate a peak from noise, got {n_samples}"
        )

    interval_s = _sample_interval_s(time_s)
    # Hann tapers the window ends to zero so that a periodicity which does not
    # complete a whole number of cycles inside the window stays in its own bins
    # instead of leaking across all of them. Being symmetric, it does not move
    # the peak.
    tapered = _detrended(force_n, time_s) * np.hanning(n_samples)
    return (
        np.fft.rfftfreq(n_samples, d=interval_s),
        np.abs(np.fft.rfft(tapered)) ** 2,
    )


def _peak_bin(power: NDArray[np.float64]) -> int:
    """Index of the largest bin that could be a periodicity rather than a trend."""
    return MIN_RESOLVED_CYCLES + int(np.argmax(power[MIN_RESOLVED_CYCLES:]))


def force_dominant_freq_hz(series: Series, window: slice | None = None) -> float:
    """Frequency of the strongest oscillation in the steady cutting force, in hertz.

    The DC bin is excluded because it carries the mean, and so are the first
    couple of bins because a signal that completes fewer than
    ``MIN_RESOLVED_CYCLES`` cycles inside the window is drift, not a period.

    Raises ``ValueError`` on a window too short or too unevenly sampled to
    transform. It does not report whether the peak is significant -- see
    ``morphology.segment_frequency_hz``, which is the caller that needs to know.
    """
    frequencies_hz, power = _spectrum(series, window)
    return float(frequencies_hz[_peak_bin(power)])


def force_periodicity(series: Series, window: slice | None = None) -> float:
    """How much of the steady cutting force repeats one period later (0..1).

    The number that separates a segmented chip from a fragmenting one at equal
    amplitude. A segmented chip's force is periodic — it collapses each time a
    segment shears and rebuilds as the next forms — so the record matches a
    copy of itself shifted by one segmentation period. Fragments separating at
    irregular intervals produce just as large a swing and match nothing.

    This is the normalized autocorrelation, maximised over a lag search bounded
    at both ends by named constants. The long side is the longest period that
    still repeats ``MIN_RESOLVED_CYCLES`` times inside the window. The short
    side is ``MIN_SAMPLES_PER_PERIOD``, applied to the *fundamental* lag rather
    than to the best one: every multiple of a period correlates as strongly as
    the period, so alternating-sign ringing at two samples per period would
    otherwise be reported as a clean periodicity found at lag four. The first
    sign change is still where the search starts, to skip the trivial peak at
    zero lag, but it is no longer the only thing holding the short side --
    which it was, and it is data-dependent enough to admit ringing at lag one.

    Normalizing each lag by the energy of the two overlapping segments matters:
    the textbook estimator is biased down by the overlap fraction, which would
    penalise exactly the long periods a slowly segmenting chip produces. What
    it does not remove is a small positive bias from maximising over many lags:
    about +0.03 at the 0.50 edge, shrinking as the repeat strengthens.

    For a periodic signal buried in noise the value is SNR / (1 + SNR), which
    is what makes the edge in ``morphology.SEGMENTED_PERIODICITY_MIN``
    interpretable rather than fitted.

    Raises ``ValueError`` if the window is shorter than
    ``MIN_OSCILLATION_SAMPLES``.
    """
    resolved = _resolved(series, window)
    force_n = _column(series, "cutting_force_n", resolved)
    n_samples = len(force_n)
    if n_samples < MIN_OSCILLATION_SAMPLES:
        raise ValueError(
            f"periodicity needs at least {MIN_OSCILLATION_SAMPLES} steady samples "
            f"before chance correlation falls clear of a real repeat, got {n_samples}"
        )

    signal = _detrended(force_n, _column(series, "time_s", resolved))
    energy = float(signal @ signal)
    if energy == 0.0:
        return 0.0

    max_lag = n_samples // MIN_RESOLVED_CYCLES
    lags = np.arange(max_lag + 1)
    overlap = np.correlate(signal, signal, mode="full")[n_samples - 1 : n_samples + max_lag]
    cumulative = np.cumsum(signal**2)
    leading = cumulative[n_samples - lags - 1]
    trailing = np.concatenate([[energy], energy - cumulative[lags[1:] - 1]])
    correlation = overlap / np.sqrt(leading * trailing)

    sign_change = np.flatnonzero(correlation < 0.0)
    if len(sign_change) == 0:
        return 0.0

    searched = correlation[sign_change[0] :]
    best = float(np.max(searched))
    if best <= 0.0:
        return 0.0

    strong = np.flatnonzero(searched >= FUNDAMENTAL_LAG_TOLERANCE * best)
    fundamental_lag = int(sign_change[0] + strong[0])
    if fundamental_lag < MIN_SAMPLES_PER_PERIOD:
        return 0.0
    return best
