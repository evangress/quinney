"""Deletion health and numerical health — is the run trustworthy, and why did mass leave?

Spec section 1's question is whether element degeneration is a physical shear
band or numerical mesh collapse. Section 5 splits the evidence in two, and this
module owns the half that lives in *deletion* and in *integration health*:

* Deletion health asks **where and when** material points separated. Chip
  formation removes points along one line — the separation path ahead of the
  tool tip, running through the hot primary shear zone. Numerical erosion
  removes them wherever the integration is unhappy, which is everywhere. Same
  count, entirely different meaning, and the difference is spatial clustering
  plus correlation with the temperature field.

* Numerical health asks whether the numbers are worth reading at all. A run
  that has invented 20% of its own energy, or whose stable step has been
  collapsing for ten thousand steps, produces a force history that describes
  nothing. Those checks gate every other diagnostic.

The coherence estimator itself lives in :mod:`quinney.diagnostics.bands` and is
imported, not reimplemented: "clustered along a line" is one question whether
the points in question carry high plastic strain or have been deleted, and two
implementations would drift apart and quietly disagree about the same episode.

Four fields spec section 5 lists are Lagrangian-mesh quantities that MPM cannot
measure — ``hourglass_ie_ratio_pct``, ``jacobian_min``, ``aspect_ratio_p99``,
``plastic_work_delta_last_remesh``. They return ``None`` rather than a plausible
number, and they stay in the API so the contract survives a formulation change.

Deliberately absent: any threshold *verdict*. These functions return numbers and
the bounds beside them; deciding that 6% drift warrants an abort is the
supervisor's judgement, and moving it into a boolean here would hide the
evidence the supervisor is scored on.

Every function is pure: telemetry in, number out. No I/O, no wall-clock, no RNG.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quinney.diagnostics.bands import band_coherence
from quinney.telemetry import Deletion, Series, Snapshot

# Spec section 5, "Deletion health": deleted mass is a hard bound with a target
# below 1-2%. Past this the body has lost enough material that the cutting and
# thrust forces no longer describe the case that was set up, so the run is not
# comparable to experiment whatever its chip looks like.
MAX_DELETED_MASS_PCT = 2.0

# Spec section 5, "Numerical health": explicit central-difference integration
# never conserves energy exactly, but a residual past 5% of the work done means
# the integration has stopped tracking the physics.
MAX_ENERGY_DRIFT_PCT = 5.0

# Spec section 5, "Numerical health": hourglass energy past 10% of internal
# energy is fake stiffening — the reduced-integration control is carrying load
# the material should be carrying. Unreachable under MPM; kept because it is the
# bound a future Lagrangian arm is measured against.
MAX_HOURGLASS_RATIO_PCT = 10.0

# Below this the run has done no measurable external work, so a drift expressed
# as a fraction of work done is undefined rather than large. Twelve orders below
# the order-joule input of the bench case: no live run can fall under it, and a
# record that does is a record from before the tool engaged.
MIN_REFERENCE_WORK_J = 1e-12

# A trend needs two points. Stated once so the slope and the window agree.
MIN_TREND_SAMPLES = 2


def _validate_window_steps(window_steps: int | None) -> None:
    """A window is a positive count of solver steps, or ``None`` for the whole record."""
    if window_steps is not None and window_steps <= 0:
        raise ValueError(f"window must span at least one step, got window_steps={window_steps}")


def _require_samples(series: Series, quantity: str, minimum: int) -> None:
    """Raise naming ``quantity`` if the series is too short to compute it.

    A halt before the first sample produces an empty but perfectly loadable
    ``Series``, and indexing it would raise a bare ``IndexError`` naming neither
    the diagnostic that failed nor the record it failed on. The supervisor reads
    these messages; they have to say which question could not be answered.
    """
    n_samples = len(series.step)
    if n_samples < minimum:
        plural = "step" if minimum == 1 else "steps"
        raise ValueError(
            f"{quantity} needs at least {minimum} sampled {plural}, but the series holds "
            f"{n_samples}"
        )


def _require_increasing_steps(series: Series, quantity: str) -> None:
    """Raise naming ``quantity`` if the series' steps do not increase row to row.

    ``Series.__post_init__`` checks column lengths, not step order, and a writer
    that resumed from a checkpoint and re-based its step counter can emit a
    record that goes backwards. Every windowed quantity here divides by a
    difference of steps, so a backwards record does not merely lose precision:
    it flips the sign, and a run leaking energy would report as creating it —
    the wrong side of any bound tested for excess.
    """
    if len(series.step) < MIN_TREND_SAMPLES:
        return
    backwards = np.diff(series.step) <= 0
    if backwards.any():
        row = int(np.argmax(backwards)) + 1
        raise ValueError(
            f"{quantity} needs a series whose steps increase, but row {row} is at step "
            f"{int(series.step[row])}, after row {row - 1} at step {int(series.step[row - 1])}"
        )


def _recent_mask(
    steps: NDArray[np.int64],
    end_step: int,
    window_steps: int | None,
) -> NDArray[np.bool_]:
    """Records inside the last ``window_steps`` solver steps ending at ``end_step``.

    Windows are counted in *solver* steps rather than in stored rows so that the
    answer does not move when the adapter changes how often it samples. Records
    after ``end_step`` are excluded: a point that separates later has not
    separated yet at the moment being examined.
    """
    _validate_window_steps(window_steps)
    within_end = steps <= end_step
    if window_steps is None:
        return within_end
    return within_end & (steps > end_step - window_steps)


def _least_squares_slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Slope of the least-squares line through (x, y), in y-units per x-unit."""
    x_centred = x - x.mean()
    spread = float((x_centred * x_centred).sum())
    if spread == 0.0:
        raise ValueError("cannot fit a trend: every sample sits at the same step")
    return float((x_centred * (y - y.mean())).sum() / spread)


def deleted_mass_pct(series: Series, initial_mass_kg: float) -> float:
    """Cumulative separated mass as a percentage of the starting body (0..100).

    ``initial_mass_kg`` is the total material-point mass before anything was
    deleted — ``snapshot.mass_kg.sum()`` of the run's first snapshot. It is a
    parameter rather than a series column because the telemetry contract stores
    the loss, not the total, and inferring the total from a later snapshot would
    quietly assume the loss record is complete.

    Compare against :data:`MAX_DELETED_MASS_PCT`.
    """
    if initial_mass_kg <= 0.0:
        raise ValueError(f"initial mass must be positive, got {initial_mass_kg} kg")
    _require_samples(series, "deleted mass", 1)
    return 100.0 * float(series.deleted_mass_kg[-1]) / initial_mass_kg


def deletion_rate_per_s(deletions: Deletion, series: Series, window_s: float) -> float:
    """Material points separating per second over the last ``window_s`` of the record.

    Rate is per second rather than per step because it is the physical quantity:
    a chip separating at 20 m/s past 25 um point spacing has a rate the step size
    must not change. Deletion records carry a step, so the series supplies the
    step-to-time mapping.

    Raises ``ValueError`` if the window is not positive or reaches back past the
    start of the record — a rate over time that was never simulated is fiction —
    and if a deletion is recorded at a step the series does not span, since
    interpolation would silently clamp it to an endpoint and count it in a window
    it may not belong to.
    """
    if window_s <= 0.0:
        raise ValueError(f"window must have positive duration, got window_s={window_s}")

    _require_samples(series, "a deletion rate", 1)
    _require_increasing_steps(series, "a deletion rate")
    time_s = series.time_s

    end_time_s = float(time_s[-1])
    start_time_s = end_time_s - window_s
    if start_time_s < float(time_s[0]):
        raise ValueError(
            f"window of {window_s} s reaches past the start of a record spanning "
            f"{end_time_s - float(time_s[0])} s"
        )

    outside = (deletions.step < series.step[0]) | (deletions.step > series.step[-1])
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} deletion(s) fall outside the series' step range "
            f"[{int(series.step[0])}, {int(series.step[-1])}], first at step "
            f"{int(deletions.step[outside][0])}: their time cannot be resolved"
        )

    deletion_time_s = np.interp(deletions.step.astype(np.float64), series.step, time_s)
    inside = (deletion_time_s >= start_time_s) & (deletion_time_s <= end_time_s)
    return float(inside.sum()) / window_s


def deletion_spatial_coherence(
    deletions: Deletion,
    snapshot: Snapshot,
    cell_size_m: float,
    window_steps: int | None = None,
) -> float:
    """How clustered the separated points are within the body they left (0..1).

    High means the deletions string along one line — chip separation ahead of
    the tool. Low means they are sprinkled through the bulk — numerical erosion.
    This is the deletion-side half of spec section 1's question.

    **The score is a ranking, not an absolute fraction, and its range is
    deletion-specific.** ``band_coherence`` normalizes a join count against a
    random relabelling of the same parent cloud, so its scale moves with how
    much of the cloud the mask carries. A dense high-plastic-strain band scores
    around 0.75 against 0.007 for equal-sized scatter, while a one-point-wide
    deletion trail — which is what this function actually sees — scores around
    0.2 against 0.0002 for diffuse deletion. Same qualitative separation, an
    order of magnitude apart in absolute terms, purely because deletion sets are
    sparse. Compare deletion coherence against other deletion coherence, never
    against a plastic-strain band figure; this module deliberately names no
    cut-off, because a cut-off calibrated on synthetic clouds would be a guess
    dressed as a constant, and the call belongs to the rule baseline and the
    supervisor that are scored on it.

    The reference point set is the body reconstructed at the deletion positions:
    surviving points from ``snapshot`` plus the windowed deletions put back where
    they left. Coherence is a property of a subset *within* a cloud, so the
    surviving points are what makes "clustered" mean anything; scoring the
    deletion cloud on its own would call any deletion set perfectly coherent.

    Returns 0.0 when nothing has separated inside the window — the honest score
    for a run in which nothing has localized, and the state a healthy run sits in
    before the chip starts to form.

    ``snapshot`` also fixes the moment: only points that left at or before its
    step are counted, since a point deleted later is still in the snapshot and
    would otherwise be placed twice.

    **Pass a window.** The default ``window_steps=None`` reinstates every point
    the run has ever deleted, at coordinates from as far back as the first step,
    against a body that has since moved and been cut through. Early scatter then
    sits on top of the current separation trail and drags the score down. Use a
    window comparable to the snapshot sampling interval, so the deletions being
    scored left a body close to the one they are placed back into.
    """
    selected = _recent_mask(deletions.step, snapshot.step, window_steps)
    if not selected.any():
        return 0.0

    deleted_coords_m = deletions.coords_m[selected]
    coords_m = np.vstack([snapshot.coords_m, deleted_coords_m])
    mask = np.zeros(len(coords_m), dtype=bool)
    mask[len(snapshot.coords_m) :] = True
    return band_coherence(coords_m, mask, cell_size_m)


def deletion_temp_correlation(
    deletions: Deletion,
    snapshot: Snapshot,
    window_steps: int | None = None,
) -> float:
    """Correlation between leaving the body and being hot (-1..1).

    The point-biserial correlation between a "was deleted" indicator and
    temperature, over the surviving points of ``snapshot`` pooled with the
    windowed deletions. Physical separation happens inside the primary shear
    zone, which adiabatic heating lifts order 100 K above the 294 K ambient, so
    it should come back strongly positive. Numerical erosion has no reason to
    prefer hot material and comes back near zero.

    Returns 0.0 when nothing has separated inside the window, and when the pooled
    temperature field is uniform — an isothermal body offers nothing to
    correlate against, and reporting a number there would be division noise.

    As with :func:`deletion_spatial_coherence`, only points that left at or
    before the snapshot's step count.

    **Pass a window.** The pool mixes two instants of the temperature field:
    deleted points carry the temperature they had *when they left*, survivors
    carry theirs at ``snapshot.step``. The default ``window_steps=None`` spans
    the whole run, so a run halted deep into the cut pools deletions from early
    steps — recorded off a barely-heated body — against a snapshot hundreds of
    kelvin hotter. That dilutes a real correlation toward zero, and reading a
    diluted 0.3 as diffuse erosion when a windowed evaluation would have said
    0.9 and physical separation is exactly the misclassification spec section 1
    exists to prevent. Use a window comparable to the snapshot sampling
    interval, so the deletions being scored left a body close to the one they
    are scored against.
    """
    selected = _recent_mask(deletions.step, snapshot.step, window_steps)
    if not selected.any():
        return 0.0

    deleted_temperature_k = deletions.temperature_k[selected]
    temperature_k = np.concatenate([snapshot.temperature_k, deleted_temperature_k])
    was_deleted = np.zeros(len(temperature_k), dtype=np.float64)
    was_deleted[len(snapshot.temperature_k) :] = 1.0

    temperature_spread = temperature_k - temperature_k.mean()
    deleted_spread = was_deleted - was_deleted.mean()
    denominator = np.sqrt(float((temperature_spread**2).sum()) * float((deleted_spread**2).sum()))
    if denominator == 0.0:
        return 0.0
    return float((temperature_spread * deleted_spread).sum() / denominator)


def energy_drift_pct(series: Series, window_steps: int | None = None) -> float:
    """Net energy-balance violation as a percentage of the work done (signed).

    The balance an explicit run must satisfy is that the residual

        R = kinetic_energy_j + internal_energy_j - external_work_j

    stays constant: everything the boundary puts in ends up as motion or as
    stored and dissipated internal energy. Drift is what R does anyway.

    Precisely, with rows restricted to the last ``window_steps`` solver steps
    (or the whole record when ``window_steps`` is ``None``):

        spanned_steps   = step[last row in window] - step[first row in window]
        record_steps    = step[last row] - step[first row]
        requested_steps = record_steps                 if window_steps is None
                          else min(window_steps, record_steps)
        drift_j         = (R[last row in window] - R[first row in window])
                          / spanned_steps * requested_steps
        reference_j     = |external_work_j[last row] - external_work_j[first row]|
                          over the **full** record, not the window

    and the result is ``100 * drift_j / reference_j``.

    Two normalizations, for two different reasons.

    The **reference is not windowed**, because "relative to total work done" is
    what makes 5% mean the same thing in a short window as in a long one. A short
    window measured against its own small work increment would report an alarming
    percentage of nearly nothing.

    The **drift is scaled to the requested window**, because the rows that land
    inside a window are set by how often the adapter sampled, not by the physics.
    Without the scaling, a residual climbing 1e-5 J/step against 1.0 J of work
    reports 0.199% over a 200-step window when every step was stored and 0.100%
    when every hundredth was — same run, factor of two, and a verdict that flips
    if the truth straddles :data:`MAX_ENERGY_DRIFT_PCT`. Episodes stored a year
    apart under different sampling policies have to stay comparable, so the
    measured change is scaled from the steps it was actually observed over to the
    window that was asked for. This treats drift as a rate across the window,
    which is the assumption the diagnostic already makes by taking endpoints.

    A window longer than the record is truncated to the record rather than
    extrapolated past it: drift over steps that were never simulated is fiction.

    The scaling is a two-point extrapolation, so it is only as good as the rows
    it has. Under *uniform* sampling the stretch is bounded near 2x, because the
    window always holds at least half the steps it asked for. Under *non-uniform*
    sampling it is unbounded: two rows at steps 20000 and 20001 inside a
    200-step window would amplify whatever per-row noise they carry 200 times.
    That cannot happen today — the telemetry writer appends a series row every
    step unconditionally, and its ``sample_every`` gates snapshots only — but it
    goes live the moment anyone thins the series, which is exactly the change
    this scaling exists to survive. The fix then is to interpolate the residual
    to the window boundary between the two rows that straddle it, rather than to
    stretch from the innermost row.

    Positive means energy appeared from nowhere, negative means it leaked away.
    Compare the magnitude against :data:`MAX_ENERGY_DRIFT_PCT`.

    Raises ``ValueError`` if the record shows no external work: drift as a
    fraction of work done is undefined before the tool has done any.
    """
    _require_samples(series, "energy drift", MIN_TREND_SAMPLES)
    _require_increasing_steps(series, "energy drift")

    external_work_j = series.external_work_j
    reference_j = abs(float(external_work_j[-1]) - float(external_work_j[0]))
    if reference_j < MIN_REFERENCE_WORK_J:
        raise ValueError(
            "energy drift is undefined before any external work has been done: "
            f"the record shows {reference_j} J"
        )

    residual_j = series.kinetic_energy_j + series.internal_energy_j - external_work_j
    inside = _recent_mask(series.step, int(series.step[-1]), window_steps)
    windowed_steps = series.step[inside]
    if len(windowed_steps) < MIN_TREND_SAMPLES:
        raise ValueError(
            f"energy drift needs at least {MIN_TREND_SAMPLES} sampled steps in the window, "
            f"got {len(windowed_steps)}"
        )

    spanned_steps = int(windowed_steps[-1]) - int(windowed_steps[0])
    if spanned_steps <= 0:
        raise ValueError(
            f"energy drift cannot be scaled: every sampled step in the window sits at "
            f"step {int(windowed_steps[0])}"
        )
    record_steps = int(series.step[-1]) - int(series.step[0])
    requested_steps = record_steps if window_steps is None else min(window_steps, record_steps)

    windowed_j = residual_j[inside]
    drift_per_step_j = (float(windowed_j[-1]) - float(windowed_j[0])) / spanned_steps
    drift_j = drift_per_step_j * requested_steps
    return 100.0 * drift_j / reference_j


def dt_slope_per_step(series: Series, window_steps: int | None = None) -> float:
    """Trend in the stable time step, in seconds of dt per solver step.

    Least-squares slope of ``time_step_s`` against ``step`` over the last
    ``window_steps`` solver steps. Sign convention: **negative means the step is
    collapsing** — dt shrinking over thousands of steps is spec section 0's
    signature of a discretization degenerating under the solver, which the
    instantaneous CFL evaluation cannot see because it is satisfied at every
    individual step. Positive means the step is recovering.

    The magnitude is tiny in absolute terms (a nanosecond step losing a
    picosecond per step is a fast collapse), so read it against the current dt,
    not against zero.

    A slope is per step by construction, so unlike :func:`energy_drift_pct` this
    needs no scaling to stay comparable across sampling policies — the regression
    is against ``step``, and coarser sampling costs precision rather than shifting
    the answer. A window longer than the record is truncated to the record, which
    is harmless for the same reason.

    Raises ``ValueError`` if the window holds fewer than two samples.
    """
    _require_samples(series, "a time-step trend", MIN_TREND_SAMPLES)
    inside = _recent_mask(series.step, int(series.step[-1]), window_steps)
    steps = series.step[inside].astype(np.float64)
    if len(steps) < MIN_TREND_SAMPLES:
        raise ValueError(
            f"a time-step trend needs at least {MIN_TREND_SAMPLES} sampled steps in the "
            f"window, got {len(steps)}"
        )
    return _least_squares_slope(steps, series.time_step_s[inside])


def hourglass_ie_ratio_pct(series: Series) -> None:
    """``None`` under MPM: single-point MPM elements have no hourglass modes to control.

    It would become meaningful under a reduced-integration Lagrangian mesh, where
    hourglass control energy past :data:`MAX_HOURGLASS_RATIO_PCT` of internal
    energy means the stabilization is carrying load the material should carry.
    """
    return None  # noqa: RET501 (explicit: a ruling, not an unimplemented stub)


def jacobian_min(snapshot: Snapshot) -> None:
    """``None`` under MPM: material points carry no element connectivity, so no Jacobian.

    It would become meaningful under a Lagrangian mesh, where the minimum element
    Jacobian going negative is the definition of mesh inversion.
    """
    return None  # noqa: RET501 (explicit: a ruling, not an unimplemented stub)


def aspect_ratio_p99(snapshot: Snapshot) -> None:
    """``None`` under MPM: a material point is a volume sample with no shape to distort.

    It would become meaningful under a Lagrangian mesh, where the worst-percentile
    element aspect ratio is how mesh distortion is measured before it inverts.
    """
    return None  # noqa: RET501 (explicit: a ruling, not an unimplemented stub)


def plastic_work_delta_last_remesh(series: Series) -> None:
    """``None`` under MPM: the background grid is reset every step and never remeshed.

    It would become meaningful under a remeshing Lagrangian scheme, where the
    plastic work lost across a history-mapping step measures how much state the
    remesh threw away.
    """
    return None  # noqa: RET501 (explicit: a ruling, not an unimplemented stub)
