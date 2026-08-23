"""When the solver should stop and hand the run to a supervisor.

The halt decision is the one part of the loop that has to be reproducible
without a solver: `replay.py` and the model ladder re-run stored episodes on
machines with no Kratos install, and they need to know *why* each episode
stopped where it did. So the predicates live here, take a step counter and a
plain mapping of diagnostics, and return a decision — no Kratos, no file
access, no clock.

Two triggers, and they are deliberately different in kind. The step interval is
a metronome: it exists so a long run is supervised at all, not because step
1000 is interesting. A diagnostic trip is the opposite — it fires because
something has gone out of bounds, and it outranks the metronome so the recorded
reason names the event rather than the clock.

The metronome is disarmed until its interval has elapsed again. Each trip is
disarmed for a window after **the halt that trip itself caused**, and for the
trip that is not a nicety. The quantities worth tripping on are cumulative:
energy drift cannot un-drift, a shear band that has collapsed to two elements
does not re-widen within a step. So a trip is still out of bounds on the first
step after the resume, and a trip path that consulted only the diagnostic would
checkpoint, exit and resume on every single step forever — burning one
supervision episode per step and never advancing the physics.

The two windows are tracked separately, and that separation is load-bearing. A
single shared timer keyed on "the last halt of any kind" is deaf in the common
case: with a 20-step metronome and a 50-step trip window, a routine clock halt
lands every 20 steps and the trip window never elapses again for the life of
the run. Energy drift could pass 500 percent with every halt still recorded as
a routine clock tick. Per-trip state also keeps one trip's firing from
silencing another's.

Deliberately absent: the thresholds themselves. A trip carries the bound it was
constructed with, because the canonical spec-§5 numbers belong to the
diagnostics that compute the quantities, not to the code that compares them.

Also deliberately absent: any notion of how long a diagnostic has been
answerless. A ``band_width_elements`` that is ``None`` for a whole run means the
band never formed, which is a real finding — but detecting it needs history, and
these predicates are per-step and stateless so that replay reproduces them
exactly. It belongs to whatever summarizes an episode, not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

# Recorded in the checkpoint manifest and in the episode record, so they are
# constants rather than literals typed at each call site.
REASON_STEP_INTERVAL = "step_interval"
REASON_DIAGNOSTIC_TRIP = "diagnostic_trip"

# Steps one diagnostic trip stays disarmed for after a halt *that trip* caused.
# The supervisor cannot be relied on to clear or widen the trip before resuming
# — that contract is unenforceable, and the failure it permits is an unbounded
# halt loop rather than a degraded run. This bounds the damage instead: a
# supervisor that leaves a trip standing loses this many steps of supervision
# on that one diagnostic, not the run and not the other trips. 50 steps is
# short enough that a genuinely worsening run is caught again well inside any
# sane metronome interval, and long enough that a time-step or boundary change
# has room to take effect.
TRIP_REARM_STEPS = 50

TripDirection = Literal["above", "below"]


@dataclass(frozen=True)
class DiagnosticTrip:
    """A named diagnostic and the bound that makes it worth stopping for.

    ``direction`` is not inferable from the number: energy drift is a fault
    when it rises past 5 percent, shear-band width is a fault when it falls
    below 3 elements. The bound itself is exclusive — a value sitting exactly
    on its limit has not yet crossed it.
    """

    name: str
    threshold: float
    direction: TripDirection


@dataclass(frozen=True)
class HaltDecision:
    """Whether to stop, and the reason to record if so.

    ``detail`` is prose for the episode record and the supervisor prompt. It is
    never parsed; ``reason`` is the machine-readable half.

    ``fired_trips`` names the diagnostics that caused *this* halt, and is empty
    for a metronome halt or no halt at all. The caller advances only these
    trips' re-arm windows: a routine clock halt must not disarm a trip that had
    nothing to do with it.
    """

    halt: bool
    reason: str | None
    detail: str
    fired_trips: tuple[str, ...] = ()


def tripped(trip: DiagnosticTrip, value: float | None) -> bool:
    """Whether ``value`` has crossed ``trip``'s bound in the fault direction.

    ``None`` means the diagnostic has no answer at this step and is never a
    trip. Shear-band width and orientation return ``None`` before the band
    localizes, and an unlocalized body is a normal physical state for most of a
    run. Substituting a number for it — 0.0 elements of band width — would fire
    a "below 3 elements" trip on every pre-localization step, which is the
    misdiagnosis this whole loop exists to avoid.

    A non-finite value raises: NaN compares false against every bound, so a
    diverging run would otherwise sail straight past the trip that exists to
    catch it.
    """
    if value is None:
        return False
    if not math.isfinite(value):
        raise ValueError(f"diagnostic {trip.name!r} is not finite: {value!r}")
    if trip.direction == "above":
        return value > trip.threshold
    if trip.direction == "below":
        return value < trip.threshold
    raise ValueError(
        f"trip {trip.name!r} has unknown direction {trip.direction!r}; expected 'above' or 'below'"
    )


def should_halt(
    step: int,
    *,
    interval_steps: int | None,
    last_halt_step: int | None = None,
    last_trip_steps: Mapping[str, int] | None = None,
    diagnostics: Mapping[str, float | None] | None = None,
    trips: Sequence[DiagnosticTrip] = (),
    trip_rearm_steps: int = TRIP_REARM_STEPS,
) -> HaltDecision:
    """Decide whether the run stops at ``step``.

    ``last_halt_step`` is the step the previous halt of *any* kind fired at, or
    ``None`` for a run that has not been halted yet, and it drives the metronome
    only. It is not defaulted to 0, because 0 is a step a trip can legitimately
    halt at — a diagnostic already out of bounds before the first step is a
    broken setup worth stopping for — and the two cases have to stay
    distinguishable.

    ``last_trip_steps`` maps a diagnostic name to the step at which *that trip*
    last fired; a name absent from it is armed. It is deliberately not derived
    from ``last_halt_step``: a metronome halt every 20 steps would otherwise
    hold a 50-step trip window permanently open and leave every trip silently
    deaf for the life of the run.

    Three states of a watched diagnostic, deliberately kept distinct — a later
    reader will want to flatten them, and flattening any pair loses a real
    failure mode:

    * **absent from the mapping** raises ``KeyError``. That is a wiring error:
      a trip was registered for something nothing computes, and treating it as
      in-bounds turns a broken diagnostic into a run that is never supervised.
    * **present as ``None``** is "no answer at this step", and is neither a trip
      nor an error. Band width has no value before the band localizes, which is
      most of a normal run.
    * **present as NaN or infinity** raises ``ValueError``. That is a corrupt
      computation, and it is the state a diverging run reaches just as the trip
      becomes most necessary.
    """
    if interval_steps is not None and interval_steps <= 0:
        raise ValueError(f"interval_steps must be positive, got {interval_steps}")
    if trip_rearm_steps < 0:
        raise ValueError(f"trip_rearm_steps must not be negative, got {trip_rearm_steps}")
    origin = 0 if last_halt_step is None else last_halt_step
    if step < origin:
        raise ValueError(f"step {step} precedes last_halt_step {origin}")

    diagnostics = {} if diagnostics is None else diagnostics
    last_trip_steps = {} if last_trip_steps is None else last_trip_steps

    fired_names: list[str] = []
    fired_detail: list[str] = []
    for trip in trips:
        if trip.name not in diagnostics:
            raise KeyError(f"watched diagnostic {trip.name!r} was not computed for step {step}")
        value = diagnostics[trip.name]
        # Evaluated even while disarmed, so a missing or non-finite diagnostic
        # is still loud during the re-arm window rather than hidden by it.
        crossed = tripped(trip, value)
        fired_at = last_trip_steps.get(trip.name)
        armed = fired_at is None or step - fired_at >= trip_rearm_steps
        if crossed and armed:
            fired_names.append(trip.name)
            fired_detail.append(f"{trip.name}={value:g} {trip.direction} {trip.threshold:g}")

    if fired_names:
        return HaltDecision(
            True, REASON_DIAGNOSTIC_TRIP, "; ".join(fired_detail), tuple(fired_names)
        )

    # Step 0 is excluded from the metronome: the initial state is already on
    # disk as the case itself, and the supervisor has no history to reason over.
    if interval_steps is not None and step > 0 and step - origin >= interval_steps:
        return HaltDecision(
            True,
            REASON_STEP_INTERVAL,
            f"step {step} is {step - origin} steps past the last halt (interval {interval_steps})",
        )

    return HaltDecision(False, None, "")
