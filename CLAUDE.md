# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Quinney lets an LLM supervise **explicit finite element analysis of metal chip formation** —
2D orthogonal cutting in Kratos Multiphysics, thermo-mechanically coupled, OFHC copper.

The full design brief is [fea-supervisor-spec-v2.md](fea-supervisor-spec-v2.md). **Read it before
writing code.** It is the source of truth for the physics, the diagnostic vector, the action
schema, the repository layout, and the phase plan. This file does not restate it.

The repository is currently pre-implementation: no source, no tests, no commits. The layout in
§9 of the spec is the target, not the present state.

## Framing that changes how you write code

Three premises from the spec that are easy to violate by accident:

**Explicit dynamics does not converge.** There is no equilibrium iteration to accelerate. Do not
write code that reasons about convergence rate, residual tolerance, or iteration counts. Success
is run completion, energy fidelity, and physical chip morphology.

**The LLM never computes the time step.** `Δt ≤ L_min / c_d` is a closed form evaluated every step
in microseconds. The supervisor's job is the judgment the closed form cannot make — is this
element degeneration a *physical shear band* or *numerical mesh collapse*? That distinction is
the central diagnostic task and the reason the project exists.

**The pause is the point.** The solver exits so the supervisor has unbounded time and a set of
*investigation tools* (§4), not a fixed feature vector delivered slowly. When adding to the query
interface, ask whether it lets the supervisor ask a new question — if it just enlarges the
standing summary, it misses the architecture.

## Architecture

A halt-and-restart loop across process boundaries, not an in-process callback:

```
Kratos solver → trigger → checkpoint + telemetry dump → solver EXITS
                                                            ↓
                       supervisor process (unbounded time, tool access)
                                                            ↓
              modified ProjectParameters → solver RESTARTS from checkpoint
```

Consequences that constrain every design decision:

- **Every decision is replayable.** The supervisor sees only files on disk (checkpoint +
  telemetry). That is what makes `replay.py` and the Phase 5 model-ladder descent possible —
  stored episodes get re-run against 14B → 8B → 7B models with no solver in the loop. Never let
  the supervisor reach into live solver state.
- **The supervisor is any process at all** — an API model, a local model, a rule baseline, or a
  human. Keep the interface identical for all four. Phase 2 has *you* acting as supervisor; if
  the tool surface only works for an LLM, it is wrong.
- **Unbounded time ≠ unbounded authority.** Every action passes `envelope/validate.py`: schema
  validation and hard parameter clamping. Δt may never go below CFL. `abort_with_diagnosis` is
  always available and always safe.
- **The rule baseline is permanent**, not scaffolding. It is the control arm; rules winning on
  some diagnostics is a valid finding, and undetectable without it.
- **Actions carry `evidence`.** It is required, and it is how you audit whether a correct action
  came from correct reasoning or from luck.
- **Do not rely on native function-calling** at the 14B scale. Use JSON-schema-constrained output
  with a hand-rolled dispatch loop, routed through the OpenAI-compatible chat completions
  interface (via LiteLLM) so Ollama and vLLM are a config change.
- **`render_field` is vision-only.** Text-only rungs of the ladder must be scored without it, or
  the ceiling run must drop vision entirely, or the comparison is unfair.

## Environment and commands

Python **3.14** on Windows, managed with `uv`. Source lives under `src/quinney/`.

```bash
uv sync                              # env + deps, installs quinney editable
uv run pytest                        # all tests
uv run pytest tests/test_x.py::TestY::test_z   # one test
uv run --with ruff ruff check src tests
uv run --with ruff ruff format src tests
```

Kratos 10.4.3 is installed and working (core + StructuralMechanics,
ConstitutiveLaws, ConvectionDiffusion, MPM, Meshing, LinearSolvers), pinned per
application — the `KratosMultiphysics-all` metapackage lags and has no cp314
wheels. Kratos is CPU-bound, which is what spec §7 assumes; no GPU is needed to
run it.

**Read [docs/kratos_api_notes.md](docs/kratos_api_notes.md) before touching the
solver boundary.** It records what the installed wheels actually expose — most
importantly that Johnson-Cook exists only in `MPMApplication` and that JC damage
does not exist at all, which is why `quinney/materials/johnson_cook.py` owns the
damage law. `MP_*` variables are on the `MPMApplication` namespace, not the core
one; `TO_ERASE` is on the core one.

Kratos prints banners to stdout on import, so grep test output for what you want
rather than reading the tail.

## Coding principles

These exist to keep the seams above intact — each one is stated in terms of a boundary
this project actually has.

**Orthogonality.** Telemetry extraction, diagnostic computation, supervision, safety validation,
and storage are independent. Re-tuning a band-coherence threshold must not require re-running the
solver. Changing the prompt must not touch the numbers. One reason to ripple, per change.

**DRY, but earn it.** One canonical definition per concept — the diagnostic field list, the
parameter clamp bounds, the chip-regime edges each live in *one* place and are imported, never
re-typed. But do not abstract two things that merely look alike today. Extract on the third
repetition, not the first.

**Single responsibility.** A function does one thing at one level of abstraction. The agent loop
orchestrates; the helpers it calls each own a single diagnostic or transform. If a function needs
"and" to describe it, split it.

**Side-effect isolation.** Pure functions everywhere except the Kratos adapter, `memory/store.py`,
and `replay.py`. Diagnostics take telemetry and return numbers — no printing, no file writes,
no network, no globals. This is what makes replay reproducible.

**Dependency inversion at the model and solver boundaries.** The supervisor depends on an
*interface*, not on Ollama, vLLM, or Anthropic. The diagnostics depend on dumped
telemetry, not on Kratos. Never `import KratosMultiphysics` from `diagnostics/` or `supervisor/`; a
machine with no Kratos install must still be able to replay stored episodes.

**Determinism & reproducibility.** Same episode + same schema version → the same diagnostics,
always. No wall-clock and no RNG in the diagnostic path. Model sampling is the sole intentional
nondeterminism, and its seed/temperature belong in the episode record. Episodes stored a year
apart must be comparable, or the ladder measurement is worthless.

**Stable data contracts.** The telemetry dump, the episode record, and the action JSON schema are
a *public API* — replay, the ladder, and the scoring all depend on their shape. Version them:
additive changes bump minor; any rename or removal bumps major and needs a migration note. Never silently change a field's meaning or units.

**Explicit over implicit.** Full type hints on every public function. Real units in names or
docstrings (`_deg`, `_hz`, `_pct`, `_mm`, `0..1`). No magic numbers — clamp bounds, band-width
minimums, and energy-drift limits are named module constants with a comment on *why* (the spec's
§5 thresholds are the source: >5% energy drift, >10% hourglass ratio, <3 elements band width).

**Fail loud in the small, resilient in the large.** A single supervision decision raises on
malformed telemetry — the caller wants to know. A batch replay over hundreds of episodes catches
per-episode, logs the skip to stderr, and keeps going. This asymmetry is intentional.

**YAGNI / right-sized design.** The spec's §9 layout is the ceiling, not a starting skeleton —
create a module when there is code to put in it. No plugin system, no config framework,
no abstract base class hierarchy until a second real implementation demands it. Prefer a boring
function to a clever class.

**Composability.** Public functions take and return plain data so callers can pipe them:
`telemetry → diagnostics → action`, `action → envelope → parameters`, `episode → replay → score`.
Keep them free-standing.
