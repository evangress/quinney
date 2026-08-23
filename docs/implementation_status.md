# Implementation status — feature tracker

Derived from [fea-supervisor-spec-v2.md](../fea-supervisor-spec-v2.md). This file is the
running marker of what exists and what does not. **Update it in the same commit as the work
it describes** — a stale tracker is worse than none.

Active plan: [plans/features-3-6.md](plans/features-3-6.md) — features 5, 6, 4, 3, in that
(dependency) order, on branch `features-3-6`.

Status vocabulary: `DONE` (implemented + tested) · `PARTIAL` (exists but incomplete or
ad hoc) · `TODO` (nothing on disk) · `N/A` (spec item retired, with reason).

| # | Feature | Spec | Phase | Status |
|---|---|---|---|---|
| 1 | Formulation spike (Lagrangian vs MPM) | §3, §10 | 0 | **DONE** |
| 2 | Constitutive physics (JC flow + damage, thermal, Zorev, OFHC Cu) | §2 | 0 | **DONE** |
| 3 | Orthogonal cutting bench case | §10 | 0 | **TODO** (task 8, not started) |
| 4 | Halt / checkpoint / restart loop | §4 | 1 | **IN REVIEW** — resume is bit-exact |
| 5 | Telemetry dump — versioned data contract | §4, §5 | 1 | **DONE** — schema 1.1 |
| 6 | Diagnostics — the standing vector | §5 | 1 | **IN REVIEW** — parts done, assembly not started |
| 7 | Query interface (`diagnostics/api.py`) | §4 | 2 | **TODO** |
| 8 | Action schema | §6 | 2 | **TODO** |
| 9 | Safety envelope (`envelope/validate.py`) | §4, §6 | 3 | **TODO** |
| 10 | Episode memory (`memory/store.py`, SQLite) | §9 | 2 | **TODO** |
| 11 | `replay.py` | §9, §10 | 1 | **TODO** |
| 12 | Rule-based baseline supervisor | §8 | 3 | **TODO** |
| 13 | Seeded pathology matrix | §9, §11 | 3 | **TODO** |
| 14 | Agent supervisor (LLM tool-use loop) | §7 | 4 | **TODO** |
| 15 | Model ladder descent | §7a | 5 | **TODO** |
| 16 | Scoring against success criteria | §11 | 4-5 | **TODO** |

## What landed in this batch

**Feature 5 — `src/quinney/telemetry.py`, schema 1.1. DONE.** The versioned on-disk episode
contract: `CaseMetadata`, `Snapshot`, `Series`, `Deletion`, `Episode`, with per-field
`INTRODUCED_IN` metadata so an older episode loads under a newer reader with later columns
NaN-filled. NaN means *not recorded at this schema version*, never *measured as zero*. Kratos-free
by test. `src/quinney/kratos_adapter/telemetry_writer.py` is the solver-side half.

**Feature 4 — halt / checkpoint / restart. Resume is bit-exact.** `src/quinney/halt.py` (pure,
Kratos-free trigger predicates), `src/quinney/checkpoint_format.py`, `kratos_adapter/checkpoint.py`,
`kratos_adapter/restart.py`, `bench/restart_fidelity/`. Sixteen material-point fields compare at
`max|d| = 0.0` at one thread, measured twenty steps *after* the resume and across a genuine
two-subprocess topology. Spec §10 Phase 1's "bit-comparable continuation" is met.

**Feature 6 — diagnostics. Modules built, assembly outstanding.** `diagnostics/bands.py`,
`merchant.py`, `morphology.py`, `forces.py`, `health.py`. `summary.py` — the standing §5 vector —
is not written yet.

## Blocking issues found, not yet fixed

- **`DamageSofteningProcess` segfaults the MPM solver** on the step after it clones per-point
  Properties. Reproduced standalone, no restart involved. Pre-existing Phase 0 code. The
  progressive-softening path cannot currently run, and `bench/orthogonal_cutting/run.py` defaults
  to `soften=True`. Needs its own task.
- **Softening state is not carried across a checkpoint and there is no seam to carry it.** Worse
  than a reset: the degraded per-point `Properties` clones *do* survive the serializer, so a naive
  resume records them as pristine and degrades again, compounding `(1-D)²` per halt. Recorded in
  `docs/kratos_api_notes.md`.
- **Open empirical question: does `MP_STRAIN_ENERGY` include plastic dissipation?** If it is
  elastic-only, `KE + IE - W_ext` in a cutting run is dominated by plastic work converted to heat,
  which lives only in `MP_TEMPERATURE` — and the 5% `energy_drift_pct` bound would trip on correct
  physics in the first hundred steps of every run. Must be settled before any threshold is tuned
  on it.
- **Deferred to the user:** the active ruff rule set is not in `pyproject.toml`; it comes from a
  user-level config outside the repo, so CI will not match what a contributor sees.

## Calibration owed to a real run

Every threshold below was set against synthetic data and has never seen real telemetry. Each needs
re-checking during feature 3:

- `MAX_ISOTROPIC_EIGENVALUE_RATIO`, `MIN_ON_AXIS_FRACTION`, `BAND_LINK_SPACINGS`,
  `INLIER_HALF_WIDTHS` and the 90th-percentile strain mask (`bands.py`)
- the four `REGIME_EDGES` and `MIN_OSCILLATION_SAMPLES` (`morphology.py`, `forces.py`)
- the deletion-coherence scale, which has no absolute meaning and no real-run yardstick
  (`health.py`)


---

## 1. Formulation spike — DONE

`docs/formulation_spike.md`. MPM chosen; the Lagrangian arm could not be built with the same
constitutive model, so the comparison as literally specified was retired with reasoning
recorded. `bench/jc_compression/` is the smoke case that settled it.

## 2. Constitutive physics — DONE

- `src/quinney/materials/johnson_cook.py` — fracture strain, damage accumulation, Hillerborg
  evolution, degradation factor. Pure.
- `src/quinney/materials/ofhc_copper.py` — published constants. **Carries an unverified-source
  warning that is still live.**
- `src/quinney/materials/zorev.py` — sticking/sliding shear cap. Pure.
- `src/quinney/kratos_adapter/{damage_process,softening,zorev_friction,rigid_tool}.py` —
  the solver-side halves.
- `src/quinney/point_ledger.py` — per-point state across erasure.
- Tests: `test_johnson_cook`, `test_damage_evolution`, `test_ofhc_copper`, `test_zorev`,
  `test_damage_process`, `test_softening`, `test_zorev_friction`, `test_rigid_tool`,
  `test_point_ledger`.

Thermal coupling comes from the compiled `JohnsonCookThermalPlastic2DPlaneStrainLaw`
(Taylor-Quinney verified in the spike, 140 K measured vs 135 K predicted).

## 3. Orthogonal cutting bench case — PARTIAL

`bench/orthogonal_cutting/` builds and runs, and `bench/zorev_friction/` exercises the
friction boundary. Not yet done: validation against published OFHC copper cutting force and
chip thickness data (§11), and a parametric speed sweep to hit the regime transition (§2).

## 6. Diagnostics — PARTIAL

`bench/orthogonal_cutting/analyze.py` computes chip thickness ratio and a band orientation
ad hoc, in a bench script, against a bench-specific `.npz`. That is a prototype, not the
`diagnostics/` package §5 and §9 call for: no band coherence, no band width, no temperature
alignment, no segment frequency, no regime classification, no deletion-health group, no
numerical-health group. Everything §5 lists still needs a home in a pure, Kratos-free module.

---

## Notes on ordering

Features 4 and 5 gate almost everything downstream: the query interface (7), replay (11), and
the whole supervisor stack read *files on disk*, never live solver state (CLAUDE.md,
"every decision is replayable"). Building 7 before 5 means inventing the telemetry contract
implicitly inside the query layer, which is the wrong seam.
