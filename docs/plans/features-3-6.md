# Plan — features 3 through 6

**Spec:** [fea-supervisor-spec-v2.md](../../fea-supervisor-spec-v2.md) (binding authority)
**Tracker:** [docs/implementation_status.md](../implementation_status.md)

Covers tracker features **5** (telemetry contract), **6** (diagnostics), **4** (halt/checkpoint/
restart) and **3** (bench case validation) — in dependency order, which is not tracker order.

## Global Constraints

Binding on every task. Copy verbatim into every reviewer dispatch.

1. **No Kratos import outside `kratos_adapter/`.** `diagnostics/`, `telemetry.py`, and anything
   under `materials/` must import cleanly on a machine with no Kratos install. This is what makes
   replay and the model ladder possible (CLAUDE.md, "dependency inversion at the solver
   boundary"). A test that needs Kratos lives in a test module that already imports it.
2. **Diagnostics are pure.** Array in, number out. No printing, no file writes, no globals, no
   wall-clock, no RNG. Same telemetry + same schema version yields identical numbers, always.
3. **One canonical definition per concept.** Threshold constants (5% energy drift, 10% hourglass,
   3-element band width, 1-2% deleted mass) are named module constants defined **once**, with a
   comment naming the spec section they come from. Never re-typed in a second module or a test.
4. **Units in names.** `_m`, `_s`, `_k`, `_pa`, `_deg`, `_hz`, `_pct`, `_n`, or a `0..1` note in
   the docstring. Full type hints on every public function.
5. **The telemetry contract is a public API.** Additive change bumps minor. Rename or removal
   bumps major and needs a migration note. Never silently change a field's meaning or units.
6. **No magic numbers.** If a number appears in logic, it is a named constant with a comment on
   why that value.
7. **Fail loud in the small.** A single malformed telemetry record raises. Batch paths (later,
   not in this plan) catch per-episode. Do not add try/except that swallows.
8. **YAGNI.** Create a module when there is code for it. No ABCs, no plugin registry, no config
   framework. A boring function beats a clever class.
9. **Do not commit.** The controller commits each wave. Leave the working tree clean of
   `git add` / `git commit` calls.
10. **Tests are behavioral.** Name what the physics does, not what the function is called. No
    test that asserts a function was called; no assertion-free test.
11. **Match the house style.** Read two existing modules before writing
    (`src/quinney/materials/johnson_cook.py` and `src/quinney/point_ledger.py`). Module docstrings
    explain *why the module exists and what it deliberately does not do*. Follow that.

## Ruling — what MPM makes inapplicable

Spec section 5 lists `hourglass_ie_ratio`, `jacobian_min`, `aspect_ratio_p99`, and
`plastic_work_delta_last_remesh`. All four are Lagrangian-mesh quantities. MPM was chosen in
Phase 0 (docs/formulation_spike.md), and it has no element Jacobian to degenerate, no hourglass
modes, and no remesh. **These four fields are `None` in the v1 summary, with a documented reason
on each.** They stay in the schema so the contract survives a future formulation change; they are
not faked. Cost if wrong: a future Lagrangian arm has to populate four already-reserved fields.

---

## Task 1 — Telemetry contract v1 (feature 5, pure half)

**Files:** `src/quinney/telemetry.py`, `tests/test_telemetry.py`

The versioned, Kratos-free data contract that every downstream consumer reads. Nothing else in
this plan can be built without it.

### On-disk layout

An *episode* is one directory:

```
<episode_dir>/
  manifest.json                 # schema version, case metadata, snapshot index
  series.npz                    # homogeneous time series (one row per sampled step)
  snapshots/step_000250.npz     # ragged per-material-point state, one file per sample
```

Ragged point state gets its own file per sample because erasure changes the point count between
samples, and because diagnostics want to load one snapshot without reading the whole run.

### Types

```python
SCHEMA_VERSION = "1.0"   # major.minor; major mismatch on load is an error

@dataclass(frozen=True)
class CaseMetadata:
    case_name: str
    material: str                    # "ofhc_copper"
    formulation: str                 # "mpm"
    cutting_speed_m_s: float
    rake_angle_deg: float
    depth_of_cut_m: float            # t0, the uncut chip thickness
    cell_size_m: float               # background grid cell; the mesh-resolution yardstick
    time_step_s: float
    workpiece_bounds_m: tuple[float, float, float, float]   # x0, x1, y0, y1
    tool_tip_m: tuple[float, float]
    friction_coefficient: float
    friction_factor: float           # Zorev m, 0..1
    deletion_threshold: float        # damage at which a point separates

@dataclass(frozen=True)
class Snapshot:
    """Per-material-point state at one sampled step. All arrays share length n_points."""
    step: int
    time_s: float
    point_ids: NDArray[np.int64]
    coords_m: NDArray[np.float64]            # (n, 2)
    velocity_m_s: NDArray[np.float64]        # (n, 2)
    eps_p: NDArray[np.float64]               # equivalent plastic strain, dimensionless
    eps_rate_per_s: NDArray[np.float64]
    temperature_k: NDArray[np.float64]
    damage: NDArray[np.float64]              # 0..1
    mass_kg: NDArray[np.float64]
    equivalent_stress_pa: NDArray[np.float64]

@dataclass(frozen=True)
class Series:
    """One row per sampled step. All arrays share length n_steps."""
    step: NDArray[np.int64]
    time_s: NDArray[np.float64]
    cutting_force_n: NDArray[np.float64]     # Fx on the tool
    thrust_force_n: NDArray[np.float64]      # Fy on the tool
    time_step_s: NDArray[np.float64]         # the actual dt used, for dt_slope_window
    kinetic_energy_j: NDArray[np.float64]
    internal_energy_j: NDArray[np.float64]
    external_work_j: NDArray[np.float64]
    n_points: NDArray[np.int64]
    deleted_mass_kg: NDArray[np.float64]     # cumulative

@dataclass(frozen=True)
class Deletion:
    """One record per separated material point — the deletion-health input."""
    step: NDArray[np.int64]
    point_id: NDArray[np.int64]
    coords_m: NDArray[np.float64]            # (n, 2), position at separation
    temperature_k: NDArray[np.float64]
    mass_kg: NDArray[np.float64]

@dataclass(frozen=True)
class Episode:
    schema_version: str
    metadata: CaseMetadata
    series: Series
    deletions: Deletion
    snapshot_steps: tuple[int, ...]          # index; snapshots load lazily
```

### Functions

```python
def write_manifest(episode_dir: Path, metadata: CaseMetadata, snapshot_steps: Sequence[int]) -> None
def append_snapshot(episode_dir: Path, snapshot: Snapshot) -> None
def write_series(episode_dir: Path, series: Series, deletions: Deletion) -> None
def load_episode(episode_dir: Path) -> Episode
def load_snapshot(episode_dir: Path, step: int) -> Snapshot
def final_snapshot(episode_dir: Path) -> Snapshot
```

`load_episode` raises on a major-version mismatch with a message naming both versions. A minor
mismatch loads, because minor changes are additive by contract.

Every dataclass validates its own array-length invariant on construction (`__post_init__`) and
raises `ValueError` naming the two disagreeing fields. That is the "fail loud in the small" rule
and it is what stops a silently truncated array reaching a diagnostic.

### Tests

Round-trip every type through disk and compare exactly. A snapshot whose arrays disagree in
length raises. A manifest claiming `"2.0"` raises on load; `"1.1"` loads. Loading a step that was
never written raises with the step number in the message.

---

## Task 2 — Telemetry writer (feature 5, solver half)

**Files:** `src/quinney/kratos_adapter/telemetry_writer.py`, `tests/test_telemetry_writer.py`

The only Kratos-touching half. A process that reads material point state and emits Task 1's types.

- `snapshot_from_part(part, step, time_s, damage_ledger) -> Snapshot` — reads `MP_COORD`,
  `MP_VELOCITY`, `MP_EQUIVALENT_PLASTIC_STRAIN`, `MP_EQUIVALENT_PLASTIC_STRAIN_RATE`,
  `MP_TEMPERATURE`, `MP_MASS`, `MP_EQUIVALENT_STRESS` via `CalculateOnIntegrationPoints`, and
  pulls damage from the ledger by point id. Read `docs/kratos_api_notes.md`, section "Material
  point variables", for what is actually registered and what the pybind slicing gotcha is — do
  not guess variable names. If one is genuinely missing from the install, record that in the
  notes and set the field to zeros with a comment, rather than crashing the run.
- `TelemetryWriterProcess` — a Kratos process with the same `ExecuteFinalizeSolutionStep` shape as
  `JohnsonCookDamageProcess`. Samples every `sample_every` steps, accumulates series rows every
  step, records deletions as they happen, and flushes on `ExecuteFinalize`.
- Deletion records come from the ids that `JohnsonCookDamageProcess` flags. Coordinate with its
  existing interface (read `damage_process.py`); extend it if it does not currently expose which
  ids were erased on a given step, keeping its existing tests passing.

Test against a real `KM.ModelPart` with real elements, the way `tests/test_damage_process.py`
already does — including the injected-read pattern it uses for `MP_` variables.

---

## Task 3 — Diagnostics: shear bands (feature 6)

**Files:** `src/quinney/diagnostics/__init__.py`, `src/quinney/diagnostics/bands.py`,
`src/quinney/diagnostics/merchant.py`, `tests/test_bands.py`, `tests/test_merchant.py`

Spec section 5, "Shear localization — primary". **This is the diagnostic group the whole project
exists for** (spec section 1): telling a physical shear band from numerical mesh collapse. Take
it seriously.

`merchant.py` — pure trigonometry, no telemetry dependency:
- `merchant_shear_angle_deg(rake_angle_deg, friction_angle_deg) -> float` —
  `phi = 45 + alpha/2 - beta/2`.
- `friction_angle_deg(cutting_force_n, thrust_force_n, rake_angle_deg) -> float` — from the
  measured force ratio, so the reference angle is derivable from telemetry alone.
- `shear_angle_from_thickness_ratio_deg(r, rake_angle_deg) -> float` —
  `tan(phi) = r cos(alpha) / (1 - r sin(alpha))`. Two independent routes to the same angle is
  itself a diagnostic: them disagreeing means the chip is not steady.

`bands.py` — takes a `Snapshot` plus `CaseMetadata`:
- `high_strain_mask(eps_p, percentile) -> NDArray[bool]` — the band candidate set. The percentile
  is a named constant with a stated reason, not a bare literal at a call site.
- `band_coherence(coords_m, mask, cell_size_m) -> float` (0..1) — spatial autocorrelation of the
  high-strain set. A coherent band scores near 1; scattered high-strain points score near 0.
  Document the estimator you chose in the docstring, in one sentence, with why. It must be
  scale-free (independent of point count) or the number is not comparable between runs.
- `band_orientation_deg(coords_m, mask) -> float | None` — principal axis of the high-strain
  cloud, measured from the cutting direction, in `0..180`. Undefined for a near-isotropic cloud:
  return `None` when the eigenvalue ratio is below a named threshold, rather than reporting the
  orientation of noise.
- `band_width_elements(coords_m, mask, cell_size_m) -> float` — extent perpendicular to the band
  axis, in units of background cells. `MIN_RESOLVED_BAND_ELEMENTS = 3.0` (spec section 5: thinner
  is mesh-resolution-limited, not physics) lives here and nowhere else.
- `temperature_band_alignment(coords_m, mask, temperature_k) -> float` (0..1) — does the hot
  region coincide with the high-strain region? This is the corroborating signal spec section 1
  names. Point-biserial correlation between band membership and temperature is acceptable; state
  the choice.
- `temp_gradient_max_k_per_m(coords_m, temperature_k) -> float` — from nearest-neighbour
  differences; document the neighbourhood.

**Tests must include synthetic ground truth**, because that is the only way to know the
estimators are right: a hand-built diagonal band of high-strain points at a known angle must
recover that angle within a stated tolerance and score high coherence; a uniform cloud must score
low coherence and return `None` for orientation; a band one cell wide must report width below
`MIN_RESOLVED_BAND_ELEMENTS`. Build the synthetic point clouds deterministically — `np.linspace`
and fixed offsets, or `np.random.default_rng(SEED)` with `SEED` a named constant. Never unseeded
randomness.

---

## Task 4 — Diagnostics: chip morphology and forces (feature 6)

**Files:** `src/quinney/diagnostics/morphology.py`, `src/quinney/diagnostics/forces.py`,
`tests/test_morphology.py`, `tests/test_forces.py`

Spec section 5, "Chip morphology" and "Forces". Morphology is the **outcome variable** and the
regime classification is scored against ground truth as the cleanest measure of understanding
(spec section 6), so its edges must be defined in exactly one place and be inspectable.

`forces.py` — takes a `Series`:
- `steady_window(series, discard_fraction) -> slice` — the tool approaches before it cuts, so
  every force statistic is computed over a steady window, never the whole record.
  `STEADY_STATE_DISCARD_FRACTION` is a named constant with the reason.
- `cutting_force_mean_n`, `thrust_force_mean_n`, `force_oscillation_amplitude_n` (about the
  steady mean), `force_dominant_freq_hz` — the last from an FFT of the detrended steady signal,
  ignoring the DC bin, with sample spacing taken from the series' own time column rather than
  assumed uniform.

`morphology.py`:
- `chip_thickness_m(snapshot, metadata) -> float` and `chip_thickness_ratio(...) -> float`
  (`r = t0 / tc`, so `r < 1` for a real chip). Identify chip material by displacement above the
  original workpiece surface, as `bench/orthogonal_cutting/analyze.py` prototypes — read it, then
  do it properly against the contract.
- `segment_frequency_hz(series) -> float | None` — reuses `force_dominant_freq_hz`. Return `None`
  when the oscillation amplitude is below a named significance floor: a continuous chip has no
  segment frequency, and reporting FFT noise as one is the failure mode to avoid.
- `tool_chip_contact_length_m(snapshot, metadata) -> float` — extent of chip material within a
  named contact tolerance of the rake face.
- `classify_chip_regime(...) -> ChipRegime` — a `StrEnum` of `CONTINUOUS | SEGMENTED |
  DISCONTINUOUS | INDETERMINATE`. **The edges live in one module-level table with a cited reason
  per edge**, and the function returns the regime together with the values that decided it, so a
  supervisor can cite evidence (spec section 6 requires `evidence`). `INDETERMINATE` is a real
  answer and must be reachable — a classifier that always commits is not measuring anything.

Tests: a synthetic square-wave force signal recovers its own frequency; a flat signal returns
`None` for segment frequency; a chip cloud at a known thickness recovers its ratio; each regime
edge is exercised from both sides.

---

## Task 5 — Diagnostics: deletion and numerical health (feature 6)

**Files:** `src/quinney/diagnostics/health.py`, `tests/test_health.py`

Spec section 5, "Deletion health" and "Numerical health". This group is what tells physical
separation from numerical erosion — the second half of spec section 1's central question.

Thresholds, each a named constant with the spec section in a comment, defined here only:
`MAX_DELETED_MASS_PCT = 2.0`, `MAX_ENERGY_DRIFT_PCT = 5.0`, `MAX_HOURGLASS_RATIO_PCT = 10.0`.

- `deleted_mass_pct(series) -> float` — cumulative deleted mass over initial total.
- `deletion_rate_per_s(deletions, window_s) -> float`.
- `deletion_spatial_coherence(deletions, cell_size_m, window) -> float` (0..1) — clustered means
  separation, diffuse means numerical erosion. Reuse the coherence estimator from `bands.py`
  rather than writing a second one; if its signature does not fit, say so in your report and use
  the smallest shared helper that serves both. Do not duplicate the algorithm.
- `deletion_temp_correlation(deletions, snapshot) -> float` — are deleted points hotter than the
  body? Physical separation should correlate with the hot band.
- `energy_drift_pct(series) -> float` — drift of `(KE + IE - W_ext)` relative to total work done,
  over the steady window.
- `dt_slope_per_step(series, window) -> float` — the trend spec section 0 calls out: dt
  collapsing over thousands of steps means the mesh is degenerating. State the sign convention in
  the docstring.
- `hourglass_ie_ratio_pct`, `jacobian_min`, `aspect_ratio_p99`, `plastic_work_delta_last_remesh`
  — **return `None` under MPM**, per the ruling above. Each carries a one-line docstring saying
  why it is `None` and what would make it meaningful.

---

## Task 6 — The standing diagnostic vector (feature 6, assembly)

**Files:** `src/quinney/diagnostics/summary.py`, `tests/test_summary.py`

One frozen dataclass, `DiagnosticSummary`, carrying **every field spec section 5 names, under the
spec's own names**, plus `schema_version` and the episode's step and time. Optional fields are
`float | None`, never a sentinel like `-1` or `nan`.

`summarize(episode_dir) -> DiagnosticSummary` composes Tasks 3-5 and does no computation of its
own. It is the only function in the package that touches the filesystem, and it does so through
`telemetry.load_episode` / `load_snapshot` only.

`as_dict(summary) -> dict[str, float | str | None]` — the flat, JSON-serializable form the query
interface and the episode store will hand to a model. Field order is stable and defined once.

The field list is the canonical one: bands, morphology, forces, and health modules must not each
keep their own copy of the section 5 names. Test that every section 5 field name appears exactly
once, and that a summary over a synthetic episode round-trips through `as_dict` unchanged.

**Dependency note:** Tasks 3, 4, and 5 are built in the same wave as each other but *before* this
one. Their signatures as written in this plan are what you code against; if one landed with a
different signature, adapt to what is on disk and note the divergence in your report.

---

## Task 7 — Halt, checkpoint, restart (feature 4)

**Files:** `docs/kratos_api_notes.md` (new section), `src/quinney/kratos_adapter/checkpoint.py`,
`src/quinney/kratos_adapter/restart.py`, `tests/test_checkpoint.py`, `tests/test_restart.py`

Spec section 4. The halt-and-restart loop: trigger fires, checkpoint written and telemetry
dumped, solver **exits**, parameters modified, solver **restarts from the checkpoint**.

**Spike first, and report what you find before building on it.** Whether Kratos' own restart
machinery covers MPM material point state is an open question — nothing in
`docs/kratos_api_notes.md` answers it, and MP_ state lives on elements, not nodes, so standard
nodal restart may well not carry it. Determine empirically:
- Does `KratosMultiphysics.RestartUtility` / `SaveRestartProcess` exist in this install, and does
  `MpmAnalysis` cooperate with it?
- If not, what is the minimum state a resume needs to be *bit-comparable*: point coordinates,
  velocities, mass, volume, deformation gradient, plastic strain, temperature, per-point damage
  ledger, per-point `Properties` (the softening path mutates them), plus grid state?

Add a section to `docs/kratos_api_notes.md` recording exactly what you found, in the style of the
existing sections — this is the Phase 1 introspection output the spec asks for and it is as much
a deliverable as the code.

Then build whichever of these the spike supports:
- `checkpoint.py` — `should_halt(...)` trigger predicates (a fixed step interval at minimum, and
  a named diagnostic trip), `write_checkpoint(...)`, and the process that fires them and requests
  exit. **The trigger predicates are pure and belong in a Kratos-free helper** so they can be
  tested and replayed without a solver.
- `restart.py` — `inject_parameters(project_parameters, overrides)` (a pure dict-level transform,
  no Kratos types in the signature if avoidable) and `resume_from(checkpoint_dir)`.

**Acceptance:** a short run halted at step N and resumed reaches the same state at step 2N as an
uninterrupted run of 2N steps, within a stated tolerance. State the tolerance and why it is not
exactly zero if it is not. If the spike shows bit-comparable resume is not achievable with the
installed wheels, **say so in the report and in the notes, implement the closest achievable
thing, and state precisely what diverges** — an honest negative result here is worth more than a
resume path that silently loses plastic strain.

---

## Task 8 — Bench case on the contract, and validation (feature 3)

**Files:** `bench/orthogonal_cutting/run.py`, `bench/orthogonal_cutting/analyze.py`,
`bench/orthogonal_cutting/sweep.py`, `docs/cutting_validation.md`

1. Replace the ad hoc `np.savez_compressed(...)` in `run.py` with `TelemetryWriterProcess`, so the
   bench emits the v1 contract. Keep the run's console progress output.
2. Rewrite `analyze.py` to print `DiagnosticSummary` from the diagnostics package. It must become
   a thin reporter: no diagnostic computation survives in the bench directory. Delete the
   prototype band and thickness code it currently holds — that logic now lives in `diagnostics/`.
3. `sweep.py` — run the case at a small set of cutting speeds and tabulate the summary per speed.
   Spec section 2 says the regime transition is speed-dependent; this is what makes a regime
   classification testable at all. Keep the sweep small enough to finish in an evening on 12 CPU
   threads and say in the docstring roughly how long it takes.
4. `docs/cutting_validation.md` — the measured cutting force, thrust force, and chip thickness
   ratio against published OFHC copper orthogonal cutting data.

**On published data — the honesty constraint.** Do not invent literature values, and do not cite a
paper you have not read. If you cannot obtain real published numbers in this environment, write
the document with the measured values, an explicit **"reference data not yet obtained"** section
naming what is needed (source, cutting speed, rake angle, depth of cut), and leave the comparison
column empty. A fabricated validation is worse than an absent one — it would silently become the
ground truth that spec section 11 scores everything against.

---

## Waves

Tasks in a wave own disjoint files and run in parallel. The controller commits after each wave.

- **Wave 1:** Task 1 — alone; everything reads this contract.
- **Wave 2:** Tasks 2, 3, 4, 5 — parallel.
- **Wave 3:** Tasks 6, 7 — parallel.
- **Wave 4:** Task 8 — alone; integrates all of the above.
