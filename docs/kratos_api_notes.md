# Kratos API notes

Introspection output, per spec §10 Phase 1. Everything here was verified
against the installed wheels on this machine (Windows, CPython 3.14.6,
Kratos 10.4.3) — not read from documentation.

## Install

All required applications publish `cp314` / `win_amd64` wheels at 10.4.3:

```
KratosMultiphysics                    core
KratosStructuralMechanicsApplication  Lagrangian
KratosConstitutiveLawsApplication     generic plasticity/damage
KratosConvectionDiffusionApplication  thermal
KratosMPMApplication                  material point method
KratosMeshingApplication              MMG remesh
KratosLinearSolversApplication
```

Pin the applications individually. The `KratosMultiphysics-all` metapackage
lags at 10.3.1 and ships no cp314 wheels.

## Johnson-Cook is MPM-only

This is the finding that reshapes spec §3.

```
JohnsonCookThermalPlastic2DPlaneStrainLaw   <- 2D orthogonal cutting
JohnsonCookThermalPlastic2DAxisymLaw
JohnsonCookThermalPlastic3DLaw
```

All three live in `KratosMPMApplication` and **nowhere else**. A search for
"johnson" across the entire install hits only MPM binaries;
`StructuralMechanicsApplication` and `ConstitutiveLawsApplication` contain no
Johnson-Cook of any kind.

A string in the binary reads:

> The Johnson Cook MPM material law is currently limited to explicit time
> integration only

Explicit-only, thermally coupled, plane-strain available. It matches spec §2's
requirements, and `TAYLOR_QUINNEY_COEFFICIENT` is a registered variable.

**Consequence.** Spec §3 lists "Lagrangian + element deletion" as the baseline
to start from and MPM as "possibly best". Reality inverts this: the Lagrangian
arm cannot use Johnson-Cook without a custom C++ law, so a Lagrangian-vs-MPM
spike would be comparing two different constitutive models and could not
settle the formulation question. MPM is the only arm where the required
physics ships.

## Johnson-Cook damage does not exist

There is no JC damage law anywhere in the install, and no damage variable in
MPM at all — a scan for `Damage` symbols in the MPM binaries returns nothing.
`ConstitutiveLawsApplication` has generic `SmallStrainIsotropicDamage`, which
is a different model.

This is why `quinney.materials.johnson_cook` exists: quinney owns the damage
law, computes it from exposed material-point state, and drives separation
through `TO_ERASE` + `MaterialPointEraseProcess`.

The limit of that approach: no progressive stress degradation. The compiled
plasticity law computes stress internally and exposes no hook to scale it, so
damage removes points rather than softening them. Full continuum coupling
would need a C++ constitutive law and a Kratos build from source.

## Material point variables

47 `MP_` variables are registered. Registered is not the same as readable —
see the next section — so this table gives the *use*, and readability is stated
there.

| Variable | Use |
|---|---|
| `MP_EQUIVALENT_PLASTIC_STRAIN` | band coherence, §5 |
| `MP_DELTA_PLASTIC_STRAIN` | would be the damage increment — NOT IMPLEMENTED, differenced instead |
| `MP_EQUIVALENT_PLASTIC_STRAIN_RATE` | JC rate term; 10^4–10^6 /s regime check |
| `MP_PRESSURE` | would give mean stress — NOT IMPLEMENTED |
| `MP_EQUIVALENT_STRESS` | triaxiality denominator; readable |
| `MP_CAUCHY_STRESS_VECTOR` | (σxx, σyy, σxy); the actual source of mean stress |
| `MP_TEMPERATURE`, `REFERENCE_TEMPERATURE`, `MELD_TEMPERATURE` | JC thermal term |
| `MP_MASS`, `MP_VOLUME` | `deleted_mass_pct`, §5 |
| `MP_KINETIC_ENERGY`, `MP_STRAIN_ENERGY`, `MP_TOTAL_ENERGY` | `energy_drift_pct`, §5 |
| `MP_COORD` | spatial coherence, band orientation |

**Namespace:** `MP_*` variables are on `KratosMultiphysics.MPMApplication`,
*not* on `KratosMultiphysics`. `TO_ERASE` is on the core namespace.

**Access:** `element.CalculateOnIntegrationPoints(variable, process_info)`
returns a list; `SetValuesOnIntegrationPoints` writes.

## Erasure — verified working

```python
element.Set(KM.TO_ERASE, True)
MPM.MaterialPointEraseProcess(model_part).Execute()
```

Confirmed on a real `ModelPart`: flagged elements are removed, unflagged ones
survive, and the process logs the count it erased. This path is covered by
`tests/test_damage_process.py` against real Kratos objects.

## Readability — verified on live material points

Registered does **not** mean readable. These raise
`Variable ... is called in CalculateOnIntegrationPoints, but is not implemented`
on a real MPM element:

```
MP_PRESSURE
MP_DELTA_PLASTIC_STRAIN
MP_ACCUMULATED_PLASTIC_DEVIATORIC_STRAIN
MP_DELTA_PLASTIC_DEVIATORIC_STRAIN
```

Confirmed readable, with live values from `bench/jc_compression`:

```
MP_EQUIVALENT_PLASTIC_STRAIN        MP_EQUIVALENT_PLASTIC_STRAIN_RATE
MP_TEMPERATURE                      MP_EQUIVALENT_STRESS
MP_CAUCHY_STRESS_VECTOR             MP_MASS, MP_VOLUME, MP_DENSITY
MP_COORD, MP_VELOCITY, MP_DISPLACEMENT
MP_HARDENING_RATIO                  MP_KINETIC_ENERGY, MP_STRAIN_ENERGY, MP_TOTAL_ENERGY
```

`MP_CAUCHY_STRESS_VECTOR` is (σxx, σyy, σxy) in plane strain — three
components. σzz is never reported, so mean stress is reconstructed as
(σxx + σyy)/2, exact under plastic flow where σzz is the intermediate
principal stress. **Compression is negative** in that reconstruction.

## Running a case — required settings

Found by iteration; each of these fails the run if omitted:

- `"analysis_type": "linear"` — the explicit solver accepts nothing else,
  despite `"non_linear"` being the base default.
- `"TEMPERATURE"` as a material property, distinct from
  `"REFERENCE_TEMPERATURE"`. The law's `Check()` demands a positive value near
  293 K and aborts across all OpenMP threads without it.
- JC constants are named `JC_PARAMETER_A`, `_B`, `_C`, `_n`, `_m` — the last
  two lowercase.
- Body elements are `MPMUpdatedLagrangian2D3N`; the background grid uses plain
  `Element2D3N`. Model parts must be named `Background_Grid` and
  `Initial_MPM_Material`; material points land in `MPM_Material`.
- `MATERIAL_POINTS_PER_ELEMENT` goes in the material Variables block.

Note that `MaterialPointEraseProcess` is already run by the solver's own
element search (`remove_entities_not_found`), so points flagged `TO_ERASE` are
removed without quinney calling the erase process itself.

## pybind gotcha — slicing array_1d

`MP_COORD`, `MP_VELOCITY` and friends return `array_1d<double,3>`. Slicing one
in Python (`value[:2]`) produces a `ublas::vector_slice`, which pybind cannot
convert, and the call fails with an unregistered-type `TypeError` at runtime —
not at import. Read components individually (`value[0]`, `value[1]`) instead.

## Friction — slip boundaries only

Kratos MPM applies friction at *slip boundaries*: grid line conditions carrying
normals, marked `SLIP`, with a per-node `FRICTION_COEFFICIENT` and a tangential
penalty. It does **not** apply friction between two deformable bodies —
single-grid MPM contact is inherently no-slip and exposes no coefficient.

This decides case architecture: a tool modelled as a body of material points
cannot have interfacial friction. The tool must be a slip boundary.

Three settings are required and none is discoverable from the defaults; each
one fails differently:

- `"process_name": "ApplyMPMSlipBoundaryProcess"` on the process entry.
  `mpm_analysis` keys off this exact string to add the `FRICTION_STATE` and
  `STICK_FORCE` nodal variables. Without it the run dies mid-solve with
  *"Missing FRICTION_STATE variable in solution step data"*.
- `"auxiliary_variables_list"` must be present in `solver_settings` even when
  empty — `mpm_analysis` reads it before the solver applies its own defaults,
  and otherwise raises *"Getting a value that does not exist"*.
- `"NORMAL"` must be in that list. It is not among the solver's own nodal
  variables, and the slip process writes its computed normals there. Without it
  every boundary normal stays (0,0,0) and the friction machinery silently does
  nothing — no error, no effect.

Both friction coefficient and tangential penalty coefficient must be positive
or `ApplyMPMSlipBoundaryProcess` never sets `FRICTION_ACTIVE`.

## Open questions

1. **Whether `MP_EQUIVALENT_PLASTIC_STRAIN` is reset on remesh or point
   regeneration.** The damage increment differences it between steps, so a
   reset would register as a large negative increment. `accumulate_damage`
   floors increments at zero, so the failure mode is silently *missed* damage
   rather than spurious damage. Worth confirming.
2. **Whether the plane-strain σzz reconstruction holds near the free surface**,
   where points are closer to elastic than to the plastic limit.
3. **Whether `MPMExplicitScheme` applies slip-boundary friction at all.** With
   normals, SLIP flags, coefficients and `FRICTION_ACTIVE` all confirmed set,
   `bench/zorev_friction` produces identical results at mu = 0 and mu = 0.5.
   The Johnson-Cook law is explicit-only, so if friction is implemented only in
   the implicit path, Zorev cannot reach the cutting case through this
   mechanism. The next diagnostic is a case with genuine tangential sliding at
   the boundary — the current one impacts normally, so there may simply be
   nothing for friction to resist.

## Restart — verified on MPM material points

The open question was whether Kratos' restart machinery covers material-point
state at all, since `MP_*` state lives on elements rather than nodes. It does,
and completely.

**What exists.** `KratosMultiphysics.RestartUtility` and
`save_restart_process.SaveRestartProcess` are both present (Python, not
compiled); the C++ side is `FileSerializer` / `StreamSerializer` on the core
namespace. `mpm_solver.py` is restart-aware throughout — `is_restarted()` gates
material-point generation, buffer filling, material import and the
`ACTIVE`-flag pass, and `AssignInitialVelocityToMaterialPointProcess` checks
`IS_RESTARTED` before re-applying itself. MPM restart is a supported path in
10.4.3, not something bolted on.

**Fidelity, measured.** `bench/restart_fidelity/run.py --check`, on
`bench/jc_compression` (768 material points): halt at step 20, resume, compare
against an uninterrupted run at step 40. All **16** fields this document lists
as readable are compared, including the strain rate the damage law reads every
step and the three energy accumulators a halt trigger will watch. Comparing 20
steps *after* the resume rather than at the boundary is deliberate: dropped
state propagates into velocity and stress within a step or two.

Two resume topologies are measured. *One process* halts and resumes in the same
interpreter; *two processes* halts in one subprocess, exits, and resumes in a
second that shares no Model, no ModelPart and no loaded parameters with it.
Only the second is the architecture's actual claim.

At `OMP_NUM_THREADS=1`, **every one of the 16 fields is bit-exact — `max|d|` is
exactly 0.0 — for both topologies**, and so is the control.

At 12 threads nothing is bit-exact, including the control, and that is the
point:

| worst point, relative | resumed, 1 proc | resumed, 2 procs | *two uninterrupted runs* |
|---|---|---|---|
| `MP_EQUIVALENT_PLASTIC_STRAIN` | 3.8e-16 | 3.8e-16 | 1.9e-16 |
| `MP_EQUIVALENT_PLASTIC_STRAIN_RATE` | 1.5e-15 | 1.2e-15 | 1.3e-15 |
| `MP_TEMPERATURE` | 1.8e-16 | 1.8e-16 | 1.8e-16 |
| `MP_TOTAL_ENERGY` | 2.3e-15 | 2.5e-15 | 1.7e-15 |
| `MP_KINETIC_ENERGY` | 4.0e-15 | 4.4e-15 | 3.0e-15 |
| `MP_CAUCHY_STRESS_VECTOR` | 2.2e-15 | 2.0e-15 | 2.3e-15 |
| `MP_ALMANSI_STRAIN_VECTOR` | 2.0e-16 | 2.3e-16 | 2.3e-16 |

Kratos sums nodal contributions across OpenMP threads in a nondeterministic
order, so run-to-run reproducibility is ~1e-15 relative regardless of
restarting. The restart columns sit inside the control column's scatter; on
several fields the two uninterrupted runs disagree *more* than the resumed one
does. The restart contributes nothing measurable on top of the thread
scheduler. **Any resume-fidelity claim needs that control run**, or it blames
the restart for the thread scheduler.

The energy result matters beyond fidelity: `MP_KINETIC_ENERGY`,
`MP_STRAIN_ENERGY` and `MP_TOTAL_ENERGY` come back exactly, so a halt trigger
watching `energy_drift_pct` will not fire on a restart artefact.

`--check` exits non-zero if any field exceeds a stated relative tolerance
(default 1e-13, an order above the observed thread-order floor), so a fidelity
regression cannot pass silently.

Also verified: points erased via `TO_ERASE` before the halt stay erased after
it (761 points before, 761 after), and a supervisor's `time_step` override on
the resumed parameters takes effect — `__ComputeDeltaTime` re-reads it from
settings every step.

**A halt trigger needs a re-arm window, and this is why.** The quantities worth
tripping on are cumulative: energy drift cannot un-drift. A trip that consulted
only the diagnostic would still be over its bound on the first step after the
resume, so the loop would checkpoint, exit and resume on *every step forever* —
one supervision episode per step, and the physics never advances.
`quinney.halt.TRIP_REARM_STEPS` disarms a trip for 50 steps after a halt it
caused. This is a property of supervising an explicit run rather than of
Kratos, but it is obvious only once.

**The deformation gradient is not readable, and does not need to be.** The
brief listed it among the state a resume must carry. MPM 10.4.3 registers no
`MP_DEFORMATION_GRADIENT`; the core `DEFORMATION_GRADIENT`,
`INITIAL_DEFORMATION_GRADIENT_MATRIX` and `GREEN_LAGRANGE_STRAIN_TENSOR` are
registered, but an MPM element returns an **empty list** for all three from
`CalculateOnIntegrationPoints`, and `DETERMINANT_F` raises. It is
element-internal state with no Python accessor.

That is fine for restart, because restart does not read state back and rewrite
it — the serializer takes the element whole. It only means the *measurement*
cannot probe F directly. `MP_ALMANSI_STRAIN_VECTOR` is readable and is the
observable proxy: a strain measure derived from F, and it comes back bit-exact.
Had the serializer dropped F, Almansi strain could not have matched.

**Required settings.**

- The material-point part restarts with
  `"model_import_settings": {"input_type": "rest", ...}`; the background grid
  **cannot** — `_ModelPartReading` raises outright for `"rest"` there. MPM
  carries no state on the grid, so this costs nothing.
- The restart file must be named `MPM_Material_<step>.rest`. `mpm_solver.py`
  overwrites `input_filename` with the literal string `"MPM_Material"` on load,
  so the save has to use the model part name too.
- `RestartUtility` resolves `input_output_path` against `os.getcwd()`. Pass it
  absolute or a solver launched from another directory writes, or looks, in the
  wrong place.
- `"restart_load_file_label"` is mandatory on load and `_GetRestartSettings`
  raises without it. `restart_save_frequency: 0.0` means "save whenever asked",
  which is what a supervisor-driven halt wants.
- `problem_data.start_time` is **ignored on restart**. `analysis_stage.py`
  branches on `IS_RESTARTED` and takes TIME from the restored `ProcessInfo`
  without ever reading `start_time`. Leave it alone: rewriting it changes
  nothing and would imply it mattered.

**What the `.rest` file does not carry.** Only what Kratos itself has a
variable for. Everything quinney keeps in Python is invisible to it.

*Carried*, in `ledgers.json` beside the `.rest` file: the initiation, damage and
previous-plastic-strain ledgers from `JohnsonCookDamageProcess`, written from
its `.ledgers` property and read back into a resumed process through its
`resume_from=` argument. They exist only because MPM has no damage variable at
all. A resume that trusted the `.rest` alone would restart every point
undamaged — silently, since nothing about a zero ledger is malformed.

*Not carried, and there is no seam to carry it*: **`DamageSofteningProcess`'s
state**. Its `_pristine` record is per-point *and* per-variable, which
`PointLedger = Mapping[int, float]` structurally cannot express, and neither it
nor `_applied` has an accessor. `write_checkpoint` could not carry them if
asked.

**The hazard that creates, if anyone wires a softening resume before fixing
it.** The softened `Properties` clones themselves *do* survive the serializer
(see below). So a resumed run that rebuilds `DamageSofteningProcess` has
`_clone_properties()` record the already-degraded values as if they were
pristine, then degrade them *again* by the same damage on the next `apply()` —
(1-D) squared instead of (1-D), compounding at every halt. That is worse than
resetting to pristine, because it is monotone, physics-shaped, and reads as
accelerating damage rather than as a bug. Do not resume softening until
`softening.py` exposes both maps and the checkpoint carries them.

Element `Properties` themselves *do* survive the round trip: a restored element
reads back its `JC_PARAMETER_A`, `YOUNG_MODULUS` and constitutive law
correctly.

## Softening crashes the solver — unrelated to restart

While testing whether per-point `Properties` survive a restart,
`DamageSofteningProcess` was found to **segfault the MPM solver** on the step
*after* it clones properties, in `bench/jc_compression` at 768 points. It is
not a restart bug: the crash reproduces with no restart machinery involved at
all, and a hand-rolled clone loop that also carries `CONSTITUTIVE_LAW` across
crashes identically. `tests/test_softening.py` does not catch it because it
builds a bare `ModelPart` and never solves a step.

So the sequence `element.Properties = <a Properties created at runtime on
MPM_Material>` is not safe mid-run in MPM 10.4.3. The likely suspect is that
the element's cached constitutive-law instance and its properties get out of
step, but that was not chased down. Until it is, the progressive-softening path
cannot run at all — which is also why the double-degradation hazard described
in the restart section is latent rather than live.

## Energy — what MPM exposes, and the one column it does not

Verified by reading a live material point in `bench/jc_compression` (one step,
768 points) rather than from the registry, because registered is not readable.

Readable per material point, in joules, via `CalculateOnIntegrationPoints`:

```
MP_KINETIC_ENERGY      0.5 m v^2 for that point; summing gives the body total
MP_STRAIN_ENERGY       internal energy of that point
MP_TOTAL_ENERGY        exactly the sum of the two above
MP_POTENTIAL_ENERGY    registered and readable, but reads 0.0 with no gravity
```

A sample reading, for scale: a point of mass 3.73e-4 kg at 149.7 m/s reports
`MP_KINETIC_ENERGY` = 4.182 J, which is 0.5 m v^2 to six figures. These are
per-point quantities, not body totals — the total is the sum over points, and
it therefore drops when a point is erased, which is what the deleted-mass
diagnostic exists to account for.

**There is no external-work quantity anywhere in MPM.** A scan of the
application namespace for `WORK` returns nothing, and there is no accumulator
of work done by boundaries, by a driven body, or by the slip-boundary friction.
The core namespace has `EXTERNAL_ENERGY`, but nothing in the MPM solver or its
schemes writes to it.

Consequence for the telemetry contract, and it is worth knowing before anyone
compares this column against a Kratos-native quantity: `Series.external_work_j`
is **integrated by quinney at write time, not read from the solver**. In the
cutting case the driven tool is the only route by which work enters the
workpiece, so `telemetry_writer.tool_power_w` computes `-F.v` from the force
the tool process already measures and the velocity the tool is prescribed to
move at, and `TelemetryWriterProcess` accumulates it over the run.

Sign: `F` is the force the *workpiece* exerts on the tool, which is what
`rigid_tool.reaction_force` returns; the force on the workpiece is its
negative, acting at the tool's velocity. A tool driven at `v = (-V, 0)` into
material that resists it reads `Fx > 0`, giving `-F.v = +Fx*V > 0`. Positive
therefore means the tool is doing work on the workpiece, which is the ordinary
state of a cut; negative means the workpiece is giving energy back.

Two limits of that integral, stated so nobody reads more into the column than
is there. It counts only work entering through the tool — a boundary condition
doing work on the body (the clamped base in `bench/jc_compression`, for
instance) is not in it, so the balance closes only for cases where the tool is
the sole driver. And it assumes the tool actually moves at its prescribed
velocity, which is true by construction for `RigidToolProcess` and would not be
for a tool driven by a force boundary condition.

## The solver clock, read off ProcessInfo

`MpmAnalysis` maintains all three on the model part's `ProcessInfo`, and they
are writable from Python, which is what lets a process be tested without a
solver:

```
ProcessInfo[KM.STEP]         int, 1 after the first solution step
ProcessInfo[KM.TIME]         s
ProcessInfo[KM.DELTA_TIME]   s, the step actually taken
```

`TelemetryWriterProcess` takes its step, time and time step from there rather
than counting steps itself, so the episode cannot drift out of step with the
solver's own clock.

## Telemetry writer — variables read, and the order it must run in

Every variable `telemetry_writer.read_material_point_fields` requests was
confirmed readable on live material points: `MP_COORD`, `MP_VELOCITY`,
`MP_EQUIVALENT_PLASTIC_STRAIN`, `MP_EQUIVALENT_PLASTIC_STRAIN_RATE`,
`MP_TEMPERATURE`, `MP_MASS`, `MP_EQUIVALENT_STRESS` and
`MP_CAUCHY_STRESS_VECTOR`. Nothing in the snapshot is faked; the only column
the solver cannot supply is `damage`, which comes from the damage process's
ledger keyed by point id.

Ordering constraint, and it is a real one. `TelemetryWriterProcess` must run
**once per solution step, after `RigidToolProcess` and after
`JohnsonCookDamageProcess`**, all three in the same `FinalizeSolutionStep`.
Three separate failures make it so, and none of them would leave a mark in the
output:

- A point that reaches the separation threshold is flagged `TO_ERASE` and then
  removed by the solver's own element search (`remove_entities_not_found`) on
  the next step. Its position, temperature and mass are readable only in
  between.
- The damage process reports what separated in its *latest* update. Skip a
  writer step and those points are erased before anything records them, so
  deleted mass comes up short — in the direction that makes an eroding run look
  healthy. Running a process under a `% every_n` guard is the established idiom
  in `bench/orthogonal_cutting/run.py`, which makes this easy to do by accident.
- The tool measures an impulse and divides by dt; the writer multiplies by dt
  again for the work integral. A stale force pairs one step's impulse with
  another step's dt.

All three are refused where they happen rather than documented and hoped for:
the step counter must advance by exactly one, the tool must have measured a new
force since the last call, and a separated point that has already gone raises
rather than being dropped.

## What `MP_STRAIN_ENERGY` actually measures

**It is total internal energy, including accumulated plastic work — not the
recoverable elastic part.** This is the opposite of the usual Kratos
convention, which is why it was measured rather than assumed, and it decides
whether an energy-drift diagnostic is meaningful at all.

Measured on a live material point in `bench/jc_compression`, tracking the most
plastically strained point over 80 steps. The elastic estimate is the
plane-strain elastic energy density `(1/2E)[sxx^2+syy^2+szz^2 - 2nu(...)] +
sxy^2/2G` with `szz = nu(sxx+syy)`, times `MP_VOLUME`; the plastic estimate is
`MP_EQUIVALENT_STRESS * MP_EQUIVALENT_PLASTIC_STRAIN * MP_VOLUME`, which
overstates the true integral because the material hardens as it flows.

```
step  eps_p    sigma_eq   MP_STRAIN_ENERGY   elastic est   plastic est
  20  0.161    335 MPa        3.319 J          0.586 J       2.067 J
  40  0.276    369 MPa        3.705 J          0.370 J       3.773 J
  60  0.375    391 MPa        4.352 J          0.351 J       5.306 J
  80  0.463    406 MPa        4.689 J          0.320 J       6.709 J
```

Two readings, and they agree. The energy is 6x to 15x the elastic estimate, and
it *rises* monotonically with `eps_p` while the elastic estimate *falls* —
which recoverable elastic energy cannot do. Integrating the flow stress by hand
over the plastic strain (roughly 300 MPa average against 0.463 strain over the
point's 3.57e-8 m^3) gives about 4.96 J against the reported 4.689 J: the
reported energy is the total `∫σ:dε`, elastic plus plastic, within the accuracy
of a hand integral.

**Consequences for `energy_drift_pct`.** The good one: plastic dissipation is
already inside `Series.internal_energy_j`, so `KE + IE - W_ext` can close, and a
5% threshold on it is measuring numerical bookkeeping rather than tripping on
correct physics on the first hundred steps. The trap: the heat in
`MP_TEMPERATURE` is that same plastic work expressed as temperature, so adding
a separate thermal term to the balance would count it twice. And the balance
only closes for a case where the tool is the sole driver — see the external
work section above; `bench/jc_compression` has a clamped base doing work that
nothing records, so it is not a case to calibrate a drift threshold on.

## Atomic rewrites of the manifest and the series

`telemetry.write_manifest` and `write_series` write to a `partial-` prefixed
file and `os.replace` it into position. Both are rewritten repeatedly while a
run is in flight — the manifest on every sample, the series on every flush —
and `Path.write_text` truncates before writing, so a solver killed mid-rewrite
would leave an unparseable manifest with every snapshot file beside it intact
and unreadable. `os.replace` is atomic on NTFS and on POSIX. The prefix is not
a suffix because `np.savez` insists on appending `.npz`.
