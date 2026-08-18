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

## Open questions

1. **Whether `MP_EQUIVALENT_PLASTIC_STRAIN` is reset on remesh or point
   regeneration.** The damage increment differences it between steps, so a
   reset would register as a large negative increment. `accumulate_damage`
   floors increments at zero, so the failure mode is silently *missed* damage
   rather than spurious damage. Worth confirming.
2. **Whether the plane-strain σzz reconstruction holds near the free surface**,
   where points are closer to elastic than to the plastic limit.
