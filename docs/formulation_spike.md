# Formulation spike — Phase 0

Spec §10 Phase 0 asks for the same cutting case in Lagrangian-with-deletion and
in MPM, compared against published OFHC copper data, to pick a formulation
before investing in the supervision harness.

**The comparison cannot be run as specified, and does not need to be.** The
formulation is decided by what ships.

## The choice is made by availability

Johnson-Cook exists in `MPMApplication` and nowhere else in the install (see
[kratos_api_notes.md](kratos_api_notes.md)). A Lagrangian arm would have to use
a different constitutive model, so a Lagrangian-vs-MPM comparison would be
comparing two constitutive models, not two formulations, and could not settle
the question it was meant to settle.

**Decision: MPM, `JohnsonCookThermalPlastic2DPlaneStrainLaw`, explicit central
difference.** The spec's own reasoning already favoured it — separation is
native, and there is no Gauss-point history mapping to lose accuracy through,
which removes `plastic_work_delta_last_remesh` (§5) as a concern entirely.

## What was actually run

`bench/jc_compression/` — an OFHC copper block (4 × 8 mm, 768 material points)
driven at 150 m/s into a clamped background-grid boundary. 201 explicit steps
at Δt = 2×10⁻⁸ s. Roughly a minute on 12 CPU threads.

This is a smoke case, not a validation case. It exists to prove the toolchain
runs and to settle the API questions, both of which it did.

### Results

| Quantity | Value |
|---|---|
| Steps completed | 201 / 201 |
| Material points | 768 → 768 (no loss) |
| Max equivalent plastic strain | 1.296 |
| Max plastic strain rate | 4.85 × 10⁵ /s |
| Max temperature | 433.9 K (rise of 140 K from 294 K) |
| Max equivalent stress | 480.2 MPa |
| Mass | conserved exactly |

### The physics checks out

Two independent hand calculations against the measured output:

**Flow stress.** Johnson-Cook at ε̄ᵖ = 1.30, ε̇ = 4.85×10⁵ /s, T = 434 K:

```
(90 + 292·1.30^0.31) · (1 + 0.025·ln(4.85e5)) · (1 − 0.132^1.09) ≈ 480 MPa
```

Measured maximum: 480.2 MPa.

**Adiabatic heating.** ΔT = β·σ·ε̄ᵖ/(ρ·c) = 0.9 · 400e6 · 1.3 / (8960 · 385)
≈ 135 K. Measured: 140 K.

The constitutive law and the Taylor-Quinney coupling both behave correctly.
Strain rates land squarely in the 10⁴–10⁶ /s band spec §2 expects for the
primary shear zone.

### Damage runs end to end

With `JohnsonCookDamageProcess` attached and the published OFHC constants:

| Threshold | Points with damage > 0 | Max damage | Separated |
|---|---|---|---|
| 1.0 (textbook) | 689 / 768 | 0.297 | 0 |
| 0.05 | 689 / 768 | 0.053 | 35, first at step 87 |

No separation at D = 1 is the physically right answer here: the block is under
compression (mean triaxiality −0.64), and negative triaxiality drives the
Johnson-Cook fracture strain up steeply. Copper does not fracture when you
squash it. Lowering the threshold does separate points, which exercises the
flag-and-erase path against a live solver.

## Corrections this forced

Three assumptions in the first damage implementation were wrong, and the case
found all three:

1. **`MP_DELTA_PLASTIC_STRAIN` is not implemented** for MPM elements — it
   raises when read. The increment is now differenced from the cumulative
   `MP_EQUIVALENT_PLASTIC_STRAIN` between steps, which needs the previous value
   carried in a ledger.
2. **`MP_PRESSURE` is not implemented** either. Mean stress is reconstructed
   from `MP_CAUCHY_STRESS_VECTOR` as (σxx + σyy)/2 — exact under plastic flow
   in plane strain, where σzz is the intermediate principal stress.
3. **Compression is negative** in the reconstructed mean stress, so no sign
   flip is applied. The original code negated a pressure that does not exist.

## Caveat on energy

Kinetic + strain energy falls 29% over the run. That is **not** numerical
drift — it is plastic work leaving as heat, and the 140 K temperature rise
accounts for it. The `energy_drift_pct` diagnostic in spec §5 must therefore
include plastic dissipation (computable as Σ σ_eq · Δε̄ᵖ · volume) or it will
report a healthy explicit run as catastrophically unstable.

## Orthogonal cutting — `bench/orthogonal_cutting/`

A 1.0 × 0.3 mm OFHC copper workpiece at 50 µm resolution, zero-rake tool at
20 m/s, 100 µm depth of cut. 3125 explicit steps, 816 material points, a few
minutes on 12 threads.

The tool is a second body of material points held at cutting velocity by
`RigidToolProcess`. The impulse needed to restore that velocity each step *is*
the force the workpiece exerts, so cutting and thrust force come out of driving
the tool with nothing extra to instrument. **MPM needs no contact algorithm** —
tool and workpiece interact through the shared background grid.

### What it produced

| Quantity | Value | Reference |
|---|---|---|
| Chip | 116 points lifted above the original surface | — |
| Chip thickness ratio r = t0/tc | 0.32 | 0.3–0.5 typical, zero rake |
| Shear zone orientation | 41.1° from cutting direction | Merchant φ = 45° − β/2 |
| Cutting force Fc | 224 kN/m | — |
| Thrust force Ft | 81.8 kN/m | — |
| Fc/Ft | 2.74 | 2–3 typical at zero rake |
| Specific cutting energy | 2.24 GPa | copper 1.4–2.5 GPa |
| Force oscillation | 10.5%, dominant 80 kHz | — |
| Peak strain rate | 1.14 × 10⁶ /s | §2 expects 10⁴–10⁶ |
| Temperature | median 316 K, p95 646 K | copper machining 400–700 K |
| Plastic strain | median 0.29, p95 3.03 | chip strains 2–4 |

A chip forms, the shear zone sits at a plausible angle, and the forces land
inside the published band for copper. This is a working mechanism, not a
validated model.

### Where it is not yet credible

- **A handful of pathological points.** Max plastic strain is 33.5 and three
  points pin at the melt temperature, against a median of 0.29 and p95 of 3.03.
  So the *field* is physical and a few points at the tool tip are being
  crushed. That is worth dwelling on: this case already generates the exact
  ambiguity spec §1 is about — is a high-strain point physical localization or
  numerical degeneration? Here it is clearly numerical, which makes it a
  labelled example for the diagnostics to be tested against.
- **No friction model.** MPM's grid contact is implicit; spec §2 requires a
  Zorev sticking/sliding treatment. The measured 41.1° shear angle implies a
  friction angle near 8°, which is low — consistent with contact that carries
  almost no shear. This is the largest single gap.
- **Two cells through the depth of cut.** The high-strain zone spans 6.8 cells
  and is only 1.7:1 elongated, so it reads as a diffuse zone rather than a
  band. Continuous-chip behaviour is right for copper at this speed, but the
  resolution cannot distinguish that from an unresolved band.
- **20 m/s is fast** (1200 m/min), chosen to keep runtime to minutes.

### Damage under cutting

Every one of the 720 workpiece points accumulated damage, some reaching 1.0,
and 15 points were erased. Contrast with the compression case, where nothing
separated at D = 1: cutting produces the tensile and shear states that
Johnson-Cook damage responds to, and pure compression does not.

## Not done yet

- **Zorev friction** at the tool-chip interface (§2) — the biggest gap.
- **Rake angle** other than zero, and a mesh fine enough to resolve a band.
- **Validation against published OFHC copper data**, not just published ranges.
- **Chip regime transition** — segmented chips need conditions this case does
  not reach.
