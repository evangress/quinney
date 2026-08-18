# LLM-Supervised Chip Formation Simulation — Implementation Spec v2

**Target:** Kratos Multiphysics, explicit thermo-mechanical, 2D orthogonal cutting
**Supervisor:** Paused-solver architecture, model-agnostic
**Handoff:** Written as a build brief for Claude Code.

---

## 0. Framing (read first)

**Explicit dynamics does not converge.** No equilibrium iteration exists to accelerate.
Central-difference marching is conditionally stable. Success is measured as run completion,
energy fidelity, and *physical* chip morphology — not convergence rate.

**The LLM never computes the time step.** `Δt ≤ L_min / c_d` is evaluated analytically every
step in microseconds. The LLM sees trends the closed form cannot: Δt collapsing over thousands
of steps means the mesh is degenerating, which is a decision, not a calculation.

**This is a capability-ceiling experiment, not a performance optimization.** The solver is
deliberately halted so the supervisor has unbounded time. That is explicitly not scalable and
that is fine. The question being answered is: *what can a language model infer about a
simulation's physical state when given full access and no time pressure?* Deployment latency
is a separate, later question.

---

## 1. The core problem being supervised

In chip formation, **physical shear localization and numerical mesh collapse both present as
element degeneration.** Distinguishing them is the central diagnostic task, and it is the
reason this domain suits an LLM supervisor better than bulk forming does.

A human expert separates them using several weakly-correlated signals at once:

- Is the high-plastic-strain region **spatially coherent** (a band) or scattered?
- Does the band **orientation** match the theoretical shear plane angle?
- Does the **temperature field** corroborate adiabatic heating along that band?
- Does the **cutting force signal** oscillate at a frequency consistent with segmentation?
- Is element deletion **clustered along the band** (chip separation) or diffuse (numerical erosion)?

No single threshold captures this. That is the premise of the project.

Secondarily: the **damage/deletion threshold in cutting simulations is notoriously arbitrary
and mesh-dependent.** Analysts tune it by hand until chip morphology looks right. Automating
that is automating a judgment call that was always a judgment call.

---

## 2. Physics requirements

### Constitutive model

**Johnson-Cook flow stress** (strain, strain-rate, temperature) plus **Johnson-Cook damage**
initiation and evolution. This is the standard for machining simulation and is required for
the shear-band physics to appear at all.

### Thermal coupling is mandatory

Adiabatic heating drives thermal softening, which drives shear localization, which produces
serrated chips. A purely mechanical model will not reproduce the phenomenon most worth
supervising. Use a coupled thermo-mechanical explicit scheme with the Taylor-Quinney
coefficient for plastic work → heat conversion.

### Regime notes

- Primary shear zone strain rates reach 10⁴–10⁶ /s
- Tool-chip interface needs a sticking/sliding friction treatment (Zorev-type), not pure Coulomb
- Chip morphology regimes — continuous, segmented/serrated, discontinuous — are the outcome
  variable. Regime transition is speed- and material-dependent.

### Material

**OFHC copper.** Johnson-Cook parameters are well established from the original Johnson-Cook
work and widely reproduced. Copper is ductile enough to give continuous-to-segmented behavior
across an accessible speed range, and there is abundant orthogonal cutting validation data.

**Do not use copper for a Charpy/Izod model.** Annealed copper typically bends through rather
than fracturing, giving no separation event to supervise. Charpy is the wrong test here
regardless; orthogonal cutting is the direct target.

---

## 3. Formulation decision — spike this first

| Approach | Fit | Notes |
|---|---|---|
| **Lagrangian + element deletion** | Baseline | Simple, mesh-dependent, mass loss. Start here. |
| **Lagrangian + adaptive remesh (MMG)** | Good | Kratos `MeshingApplication`. Gauss-point state mapping is the hard part. |
| **MPM** (`ParticleMechanicsApplication`) | Possibly best | Handles separation natively, sidesteps history mapping. |
| ALE / CEL | Out of scope | Not well supported in Kratos for this. |

**Phase 0 task:** build the same cutting case in Lagrangian-with-deletion and in MPM, compare
chip morphology and cutting force against published data. Pick the winner before investing in
the supervision harness. This is a one- or two-evening spike and it de-risks everything after.

### Known caveat — history variable mapping

MMG remeshing interpolates nodal variables. **Gauss-point state (equivalent plastic strain,
damage, temperature) mapping is lossy** and is a genuine source of error on every rezone.
Log total plastic work immediately before and after every remesh and treat the delta as a
first-class quality metric. Large losses are the argument for MPM.

---

## 4. Paused-solver architecture

### Halt-and-restart, not blocking

```
Solver runs → trigger fires → checkpoint written + telemetry dumped → solver EXITS
                                                                          ↓
                                          Supervisor process (unbounded time, tool access)
                                                                          ↓
                        Modified ProjectParameters written → solver RESTARTS from checkpoint
```

Simpler than an in-process blocking call, makes every decision trivially replayable, and lets
the supervisor be any process at all — a local model, an API call, or a human.

### The query interface — this is the key piece

With the solver halted, the supervisor does not receive a fixed feature vector. It receives a
summary plus **tools to investigate**:

```python
get_summary()                              # the standing diagnostic vector
get_field_stats(field, region=None)        # percentiles, histogram
get_element_neighborhood(element_id, k)    # local topology + state
get_history(quantity, window)              # time series
get_spatial_correlation(field, threshold)  # band/cluster detection
compute_merchant_angle()                   # analytical reference for comparison
get_similar_episodes(k)                    # retrieval from L5 memory
render_field(field, region)                # PNG — vision models only
propose_action(...)                        # terminal
```

This converts the supervisor from a classifier into an agent that investigates. It is the
whole point of pausing. A fixed vector delivered slowly gains nothing over a fixed vector
delivered fast.

### Safety envelope still applies

Unbounded thinking time does not grant unbounded authority. All actions pass schema
validation and hard parameter clamping. Δt may never go below CFL. `abort_with_diagnosis`
is always available and always safe.

---

## 5. Diagnostic summary (the standing vector)

### Shear localization — primary

- `plastic_strain_band_coherence` — spatial autocorrelation of high-ε̄ᵖ elements
- `band_orientation_deg` and delta vs. Merchant-predicted shear angle
- `band_width_elements` — a band under ~3 elements wide is mesh-resolution-limited, not physics
- `temperature_max`, `temp_gradient_max`, `temp_band_alignment` — does heating corroborate the band?

### Chip morphology — the outcome variable

- `chip_thickness_ratio`
- `segment_frequency_hz` from force signal FFT
- `chip_regime_classification` — continuous / segmented / discontinuous
- `tool_chip_contact_length`

### Forces — directly comparable to experiment

- `cutting_force_mean`, `thrust_force_mean`
- `force_oscillation_amplitude`, `force_dominant_freq`

### Deletion health

- `deleted_mass_pct` — hard bound, target < 1–2%
- `deletion_rate`, `deletion_spatial_coherence` — clustered = separation, diffuse = numerical erosion
- `deletion_temp_correlation` — physical separation should correlate with the hot band

### Numerical health

- `energy_drift_pct` (> 5% is a problem)
- `hourglass_ie_ratio` (> 10% means fake stiffening)
- `jacobian_min`, `aspect_ratio_p99`, `dt_slope_window`
- `plastic_work_delta_last_remesh` — history-mapping loss

---

## 6. Action space

```json
{
  "action": "no_action | adjust_deletion_threshold | adjust_damage_evolution |
             trigger_remesh | refine_shear_zone | adjust_friction_params |
             switch_formulation | classify_chip_regime | abort_with_diagnosis",
  "params": { },
  "rationale": "...",
  "evidence": ["which diagnostics drove this"],
  "confidence": 0.0
}
```

`classify_chip_regime` is an **observation** action with no intervention. It is scored against
ground truth independently and is the cleanest measure of whether the model actually
understands the simulation state. Include it.

`evidence` is required. It is how you audit whether correct actions came from correct
reasoning or from luck.

---

## 7. Serving infrastructure

**Hardware:** desktop, 16 GB dedicated GPU. Kratos is CPU-bound and fully halted during
supervision, so there is **zero contention** — the whole card is available for inference.

### Use both backends, for different jobs

| Backend | Job | Why |
|---|---|---|
| **Ollama** | Interactive dev loop, model ladder | Trivial model swapping; native JSON-schema structured output; single-request workload where vLLM's advantage doesn't apply |
| **vLLM** | Replay batch evaluation | Continuous batching over hundreds of stored episodes; built-in guided decoding (xgrammar/outlines) |

Code against the **OpenAI-compatible chat completions interface** and route through the
existing LiteLLM instance. Backend selection then becomes a config change, and the ladder
descent costs nothing to run.

### VRAM budget (16 GB)

| Class | Quant | Weights | Verdict |
|---|---|---|---|
| 7–8B | Q8_0 | ~8 GB | Fast, generous context. Lower rung of ladder. |
| **14B** | **Q4_K_M** | **~9 GB** | **Primary target.** Room for KV cache + usable context. |
| 24B | Q4_K_M | ~14 GB | Tight; context-starved. |
| 32B | Q4_K_M | ~19 GB | Does not fit. Partial offload destroys throughput. |
| 32B | Q3_K_M | ~15 GB | Fits, but Q3 quantization damage to structured output is real. Not recommended. |

**Context is the binding constraint, not weights.** The agent loop accumulates tool results
across turns, so KV cache grows fast. Enable flash attention and KV cache quantization
(`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`) to buy headroom. Verify actual
context capacity empirically before designing prompt budgets around it.

**Starting models:** Qwen2.5-14B-Instruct or Qwen2.5-Coder-14B. Verify current best-in-class
at build time — this space moves quickly.

### Do not rely on native function-calling

Tool-calling support at the 14B scale via Ollama is inconsistent across models and fails in
ways that are hard to diagnose inside an agent loop. **Use explicit JSON-schema-constrained
output with a hand-rolled tool dispatch loop.** More code, far fewer mystery failures, and it
keeps the interface identical across every rung of the ladder.

---

## 7a. Model ladder

Because latency is irrelevant under the paused-solver architecture, **do not start small.**

1. **Establish the ceiling** with a strong model via API (Claude). 16 GB cannot host a
   frontier-class model, so the ceiling measurement is necessarily remote. This answers the
   actual research question: what is inferable given full access and no time pressure?
2. **Descend the ladder** — 14B → 8B → 7B, locally — on *identical stored episodes* via replay.
3. **Find the break point.** Which diagnostics degrade first, and at what scale?

`render_field` is only usable by vision-capable models. Text-only rungs must be scored on the
text tools alone for a fair comparison — or drop vision from the ceiling run entirely to keep
the comparison clean.

### Bootstrap: human-in-the-loop first

Halt-and-restart means **you can be the supervisor** for the first 10–20 decisions. This
seeds the episode library with expert judgment and generates the few-shot examples before any
automation exists. It also validates that the query interface exposes what a decision actually
requires — if you find yourself wanting a tool that does not exist, that is the most valuable
output of the whole phase.

---

## 8. Baseline requirement

Build a **rule-based supervisor** and keep it permanently. It is the control arm. Every LLM
result is reported against it. There is a real possibility rules win on several diagnostics —
that is a valid finding, and without the baseline it is undetectable.

---

## 9. Repository layout

```
chip-supervisor/
├── docs/
│   ├── kratos_api_notes.md        # Phase 1 introspection output
│   └── formulation_spike.md       # Phase 0 Lagrangian vs MPM comparison
├── kratos_adapter/
│   ├── processes.py               # triggers, checkpoint, telemetry dump
│   ├── restart.py                 # parameter injection + resume
│   └── project_params/
├── diagnostics/
│   ├── api.py                     # the query interface (Section 4)
│   ├── bands.py                   # shear band coherence + orientation
│   ├── morphology.py              # chip ratio, segment freq, regime
│   └── render.py                  # field → PNG
├── supervisor/
│   ├── agent.py                   # tool-use loop
│   ├── schema.py                  # pydantic action schema
│   ├── rules.py                   # baseline
│   └── prompts/
├── envelope/validate.py
├── memory/store.py                # SQLite episodes
├── bench/
│   ├── orthogonal_cutting/
│   └── seeded_pathologies/
└── replay.py
```

---

## 10. Phase plan

**Phase 0 — Formulation spike.** Orthogonal cutting in Lagrangian-with-deletion and in MPM.
Compare chip morphology, cutting force, and thrust force to published OFHC copper data.
Pick one. Document in `formulation_spike.md`.

**Phase 1 — Halt/restart + telemetry.** Kratos API introspection. Checkpoint-exit-resume loop
proven to produce bit-comparable continuation. Diagnostic summary emitted. `replay.py`
skeleton. No supervisor.

**Phase 2 — Query interface + human in the loop.** Build `diagnostics/api.py`. You act as
supervisor for 10–20 decisions. Seed the episode library. Refine the tool surface based on
what you actually reach for.

**Phase 3 — Rule baseline + envelope.** Seeded pathology matrix. Baseline performance
recorded. System safe and useful with no LLM connected.

**Phase 4 — Agent supervisor, strong model.** Tool-use loop. Establish the capability ceiling.
A/B against Phase 3 baseline.

**Phase 5 — Model ladder descent.** 32B → 14B → 7B on stored episodes via replay. Identify
break points per diagnostic. This is the deployment-feasibility answer.

---

## 11. Success criteria

Against the Phase 3 rule baseline:

- **Chip regime classification accuracy** vs. ground truth (cleanest signal of understanding)
- **Chip thickness ratio and cutting force** error vs. published experimental data
- **Run completion rate** on the seeded pathology matrix
- **Deleted mass fraction** held under bound
- **Localization call accuracy** — correctly separating physical shear bands from mesh collapse
- **Evidence quality** — did correct actions cite correct diagnostics, or get there by luck?
- **Ladder break point** — smallest model retaining acceptable performance per diagnostic
