# Roadmap: The Neuro-Symbolic Homeostat

> Synthesizing Physics, Control Theory, and Inference into a Unified Architecture.
> Ground truth for current capability status and tests is `docs/CHECKLIST.md`. This roadmap is an execution guide; if any status here disagrees, defer to the checklist.

We are building a Homeostat: a self‑regulating control system that maintains internal stability (correctness) while adapting to external stress (data/constraints).

- System 1 (Deep Learning): Fast, intuitive, prone to errors/hallucinations (LLM phenomenon)...(in the codebase we describe these strictly as errors/constraint violations,  as code is reusable we do not assume what you might need this function for, so, we stick to a general term)
- System 2 (This Framework): Slow, deliberate, guaranteed to respect constraints.

Goal: Provide the "System 2" cortex that corrects the "System 1" intuition.

---

## Status Source of Truth

- Live implementation status, tests, and capability flags are tracked exclusively in `docs/CHECKLIST.md`. This roadmap is the one‑shot execution guide (what and how); do not treat status notes here as authoritative. Always check `docs/CHECKLIST.md` for current completion and verification states.

- Windows environment is assumed. All shell examples use PowerShell.

Definition of Done (DoD)
- An item is complete only when all of the following are true:
  - Code: feature implemented and integrated
  - Tests: unit/integration tests added and passing locally
  - Docs: user/developer documentation updated or created if they do not exist (usage, flags, caveats, references if we used a reference for the info or algo, but only if it exists in our references as it might be an original idea)
- Process:
  - Run the relevant tests (or the full suite) and ensure they pass
  - When green, mark the item ✅ in `docs/CHECKLIST.md` (status source of truth)

---

## I. The Core Identity

- Logic → Energy Constraints F(η)
- Inference → Relaxation (gradient/prox/ADMM)
- Correction → Redemption Couplings (future correcting past)
- Reflexes → Weight Adapters (SmallGain, GradNorm)
- Precision → Stiffness/curvature as a first‑class signal for steps and noise
- Noise → Stabilizer and probe (orthogonal, precision‑aware)

---

## II. The Architecture Evolution

### Phase 1: The Physics Engine (Current Status: Mature; see CHECKLIST)

Goal: Treat logic as a physical energy surface and relax via safe, monotone descent.

Key Achievements (see `docs/CHECKLIST.md`):
- Analytic gradients available; vectorized quadratic/hinge coupling gradients; cached adjacency with neighbor‑only gradient evaluation; coordinate descent and proximal/ADMM modes.
- Line search (Armijo) for monotone acceptance.
- Iso‑energy orthogonal noise (IEON) and metric‑aware projection.
- Observability: RelaxationTracker, EnergyBudgetTracker, GatingMetricsLogger, contraction margins, KPI logging.

Practical Operations (fast, safe, auditable)
- Speed toggles to use by default:
  - `use_analytic=True`, `use_vectorized_quadratic=True`, `neighbor_gradients_only=True`, `line_search=True`, `normalize_grads=True`, optionally `max_grad_norm` and tuned `step_size`.
- Monotone descent: Armijo backtracking (`line_search=True`) guarantees ΔF ≤ 0 for accepted steps.
- Proximal/operator‑splitting backbones: quadratics, directed/asymmetric hinge, gate‑benefit, damped gate‑benefit, with star‑block prox and ADMM available.
- Observability (“P3”): Emit and log per‑term energy stacks, total F and ΔF trajectories, acceptance reasons, contraction margins, weight adapter activity, and compute/time KPIs.
- Dashboard: `tools/p3_dashboard.py` (Streamlit). See `docs/DASHBOARD_P3.md` for usage and panels.

Quick Windows sanity example (PowerShell; illustrative)
```powershell
uv run python - << 'PY'
from core.coordinator import EnergyCoordinator
# Build modules, couplings, and constraints here...
coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints=constraints,
    use_analytic=True,
    use_vectorized_quadratic=True,
    neighbor_gradients_only=True,
    line_search=True,
    normalize_grads=True,
    max_grad_norm=1.0,
    step_size=0.07,
)
etas0 = coord.compute_etas(inputs)
etas = coord.relax_etas(etas0, steps=40)
print("etas:", etas)
PY
```

Resilience (Phase‑1 carry‑over P5; implementation patterns)
- Disaster‑hardened operation:
  - Checkpoint/rollback ring buffer (periodic, validated states).
  - Catastrophic failure detection: NaN/Inf, monotone increases, energy explosions, deadlocks (SCC detection).
  - Circuit breaker with exponential backoff; half‑open recovery.
  - Emergency procedures: circular coupling break (Tarjan→prune weakest), module pruning on memory pressure.
- Purpose: Zero crash under catastrophic failures while preserving auditability and monotone guarantees.
- Tests: Inject NaNs/divergence/deadlock scenarios; verify rollbacks and recoveries.

Benchmarking and speed profiling
- ΔF90 harness to compare configurations and wall‑clock:
```powershell
uv run python -m experiments.benchmark_delta_f90 --configs default analytic vect coord adaptive --steps 60 --run_id dev_run
```
- Micro‑bench and end‑to‑end timers with `Measure-Command`.

---

  ### Phase 2: The Precision Layer (Current Status: Delivered; see CHECKLIST)

Goal: Treat Certainty (Precision/Stiffness) and Uncertainty (Slack) as first‑class citizens to stabilize and accelerate search, and to guide expansion.

Principles
- Uncertainty is a resource. Use it to absorb stress and explore flat valleys.
- Maximum Entropy: Prefer the flattest solution that satisfies constraints (robustness).
- Noise as Stabilizer: Orthogonal noise smooths the effective landscape; prevents “ringing” and tracks deep valleys rather than shallow potholes.

Implementation Plan

1) Interface: `SupportsPrecision` (stiffness/curvature)
- All modules exposing curvature enable precision‑aware steps and noise.
- See `docs/CHECKLIST.md` (Phase‑2 foundation present: protocol and `_precision_cache`, tests).

2) Data: Coordinator `_precision_cache`
- Cache diagonal curvature approximations per module; optional correlation strengths per edge (kept simple—do not invert full matrices).

3) Steps: Newton‑aware diagonal preconditioning
- Replace raw gradient g with preconditioned g_i ← g_i / (ε + curvature_i).
- Keep `metric_vector_product` strictly for manifold‑aware orthogonality in noise projection, not for precision inversion.

4) The Bridge: Precision‑aware noise (required)
- PrecisionNoiseController (implemented; see CHECKLIST for status)
  - Inputs: gradients, per‑coordinate (or per‑module) curvature from `_precision_cache`.
  - Logic: High curvature (stiff/certain) → low noise; low curvature (uncertain) → high noise.
  - Output: Per‑dimension noise magnitudes.
  - Optional: Project noise M‑orthogonally when a problem metric is provided; default to gradient‑null iso‑energy orthogonality when no M is provided.

5) Free‑energy guard and early‑stop (stability)
- Track F = U − T·S; accept steps with sufficient ΔF decrease; combine with early‑stop + patience using stability and cost criteria.
- Switch to smoothing if needed (AGM kernel consensus with trust‑region clipping).

6) Entropy‑regularized gating with uncertainty
- Sensitivity probes (noise‑as‑probe schedule) → dispersion measure.
- Confidence fusion: c = sigmoid(a·(ρ_max − residual) − b·sensitivity).
- Use c to:
  - Gate structural expansions (only expand if uncertainty is too high to be useful or yields large ΔE benefit).
  - Trigger early‑stop when stabilized.

7) Observability (precision‑first dashboards)
- Visualize stiffness/precision across modules and steps; stream:
  - ΔF and acceptance reasons,
  - sensitivity trajectories, confidence,
  - sharpness proxies (e.g., Hutchinson trace),
  - escape events (noise-triggered basin transitions),
  - gradient norms and normalization activity.
- Dashboard reference: `docs/DASHBOARD_P3.md`

8) Validation (adopt battery below)
- ΔF histograms negative‑skewed; sensitivity reductions across schedule; calibrated acceptance distributions; stable confidence ledger.

Windows PowerShell validator examples
```powershell
# Run precision tests (example names; see CHECKLIST for canonical test files)
uv run -m pytest tests\test_precision_core.py -v
uv run -m pytest tests\test_orthogonal_noise.py -v
uv run -m pytest tests\test_precision_noise_controller.py -v
uv run -m pytest tests\test_precision_preconditioning.py -v
uv run -m pytest tests\test_metric_aware_noise.py -v
uv run -m pytest tests\test_free_energy_guard.py -v
uv run -m pytest tests\test_early_stop_patience.py -v
uv run -m pytest tests\test_sensitivity_probes.py -v
```

Immediate Next Steps (Phase‑2 Complete)
- [x] Observability P3
- [x] Free‑Energy Guard
- [x] Precision Layer Adoption
- [x] Metric‑Aware Noise Integration
- [x] Hierarchical Inference Scaffold (Foundation to be revisited after completion of phase 4)
  - `core/hierarchy.py`: FamilyGrouping, coarse energy, selection helpers
  - `docs/HIERARCHICAL_INFERENCE.md`: Conceptual guide
- [x] Validation Wiring
- [x] Checklist Alignment

Pending (Beyond Phase‑2)
- [ ] Meta‑learning outer loop (Layer 4): RL parameter search (`EnergyLandscapeSearchEnv`) — not implemented yet; future phase
- [ ] Advanced information‑structure metrics integration: redundancy ρ, alignment a, constraint violation rate h — not yet wired into coordinator; planned alongside hierarchical inference (h is the span-level constraint violation metric defined in the Information Structures research repo...
- **“Constraint violation rate (h)”**: This metric originates from the Information Structures research, where h measures unsupported or contradicted factual spans, using provenance (document IDs, offsets, or KG triples) to audit each claim. We adopt the same idea here but use the neutral term “constraint violation rate” in code and logs so downstream tasks can map it to whatever error notion applies (LLM hallucination, schema breach, policy violation, etc.). Provenance components:
  - **Document IDs**: unique identifiers for each evidence source (e.g., `trial_2021_phase3.pdf`).
  - **Offsets**: byte/character/token ranges pointing to the exact supporting span (e.g., `2315–2420`).
  - **KG Triples**: subject–predicate–object facts `(DrugX, reduces, HbA1c)` with optional provenance metadata.).

---

### Phase 3: The Symplectic Integrator (Future Horizon)

Goal: Long‑term geometric stability (volume‑preserving dynamics).

Shift
- From dissipative solvers (Euler/Runge‑Kutta) to symplectic (Verlet/Leapfrog).
- Rationale: Conserve phase‑space volume; prevent drift over long horizons; enable infinite‑horizon active inference.

Bridging guardrails (before/while switching)
- Keep ΔF guard, monotone acceptance, and trust‑region clipping to avoid oscillations during integrator transitions.
- Prox/ADMM modes reduce hinge/gate oscillations—usefully complementary to symplectic steps.
- Keep precision‑aware preconditioning and orthogonal noise to stabilize stiff directions.

---

### Phase 4: Thermodynamic Geometry & Zipfian Adaptation

Goal: Information structure adapts to noise; the system “breathes.”

Shouting Principle
- High noise/low bandwidth → “shout” with redundancy (expand structure).
- Low noise/high bandwidth → “whisper” with compression (merging, removal of redundant constraints).

Two concrete mechanisms

1) Progressive Compute (exact, lossless)
- Interleaved Matrix Compression (IMC): banded decomposition with calibrated uncertainty thresholds.
- Inference unfolds band‑2/3 only when uncertainty warrants; preserves exactness (bands algebraically sum to full model).
- Analogy: Adaptive Mesh Refinement (AMR) — refine where error is high; exactness in the limit.

2) Constraint‑Conformance via Redundancy (ρ)
- Estimate semantic redundancy threshold ρ; require minimum redundancy for tasks; monitor constraint violation rates h (defined as span-level violations with provenance in the Information Structures research, it's a complex metric so please study it) and semantic consistency.
- Use CoT as parity checks; support verification or abstention under uncertainty.
- Coding‑theory analogy: redundancy ↔ error correction; redundancy is a design knob to target error rates under channel noise.

---

## III. Theoretical Unification

A single mathematical truth through four lenses:

| Lens | Concept | Implementation |
| :--- | :--- | :--- |
| Physics | Energy Minimization | EnergyCoordinator, Relaxation |
| Control | H∞ Robustness | SmallGainWeightAdapter, HMPO (where applicable) |
| Stats | Belief Propagation | Redemption Couplings, Injection Logic |
| Info Theory | Channel Capacity / Zipf | Dynamic Topology, Compression/Expansion Gating |

The Golden Spike (operational)
- Inject noise orthogonal to the gradient (physics) and scale by inverse precision (stats) to achieve robust exploration (control).
- With a problem metric M, project noise M‑orthogonally; otherwise, use gradient‑null orthogonality.
- Validate with sharpness proxies and escape events; show improved convergence and robustness at matched loss.

---

## IV. Technical Specification (For Developers & Agents)

Objective: Implement and operate Phase‑2 (Precision Layer) safely and measurably; keep solvers fast/stable and auditable.

1) Precision Interface
```python
class SupportsPrecision(Protocol):
    def curvature(self, eta: float) -> float:
        """Local stiffness (2nd derivative) at eta."""
        ...
```

2) Precision Cache
- Coordinator maintains `_precision_cache: Dict[ModuleID, float]` for diagonal curvatures; optional couplings’ correlation strengths kept simple.

3) Newton‑aware Step (Diagonal Preconditioning)
- Update: η ← η − α · (g / (ε + curvature))
- Do not invert full matrices; avoid conflating curvature with metric projections.
- Reference: `docs/PRECISION_LAYER.md` (usage, flags, validation)

4) PrecisionNoiseController (Bridge)
- Input: grads, curvatures.
- Logic: noise_magnitude_i ∝ 1 / (ε + curvature_i).
- Optional: project noise M‑orthogonally when a metric is supplied.
- Output: noise magnitudes vector to injection logic.
- Observability: log noise scales, orthogonality checks, and escape events.
- References: `docs/PRECISION_LAYER.md`, `docs/ISO-ENERGY_ORTHOGONAL_NOISE.md`, `docs/METRIC_AWARE_NOISE_CONTROLLER.md`

5) Noise Scheduling & Probes
- Use controlled schedules (cyclical/cosine/inverse anneal) for inputs or parameter noise; couple to sensitivity probes to quantify brittleness.

6) Free‑Energy Guard & Early‑Stop
- Track F = U − T·S; accept step if ΔF < −ε; early‑stop based on multi‑criteria stabilization and patience.
- Smoothing fallback: AGM kernel consensus with trust‑region clipping; schedule α by cosine decay; gate by TD variance and disagreement.

7) Prox/Operator‑Splitting Backbone
- Model F(η)=Σ_i f_i(η_i)+Σ_(i,j) g_ij(η_i,η_j) with proximable families; run block‑coordinate prox or ADMM on the factor graph.
- Keep η∈(0,1) via mirror/logit reparameterization; enforce conservative Lipschitz (Gershgorin) step caps; log contraction margins.

8) Observability: What to log (P3)
- Per‑term energies (data, priors, couplings), ΔF, acceptance/rejection reasons, backtrack counts.
- Sharpness proxies, sensitivity (probes), confidence c, escape events, gradient norms, normalization/clipping activity.
- Contraction margins, Lipschitz estimates, redemption‑gain, compute/time KPIs.

9) Validation Battery (adopt per release)
- Conflict/Resilience: staged batch rejections, rollbacks under injected failures, grace‑period behavior.
- Precision/Noise: sharpness reductions at matched loss; escape event counts; sensitivity decreases after schedule.
- Free energy: ΔF histograms negative‑skewed; early‑stop triggers only after stabilization; trust‑region invariants respected.
- Diversification: function‑space distances geometry stable; subset selection beats single best.
- Progressive compute: lossless split/merge parity; calibrated expansion rates near targets.

Windows PowerShell helpers
```powershell
# Setup env, run tests verbosely
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv sync --extra dev
uv run -m pytest tests -v --tb=short

# Selected tests (examples; see CHECKLIST for canonical file names)
uv run -m pytest tests\test_precision_core.py -v
uv run -m pytest tests\test_orthogonal_noise.py -v
uv run -m pytest tests\test_relaxation_energy_monotonic.py -v
```

---

## V. Hierarchical Inference (Coarse → Fine; Multi‑Scale Scaffold)

Motivation
- Speed: avoid combinatorial explosion in module selection and reduce total solve time.
- Structure: higher‑level modules set constraints/policies for lower‑level modules; enables “V‑trace” style redemption.

Two‑level workflow
1) Coarse Level (fast): group modules into families (functional groups). Optimize family‑level activations (η_family) on small probe sets with a cheap coarse energy that includes family masses, consistency couplings, and complexity penalty.
2) Selection: choose modules from active families (diversity preferred—function‑space distances).
3) Fine Level (precise): run the standard coordinator on the selected modules with full priors and couplings.

Auditability
- Persist coarse decisions: family masses, coupling terms, selected modules, and function‑space distances.
- Keep a stable, versioned probe set for comparability across runs.

---

## VI. Progressive Compute & Zipfian Breathing

Exactness with elasticity
- Lossless bands (IMC) unfold compute only when uncertainty warrants; algebraic equivalence to the full model is preserved.
- Calibrate uncertainty thresholds; prefer identical kernel/precision settings for calibration and evaluation to avoid drift.

AMR analogy
- Progressive bands ≈ grid refinement; refine where error is high; preserve exactness.

---

## VII. Constraint‑Conformance & Control‑Theoretic Prompting

Noisy channel view
- Redundancy reduces error; CoT is parity for semantic inconsistencies.
- Pipeline: measure redundancy ρ; monitor drift/entropy; apply CoT parity and verification/abstention; control prompt length to avoid context fatigue.

Design knobs
- Target redundancy (CoT depth/width) vs latency cost; calibrate to acceptable error rate.

---

## VIII. Execution Timeline & Deliverables (guidance; see CHECKLIST for status)

Phase 1 carry‑over (finish core P0/P1/P3/P5)
- P3 Observability & Visualization: event streams, dashboards for energy descent, margins, ΔF, precision/stiffness panes.
- P5 Resilience: CheckpointManager, FailureDetector, CircuitBreaker; recovery path; injected‑failure tests (NaN/divergence/deadlock).
- P0 Discrete↔continuous bridge: Snap‑to‑Grid schedule + hysteresis + reheat; optional projections; straight‑through option; snap metrics.
- P0 Polynomial enhancements: η→ξ mapping; RHS probe; basis‑decorrelation monitor (aPC‑style).
- P1 Stability allocator telemetry: margin/budget exposure and plots.

Phase 2 — Adoption & Impact
- Hello World demo: end‑to‑end solve with KPIs/trace.
- Visual dashboard: Streamlit panes for ΔF/precision/sensitivity/confidence and budgets.
- Bake‑off: standard solvers baselines; report.
- Packaging: PyPI (0.1.x) post‑demo; DX polish (decision tree, contributor guide, API docs).

---

## IX. Validation Gate (must‑pass before shipping)

- ΔF monotone acceptance under Armijo (per accepted step).
- Precision/Noise: sharpness down at matched loss; escape events tracked; sensitivity decreases across schedule.
- Free‑energy: ΔF histograms negative; early‑stop with patience; trust‑region invariants respected during smoothing.
- Hierarchical: coarse→fine within 10% of brute force on small suites; 100× speedup vs exhaustive selection.
- Progressive compute: lossless band split/merge parity; calibrated expansion rates.
- Resilience: rollbacks under injected failures; circuit breaker opens/recovers as configured; no crashes.

---

## X. Notes for Development & Operations

- Use Windows PowerShell in all examples; prefer `uv` for environment and test runs.
- Treat `docs/CHECKLIST.md` as the single source of truth for capability status and test coverage.
- Keep observability on by default: only accepted steps emit energy events; record acceptance reasons, backtracks, and contraction margins.
- Prefer analytic/vectorized paths, neighbor‑only gradients, and pre‑allocated buffers for performance.

---

## Citation

If you use this repository in your research, please cite it as below.

Authors:
- Oscar Goldman - Shogu research Group @ Datamutant.ai subsidiary of 温心重工業

---

“We are building a machine that doesn't just calculate the answer, but feels the tension of the problem until it settles into the truth.”
