# System Capability Checklist ✅

Quick reference of what the current build already ships. Each entry links back
to the corresponding documentation or subsystem so you can confirm provenance at a glance.

## How to validate (Windows PowerShell, uv)

```powershell
# Activate or create env (first time)
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv sync --extra dev

# Run full test suite (verbose, short tracebacks)
uv run -m pytest tests -v --tb=short

# Single file / single test (examples)
uv run -m pytest tests\test_precision_core.py -v
uv run -m pytest tests\test_precision_core.py::test_precision_cache_updates_from_modules -q
```

---

## Core Algorithms & Stability
- ✅ ADMM / proximal mode for **all coupling families** (quadratic, directed/asymmetric hinge, gate-benefit, damped gate-benefit) — see `docs/PROXIMAL_METHODS.md`
- ✅ Orthonormal polynomial reparameterization (Legendre + aPC) with conditioning benchmarks — see `docs/POLYNOMIAL_BASES.md`
- ✅ **SmallGain** stability-margin allocator with production validation (`docs/SMALLGAIN_VALIDATION_FINAL.md`)
- ✅ Stability guard + margin warnings (`warn_on_margin_shrink`, `margin_warn_threshold`)
- ✅ Homotopy scaling with oscillation guard/backoff logic
- ✅ Gershgorin Lipschitz estimator (row-sum bound) + optional coupling auto-cap
- ✅ Energy monotonicity assertion with automatic guards (noise/line-search/adapters/homotopy) — see `docs/ENERGY_CONSERVATION_AND_MONOTONICITY.md`

## Meta-Learning & Adapters
- ✅ GradNormWeightAdapter
- ✅ AGMPhaseWeightAdapter
- ✅ GSPOTokenWeightAdapter (sequence/token GSPO trainer)
- ✅ SmallGainWeightAdapter (stability-aware allocator)
- ✅ StructuredTextLLMAdapter + warm-start proposer (System-2 reasoning bridge)

## Phase 2: Precision & Stability
- ✅ `SupportsPrecision` protocol for modules (stiffness/curvature)
- ✅ `_precision_cache` wired into `EnergyCoordinator`
- ✅ Regression test `tests/test_precision_core.py`
- ✅ `PrecisionNoiseController` (inverse-curvature redistribution)
- ✅ Diagonal preconditioning (use_precision_preconditioning + precision_epsilon)
- ✅ Free-Energy Guard (F = U - T·S acceptance) with `use_free_energy_guard`
- ✅ Early-Stop with Patience (`enable_early_stop`, `early_stop_patience`)
- ✅ Sensitivity Probes (`enable_sensitivity_probes`) & Dispersion metrics
- ✅ Observability P3:
  - ✅ Precision/stiffness logging (min/max/mean, per-η)
  - ✅ Free Energy decomposition (U, S, F) logging
  - ✅ Sensitivity dispersion logging (`sensitivity:dispersion`)
  - ✅ Acceptance reasons & backtracks logging
  - ✅ Dashboard panels for all the above (`tools/p3_dashboard.py`)
- ✅ Documentation: `docs/PHASE2_PRECISION_LAYER.md`
- ✅ Documentation: `docs/CONSTRAINT_PROGRAMMING.md` (The Paradigm)
- ✅ Documentation: `docs/README_CONSTRAINT_DICTIONARY.md` (Runtime Usage)

## Optimization Modes
- ✅ Line search (Armijo backtracking) with trial projection and backtrack counters
- ✅ Coordinate descent (active coordinate updates, neighbor-aware deltas)
- ✅ Operator-splitting / proximal mode (locals + incident couplings)
- ✅ Proximal “star” blocks (`prox_block_mode="star"`)
- ✅ Mirror/logit η updates (bounded updates in ζ-space)
- ✅ ADMM: quadratic, hinge (directed/asymmetric), gate-benefit (prox-linear on gate with damping)

## Noise & Exploration
- ✅ Iso-energy orthogonal noise (IEON) with analytic projection to gradient null space
- ✅ Metric-aware noise projection (supports custom metric matrix or vector product)
- ✅ OrthogonalNoiseController (adaptive noise schedule) — enable with `auto_noise_controller=True`
- ✅ Uncertainty-gated gate costs (AGM-driven relax/tighten)
- ✅ PrecisionNoiseController (inverse-precision noise redistribution with orthogonal re-projection) — tests present

## Observability & Instrumentation
- ✅ RelaxationTracker (per-step ΔF/η traces, optional per-η logging)
- ✅ EnergyBudgetTracker (per-term energy & gradient budgets, contraction margins, compute cost)
- ✅ Contraction margin telemetry + history (`last_relaxation_metrics`)
- ✅ Lipschitz detail exposure for allocator telemetry (`expose_lipschitz_details=True`)
- ✅ KPI logging: `compute_cost`, `redemption_gain`
- ✅ GatingMetricsLogger (hazard/η/redemption CSVs) — tests present
- ✅ Dashboard (Streamlit `tools/p3_dashboard.py`) documented in `docs/DASHBOARD_P3.md`[needs deep tests skipped for now, maybe Zig integration, dunno yet]
  - Usage: `uv run streamlit run tools/p3_dashboard.py`
  - Tests: `tests/test_dashboard_app.py` (skips if Streamlit not installed)
- ✅ Escape events logging (noise-triggered basin transitions)
- ✅ Confidence trajectory logging (`confidence:c`)
- ✅ ΔF/ΔE histogram validator (`tools/validate_delta_f.py`) and tests (`tests/test_delta_f_histograms.py`)
- ✅ Information-structure metrics logging (alignment `info:alignment`, drift `info:drift`, constraint violation rate h via `info:constraint_violation_rate`) — see `docs/INFORMATION_METRICS.md`

## Inference & Scale (Warm-start / P2)
- ✅ `run_warm_start_relaxation(...)` pipeline with contraction-margin metrics
- ✅ `MLPWarmStartProposer` + serialization
- ✅ Cached active-set amortizer with similarity cache + stage planning
- ✅ Compile-time vectorization cache for coupling families

## Scheduling & Curriculum
- ✅ Curriculum scheduler scaffolding (`core/curriculum.py`) with tests (`tests/test_curriculum_scheduler.py`)

## Backends
- ✅ Torch backend integration (`core/torch_backend.py`) with tests (`tests/test_torch_backend.py`)
- ✅ JAX backend integration (`core/jax_backend.py`) with tests (`tests/test_jax_backend.py`)

## Modules Library
- ✅ Gating: `modules/gating/energy_gating.py` (+ straight-through option) — tests present
- ✅ Sequence: `modules/sequence/monotonic_eta.py` — tests present
- ✅ Connectivity: `modules/connectivity/nl_threshold_shift.py` — tests present
- ✅ Game/Nash: `modules/game/emergent_nash.py` — tests present
- ✅ Polynomial Energy: `modules/polynomial/polynomial_energy.py` (Legendre/aPC) — tests present
- ✅ aPC utilities: `modules/polynomial/apc.py` — tests present
- ✅ RHS probe (diagnostic): `modules/polynomial/rhs_probe.py` — tests present
- ✅ Compression: `modules/compression/compression_energy.py` — tests present
- ✅ Optional attention demo: `models/nonlocal_attention.py` (opt-in)

## Entity & Adapters Infrastructure
- ✅ Lightweight entity wrapper: `core/entity_adapter.py` (ECS-style identity/version)
- ✅ LLM adapter boundary: `core/llm_adapter.py` with `StructuredTextLLMAdapter`
- ✅ Amortizer interfaces & cache: `core/amortizer.py` (+ tests)

## Documentation Highlights
- ✅ `docs/PHASE1_COMPLETION_SUMMARY.md` (snapshot of completed Phase 0/1 work)
- ✅ `docs/README_WORMHOLE.md` (Non-local gradient teleportation / Wormhole Effect)
- ✅ `docs/README_SMALLGAIN.md`
- ✅ `docs/README_MODULES.md` (module quick reference + precision-aware note)
- ✅ `Complexity_from_Constraints.md` (philosophy + five equations)
- ✅ `docs/ROADMAP_NEUROSYMBOLIC_HOMEOSTAT.md` (living roadmap moving forward)
- ✅ Noise docs: `docs/ISO-ENERGY_ORTHOGONAL_NOISE.md`, `docs/METRIC_AWARE_NOISE_CONTROLLER.md`, `docs/README_WHEN_TO_USE_NOISE.md`
- ✅ Stability docs: `docs/STABILITY_GUARANTEES.md`, `docs/MONOTONIC_ENERGY_SUMMARY.md`
- ✅ Prox/ADMM docs: `docs/PROXIMAL_METHODS.md`
- ✅ Meta-learning docs: `docs/META_LEARNING.md`, `docs/SMALLGAIN_VALIDATION_FINAL.md`

## Test Suite
- ✅ 130+ pytest cases (per Phase 1 summary; backend-specific skip where noted)
- ✅ Dedicated conditioning tests (`tests/test_polynomial_conditioning.py`)
- ✅ Stability warning tests (`tests/test_stability_margin_warnings.py`)
- ✅ ADMM parity suites (`tests/test_admm_*.py`)
- ✅ Warm-start / amortizer suites (`tests/test_warm_start_proposer.py`, `tests/test_cached_amortizer.py`, `tests/test_amortized_inference_validation.py`)
- ✅ Noise controller coverage (`tests/test_orthogonal_noise.py`)
- ✅ Metric-aware noise projection tests (`tests/test_metric_aware_noise.py`)
- ✅ Precision-aware noise controller tests (`tests/test_precision_noise_controller.py`)
- ✅ Precision preconditioning tests (`tests/test_precision_preconditioning.py`)
- ✅ Backends: `tests/test_torch_backend.py`, `tests/test_jax_backend.py`
- ✅ Vectorization cache: `tests/test_vectorization_cache.py`
- ✅ Homotopy helpers & logging: `tests/test_homotopy_schedule.py`, `tests/test_homotopy_logging.py`, `tests/test_homotopy_gate_and_term_scales.py`
- ✅ Term weights & adapters: `tests/test_term_weights.py`, `tests/test_weight_adapter.py`, `tests/test_weight_adapter_hook.py`, `tests/test_gspo_weight_adapter.py`, `tests/test_small_gain_weight_adapter.py`
- ✅ Coordinator invariants & monotonic energy: `tests/test_coordinator_invariants.py`, `tests/test_relaxation_energy_monotonic.py`, `tests/test_monotonic_energy.py`
- ✅ Coordinate descent & gradients: `tests/test_coordinate_descent_incremental.py`, `tests/test_core_coordinator_gradients.py`
- ✅ Couplings & gate effects: `tests/test_coupling_gradients.py`, `tests/test_gate_coupling.py`, `tests/test_damped_gate_benefit_coupling.py`, `tests/test_gating_module.py`, `tests/test_soft_gating_effect.py`
- ✅ Escape events logging tests (`tests/test_escape_events.py`)
- ✅ Confidence logging tests (`tests/test_confidence_logging.py`)
- ✅ ΔF histogram skewness tests (`tests/test_delta_f_histograms.py`)
- ✅ Information metrics unit tests (`tests/test_info_metrics.py`) and logging tests (`tests/test_info_metrics_logging.py`)

## Meta-Learning (Outer Loop)
- ✅ RL environment scaffold for parameter search (`core/meta_env.py`)
- ✅ Smoke test for environment (`tests/test_meta_env.py`)

---

For any new feature or doc, add a matching bullet here with ❌ until it is implemented and verified; once complete, flip it to ✅ and reference the validating doc/test. This keeps every live document aligned with an authoritative checklist.***

