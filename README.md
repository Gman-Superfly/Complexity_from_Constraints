# Complexity from Constraints

**📖 MUST READ**: [Complexity_from_Constraints.md](Complexity_from_Constraints.md) — Philosophy, motivation, and the five equations.


## Code in this repo
Small, composable modules coordinated by a global free-energy objective, with sparse non-local couplings that provide "future-like" corrections. Each module exposes an order parameter (η) and a local energy F(η; c). Composition = Σ F_local + Σ F_couple. The system seeks low-energy, coherent behavior without hard-coding global rules.

We keep the design tight and exact. This is an MVP intended for learning and iteration.

contains ideas and code from other Gman-Superfly repos and Abstractions by Furlat

"When to use this framework" guide: "If you have N independent modules and want global coherence without backprop through all of them, use this. If you need differentiable end-to-end training, use standard deep learning, check the "Notes" section at the end of this document.

##CODE
## Why this exists (short)
- **Non-locality**: Distant parts can influence each other to redeem provisional mistakes.
- **Free-energy lens**: A single scalar objective coordinates tiny modules without making them big.
- **Composability**: Add/remove modules and couplings without rewriting the system.

## Conceptual overview (why η, why F)
- Each module emits an order parameter `η ∈ [0, 1]` plus a Landau-style local energy `F_local(η; a, b, h)` that measures how “coherent” that module feels.
- Sparse couplings (quadratic, hinge, gate–benefit) capture non-local redemption: future context can lower past energy if it truly helps (`Δη_domain > 0`).
- The coordinator sums `Σ F_local + Σ F_couple = 𝓕` and relaxes all `η` simultaneously (analytic gradients + damping/backtracking) so total energy never increases.
- Observability is explicit: `RelaxationTracker` logs `η` bounds and ΔF; gating metrics (`hazard_mean`, `μ̂`, `good_bad_ratio`) show whether expansions stay rare but impactful.
- Read the philosophy + five equations here: **[Complexity_from_Constraints.md](Complexity_from_Constraints.md)** and see module/experiment guides in `docs/`.
- Gradient discipline: coordinator can auto-rescale term weights when a family’s gradient norm explodes (opt-in `auto_balance_term_weights`). This avoids “energy wars” while emitting warnings so you know it happened. Longer term we’ll replace this heuristic with a principled balancing method, but it keeps runs sane today.

  Meta-training note: `WeightAdapter` hooks let an outer loop learn term weights instead of fixing them. Use this when you want GradNorm-style balancing or integration with other trainers.

  Tunable knobs:
  - `term_norm_target`: desired per-term gradient norm; smaller → stricter balancing.
  - `max_term_norm_ratio`: tolerance above the target before scaling kicks in.
  - `term_weight_floor` / `term_weight_ceiling`: clamp how far auto-scaling can push weights (prevents runaway boosts).
  - For critical terms, override `constraints["term_weights"]` or supply a custom `WeightAdapter` to run your own strategy.

## The "Wormhole Effect" (Gradient Teleportation)

(needs to be tested at scale!, for now it's cute.. motivation: I wanted the Flow to be intelligent and the components to be dumb.)

Why does this system solve problems that standard sparse networks get stuck on? It uses a mechanism we call **Non-Local Gradient Teleportation** (or the "Wormhole Effect").

- **Standard Physics**: If a gate is closed ($\eta=0$), the connection is broken. No force can pass through, so the system can't "feel" that opening the gate would be good. It gets stuck in a local minimum.
- **This Framework**: The `GateBenefitCoupling` applies a gradient force to a closed gate proportional to the **potential** energy saving of opening it (`contrib = -weights * delta`).
- **The Result**: The future "reaches back" and pulls a completely inactive module into existence if the *predicted* benefit is high enough. This is **causal retro-propagation without an active channel**.

It solves the "Zero-Gradient Problem": *How do you learn to open a door if you never walk through it?* 
**Answer**: You let the value of the room behind it pull the handle.

### WeightAdapter hook (meta-training)

`EnergyCoordinator` exposes an optional `WeightAdapter` protocol so external trainers can tune per-term weights (GradNorm-style). 
(fun stuff to be added here)
```python
# Small-Gain stability-margin allocator ✅ PRODUCTION READY
from core.weight_adapters import SmallGainWeightAdapter
coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={},
    weight_adapter=SmallGainWeightAdapter(
        budget_fraction=0.7,   # spend ≤ 70% of available margin (validated optimal)
        max_step_change=0.10,  # per-step clamp (validated: 0.10 for energy, 0.20 for speed)
        floor=0.1,             # hard lower bound for weights
        ceiling=3.0,           # hard upper bound for weights
        ema_alpha=0.3,         # smooth value/cost ratios
    ),
    stability_guard=True,      # required for margin tracking
)
```

```python
class MyAdapter:
    def step(self, term_grad_norms, energy, current):
        updated = dict(current)
        for key, norm in term_grad_norms.items():
            updated[key] = max(0.1, min(2.0, norm / (energy + 1e-6)))
        return updated

coord = EnergyCoordinator(..., weight_adapter=MyAdapter())
```
Term keys follow the pattern `local:ClassName` / `coup:ClassName`.

Built-in option: `GradNormWeightAdapter`
```python
from core.weight_adapters import GradNormWeightAdapter

coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={"term_weights": {"local:EnergyGatingModule": 0.8}},
    auto_balance_term_weights=False,  # optional heuristic stays available
    weight_adapter=GradNormWeightAdapter(
        target_norm=1.0,
        alpha=1.2,
        update_rate=0.15,
        floor=0.2,
        ceiling=3.0,
    ),
)
```
`GradNormWeightAdapter` lives in `core/weight_adapters.py` and keeps local/coupling gradients at comparable magnitudes by increasing weights for under-powered terms and reducing ones that dominate.

GSPO-token option (global stability + local advantages):
```python
from core.weight_adapters import GSPOTokenWeightAdapter

coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={"term_weights": {"local:EnergyGatingModule": 0.8}},
    weight_adapter=GSPOTokenWeightAdapter(
        target_norm=1.0,
        num_buckets=12,
        group_size=4,          # capacity/variance knob (sequence-level sampling)
        batch_size=2,          # outer-loop batch for prompts
        hidden_size=64,        # 2-10 terms; raise to 128-256 for 10-20+ terms
        update_every_n_steps=4,# throttle RL updates in long runs (decode-only when skipped)
        ema_reference_alpha=0.99, # EMA sync reference toward policy each update
        use_token_level=True,
        enable_throttling=True,    # set False to train every step regardless of update_every_n_steps
        logging_callback=lambda m: print("gspo metrics:", m), # optional dashboard hook
    ),
)
```
`GSPOTokenWeightAdapter` wraps `core/gspo_token_vectorized.py` so the coordinator can learn term weights via the GSPO-token objective described in Zheng et al. (2025). Gradient ratios become prompts, candidate weight sequences become responses, and the reward is the redemption-style improvement toward balanced norms. Requires the `torch` extra.

**Performance note**: `GSPOTokenWeightAdapter` runs a mini RL training step per coordinator relaxation (5-20x slower than `GradNormWeightAdapter`). Intended for outer-loop meta-training (30-100 steps) or weight search sweeps, not tight inner loops with 1000s of steps. The default `hidden_size=64` GRU supports 2-10 term families; for 10-20+ terms, increase to 128-256 or see P4 in `docs/fixes_and__related_todos.md` for scaling plans.

Phase-adaptive option: `AGMPhaseWeightAdapter`
```python
from core.weight_adapters import AGMPhaseWeightAdapter
coord = EnergyCoordinator(..., weight_adapter=AGMPhaseWeightAdapter())
# Policy: in stable/improving regimes → gently boost couplings, soften gate local energy;
# in unstable/slow regimes → gently tame couplings, strengthen gate local energy.
```

**What “meta-training” means here**
- The inner loop (coordinator) relaxes η to lower the current total energy.
- The outer loop (your adapter or another trainer) observes per-term gradient norms and total energy, then updates weights to reflect higher-level goals (e.g., GradNorm balancing, curriculum scheduling, downstream accuracy).
- This decouples modeling from tuning: instead of hard-coding all λ’s, you can learn them online, integrate with other optimizers, or plug into a broader training regimen (AGM trainer, reinforcement learning, etc.).
- Auto-balancing (`auto_balance_term_weights`) is a simple built-in heuristic; `WeightAdapter` lets you replace it with something principled or domain-specific when needed.
- When running multi-module experiments (e.g., `sequence_gating_hypothesis.py`, `energy_gated_expansion.py`, or bespoke coordinator flows), you can swap in any `WeightAdapter` to collect redemption metrics at scale. For example:
  - Baseline: `weight_adapter=None`
  - GradNorm: `weight_adapter=GradNormWeightAdapter(...)`
  - GSPO-token: `weight_adapter=GSPOTokenWeightAdapter(...)` to tie sparse redemption signals directly into per-term weights via GSPO-token clipping.
  - Document runs by noting the adapter used and logging ΔF plus redemption statistics (see `docs/README_EXPERIMENTS.md` for scenario guidance).

### Structure-Preserving Noise (Orthogonal Exploration)
`EnergyCoordinator` supports exploration noise that is projected onto the null space of the gradient, so it does not increase energy to first order (explores along level sets). This is enabled by default but ships with a conservative default magnitude of `0.0` to preserve determinism unless explicitly activated.

Example:
```python
coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={},
    enable_orthogonal_noise=True,  # default
    noise_magnitude=1e-2,          # activate exploration
    noise_schedule_decay=0.99       # optional exponential decay
)
```
Notes:
- Safe-by-construction: noise is orthogonal to ∇F at each step.
- Keep `noise_magnitude=0.0` for reproducible baselines; set > 0 for search/robustness runs.
- See `tests/test_orthogonal_noise.py` for math checks and integration behavior.

#### Automatic orthogonal-noise controller
Set `auto_noise_controller=True` to let the coordinator adapt the instantaneous noise magnitude based on convergence signals (gradient rotation, stalled ΔF, and recent backtracks). The controller treats `noise_magnitude` as the maximum budget and anneals it via `noise_schedule_decay`.

```python
coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={},
    enable_orthogonal_noise=True,
    auto_noise_controller=True,
    noise_magnitude=5e-2,      # max magnitude (controller scales [0, 1])
    noise_schedule_decay=0.995 # optional annealing
)
```

The controller keeps determinism when progress is healthy (noise≈0) and automatically boosts exploration when ΔF stalls, gradients rotate sharply, or repeated backtracks indicate a curved valley. Set `auto_noise_controller=False` to fall back to the static magnitude path.

#### Uncertainty-gated gate costs
Set `enable_uncertainty_gate=True` to tighten or relax gate costs automatically using AGM/uncertainty metrics computed from the accepted energy history. When convergence is smooth (high rate, low uncertainty) gate costs are relaxed to encourage exploitation; when energy stalls or uncertainty spikes they are tightened to keep expansion rare.

```python
coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={},
    enable_uncertainty_gate=True,
    gate_cost_relax_scale=0.85,   # multiplier applied in stable regimes
    gate_cost_tighten_scale=1.2,  # multiplier when rate is low / uncertainty high
    gate_cost_floor=5e-3,
    gate_cost_smoothing=0.3,
)
```

The controller works per-gate and composes with the homotopy helpers, so continuation schedules and uncertainty control can be active simultaneously.

### Energy Conservation and Monotonic Energy Assertions
`EnergyCoordinator` enforces a **monotonic energy assertion** by default (`assert_monotonic_energy=True`) during deterministic gradient descent. This aligns with the repo’s energy‑minimization goal and catches gradient bugs, numerical instability, or misconfigured couplings early. Built‑in guards auto‑skip the assertion when it doesn’t conceptually apply.

NOTE it's actually optional **monotonic energy assertion** 
(`assert_monotonic_energy= False or True`) that enforces strict energy conservation during gradient descent is on by default. This is a powerful debugging and validation tool for ethos conformity, it's not strictly for everything you may need.
it's for catching gradient bugs, numerical instability, or misconfigured couplings early. but remember this is changeable!!

Example:
```python
coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={},
    assert_monotonic_energy=True,      # Enable strict check
    monotonic_energy_tol=1e-10,        # Tolerance for numeric jitter
    noise_magnitude=0.0,               # Must be deterministic
    line_search=False                  # Line search has its own logic
)
```

**When to enable**:
- Unit tests and CI/CD (catch regressions)
- Debugging new gradient implementations
- Validating coupling configurations
- Benchmarking deterministic baselines

**Auto‑skip or disable when**:
- Using exploration noise (`noise_magnitude > 0`) — second-order effects can increase energy
- Using line search — it expects trial steps to fail
- Using adaptive methods (ADMM, operator-splitting) — transient increases are part of convergence
- Using homotopy schedules or dynamic term weights (weight adapters) — the energy function changes over time
- Production deployments — use soft monitoring instead, or leave on and rely on guards

Guards automatically skip the assertion when noise, line search, or homotopy/weight‑adaptation are active, so it is safe as a default. You can still set `assert_monotonic_energy=False` to explicitly disable.

See `docs/ENERGY_CONSERVATION_AND_MONOTONICITY.md` for detailed guidance, mathematical background, and troubleshooting.

#### Normalized Dynamics: why orthogonal (tangent-plane) noise
Short answer: In Normalized Dynamics, the update uses the unit gradient direction. The gradient defines the normal to the energy level set, so the orthogonal complement is the tangent space. Injecting noise in that tangent space is structure‑preserving: it explores along the level set and doesn’t raise energy to first order. That’s the geometric reason a normalized, direction‑only flow naturally pairs with orthogonal (tangent‑plane) noise.

#### When to turn it up (signals)
- Gradient rotation: large angle between successive gradients (curved valleys).
- Stall vs first‑order model: observed ΔF much smaller than α‖g‖ under normalized step.
- Backtracks / low contraction margin: line search repeatedly trims steps, or stability margin is tight.
- Flat‑but‑anisotropic: very small ‖g‖ but high anisotropy proxy (e.g., gradient variance).

See also: `core/coordinator.py` (enable_orthogonal_noise, noise_magnitude), `tests/test_orthogonal_noise.py`, and `docs/meta_learning_for_energy_landscapes.md` (Normalized Dynamics geometric trigger).

### Gradient + backend fast paths

- `EnergyCoordinator` now defaults to `use_analytic=True`. All shipped modules/couplings implement analytic gradients, and quadratic / hinge / gate-benefit families have vectorized paths (`use_vectorized_*`) to avoid Python loops on large graphs.
- Adaptive coordinate descent is built in: set `adaptive_coordinate_descent=True` to warm-start with coordinate updates when ΔF stalls, then fall back to gradient steps.
- Operator-splitting/prox mode: set `operator_splitting=True` to enable block‑prox updates (locals + incident couplings). Tunables: `prox_tau`, `prox_steps`, and optional `prox_block_mode="star"` to update each module together with its adjacent couplings (Jacobi-style block pass). Quadratic/hinge use closed‑form pairwise prox; gate‑benefit uses projected update.
- ADMM mode (production-ready ✅): set `use_admm=True` with `admm_rho`, `admm_steps`, `admm_step_size`. Introduces auxiliary differences per quadratic edge and alternates s/η/u updates with a monotone acceptance guard.
  - **All coupling families supported**: Quadratic, DirectedHinge, AsymmetricHinge, GateBenefitCoupling, DampedGateBenefitCoupling ✅
  - Hinge family: nonnegative auxiliary gaps on `β η_j − α η_i`.
  - Gate-benefit family: prox‑linear gate update with damping (set `admm_gate_prox=True`, `admm_gate_damping≈0.5`).
  - Tests: `test_admm_*.py` validate energy parity vs gradient descent across all coupling types.
- Stability guard (optional): set `stability_guard=True` to cap step size using a conservative Gershgorin-style Lipschitz bound. Tunables: `stability_cap_fraction` (default 0.9), `log_contraction_margin=True` to record a per‑step margin; compatible with line search.
- Coupling auto-cap: set `stability_coupling_auto_cap=True` with `stability_coupling_target` (desired Lipschitz bound). Coupling term weights are temporarily scaled so the estimated Lipschitz stays below the target; shows up in EnergyBudgetTracker logs as lower `energy:coup:*`.
- Homotopy / continuation: set `homotopy_coupling_scale_start=<0..1>` and `homotopy_steps` to scale coupling term weights from a gentle start (e.g., 0.2) up to 1.0 over the first `homotopy_steps` iterations. Use `homotopy_term_scale_starts={"coup:GateBenefitCoupling":0.3, ...}` for per-term ramps, and `homotopy_gate_cost_scale_start` to temporarily raise/lower gate costs before settling to the target configuration.
- Mirror/logit η updates (optional, gradient mode): set `use_logit_updates=True` to update in ζ=logit(η) space (bounded mirror map). This is intended for boundary stability (reduces clamp artifacts near 0/1), not speed. Tune with a smaller `step_size` than usual. Keep off by default unless you specifically see boundary issues.

### Observability helpers
- Per‑step relaxation traces: `RelaxationTracker(name, run_id).attach(coord)` then `flush()` after relaxation.
- Per‑η logging (optional): `RelaxationTracker(log_per_eta=True)` adds `eta:<idx>` columns for APC fitting/diagnostics.
- Per-step energy budget: `EnergyBudgetTracker(run_id="...").attach(coord)` logs per-term `energy:*`, `grad_norm:*`, backtracks and optional `contraction_margin` to CSV; call `flush()` after relaxation. Plot with `uv run python -m experiments.plot_energy_budget --input logs/energy_budget.csv --metric energy:local:YourModule`.
- Additional telemetry fields (when available):
  - `homotopy_scale`, `homotopy_backoffs` (continuation schedule and backoffs)
  - `poly_corr_max:poly:<idx>`, `poly_corr_warn:poly:<idx>` (basis decorrelation monitor)
  - `margin_warn` (1 when `contraction_margin` < configured threshold)
- KPI fields: trackers now emit `compute_cost` (wall-time delta between accepted steps) and `redemption_gain = max(ΔF, 0)/compute_cost`, so you can rank runs by energy saved per unit compute.
- Lipschitz details for allocator/telemetry: set `expose_lipschitz_details=True` to compute a Gershgorin-like bound with components exposed as a dict: `L_est`, `row_sums`, `row_targets`, `row_margins`, `global_margin`, `family_costs`, and `edge_costs`. If your `weight_adapter` defines attributes `edge_costs`, `row_margins`, or `global_margin`, the coordinator injects these per-step for allocator policies.
- Torch/JAX backends support the same Landau-style modules (gating, sequence, connectivity, Nash) plus quadratic/hinge/gate-benefit couplings, so you can offload relaxation with `uv run pytest tests/test_torch_backend.py` / `tests/test_jax_backend.py` when those extras are installed.
- For a complete performance playbook (ΔF90 benchmarks, profiler snippets, remaining ideas) see `docs/speed_up_tweaks.md`.

SmallGain allocator — Production Status ✅
- **Status**: PRODUCTION READY (validated on baseline + dense scenarios)
- **Performance**: 
  - Matches GradNorm on baseline (ΔF90=10) with **4x better final energy**
  - 40% faster than GradNorm on dense graphs (ΔF90=12 vs 20) with **4.4x better final energy**
  - 50-55% reduction in ΔF90 vs vanilla analytic baseline
- **Validated Defaults**: 
  - Budget fraction ρ = 0.7 (spend ≤ 70% of stability margin)
  - Per-step max Δweight = 0.10 (optimal for final energy quality)
  - Speed variant: Δweight = 0.20 (30% faster ΔF90 with 6% energy loss)
- **Full validation results**: See `docs/SMALLGAIN_VALIDATION_FINAL.md`
- **Tuning**: Use `experiments/sweeps/sweep_smallgain.py` for domain-specific optimization

### Running tests (uv environment)
- Full suite:
  - `uv run -m pytest tests -v --tb=short`
- Single file:
  - `uv run -m pytest tests\test_monotonic_energy.py -v`
- Single test:
  - `uv run -m pytest tests\test_monotonic_energy.py::test_monotonic_energy_can_be_disabled -q`

If pytest isn’t installed in the current env:
- `uv run --with pytest -m pytest tests -v`

Script alias:
- After `pyproject.toml` adds `[tool.uv.scripts] test = "pytest -v --tb=short"`, you can run:
  - `uv run -s test`

### Homotopy scheduler (simple helpers)
Use `core/homotopy.py` to linearly scale term weights from a gentle start to 1.0 over N steps.
```python
from core.homotopy import linear_scale, term_scales_from_starts
scale = linear_scale(start=0.2, total_steps=50, iter_idx=t)  # → [0.2..1.0]
per_term = term_scales_from_starts({"coup:QuadraticCoupling": 0.3}, total_steps=50, iter_idx=t)
```

### CompressionEnergyModule
Add a compression target with an optional quartic term:
```python
from modules.compression.compression_energy import CompressionEnergyModule
mod = CompressionEnergyModule(a=1.0, b=0.2, target_default=0.6)
eta = mod.compute_eta(x=0.55)  # treat x as observed compression ratio in [0,1]
E = mod.local_energy(eta, constraints={"compression_target": 0.6})
```

### Plotting scripts (observability)
Quick plots for per-step metrics from CSV logs:
- Budget vs spend and contraction margin:
  - `uv run python -m experiments.plots.plot_budget_vs_spend --input logs/energy_budget.csv --run_id demo`
- AGM and uncertainty timelines:
  - `uv run python -m experiments.plots.plot_agm_uncertainty --input logs/energy_budget.csv --run_id demo`

## Hypothesis tests to run
- **Sequence redemption vs local baseline**: Compare prefix-only scoring to non-local coupling + gating; measure ΔF and earlier-position fixes with `RelaxationTracker` + `GatingMetricsLogger`.
- **Connectivity threshold shift**: With/without gate-benefit coupling and shortcuts, show that sparse non-local edges plus gating lower the apparent percolation threshold (compare ΔF, hazard stats).
- **Coupling/gate sweeps**: Grid over coupling weights and gate costs to map when “future-like” corrections happen vs when behavior collapses to local. Plot ΔF, `hazard_mean`, `μ̂`.
- **Multi-module scaling**: Chain sequence + connectivity + gating modules to demonstrate coherent global behavior emerges without bespoke heuristics when couplings are active. Log energy/η traces for both disabled/enabled couplings.


## What’s here (MVP)
- Core
  - `core/interfaces.py`: typed protocols for modules and couplings
  - `core/energy.py`: Landau-style utilities; total energy helpers
  - `core/couplings.py`: quadratic, hinge, and gate–benefit couplings
  - `core/coordinator.py`: energy evaluation and relaxation (finite-diff + analytic fallback)
  - `core/entity_adapter.py`: lightweight entity wrapper with versioning and events
- Modules
  - `modules/sequence/monotonic_eta.py`: sublinear sequence consistency η and F_local
  - `modules/connectivity/nl_threshold_shift.py`: connectivity η on grid graphs
  - `modules/gating/energy_gating.py`: energy-gated expansion (rare but impactful decisions)
  - `models/nonlocal_attention.py`: energy-regularized attention (optional, PyTorch)
- `core/gspo_token_vectorized.py`: self-contained GSPO / GSPO-token trainer (sequence-level and token-level objectives per Zheng et al. 2025) for reinforcement-style order-parameter updates.
- Experiments
  - `experiments/landau_sweep.py`: disorder→order sweep on parameter a
  - `experiments/non_local_connectivity_threshold_shift.py`: shifted connectivity threshold via non-local shortcuts
  - `experiments/sequence_redemption.py`: non-local redemption vs prefix-only baseline
- `experiments/sequence_gating_hypothesis.py`: coordinator + gating hypothesis test (tracks ΔF/η and gating metrics)
  - `experiments/energy_reg_attn_ablation.py`: energy-regularized attention ablation (optional)
  - `experiments/energy_gated_expansion.py`: cost-vs-benefit expansion with redemption
  - `experiments/analyze_first_three.py`: quick summaries from logs
- Logging and tests
- `cf_logging/metrics_log.py`: Polars-based CSV logs
  - `tests/`: lightweight tests for core behaviors and invariants

## Install (Windows, macOS, Linux)
We recommend `uv` for environments (Python >= 3.11).

Windows PowerShell:
```
uv venv .venv
.\.venv\Scripts\Activate.ps1
# Option A: use lockfile (recommended)
uv sync                  # installs from uv.lock
# Include dev extras (tests) if desired:
uv sync --extra dev

# Option B: ad-hoc install (no lock)
uv pip install -e .      # or: uv pip install -e .[dev]
# Optional examples (matplotlib plots)
uv pip install -e .[examples]
# optional for attention ablation:
uv pip install torch
```

macOS/Linux:
```
uv venv .venv
source .venv/bin/activate
# Option A: use lockfile (recommended)
uv sync
# Include dev extras (tests) if desired:
uv sync --extra dev

# Option B: ad-hoc install (no lock)
uv pip install -e .      # or: uv pip install -e .[dev]
# Optional examples (matplotlib plots)
uv pip install -e .[examples]
# optional:
uv pip install torch
```

## Run experiments
```
uv run python -m experiments.landau_sweep
uv run python -m experiments.non_local_connectivity_threshold_shift
uv run python -m experiments.sequence_redemption
uv run python -m experiments.sequence_gating_hypothesis [--track_relaxation --log_gating_metrics]
uv run python -m experiments.energy_gated_expansion [--log_gating_metrics]
uv run python -m experiments.auto_balance_demo [--scenarios baseline gradnorm]
# optional if torch available
uv run python -m experiments.energy_reg_attn_ablation
uv run python -m experiments.emergent_nash_learning
uv run python -m experiments.branching_coexistence [--log_gating_metrics]
uv run python -m experiments.benchmark_delta_f90 --configs default analytic vect coord adaptive prox prox_star gradnorm agm --steps 60
uv run python -m experiments.benchmark_delta_f90 --configs vect smallgain prox prox_star --scenario dense --dense_size 32 --log_budget
```

Summaries:
```
uv run python -m experiments.analyze_first_three
```

## Examples
```
# quick Landau free-energy plot (open window or save PNG)
uv run python examples.landau_plot --a -0.5 --b 1.0 --save plots/landau.png
```

## Direction
- Keep modules small and typed; push global coordination into energy and sparse couplings.
- Use non-locality to enable “future-like” corrections (redemption), measured explicitly.
- Treat gating as an energy-driven decision so expansion is rare but impactful.
- Prefer Polars for metrics and logs; avoid heavy frameworks unless clearly needed.

## Documentation quick links

### Core Philosophy & Theory

- [Complexity_from_Constraints.md](Complexity_from_Constraints.md) — philosophy + five equations (must read) + "Wormhole Effect"
- [docs/PHASE1_COMPLETION_SUMMARY.md](docs/PHASE1_COMPLETION_SUMMARY.md) — Phase 1 completion status + roadmap

### Technical Deep-Dives ✅ NEW

- [docs/PROXIMAL_METHODS.md](docs/PROXIMAL_METHODS.md) — ADMM, prox operators, when to use ✅
- [docs/STABILITY_GUARANTEES.md](docs/STABILITY_GUARANTEES.md) — Lyapunov stability, SmallGain theorem, tuning ✅
- [docs/META_LEARNING.md](docs/META_LEARNING.md) — Adapter hierarchy (GradNorm, AGM, SmallGain, GSPO-token) ✅
- [docs/POLYNOMIAL_BASES.md](docs/POLYNOMIAL_BASES.md) — Legendre vs aPC, conditioning benefits
- [docs/SMALLGAIN_VALIDATION_FINAL.md](docs/SMALLGAIN_VALIDATION_FINAL.md) — SmallGain production validation ✅

### Module & Experiment Guides

- [docs/README_MODULES.md](docs/README_MODULES.md) — module quick reference (interfaces, invariants)
- [docs/README_EXPERIMENTS.md](docs/README_EXPERIMENTS.md) — experiment intent, what to log, expected signals
- [experiments/benchmark_delta_f90.py](experiments/benchmark_delta_f90.py) — ΔF90 benchmark harness
  - Presets: `prox`, `gradnorm`, `agm`, `smallgain`
  - CSV fields: `energy:*`, `grad_norm:*`, `operator_splitting`, `adapter` flags

### Code Reference

- [PYDANTIC_V2_VALIDATION_GUIDE.md](PYDANTIC_V2_VALIDATION_GUIDE.md) — entity construction/validation patterns
- [cf_logging/observability.py](cf_logging/observability.py) — `RelaxationTracker`, `GatingMetricsLogger`, `EnergyBudgetTracker`
- Optional backends: `core/torch_backend.py`, `core/jax_backend.py`


## Notes:
Novelty vs. reinvention: Energy-based models, graphical models, and modular RL are well-studied. The sketptic view: "This is just EBMs + sparse factor graphs + active inference, rebranded." 
What's new?... we would argue: it's the specific combination hazard-based gating + typed micro-modules + non-local couplings + explicit redemption metrics

## Why this repo fills an interesting gap

1. **Explicit mechanics over black boxes**  
   - Typical EBM tutorials hide constraints inside monolithic neural nets. Here every constraint is an entity (`QuadraticCoupling`, `HingeCoupling`, `GateBenefitCoupling`, etc.) and `EnergyCoordinator` reports the gradient for each one. You literally encode “if node A rises, node B must fall” and see that tension in the logs.  
   - Because constraints stay explicit, observability is trivial: `RelaxationTracker` and `EnergyBudgetTracker` tell you which rule is stressed, which makes the framework feel like tinkering with springs and latches instead of guessing inside a transformer.

2. **Active Inference without the Bayesian wall**  
   - Libraries such as `pymdp` demand fluency in variational free energy, factorized beliefs, and dense notation. This repo implements the same “future evidence redeems past predictions” using physics: gradients, damping, and redemption gates that lower energy when later context fixes earlier mistakes.  
   - Engineers can watch postdiction happen numerically (`Δη_domain`, `hazard_mean`) without deriving variational bounds, making this a friendlier entry point for Active Inference behavior.

3. **Control-theory safety as a first-class lesson**  
   - Most “intro AI” repos hand-wave stability (“drop the learning rate if it explodes”). Here the roadmap ships Gershgorin bounds, small-gain allocators, Lipschitz estimators, and contraction-margin telemetry. Turning on `stability_guard=True` literally applies a Lyapunov-style cap every step.  
   - Learners therefore see passivity and safety instrumentation inside runnable code, bridging AI practice with control theory—a gap rarely covered in public repos.

4. **White-box meta-learning** ✅ PRODUCTION READY
   - P4 adapters (GradNorm ✅, AGM ✅, GSPO-token ✅, **SmallGain** ✅) are typed strategies that observe per-term gradients and adjust weights with full transparency. When the "grammar constraint" is violated, the adapter announces the +10% weight bump; when a coupling dominates, it gets throttled.  
   - **SmallGain** achieves 40% faster convergence on dense graphs with formal stability guarantees (validated: `docs/SMALLGAIN_VALIDATION_FINAL.md`).
   - That demystifies "attention" and "learning rates" by showing that meta-learning is simply deciding which explicit constraint matters most at each step.

5. **Accessibility through mechanics**  
   - The combination of explicit components, physics-style redemption, built-in stability guards, and planned visual dashboards turns the repo into a “watch the physics of intelligence” lab. Once the roadmap’s real-data demo and visualizer land, it becomes the only place you can literally see an energy surface relax while tracing which rule fired. Mechanics—not tensors or Bayesian integrals—becomes the intuitive language for EBMs.

## Isn’t this just a physics engine?
Optimization makes the two domains identical: minimizing loss in ML equals lowering potential energy in physics. This repo exposes the visible springs so you can inspect them.

1. **Loss ↔ potential energy** — Violated constraints stretch a spring, raising energy exactly like an error term. Relaxation is gradient descent in disguise.
2. **Latches ↔ non-linear activations** — Mechanical latches mirror ReLU/sigmoid behavior; mixing linear springs with non-linear gates yields universal computation (Hopfield/HNN heritage).
3. **Inference ↔ equilibrium** — We pin known values, let the whole system settle, and read the equilibrium. Stable Diffusion does this for pixels; here we do it for logic.
4. **Historical precedent** — Hopfield and Hinton’s Nobel-recognized work formalized neural nets as statistical physics. We simply expose the forces.
5. **Why this view matters** — Because you can point to the exact spring that’s too tight and adjust it (or let adapters do it). Black-box nets bury that tension inside millions of weights.

......Boing...Boing...Boing...Boing...Boing...Boing...

## How this differs from classic Hopfield networks (The "One-Shot" Explanation)

Both are Energy-Based Models (EBMs) that minimize a scalar function, but they serve opposite purposes. 

**Hopfield Networks** are "associative memories": they use identical units to store patterns and recall them (like a mattress relaxing into a shape).
**Complexity from Constraints** is a "reasoning engine": it uses diverse, typed parts to solve logic puzzles, balancing conflicting rules in real-time.

| Feature | Hopfield / Boltzmann Machines | Complexity from Constraints |
| --- | --- | --- |
| **The "Atom"** | **Identical Neurons**: Scalar pixels or spins ($s_i \in \{-1, 1\}$) interacting via symmetric weights. | **Typed Modules**: Semantic units (e.g., `SequenceModule`, `GatingModule`) that encapsulate specific behaviors (ordering, sparsity, compression). |
| **The Physics** | **Spin Glass**: A homogeneous field of magnetic spins aligning or anti-aligning. | **Clockwork Linkage**: A heterogeneous machine with springs (quadratic costs), latches (hinges), gears (couplings), and tilting platforms (dynamic weights). |
| **The Landscape** | **Static**: The energy landscape is fixed after training. Inference just rolls the ball downhill. | **Dynamic**: The landscape *moves* during inference. Adapters (GradNorm, AGM) reshape the hills in real-time to break deadlocks or prioritize specific constraints. |
| **The Logic** | **Correlations**: "If A is on, B should be on" ($E = -w x_i x_j$). | **Predicates**: "If A > B, then C must cost more" (Inequalities, gate-benefits, redemption hinges). |
| **The Goal** | **Recall**: Retrieve a stored memory from a noisy cue. | **Reasoning**: Satisfy a set of explicit, conflicting logical rules or safety constraints. |
| **Debuggability**| **Opaque**: Why did it settle here? "Because the tensor said so." | **Transparent**: Why did it settle here? "Because the 'safety_spring' pushed back with 5.2 units of force against the 'profit_coupling'." |

**Summary**: If you want to study *distributed memory*, Hopfield nets are the gold standard. If you want to engineer systems that obey *explicit rules*, explain *why* a rule triggered, and adapt those rules mid-thought (Meta-Learning), this framework is your laboratory.

### Related evidence (recent work)
We note independent, recent work that aligns with parts of this framework. We converged on similar ideas separately; we cite these as related evidence without claiming priority:

- **Data-Driven Ginzburg-Landau ROMs** (Williams et al., 2024): Validates the use of **Landau order parameters** and **SINDy-discovered sparse dynamics** for modeling oscillatory instabilities (vortex shedding). Their findings—that complex high-order terms vanish under sparsity, leaving a core Landau-Ginzburg structure—strongly support our minimalist "quadratic/quartic + sparse coupling" design. Link: https://arxiv.org/html/2411.08277v1
- **Dynamic Chunking** (H-Net): Dynamic boundary “gating”, smoothing (soft application) and ratio-style compression targets echo our gating + gate-benefit + complexity patterns. Link: https://arxiv.org/html/2507.07955v2
- **EXPO**: Stable Reinforcement Learning with Expressive Policies — base policy plus edit-selection mirrors our “provisional + redemption” coupling with explicit benefit vs cost. Link: https://arxiv.org/html/2507.07986v2

### Discrete–continuous optimization bridge
- We use hazard-based gating (η_gate = 1 − exp(−softplus(k·net))) as a smooth relaxation of discrete open/close decisions. During measurement we apply soft blending; for attribution or deployment we can apply a hard threshold (straight‑through–like), yielding a practical route to solve discrete selection with continuous optimization inside the coordinator (no REINFORCE).
- This pattern mirrors “smoothing/STE” in dynamic chunking architectures (cf. H‑Net) and fits an operator‑splitting/proximal view: the gate is a differentiable control variable, while the final decision can be snapped to a discrete action without changing the inner energy formulation. See also “Related evidence (recent work)”. 

Straight‑Through option (optional):
- Construct `EnergyGatingModule(..., straight_through=True, st_threshold=0.5)` to return a hard forward decision (0/1) while keeping the smooth formulation available internally for analysis and attribution. Default remains soft (straight_through=False).

### Stability "Nugget": Orthonormal Polynomials (aPC / CODE) ✅ VALIDATED
Standard energy functions on $\eta \in [0,1]$ (using monomials $1, \eta, \eta^2...$) are often ill-conditioned, leading to optimization instability ("energy wars").
- **The Fix**: Map $\eta \to \xi = 2\eta - 1$ and use an **orthonormal polynomial basis** (Legendre for uniform, Arbitrary Polynomial Chaos for data-driven distributions).
- **Impact**: This diagonalizes the Hessian, smoothing the landscape and drastically reducing backtracks. ✅ **Validated via conditioning benchmarks** (`test_polynomial_conditioning.py`).
- **Reference**: Wildt, N., et al. (2025). "CODE: A global approach to ODE dynamics learning." arXiv:2511.15619.
- **Implementation**: See `modules/polynomial/apc.py` and `modules/polynomial/polynomial_energy.py`. Validation & usage: `docs/POLYNOMIAL_BASES.md`.
- **Tests**: Minima parity, gradient parity, orthonormality, and **conditioning improvements** (ΔF smoothness, backtrack reduction) all validated ✅.

### Architecture philosophy: dumb core, intelligent updates & events
- The coordinator is intentionally simple: a typed, auditable projector that minimizes an explicit energy with gates/couplings. It does not “learn”; it enforces constraints.
- The system’s intelligence lives in the updates and event‑driven coordination: redemption couplings, gating decisions, weight/adaptation policies, amortizers that propose good initial η, homotopy/stability schedules, and explicit event logs/metrics that close the loop.
- In production, amortizers are first‑class (not optional): they propose fast, task‑aware η₀ and active sets; the coordinator then performs principled projection (prox/ADMM/line‑search) to enforce the global energy and constraints.

## Utility
Where it helps:
Composability over monoliths: The micro-module + energy coordinator pattern is a practical alternative to end-to-end deep models. For domains where interpretability and modularity matter (safety-critical systems, scientific ML, explainable AI).

Non-local correction mechanisms: Redemption couplings are underexplored in mainstream ML. The gating + coupling design could be useful for new architectures for sequence modeling, retrieval-augmented generation, or multi-agent systems where "future context redeems past errors."

Energy-based coordination without neural nets: Many researchers default to learned weights/attention. This framework shows that explicit energy minimization with typed protocols can achieve coherent behavior. This is valuable for hybrid systems (symbolic + subsymbolic) and for researchers who want more control than "train a bigger model."

Code as pedagogy: The small, typed, tested codebase is a teaching tool. A grad student learning about Landau theory, variational methods, or active inference could run These experiments in an afternoon. 

### Examples where to use (and not)
- Use when
  - You need small, interpretable pieces to coordinate coherently (audio agents, scientific ML, hybrid symbolic workflows, retrieval‑augmented pipelines).
  - Cutting‑edge cases
    - Multimodal tool‑using agents: coordinate vision/audio/LLM modules with non‑local redemption across modalities.
    - RAG planning at document scale: later evidence redeems earlier retrieval/summary chunks; gate costly retrieval only when ΔF drops.
    - Program synthesis/repair: non‑local constraints reward edits that improve test/contract energy; gate merges for impactful fixes.
    - Embodied/robotic planning: sparse non‑local couplings enforce long‑horizon constraints; gates open rare action branches.
    - Continual/active learning: dataset curation where gates accept examples that reduce global energy (curriculum/robustness).
    - Sensor fusion: asynchronous sensors (vision/LiDAR/IMU/audio) coupled non‑locally; gates spawn/merge hypotheses adaptively.
    - Audio agents: streaming ASR + enhancement + diarization; redemption rescoring fixes earlier segments using later context.
  - Classical cases
    - Graph connectivity/percolation with shortcuts; study threshold shifts under sparse non‑local structure.
    - Scheduling/operations research: add resources/capacity only when global constraint energy decreases.
    - Factor‑graph‑like inference with typed micro‑modules and explicit ΔF traces (message‑passing analogue).
    - Time‑series anomaly detection: redeem or confirm early flags using non‑local future context; gate escalations.
    - Multi‑agent games/regret dynamics: sparse couplings plus gating to model rare but decisive strategic shifts.

- Not ideal when
  - In current state the backends are MVP‑level, if You need SOTA accuracy on large benchmarks immediately please be patient, use something else or extend it yourself.
  - You must operate at web scale without investing in analytic grads/vectorized couplings/JAX/Torch fast paths... we are still at MVP!!!
  We working as fast as possible to make this "un Cinghialone"


## Reactivity
- What it means
  - Step‑reactive: η and total energy update each accepted iteration.
  - Decision‑reactive: gates respond instantly to net = (gain − cost) via hazard: η_gate = 1 − exp(−softplus(k·net)).
  - Weight‑reactive: optional `auto_balance_term_weights` and `WeightAdapter` adjust per‑term weights online.

- How to measure
  - Enable ΔF/η traces with `--track_relaxation` (uses `RelaxationTracker`). Only accepted steps emit energy events.
  - Log gating responsiveness with `--log_gating_metrics` (hazard_mean, μ̂, good_bad_ratio).
  - Suggested KPI: “ΔF90” = steps to reach 90% of total energy drop; repo experiments typically converge in ~30–50 steps. With analytic grads + line search + vectorized quadratic, expect ~15–30.

- How to increase reactivity
  - Coordinator toggles: `use_analytic=True`, `use_vectorized_quadratic=True`, `line_search=True`, `normalize_grads=True`, optional `max_grad_norm`, `neighbor_gradients_only=True` or use `relax_etas_coordinate(...)`.
  - Gating crispness: increase `k`, tune `cost`, and, if appropriate, reduce `gate_alpha/gate_beta` via `constraints`.
  - Term‑weight emphasis: increase `coup:GateBenefitCoupling` (or `coup:DampedGateBenefitCoupling`) relative to `local:EnergyGatingModule` when Δη should drive openings.

- Quick run (Windows PowerShell, our system setup)
  - `uv run python -m experiments.sequence_gating_hypothesis --steps 40 --track_relaxation --log_gating_metrics`

See also: `docs/speed_up_tweaks.md` for deeper performance guidance and profiling snippets. (this small note is for Datamutant devs to implement the ref'd doc might not be on git)

## Contributing (light)
Please wait until we have some substance here to get our teeth into, this note might be old, contact OG on twitter.




