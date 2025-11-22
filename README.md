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

### WeightAdapter hook (meta-training)

`EnergyCoordinator` exposes an optional `WeightAdapter` protocol so external trainers can tune per-term weights (GradNorm-style). 
(fun stuff to be added here)
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

### Gradient + backend fast paths

- `EnergyCoordinator` now defaults to `use_analytic=True`. All shipped modules/couplings implement analytic gradients, and quadratic / hinge / gate-benefit families have vectorized paths (`use_vectorized_*`) to avoid Python loops on large graphs.
- Adaptive coordinate descent is built in: set `adaptive_coordinate_descent=True` to warm-start with coordinate updates when ΔF stalls, then fall back to gradient steps.
- Operator-splitting/prox mode: set `operator_splitting=True` to enable block‑prox updates (locals + incident couplings). Tunables: `prox_tau`, `prox_steps`. Quadratic/hinge use closed‑form pairwise prox; gate‑benefit uses projected update.
- ADMM mode (experimental, quadratic focus): set `use_admm=True` with `admm_rho`, `admm_steps`, `admm_step_size`. Introduces auxiliary differences per quadratic edge and alternates s/η/u updates with a monotone acceptance guard.
  - Hinge family is supported in ADMM via nonnegative auxiliary gaps on `β η_j − α η_i`; other nonquadratic factors (e.g., gate‑benefit) participate via their gradients in the η‑update.
- Stability guard (optional): set `stability_guard=True` to cap step size using a conservative Gershgorin-style Lipschitz bound. Tunables: `stability_cap_fraction` (default 0.9), `log_contraction_margin=True` to record a per‑step margin; compatible with line search.
- Coupling auto-cap: set `stability_coupling_auto_cap=True` with `stability_coupling_target` (desired Lipschitz bound). Coupling term weights are temporarily scaled so the estimated Lipschitz stays below the target; shows up in EnergyBudgetTracker logs as lower `energy:coup:*`.
- Homotopy / continuation: set `homotopy_coupling_scale_start=<0..1>` and `homotopy_steps` to scale coupling term weights from a gentle start (e.g., 0.2) up to 1.0 over the first `homotopy_steps` iterations. Use `homotopy_term_scale_starts={"coup:GateBenefitCoupling":0.3, ...}` for per-term ramps, and `homotopy_gate_cost_scale_start` to temporarily raise/lower gate costs before settling to the target configuration.

### Observability helpers
- Per‑step relaxation traces: `RelaxationTracker(name, run_id).attach(coord)` then `flush()` after relaxation.
- Per-step energy budget: `EnergyBudgetTracker(run_id="...").attach(coord)` logs per-term `energy:*`, `grad_norm:*`, backtracks and optional `contraction_margin` to CSV; call `flush()` after relaxation. Plot with `uv run python -m experiments.plot_energy_budget --input logs/energy_budget.csv --metric energy:local:YourModule`.
- Torch/JAX backends support the same Landau-style modules (gating, sequence, connectivity, Nash) plus quadratic/hinge/gate-benefit couplings, so you can offload relaxation with `uv run pytest tests/test_torch_backend.py` / `tests/test_jax_backend.py` when those extras are installed.
- For a complete performance playbook (ΔF90 benchmarks, profiler snippets, remaining ideas) see `docs/speed_up_tweaks.md`.

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
uv run python -m experiments.benchmark_delta_f90 --configs default analytic vect coord adaptive prox gradnorm agm --steps 60
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
- [Complexity_from_Constraints.md](Complexity_from_Constraints.md) — philosophy + five equations (must read).
- [docs/README_MODULES.md](docs/README_MODULES.md) — module quick reference (interfaces, invariants).
- [docs/README_EXPERIMENTS.md](docs/README_EXPERIMENTS.md) — experiment intent, what to log, expected signals.
- [experiments/benchmark_delta_f90.py](experiments/benchmark_delta_f90.py) — ΔF90 benchmark harness for comparing coordinator configs (analytic vs vectorized vs coordinate descent).
  - Presets now include `prox` (operator-splitting), `gradnorm` and `agm` (weight adapters). The CSV includes per-term fields like `energy:local:...`, `energy:coup:...`, `grad_norm:local:...`, `grad_norm:coup:...`, plus `operator_splitting` and `adapter` flags for analysis.
- [PYDANTIC_V2_VALIDATION_GUIDE.md](PYDANTIC_V2_VALIDATION_GUIDE.md) — required patterns for entity construction/validation across repos.
- [cf_logging/observability.py](cf_logging/observability.py) — `RelaxationTracker` for ΔF/η traces and `GatingMetricsLogger` for hazard/η/redemption CSVs.
- Optional autograd backend: see `core/torch_backend.py` (install torch extra).
- JAX backend prototype: `core/jax_backend.py` (install `[jax]` extra). Note this is an initial, lightly tested pass covering gating + quadratic couplings; future work will expand support/validation.


## Notes:
Novelty vs. reinvention: Energy-based models, graphical models, and modular RL are well-studied. The sketptic view: "This is just EBMs + sparse factor graphs + active inference, rebranded." 
What's new?... we would argue: it's the specific combination hazard-based gating + typed micro-modules + non-local couplings + explicit redemption metrics

### Related evidence (recent work)
We note independent, recent work that aligns with parts of this framework. We converged on similar ideas separately; we cite these as related evidence without claiming priority:

- Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net) — dynamic boundary “gating”, smoothing (soft application) and ratio-style compression targets echo our gating + gate-benefit + complexity patterns. Link: https://arxiv.org/html/2507.07955v2
- EXPO: Stable Reinforcement Learning with Expressive Policies — base policy plus edit-selection mirrors our “provisional + redemption” coupling with explicit benefit vs cost. Link: https://arxiv.org/html/2507.07986v2

### Discrete–continuous optimization bridge
- We use hazard-based gating (η_gate = 1 − exp(−softplus(k·net))) as a smooth relaxation of discrete open/close decisions. During measurement we apply soft blending; for attribution or deployment we can apply a hard threshold (straight‑through–like), yielding a practical route to solve discrete selection with continuous optimization inside the coordinator (no REINFORCE).
- This pattern mirrors “smoothing/STE” in dynamic chunking architectures (cf. H‑Net) and fits an operator‑splitting/proximal view: the gate is a differentiable control variable, while the final decision can be snapped to a discrete action without changing the inner energy formulation. See also “Related evidence (recent work)”. 

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




