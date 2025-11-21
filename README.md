# Complexity from Constraints

**📖 MUST READ**: [Complexity_from_Constraints.md](Complexity_from_Constraints.md) — Philosophy, motivation, and the five equations.


## Code in this repo
Small, composable modules coordinated by a global free-energy objective, with sparse non-local couplings that provide "future-like" corrections. Each module exposes an order parameter (η) and a local energy F(η; c). Composition = Σ F_local + Σ F_couple. The system seeks low-energy, coherent behavior without hard-coding global rules.

We keep the design humble and exact. This is an MVP intended for learning and iteration.

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

`EnergyCoordinator` exposes an optional `WeightAdapter` protocol so external trainers can tune per-term weights (GradNorm-style). Implement
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

**What “meta-training” means here**
- The inner loop (coordinator) relaxes η to lower the current total energy.
- The outer loop (your adapter or another trainer) observes per-term gradient norms and total energy, then updates weights to reflect higher-level goals (e.g., GradNorm balancing, curriculum scheduling, downstream accuracy).
- This decouples modeling from tuning: instead of hard-coding all λ’s, you can learn them online, integrate with other optimizers, or plug into a broader training regimen (AGM trainer, reinforcement learning, etc.).
- Auto-balancing (`auto_balance_term_weights`) is a simple built-in heuristic; `WeightAdapter` lets you replace it with something principled or domain-specific when needed.

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
# optional if torch available
uv run python -m experiments.energy_reg_attn_ablation
uv run python -m experiments.emergent_nash_learning
uv run python -m experiments.branching_coexistence [--log_gating_metrics]
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
- [PYDANTIC_V2_VALIDATION_GUIDE.md](PYDANTIC_V2_VALIDATION_GUIDE.md) — required patterns for entity construction/validation across repos.
- [cf_logging/observability.py](cf_logging/observability.py) — `RelaxationTracker` for ΔF/η traces and `GatingMetricsLogger` for hazard/η/redemption CSVs.
- Optional autograd backend: see `core/torch_backend.py` (install torch extra).
- JAX backend prototype: `core/jax_backend.py` (install `[jax]` extra). Note this is an initial, lightly tested pass covering gating + quadratic couplings; future work will expand support/validation.



## Contributing (light)
Please wait until we have some substance here to get our teeth into, this note might be old, contact OG on twitter.




