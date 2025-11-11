# Complexity from Constraints

Small, composable modules coordinated by a global free-energy objective, with sparse non-local couplings that provide “future-like” corrections. Each module exposes an order parameter (η) and a local energy F(η; c). Composition = Σ F_local + Σ F_couple. The system seeks low-energy, coherent behavior without hard-coding global rules.

We keep the design humble and exact. This is an MVP intended for learning and iteration.

## Why this exists (short)
- **Non-locality**: Distant parts can influence each other to redeem provisional mistakes.
- **Free-energy lens**: A single scalar objective coordinates tiny modules without making them big.
- **Composability**: Add/remove modules and couplings without rewriting the system.

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
  - `experiments/energy_reg_attn_ablation.py`: energy-regularized attention ablation (optional)
  - `experiments/energy_gated_expansion.py`: cost-vs-benefit expansion with redemption
  - `experiments/analyze_first_three.py`: quick summaries from logs
- Logging and tests
  - `logging/metrics_log.py`: Polars-based CSV logs
  - `tests/`: lightweight tests for core behaviors and invariants

## Install (Windows, macOS, Linux)
We recommend `uv` for environments.

Windows PowerShell:
```
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install polars numpy networkx pytest
# optional for attention ablation
uv pip install torch
```

macOS/Linux:
```
uv venv .venv
source .venv/bin/activate
uv pip install polars numpy networkx pytest
# optional
uv pip install torch
```

## Run experiments
```
uv run python -m experiments.landau_sweep
uv run python -m experiments.non_local_connectivity_threshold_shift
uv run python -m experiments.sequence_redemption
uv run python -m experiments.energy_gated_expansion
# optional if torch available
uv run python -m experiments.energy_reg_attn_ablation
uv run python -m experiments.emergent_nash_learning
```

Summaries:
```
uv run python -m experiments.analyze_first_three
```

## Direction
- Keep modules small and typed; push global coordination into energy and sparse couplings.
- Use non-locality to enable “future-like” corrections (redemption), measured explicitly.
- Treat gating as an energy-driven decision so expansion is rare but impactful.
- Prefer Polars for metrics and logs; avoid heavy frameworks unless clearly needed.

## Publishing checklist (humble)
- Documentation
  - [ ] Short conceptual overview (why η, why F) with a small diagram
  - [ ] Module/experiment READMEs (1–2 screens each)
  - [ ] Reference to PYDANTIC_V2_VALIDATION_GUIDE.md if applicable
  - [ ] LLMS.txt (policy for LLM/AI crawlers) — present
- Code and tests
  - [ ] Tighten assertions at entry/exit points; ensure typed returns
  - [ ] Add production-quality tests named `test_prod_*` for key flows
  - [ ] Calibrate energy weights and add invariants for non-trivial minima
  - [ ] Optional: replace finite-diff with full analytic gradients where feasible
- Packaging and usability
  - [ ] Minimal `pyproject.toml` for installable modules (optional)
  - [ ] Small examples notebook or script reproducing core plots
  - [ ] OS notes verified for Windows/macOS/Linux
- Policy and license
  - [ ] Choose license (e.g., MIT/Apache-2.0); ensure headers as needed
  - [ ] Confirm LLMS.txt reflects intended usage and training policy

## Contributing (light)
Please wait until we have some substance here to get our teeth into, this note might be old, contact OG on twitter.




