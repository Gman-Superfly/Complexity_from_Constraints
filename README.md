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
# optional:
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
uv run python -m experiments.branching_coexistence
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

## Roadmap (Scaling‑first)
- P0 — Scaling core (near‑term)
  - [ ] Replace FD‑only steps with correct analytic gradients in the coordinator; remove double‑counting; add optional `SupportsLocalEnergyGrad` / `SupportsCouplingGrads`; clamp η∈[0,1]; optional damping/line‑search.
  - [ ] Neighbor‑only gradients via an adjacency map; add coordinate‑descent + active‑set updates to avoid O(M·(M+E)) per step.
  - [ ] Vectorized fast paths for common couplings (e.g., quadratic) using NumPy scatter‑adds.
  - [ ] Gradient‑norm normalization/clipping across term families to prevent “energy wars”.
  - [ ] Tests: analytic vs finite‑diff parity; relaxation energy non‑increase; domain‑safety (no out‑of‑range η).
- P1 — Production tests & observability
  - [ ] Add `test_prod_*` for composed flows (sequence + coupling + gating); ΔF/η traces; gating rate metrics.
- P2 — Differentiable/robust gating
  - [ ] Soft application of gating effect in `energy_gated_expansion.py`; optional damping/asymmetry in redemption couplings; strength‑sweep stability tests.
- P3 — Packaging & hooks
  - [x] `pyproject.toml` + `CITATION.cff`
  - [ ] Introduce optional `term_weights` and a `WeightAdapter` hook to enable external meta‑training without coupling repos (e.g., integration with AGM trainer later).
- P4 — Backend acceleration (optional)
  - [ ] Optional JAX/Torch backend for autograd and GPU; sparse/block‑structured couplings.

## Publishing checklist (humble)
- Documentation
  - [ ] Short conceptual overview (why η, why F) with a small diagram
  - [ ] Module/experiment READMEs (1–2 screens each)
  - [ ] Reference to PYDANTIC_V2_VALIDATION_GUIDE.md if applicable
  - [x] LLMS.txt (policy for LLM/AI crawlers) — present
  - [x] CITATION.cff
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
  - [x] Choose license (MIT); ensure headers as needed
  - [ ] Confirm LLMS.txt reflects intended usage and training policy

## Contributing (light)
Please wait until we have some substance here to get our teeth into, this note might be old, contact OG on twitter.




