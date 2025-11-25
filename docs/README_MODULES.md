# Modules Quick Reference

Short, humble summaries (≤2 screens). Every module follows the `EnergyModule` protocol (`compute_eta`, `local_energy`, optional `d_local_energy_d_eta`) and keeps η within `[0, 1]`. Couplings follow `EnergyCoupling` with optional analytic derivatives (`SupportsCouplingGrads`). Term weights can be tuned dynamically via `EnergyCoordinator` (see README meta-training section).

## Precision-aware modules 

- **What**: Modules may now implement `SupportsPrecision` to expose local stiffness/curvature.
- **Why**: The coordinator records a per-module precision diagonal every relaxation step. This enables Newton-aware step scaling and precision-controlled noise injection (see `docs/ROADMAP_NEUROSYMBOLIC_HOMEOSTAT.md`).
- **Interface**:
  ```python
  class SupportsPrecision(Protocol):
      def curvature(self, eta: float) -> float:
          """Return the local second derivative at η."""
  ```
- **Guidance**: Return non-negative curvature (stiff constraints → large values). If unsure, omit the protocol and the coordinator treats the module as “loose” (precision 0). Tests: `tests/test_precision_core.py` ensures the cache reflects module-provided curvature.

## Weight Adapters (`core/weight_adapters.py`)
- **What**: Implementations of the `WeightAdapter` protocol that dynamically adjust term weights during relaxation.
- **GradNormWeightAdapter**: Equalizes gradient magnitudes across local/coupling terms to prevent "energy wars"; adjustable target norm, alpha (restoring force), update rate, floor/ceiling clamps.
- **Use when**: Multiple energy terms with different scales compete; one term consistently dominates; you want balanced contribution from all constraint types without manual tuning.
- **Demo**: See `experiments/auto_balance_demo.py` comparing fixed weights vs GradNorm adaptation.
- **Integration**: Pass to `EnergyCoordinator(weight_adapter=GradNormWeightAdapter(...))` and adapter.step() runs per iteration.

## SequenceConsistencyModule (`modules/sequence/monotonic_eta.py`)
- **What**: Sublinear monotonicity score via random pair sampling (`O(samples)`).
- **Order parameter**: Fraction of sampled pairs that satisfy `v[i] <= v[j]`.
- **Local energy**: Landau around target 1.0 → `a (1-η)^2 + b (1-η)^4`.
- **Use when**: You need a quick monotone/coherence estimate for sequences (music, trajectories, logits).
- **Constraints**: `seq_alpha`, `seq_beta` override local coefficients; asserts ≥ 0 and sample count ≥ 1.
- **Need fully vectorized PyTorch version?** Use the optional sublinear monotonicity repo: [Sublinear Monotonicity Score Extractor](https://github.com/Gman-Superfly/sublinear_monotonicity_score) (install torch). It provides a GPU-ready sampler plus optional detail stats; integrate it via a custom module if you need more throughput.

## NLThresholdShift (`modules/connectivity/nl_threshold_shift.py`)
- **What**: Connectivity order parameter on grid graphs with optional “shortcut” edges.
- **Order parameter**: Fraction of nodes in the giant connected component after adding shortcuts.
- **Local energy**: Encourages staying near configured target connectivity; punishes brittle graphs.
- **Use when**: Demonstrating non-local redemption in percolation/graph contexts.
- **Notes**: Shortcuts are explicit entities; use conflict resolution helpers if mutating shared graphs.

## EnergyGatingModule (`modules/gating/energy_gating.py`)
- **What**: Hazard-based gate (`η_gate = 1 - exp(-softplus(k·(gain - cost)))`).
- **Local energy**: `a η^2 + b η^4` discourages casual expansion; analytic derivative available.
- **Extras**: `hazard_rate(...)`, optional logistic fallback, integrates with gate–benefit couplings.
- **Use when**: Expansion must be rare but impactful; log `hazard_mean`, `μ̂`, `good_bad_ratio`.
- **Constraints**: `gate_alpha`, `gate_beta` override local coefficients for calibration (defaults from module if not set).

## NashModule (`modules/game/emergent_nash.py`)
- **What**: Game-theoretic equilibrium module using regret-based order parameter.
- **Order parameter**: `η = 1 - normalized_regret` where regret measures deviation from best-response.
- **Local energy**: Landau around target 1.0 (minimal regret) → `a (1-η)^2 + b (1-η)^4`.
- **Use when**: Multi-agent games, regret dynamics, emergent Nash equilibrium experiments.
- **Constraints**: `nash_alpha`, `nash_beta` override local coefficients; asserts ≥ 0.
- **Extras**: Includes `symmetric_2x2_payoff`, `strategy_regret`, `replicator_step` helpers for game theory experiments.

## NonlocalAttention (`models/nonlocal_attention.py`)
- **What**: PyTorch module that regularizes attention weights via an energy penalty.
- **Order parameter**: Softmax attention statistics; energy discourages entropy collapse.
- **Status**: Optional demo; requires `torch`. Keep off by default unless you need the ablation.

## PolynomialEnergyModule (`modules/polynomial/polynomial_energy.py`)
- **What**: Local energy on η using a small Legendre basis on ξ = 2η − 1 (degree ≤ 4).
- **Order parameter**: Pass-through (`compute_eta(x) → η`), expects a float in `[0,1]`.
- **Local energy**: `E(η) = Σ c_k φ_k(ξ)`, with basis selectable via `basis ∈ {"legendre","apc"}`:
  - `basis="legendre"`: `φ_k = P_k(ξ)` (Legendre); analytic derivative via chain rule (`dξ/dη = 2`).
  - `basis="apc"`: `φ_k` are APC orthonormal polynomials fitted from empirical ξ samples (Gram–Schmidt).
- **Use when**: You want a better-conditioned alternative to monomial Landau forms (aPC/CODE-style calibration); start with Legendre (fixed) before moving to aPC (data-adaptive).
- **Constraints**:
  - Weights: `constraints["poly_coeffs"]` (list of `degree+1` floats)
  - APC basis (when `basis="apc"`): `constraints["apc_basis"]` is a matrix of shape `(degree+1, degree+1)` with monomial coefficients per basis function (see `modules/polynomial/apc.py`).
  - Even-order coefficients positive typically promote stability near boundaries.
- **Notes**: Degree capped at 4 for simplicity and speed; future work may add aPC (orthonormalization on empirical η traces).

## ModuleEntity (`core/entity_adapter.py`)
- **What**: Lightweight wrapper that adds ECS-style identity and versioning to any `EnergyModule`.
- **Fields**: `ecs_id` (UUID), `version` (int), `module` (wrapped EnergyModule).
- **Methods**: `compute_eta`, `local_energy` (delegates to wrapped module), `bump_version()`, `emit(...)` for event-style hooks.
- **Use when**: You need provenance tracking, versioning, or event emission without full Abstractions framework integration.
- **Integration**: Pass `ModuleEntity` instances to coordinator; they satisfy the `EnergyModule` protocol.

## Common coupling helpers (`core/couplings.py`)
- **Quadratic/hinge**: Symmetric or directed smoothness constraints between η’s.
- **GateBenefit / DampedGateBenefit**: Reward opening gates when `Δη_domain > 0`, with optional damping/asymmetry (`positive_scale`, `negative_scale`, `eta_power`).
- **AsymmetricHinge**: Scale α/β when one module should dominate.

### Semantics (gate–benefit family)
- Positive Δη: lowers energy proportionally to the gate opening and the configured coupling weight.
- Negative Δη: increases energy; as harm lessens (Δη→0⁻), the penalty relaxes monotonically. Tests reflect this monotonicity.

### Calibration notes
- To ensure intended behavior in composed flows, set `constraints["term_weights"]`:
  - Increase `coup:<CouplingClass>` weight, decrease `local:EnergyGatingModule` when non‑local redemption should drive gate opening.
  - Adjust `gate_alpha`, `gate_beta` to modulate the cost of opening gates.

### Patterns & guarantees
- All modules assert inputs, sample counts, and parameter ranges up front (“assert early, assert often”).
- Return values are plain floats; coordinator clamps η to `[0,1]` during relaxation.
- When adding new modules, follow the same template and reference `PYDANTIC_V2_VALIDATION_GUIDE.md` for entity construction patterns.


