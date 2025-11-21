# Modules Quick Reference

Short, humble summaries (≤2 screens). Every module follows the `EnergyModule` protocol (`compute_eta`, `local_energy`, optional `d_local_energy_d_eta`) and keeps η within `[0, 1]`. Couplings follow `EnergyCoupling` with optional analytic derivatives (`SupportsCouplingGrads`). Term weights can be tuned dynamically via `EnergyCoordinator` (see README meta-training section).

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

## NonlocalAttention (`models/nonlocal_attention.py`)
- **What**: PyTorch module that regularizes attention weights via an energy penalty.
- **Order parameter**: Softmax attention statistics; energy discourages entropy collapse.
- **Status**: Optional demo; requires `torch`. Keep off by default unless you need the ablation.

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


