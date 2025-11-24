# Energy-Gated Expansion (Non-Local, Free-Energy Aligned)

This module implements rare-but-impactful expansion decisions as part of a small, composable, energy-coordinated system.

## Concepts
- Order parameter: `η_gate ∈ (0,1)` as a one-step open probability from a hazard:
  `net = gain - cost`, `λ = softplus(k·net)`, `η_gate = 1 - exp(-λ)`.
- Local energy: `F_gate(η) = a η^2 + b η^4` discourages casual expansion.
- Coupling: `F = - w · η_gate · Δη_domain` rewards opening the gate only when domain order improves.

## Usage
1. Define a `gain_fn(x)` returning positive improvement (e.g., `η_after - η_before`).
2. Create `EnergyGatingModule(gain_fn, cost=...)`.
3. Optionally add `GateBenefitCoupling(weight, delta_key)` and pass a `delta_eta_domain` in constraints.
4. Minimize total energy with the coordinator (finite-diff or analytic fallback).

## Experiment
`experiments/energy_gated_expansion.py` varies cost and tracks:
- `expansion_rate`: fraction of trials where `η_gate > 0.5`
- `redemption_mean`: improvement in full-sequence order when expansion triggers
- `hazard_mean`: mean instantaneous rate `λ` before decision
- `mu_hat`: expansions per unit redemption (compute efficiency surrogate)
- `good_bad_ratio`: redemption-positive vs non-positive expansions

## FPP lens: why hazard-based gating?
- Memoryless decisions: with `λ = softplus(k·(gain − cost))`, `η_gate = 1 − exp(−λ)` emulates exponential waiting times → stable single-pass unfolding.
- Unique-parent activation tree: record the parent (argmax Δbenefit) for observability and attribution, mirroring geodesic trees.
- Coexistence via sparse top‑k: allow rare top‑2 survivors at band boundaries so alternate hypotheses persist when signals are close.
- Critical calibration: tune `cost`, `(a,b,k)` to keep expansions rare but impactful; monitor `mu_hat`, `good_bad_ratio`, and `hazard_mean`.

Ref: Häggström & Pemantle (1997), “First passage percolation and a model for competing spatial growth.” [arXiv PDF](https://arxiv.org/pdf/math/9701226)

## Related AGM references
- How AGM is used here (phase‑adaptive weighting + uncertainty‑gated thresholds): see `docs/README_AGM.md`
- Origin of the adaptive idea (off‑policy, offline log ingestion scaffold): [AGM_Training](https://github.com/Gman-Superfly/AGM_Training)