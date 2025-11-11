# Energy-Gated Expansion (Non-Local, Free-Energy Aligned)

This module implements rare-but-impactful expansion decisions as part of a small, composable, energy-coordinated system.

## Concepts
- Order parameter: `η_gate ∈ (0,1)` from a sigmoid over `(gain - cost)`.
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


