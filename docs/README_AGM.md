# AGM in This Repository: Phase‑Adaptive Weighting and Uncertainty‑Gated Thresholds

This document explains how we use AGM (adaptive, phase‑aware modulation) inside our energy‑based coordination framework, where the coordinator minimizes a composite free‑energy over modules and couplings. We also link the original scaffold that inspired the adaptive idea and describe how we adapted it from an off‑policy/offline RL setting to our online, per‑step energy relaxation setting.

## Origin
- Original adaptive idea and scaffolding: `AGM_Training` (off‑policy RL, supports offline log ingestion and adaptive controllers).  
  Link: [AGM_Training](https://github.com/Gman-Superfly/AGM_Training)

## What “AGM” means here
- We implement **phase‑adaptive term weighting** during energy relaxation:
  - The coordinator runs iterative relaxation on the order parameters `η`.
  - A weight adapter adjusts the term weights by phase to emphasize whichever terms are most useful at each stage of optimization (`explore → align → stabilize` intuition).
  - This keeps the optimization well‑conditioned without hardcoding a single static set of weights.
- We optionally apply **uncertainty‑gated thresholding** for non‑local decisions:
  - When epistemic uncertainty is high, we lower the effective cost to allow cautious exploration; when confidence improves, we raise cost to become conservative.

Concretely, this is wired by:
- `AGMPhaseWeightAdapter` injected into `EnergyCoordinator` to modulate term weights per step.
- An `EnergyGatingModule` whose `cost` can be adjusted online based on streaming uncertainty metrics.

See the runnable demo in `experiments/agm_phase_demo.py`.

## How it differs from the original repo
- **Original** (`AGM_Training`):
  - Off‑policy RL scaffold with kernel smoothing, adaptive controllers, early stopping, and (optionally) offline/HMPO log ingestion.
  - Adaptation logic tunes controller knobs given telemetry over time (episodes/steps), oriented around policy/value updates.
- **Here** (this repo):
  - Online, **per‑step energy relaxation** over composite terms (modules + couplings).
  - AGM provides:  
    - Phase‑adaptive term weighting during relaxation (`AGMPhaseWeightAdapter`).  
    - Optional uncertainty‑driven adjustment of gating cost (`EnergyGatingModule.cost`) from streaming energy.
  - Target is not a policy gradient; it’s minimizing a free‑energy with analytical/finite‑difference updates and invariant enforcement.

## How it’s wired in code
- `EnergyCoordinator(..., weight_adapter=AGMPhaseWeightAdapter(), ...)`  
  The adapter is called each step to produce updated term weights based on the current relaxation phase/telemetry.
- `EnergyGatingModule`: Non‑local expansion decision with hazard‑based open probability `η_gate = 1 - exp(-softplus(k·(gain − cost)))`.
- Optional cost adaptation:
  - We compute streaming uncertainty from recent energy values.
  - We adjust `gate.cost` down when epistemic uncertainty is high (encourage exploration) and up when low (conserve).

Example (simplified) of cost adaptation used in the demo:

```python
def on_energy(F: float) -> None:
    energies.append(float(F))
    u = compute_uncertainty_metrics(energies, recent_performance=None)
    base = gate_module.cost
    multiplier = 0.5 + 1.5 * (1.0 - u.epistemic)  # high uncertainty → lower cost
    gate_module.cost = float(base * multiplier)
```

## Demo: AGM phase adaptation + uncertainty‑gated thresholds
- Script: `experiments/agm_phase_demo.py`
- What it does:
  - Builds a noisy sequence with a planted “glitch”.
  - Uses `SequenceConsistencyModule` to push toward monotone consistency.
  - Uses `EnergyGatingModule` to make rare but impactful expansion decisions.
  - Adds couplings: `GateBenefitCoupling` (rewards domain improvement) and `QuadraticCoupling` (stability).
  - Runs the coordinator with `AGMPhaseWeightAdapter` so term weights evolve by phase.
  - If enabled, adapts the gate `cost` online from uncertainty computed over the energy trace.
- Logs per‑step weights and energy for inspection.

### Run it (Windows PowerShell)
```powershell
cd C:\Git\Complexity_from_constraints
python .\experiments\agm_phase_demo.py
# With uncertainty-gated thresholds:
python .\experiments\agm_phase_demo.py --use_uncertainty_gate
```

## Observability and what to look for
- Energy should decrease across steps.
- The logs include columns like `w:<term>` representing AGM’s dynamic term weights; you should see them shift as phases change.
- When `--use_uncertainty_gate` is on, the effective gating behavior will respond to uncertainty (more permissive under high epistemic uncertainty).

## Related modules and docs
- Gating details and rationale: `docs/README_GATING.md`
- Demo wiring: `experiments/agm_phase_demo.py`

## References
- Original scaffold and adaptive idea: [AGM_Training](https://github.com/Gman-Superfly/AGM_Training)


