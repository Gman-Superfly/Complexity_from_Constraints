# Experiments Quick Reference

Each experiment is ≤2 screens, deterministic seeds, and logs via `cf_logging.metrics_log`. Run with `uv run python -m experiments.<name>`.

## `landau_sweep.py`
- **Goal**: Show double-well behavior as parameter `a` crosses zero.
- **Key args**: `--a_min`, `--a_max`, `--num`, `--b`, `--eta0`, `--steps`.
- **Logged fields**: `a`, `eta_final`, `F_final`, optimizer hyperparameters.
- **Signals**: `eta_final` bifurcates for negative `a` (ordered phase).

## `non_local_connectivity_threshold_shift.py`
- **Goal**: Demonstrate how sparse shortcuts lower the apparent percolation threshold.
- **Signals**: `giant_component_frac` vs Bernoulli `p`, with and without shortcuts.
- **Tips**: Use Polars to plot `Δthreshold`.

## `sequence_redemption.py`
- **Goal**: Compare prefix-only scoring vs non-local redemption on sequences with planted mistakes.
- **Signals**: `redemption_score` (loss drop at earlier positions).

## `sequence_gating_hypothesis.py`
- **Goal**: Full coordinator + gate hypothesis test (sequence module + gate-benefit coupling).
- **Metrics**: `eta_before`, `eta_after`, `eta_gate_final`, `energy_before`, `energy_after`, `redemption`.
- **Observability**: `--track_relaxation` enables `RelaxationTracker`; `--log_gating_metrics` writes hazard/η/redemption CSVs.

## `energy_gated_expansion.py`
- **Goal**: Calibrate hazard-based gating (rare but impactful expansion).
- **Metrics**: `expansion_rate`, `redemption_mean`, `hazard_mean`, `mu_hat`, `good_bad_ratio`.
- **Soft effect**: Repairs blend by `η_gate`; expansions counted when `η_gate > 0.5`.
- **Observability**: `--log_gating_metrics` enables `GatingMetricsLogger` CSV output (hazard/η/redemption). No coordinator here, so `RelaxationTracker` is not used.

## `branching_coexistence.py`
- **Goal**: Illustrate sparse top-2 gating and coexistence.
- **Metrics**: `ends_count_mean`, `branching_rate_mean`, `hazard_mean`.
- **Observability**: `--log_gating_metrics` records hazard/η for every branch decision.

## `energy_reg_attn_ablation.py` (optional, needs torch)
- **Goal**: Inspect how energy penalties change attention distributions.
- **Use when**: GPU/Torch available; otherwise skip.

## `emergent_nash_learning.py`
- **Goal**: Game-theoretic toy (HMPO/AGM tie-in). Non-essential for MVP but useful for RL folks.

## `branching_coexistence.py`
- **Goal**: Sparse top-2 branching demo (coexistence of multiple ends).
- **Metrics**: `ends_count_mean`, `branching_rate_mean`, `hazard_mean`.
- **Mechanic**: Gumbel race over `(k·net + noise)` picks winners.

## `analyze_first_three.py`
- **Goal**: Quick Polars summaries of the first three experiments (saves time after batch runs).

### General tips
- Always log to `logs/` (Polars writes are idempotent); keep run IDs unique.
- Use `RelaxationTracker` when experiments involve the coordinator to capture ΔF/η traces. Energy events are emitted only after accepted steps; trial backtracking evaluations are not logged.
- For new experiments copy the template docstrings above and state: goal, knobs, logged fields, expected signal.

### Constraint calibration patterns (term weights)
- Non‑local benefit should dominate when the measured domain improvement Δη > 0. Calibrate via `constraints["term_weights"]`:
  - Prefer larger weights for `coup:<CouplingClass>` and smaller for `local:EnergyGatingModule` when you expect redemption to open gates.
  - Example (sequence gating):
    - `term_weights = {"coup:GateBenefitCoupling": 3.0, "local:EnergyGatingModule": 0.1}`
  - Example (connectivity gating with damping):
    - `term_weights = {"coup:DampedGateBenefitCoupling": 2.0, "local:EnergyGatingModule": 0.2}`
- Gate local energy coefficients can be gently reduced for permissive regimes:
  - `gate_alpha`, `gate_beta` in `constraints` override module defaults, e.g. `gate_alpha=0.05`, `gate_beta=0.10`.
- Keep expansions “rare but impactful”: if gates open too often, raise cost or increase gate local weights; if too conservative, raise coupling weight or reduce gate local weights slightly.

### Stability/observability
- Coordinator uses line search; trial energies are evaluated silently for acceptance. Only accepted steps emit energy events.
- A small early‑stop guard aborts iterations if energy increases, preventing oscillations from polluting traces.



