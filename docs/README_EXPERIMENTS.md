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

## `auto_balance_demo.py`
- **Goal**: Show how `GradNormWeightAdapter` keeps local/coupling term gradients balanced.
- **Scenarios**: `baseline` (fixed weights) vs `gradnorm` (adapter enabled); run both by default.
- **Metrics**: Per-step `norm:<term>` and `weight:<term>` plus energy readings, logged via `cf_logging` to `logs/auto_balance_demo.csv`.
- **Observability**: Uses coordinator hooks to record term gradient norms/weights every step; inspect CSV to verify convergence toward target norm.
- **Usage**: `uv run python -m experiments.auto_balance_demo --steps 40 --scenarios baseline gradnorm`

## `agm_phase_demo.py`
- **Goal**: Demonstrate `AGMPhaseWeightAdapter` (phase-adaptive weighting) and optional uncertainty-gated gate costs.
- **Mechanics**: Adapter gently boosts coupling weights in stable/improving regimes and softens gate locals; does the opposite when unstable/slow. Optional `--use_uncertainty_gate` adapts gate cost per-step from AGM-derived uncertainty.
- **Metrics**: Per-step weights and energy snapshots logged to `logs/agm_phase_demo.csv`.
- **Usage**: `uv run python -m experiments.agm_phase_demo --steps 40 [--use_uncertainty_gate]`

## `branching_coexistence.py`
- **Goal**: Illustrate sparse top-2 gating and coexistence.
- **Metrics**: `ends_count_mean`, `branching_rate_mean`, `hazard_mean`.
- **Observability**: `--log_gating_metrics` records hazard/η for every branch decision.

## `benchmark_delta_f90.py`
- **Goal**: Compare ΔF90 (steps to 90% of total drop) across coordinator configurations.
- **Presets**:
  - `default`, `analytic`, `vect`, `coord`, `adaptive` (existing)
  - `prox` (operator-splitting/prox mode; uses `operator_splitting=True`)
  - `gradnorm` (enables `GradNormWeightAdapter`)
  - `agm` (enables `AGMPhaseWeightAdapter`)
  - `smallgain` (enables `SmallGainWeightAdapter` with stability‑margin allocator and Lipschitz telemetry)
  - `admm` (enables experimental ADMM path for quadratic couplings)
- **Logged fields**:
  - Core: `run_id`, `config`, `steps`, `wall_time_sec`, `delta_f90_steps`, `energy_final`
  - Per-term: `energy:local:<Module>`, `energy:coup:<Coupling>`, `grad_norm:local:<Module>`, `grad_norm:coup:<Coupling>`
  - Flags: `operator_splitting`, `adapter`
- **Usage**:
  - `uv run python -m experiments.benchmark_delta_f90 --configs analytic prox gradnorm agm --steps 60`

## `energy_reg_attn_ablation.py` (optional, needs torch)
- **Goal**: Inspect how energy penalties change attention distributions.
- **Use when**: GPU/Torch available; otherwise skip.

## `apc_vs_legendre_ood.py`
- **Goal**: Compare Legendre vs APC basis under OOD splits; inspect stability and backtracks.
- **Mechanics**: Fit APC on a "train" ξ distribution; evaluate on both train/test (OOD) with a shared coefficient vector; single-module relaxation with line search logs total backtracks. Optional `--track_relaxation` records per-step traces to `logs/apc_legendre_relaxation.csv`. Optional `--track_budget` emits per-step energy budgets and contraction margins to `logs/apc_legendre_budget.csv`.
- **Logged fields**: `split`, `basis`, `energy_final`, `total_backtracks`, `steps`, `eta0` (+ per-step `energy:*`, `grad_norm:*`, `contraction_margin` in budget log).
- **Usage**:
  - `uv run python -m experiments.apc_vs_legendre_ood --degree 4 --steps 30 [--track_relaxation --track_budget]`
  - Plot summary (basis comparison): `uv run python -m experiments.plot_apc_vs_legendre --input logs/apc_vs_legendre_ood.csv --save plots/apc_vs_legendre_summary.png`
  - Plot per-step budget/contraction metric: `uv run python -m experiments.plot_energy_budget --input logs/apc_legendre_budget.csv --metric energy:local:PolynomialEnergyModule --smooth 3 --save plots/energy_budget_metric.png`

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
- For deeper per‑step insight, attach `EnergyBudgetTracker` to record per‑term energy and gradient norms, plus backtracks and optional `contraction_margin` when `stability_guard` is enabled.
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



