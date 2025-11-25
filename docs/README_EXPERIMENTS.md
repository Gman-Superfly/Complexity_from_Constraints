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
- **Adapters**: To gather redemption metrics at scale, run multiple passes with different `WeightAdapter`s (`None`, `GradNormWeightAdapter`, `GSPOTokenWeightAdapter`). Annotate logs (e.g., `run_id=sequence_gating_gspo`) so downstream analysis can compare redemption efficiency by adapter choice.
- **Gate control**: Pass `--use_uncertainty_gate` (or set `enable_uncertainty_gate=True` when constructing the coordinator) to automatically relax/tighten gate costs based on AGM/uncertainty signals.

## `energy_gated_expansion.py`
- **Goal**: Calibrate hazard-based gating (rare but impactful expansion).
- **Metrics**: `expansion_rate`, `redemption_mean`, `hazard_mean`, `mu_hat`, `good_bad_ratio`.
- **Soft effect**: Repairs blend by `η_gate`; expansions counted when `η_gate > 0.5`.
- **Observability**: `--log_gating_metrics` enables `GatingMetricsLogger` CSV output (hazard/η/redemption). No coordinator here, so `RelaxationTracker` is not used.
- **Adapters**: When wrapping the scenario in `EnergyCoordinator`, optionally enable the same adapter sweep as above to study how GSPO-token vs GradNorm affects expansion rate vs redemption gain.

## `auto_balance_demo.py`
- **Goal**: Show how `GradNormWeightAdapter` keeps local/coupling term gradients balanced.
- **Scenarios**: `baseline` (fixed weights), `gradnorm` (GradNorm adapter), `gspo` (GSPO-token adapter that drives weights via `core/gspo_token_vectorized.py`; requires the `torch` extra).
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
  - `prox_star` (prox mode with `prox_block_mode="star"` for block updates on hubs)
  - `gradnorm` (enables `GradNormWeightAdapter`)
  - `agm` (enables `AGMPhaseWeightAdapter`)
  - `smallgain` (enables `SmallGainWeightAdapter` with stability‑margin allocator and Lipschitz telemetry)
  - `admm` (enables experimental ADMM path for quadratic + hinge; supports prox-linear gate-benefit updates with damping)
- **Scenarios**:
  - `baseline` (two-module toy used historically)
  - `dense` (rings of ≥3 modules with additional skip connections; use `--scenario dense --dense_size N` to stress vectorized kernels)
- **Logged fields**:
  - Core: `run_id`, `config`, `steps`, `wall_time_sec`, `delta_f90_steps`, `energy_final`
  - KPI: `compute_cost` (equal to `wall_time_sec`) and `redemption_gain = max(F0 - F_final, 0)/compute_cost`
  - Per-term: `energy:local:<Module>`, `energy:coup:<Coupling>`, `grad_norm:local:<Module>`, `grad_norm:coup:<Coupling>`
  - Flags: `operator_splitting`, `adapter`
- **Observability**:
  - Pass `--log_budget` to attach `EnergyBudgetTracker` per run; CSVs (named via `--budget_name`) stream per-step energies/gradients for downstream adapter/meta-learning analysis.
  - Margin warning toggles (with `--log_budget`): `--warn_on_margin_shrink` to emit `margin_warn` column; `--margin_warn_threshold <float>` to set the threshold (default 1e-4).
  - Constraint violation rate (h): to emit `info:constraint_violation_rate`, include `constraint_violation_count` and `total_constraints_checked` in `coord.constraints`; see `docs/INFORMATION_METRICS.md`.
- **Usage**:
  - `uv run python -m experiments.benchmark_delta_f90 --configs analytic prox gradnorm agm --steps 60`
  - `uv run python -m experiments.benchmark_delta_f90 --configs vect smallgain prox_star --scenario dense --dense_size 32 --log_budget`
  - SmallGain parameter sweep (optional):
    - Override budget fraction ρ: `--sg_rho 0.5` (defaults to 0.7)
    - Override per‑step max Δweight: `--sg_dw 0.05` (defaults to 0.10)
    - Example: `uv run python -m experiments.benchmark_delta_f90 --configs smallgain --steps 60 --sg_rho 0.9 --sg_dw 0.2 --log_budget --run_id sg_rho0p9_dw0p2`
  - ADMM gate‑benefit options (optional):
    - Enable prox‑linear gate update: `--admm_gate_prox`
    - Set damping blend (0..1): `--admm_gate_damping 0.5` (default 0.5)
    - Example: `uv run python -m experiments.benchmark_delta_f90 --configs admm --steps 60 --admm_gate_prox --admm_gate_damping 0.5`
  - Logit updates (optional, gradient mode):
    - Enable mirror/logit η updates (avoids boundary clamp artifacts): `--use_logit_updates`
    - Applies to gradient-based configs only (default, analytic, vect, coord, adaptive, gradnorm, agm, smallgain).

### Summarize ΔF90 sweeps/logs

Compact summary table over `logs/benchmark_delta_f90.csv` with optional filtering:

```powershell
uv run python -m experiments.plots.summarize_benchmark_delta_f90 --run_id_contains sg_sweep --out plots\df90_sweep_summary.csv
uv run python -m experiments.plots.summarize_benchmark_delta_f90 --run_id_contains sg_dense
```

### Plot budget + homotopy telemetry

`plot_budget_vs_spend.py` will automatically overlay `homotopy_scale` and `homotopy_backoffs` (if present) on the right axis alongside `contraction_margin`:

```powershell
uv run python -m experiments.plots.plot_budget_vs_spend --input logs\benchmark_delta_f90_budget.csv --run_id sg_dense_smallgain --out plots\budget_vs_spend_smallgain_dense.png
```

Polynomial decorrelation telemetry
- When `PolynomialEnergyModule` is active, `EnergyBudgetTracker` now logs:
  - `poly_corr_max:poly:<idx>`: maximum absolute off‑diagonal correlation among active basis features (rolling window; default 64).
  - `poly_corr_warn:poly:<idx>`: 1 if `poly_corr_max` ≥ `poly_corr_warn_threshold` (default 0.9), else 0.
- Use these to monitor basis drift and consider re‑orthonormalization (APC) if persistent warnings appear.

Plot correlation timelines:
```powershell
uv run python -m experiments.plots.plot_poly_corr --input logs\benchmark_delta_f90_budget.csv --run_id sg_dense_smallgain --out plots\poly_corr_dense_smallgain.png --threshold 0.9
```

Energy budget timeline (per-term energies)
```powershell
uv run python -m experiments.plots.plot_energy_budget_timeline --input logs\benchmark_delta_f90_budget.csv --run_id sg_dense_smallgain --out plots\energy_budget_timeline_dense_smallgain.png --prefix energy:
```

Gain budget plot (alloc vs cost)
- Visualize adapter allocations vs Lipschitz “cost” (family curvature proxy). Includes totals and top‑K families, plus optional per‑family cost overlay.
```powershell
uv run python -m experiments.plots.plot_gain_budget --input logs\benchmark_delta_f90_budget.csv --run_id sg_dense_smallgain --out plots\gain_budget_dense_smallgain.png --topk 6 --include_cost_series
```
- Notes: requires allocator details (`alloc:*`) and Lipschitz snapshot (`cost:*`) to be logged by `EnergyBudgetTracker`.

Polynomial basis conditioning benchmark
- Compares total backtracks and ΔF for:
  - Legendre with domain normalization (ξ=2η−1)
  - Legendre without normalization (raw η)
  - Optional: aPC (provide basis fit from trace via JSON)

Run:
```powershell
# Legendre only (normalized vs raw)
uv run python -m experiments.polynomial.benchmark_basis_conditioning --degree 4 --steps 60 --step_size 0.05 --run_id cond_leg

# Include aPC (use basis from fit_apc_from_trace.py output JSON)
uv run python -m experiments.polynomial.benchmark_basis_conditioning --degree 4 --steps 60 --step_size 0.05 --run_id cond_apc --apc_basis_path logs\\apc_basis.json
```
- Output CSV name starts with `polynomial_basis_conditioning`. Inspect `total_backtracks`, `energy_final`.

SmallGain validation sweep
- Runs ΔF90 benchmark across ρ and Δweight combinations on baseline and dense scenarios; writes per-run rows and a summary CSV.
```powershell
# Quick sweep (fast): steps=40, dense_size=8
uv run python -m experiments.sweeps.sweep_smallgain --quick --log_budget --run_id_prefix sg_sw

# Full sweep (default steps=80, dense_size=16)
uv run python -m experiments.sweeps.sweep_smallgain --log_budget --run_id_prefix sg_sw_full
```
- Inspect `plots\\df90_smallgain_sweep_summary.csv` for `delta_f90_steps`, `energy_final`, `total_backtracks` by (scenario, ρ, Δweight).

Adapter comparison sweep (no‑adapt vs GradNorm vs SmallGain)
- Compares `analytic` (no adapter), `gradnorm`, and `smallgain` across baseline and dense scenarios.
```powershell
# Quick: steps=40, dense_size=8
uv run python -m experiments.sweeps.sweep_adapters_compare --quick --log_budget --run_id_prefix adapter_sw

# Full
uv run python -m experiments.sweeps.sweep_adapters_compare --log_budget --run_id_prefix adapter_sw_full
```
- Summary CSV: `plots\\df90_adapters_compare_summary.csv` (columns: scenario, config, ΔF90 steps, wall_time_sec, energy_final, total_backtracks, redemption_gain).

SmallGain defaults and tuning guidance
- Defaults (recommended starting points... for now, as repo grows they will prob change... dunno yet):
  - ρ (budget fraction) = 0.7
  - Δweight (per-step max change) = 0.10
  - Speed-leaning option: Δweight = 0.20 (faster ΔF90, typically more backtracks)
- Scope: these come from toy baseline/dense scenarios; they are not universal optima.
- What’s learned vs fixed:
  - The adapter “learns” per-step, per-edge allocations within caps.
  - ρ and Δweight are hyperparameters (outer caps), not learned by default.
- When to keep fixed:
  - Safety-critical/regulated settings, reproducibility, predictable acceptance behavior.
  - Benchmarking apples-to-apples across runs.
- When to learn/tune:
  - Changing task distributions, dynamic gate/hinge regimes, or when optimizing aggregate KPIs (e.g., redemption_gain).
  - Try outer-loop tuning (grid/Bayesian) over ρ and Δweight, objective: maximize redemption_gain or minimize ΔF90 with backtrack constraints.
  - Advanced: make ρ/Δweight state-adaptive (e.g., functions of contraction margin, hinge activity); keep caps conservative.
Relaxation traces per‑η
- To log per‑module η values for diagnostics (e.g., fitting APC from traces), construct:
  - `RelaxationTracker(name="relaxation_trace", run_id="...", log_per_eta=True)`.
- The tracker will add columns `eta:<idx>` per step.

Hierarchical + amortized inference (scaffolding)
- Interface: `core/amortizer.py` defines `AmortizedProposal` and a `SimpleHeuristicAmortizer`.
- Behavior:
  - `propose_initial_etas(modules, inputs)` uses each module’s `compute_eta(x)` when available (else defaults to 0.5).
  - `select_active_set(coord, etas, k, include_neighbors=True)` picks top‑K indices by |∂F/∂η| (finite differences) and optionally adds graph neighbors from the coupling graph.
- Status: [SCAFFOLD – expand later]
- Usage sketch:
```powershell
python - <<'PY'
from core.amortizer import SimpleHeuristicAmortizer
from experiments.benchmark_delta_f90 import make_modules_and_couplings
from core.coordinator import EnergyCoordinator

mods, coups, constraints, inputs = make_modules_and_couplings()
coord = EnergyCoordinator(mods, coups, constraints, use_analytic=True, line_search=True)
amort = SimpleHeuristicAmortizer(default_eta=0.5)
etas0 = amort.propose_initial_etas(mods, inputs)
active = amort.select_active_set(coord, etas0, k=2, include_neighbors=True)
print(\"etas0:\", etas0); print(\"active:\", sorted(active))
PY
```
- Notes:
  - This is a scaffolding utility; it does not change `EnergyCoordinator` behavior automatically.
  - You can refine only `active` indices by copying `etas`, updating those entries, and leaving others fixed per step (manual loop), or integrate deeper into your own coordinator fork.

Curriculum scheduler (scaffolding)
- Interface: `core/curriculum.py` provides `CurriculumScheduler` with a simple rule set:
  - Progress if improvement rate is positive and oscillation is low for `patience` updates
  - Regress if margin warnings occur or oscillation is high for `patience` updates
- Status: [SCAFFOLD – expand later]
- Usage sketch:
```powershell
python - <<'PY'
from core.curriculum import CurriculumScheduler

sched = CurriculumScheduler(min_level=0, max_level=5, level=0, patience=2)
energies = [1.0, 0.9, 0.82, 0.75]
dec1 = sched.update(energies, margin_warn=False)
energies2 = energies + [0.70]
dec2 = sched.update(energies2, margin_warn=False)
print(dec1, dec2)
PY
```
- Notes:
  - This does not automatically change coordinator settings. Map levels to your own knobs (e.g., stage budgets, term weight caps, or homotopy scales).
  - Keep conservative bounds to preserve stability; treat this as a controller scaffold.

Contraction margin warnings
- When `EnergyBudgetTracker.warn_on_margin_shrink=True`, the tracker emits:
  - `margin_warn=1` when `contraction_margin < margin_warn_threshold` (default 1e‑4); else 0.
  - Use this flag to flag tight/vanishing margins in dashboards.

## `polynomial_rhs_probe.py`

- Goal: Diagnostic probe that approximates the gradient/RHS `−∂E/∂η` in a Legendre basis and compares an Euler integration against the true gradient descent on a `PolynomialEnergyModule` landscape.
- Logged fields: `eta_final_true`, `eta_final_fit`, `eta_final_abs_error`, `energy_final_true`, `energy_final_fit`, `energy_monotone_*`, `rhs_coeffs`.
- Usage:

```powershell
uv run python -m experiments.polynomial_rhs_probe --degree 4 --samples 25 --steps 40 --step_size 0.05 --eta0 0.3 --run_id rhs_legendre_d4
```

## `polynomial/fit_apc_from_trace.py`

- Goal: Fit an APC basis from recent η samples in a RelaxationTracker CSV (requires tracking per‑module η columns).
- Prerequisite: construct `RelaxationTracker(log_per_eta=True)` in your experiment before relaxation.
- Usage:

```powershell
# After running a relaxation with RelaxationTracker(log_per_eta=True)
uv run python -m experiments.polynomial.fit_apc_from_trace --input logs\relaxation_trace_test.csv --module_idx 0 --degree 4 --out apc_basis_m0.json
```

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
- For deeper per-step insight, attach `EnergyBudgetTracker` to record per-term energy and gradient norms, plus backtracks and optional `contraction_margin` when `stability_guard` is enabled. The same accepted-energy history drives the uncertainty controller, so you get adaptive gate costs “for free” when `enable_uncertainty_gate=True`.
- Both trackers now emit `compute_cost` (wall-clock delta between accepted steps) and `redemption_gain`, letting you normalize ΔF improvements by compute and compare scenarios fairly.
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



