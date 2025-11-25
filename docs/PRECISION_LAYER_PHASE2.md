# Phase 2: Precision Layer & Stability

This document details the features implemented in Phase 2, focusing on precision-aware optimization, free-energy guards, and advanced observability.

## 1. Free-Energy Guard (F = U - T·S)

We have moved beyond simple energy minimization (U) to Free Energy minimization (F). This allows the system to accept steps that increase internal energy (U) if they are accompanied by a sufficient increase in entropy (S), effectively "paying" for the energy cost with uncertainty.

### Concept
$$ F(\eta) = U(\eta) - T \cdot S(\eta) $$

- **U (Internal Energy):** The standard total energy (constraints + couplings).
- **S (Entropy):** Shannon-like entropy for order parameters in $(0,1)$.
  $$ S = -\sum_i [\eta_i \ln(\eta_i) + (1-\eta_i) \ln(1-\eta_i)] $$
- **T (Temperature):** Controls the trade-off. Higher T allows more exploration (energy increases).

### Usage
Enable in `EnergyCoordinator`:

```python
coord = EnergyCoordinator(
    # ... modules/couplings ...
    use_free_energy_guard=True,
    free_energy_temperature=1.0,  # Adjust T
    free_energy_epsilon=1e-6,     # Minimum ΔF decrease to accept
)
```

## 2. Early-Stop with Patience

To avoid wasting compute on converged states, we introduced an early-stopping mechanism based on energy stabilization.

### Logic
- Monitors the absolute change in energy $|\Delta E|$ between steps.
- If $|\Delta E| < \text{threshold}$ for `patience` consecutive steps, the relaxation terminates.
- Reason logged as `early_stop_converged`.

### Usage
```python
coord = EnergyCoordinator(
    # ...
    enable_early_stop=True,
    early_stop_patience=5,        # Number of stable steps required
    early_stop_delta_threshold=1e-6,
)
```

## 3. Sensitivity Probes (Dispersion)

We quantify the "brittleness" or "uncertainty" of the solution by measuring the dispersion of the trajectory under noise.

### Concept
- **Dispersion ($\rho$):** The mean standard deviation of order parameters over a sliding window of accepted states.
- High dispersion indicates a flat valley or instability (high sensitivity).
- Low dispersion indicates a sharp, stable basin (high precision).

### Usage
```python
coord = EnergyCoordinator(
    # ...
    enable_sensitivity_probes=True,
    sensitivity_probe_window=10,  # Window size for std dev calculation
)
```

Logged as `sensitivity:dispersion` in `EnergyBudgetTracker`.

## 4. Observability Enhancements (P3)

The `EnergyBudgetTracker` and `p3_dashboard.py` have been enhanced to visualize these new metrics.

### New Logs
- **Free Energy Components:** `U_internal_energy`, `S_entropy`, `F_free_energy`, `T_temperature`.
- **Precision:** `precision:min`, `precision:max`, `precision:mean`, and optional per-$\eta$ traces.
- **Sensitivity:** `sensitivity:dispersion`.
- **Acceptance:** `acceptance_reason`, `last_backtracks`.

### Dashboard Panels
- **Free Energy Decomposition:** Visualizes the trade-off between U and S.
- **Precision/Stiffness:** Tracks the curvature of the energy landscape.
- **Sensitivity Probes:** Monitors trajectory dispersion.
- **Acceptance & Backtracking:** diagnosing optimization health.
- **Constraint Violation Rate (h):** When `coord.constraints` provides `constraint_violation_count` and `total_constraints_checked`, `EnergyBudgetTracker` logs `info:constraint_violation_rate` for dashboard visualization (see `docs/INFORMATION_METRICS.md`).

## Validation
Run the following tests to verify Phase 2 features:
```powershell
uv run -m pytest tests/test_free_energy_guard.py -v
uv run -m pytest tests/test_early_stop_patience.py -v
uv run -m pytest tests/test_sensitivity_probes.py -v
uv run -m pytest tests/test_precision_per_eta_logging.py -v
uv run -m pytest tests/test_free_energy_decomposition.py -v
```

