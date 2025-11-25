# Precision Layer Dashboard (Phase 2 P3)

Mini dashboard, but I really don't like it, for sure this won't be a full feature, I would prefer not to do this, for now streamlit can live...

Dashboard script: `tools/p3_dashboard.py`

Use Streamlit + Polars to visualize precision, stability, and energy telemetry recorded by the existing loggers.

---

## 1. Prerequisites

- Logs written by:
  - `RelaxationTracker` (`logs/relaxation_trace.csv`)
  - `EnergyBudgetTracker` (`logs/energy_budget.csv`) — enable precision logging automatically with current coordinator
  - `GatingMetricsLogger` (`logs/gating_metrics.csv`)
- Environment: Windows PowerShell, `uv` virtualenv, Polars, Streamlit (already in `pyproject.toml`).
- Dependencies: `polars`, `streamlit` (see `requirements.txt` optional section). Install by running `uv sync --extra dev` or add `streamlit` manually.

Optional:
- Metric-aware precision logging requires modules implementing `SupportsPrecision` (the coordinator cache is automatic).

---

## 2. Running the dashboard (Windows PowerShell)

```powershell
# Ensure dependencies are synced
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv sync --extra dev

# Launch the dashboard
uv run streamlit run tools/p3_dashboard.py
```

This opens a local Streamlit app (default: http://localhost:8501). It reads the CSV logs in `logs/`. The app caches data per run; refresh the browser to reload files.

---

## 3. Features

1. **Energy Timeline**
   - Total energy and ΔF vs step
   - Contraction margin, backtracks metrics
   - Accept/reject history (ΔF overlay)

2. **Precision / Stiffness Pane**
   - Plots `precision:min`, `precision:mean`, `precision:max` over steps (from EnergyBudgetTracker)

3. **Gating Metrics**
   - Hazard, η_gate, redemption time series (requires GatingMetricsLogger)

4. **Raw Table**
   - Tail of `energy_budget` CSV to inspect per-term energies/grad norms, precision stats, adapter metrics

5. **Run Selector**
   - Sidebar drop-down combining run IDs from all available logs
   - Paths to the CSVs for quick inspection/debugging

---

## 4. Data Sources & Format

All telemetry is stored under `logs/` via Polars CSVs:

| File | Source | Key columns |
|------|--------|-------------|
| `energy_budget.csv` | EnergyBudgetTracker | `step`, `energy`, `delta_energy`, `precision:min/mean/max`, `contraction_margin`, `last_backtracks`, `run_id`, per-term energies (`energy:coup:*`, `energy:local:*`), grad norms, adapter metrics |
| `gating_metrics.csv` | GatingMetricsLogger | `step`, `hazard`, `eta_gate`, `redemption`, `is_good`, `run_id` |
| `relaxation_trace.csv` | RelaxationTracker | `step`, `energy`, `delta_energy`, `compute_cost`, `redemption_gain`, optional per-η columns, `run_id` |

The dashboard reads the entire CSV each time (cached by Streamlit). For large logs, use the tail slider to limit table output.

---

## 5. Extending the dashboard

- Add new columns (e.g., `sharpness_proxy`, `escape_events`) via `EnergyBudgetTracker` or custom loggers; they will appear automatically in the table and can be plotted by adding new panes.
- For real-time streaming, wrap the existing CSV reads in a periodic `st.experimental_rerun()` loop or integrate a lightweight WebSocket later. The current design intentionally stays simple (file-based).
- You can add additional pages or tabs with `st.tabs(["Energy", "Precision", ...])` if the layout becomes crowded.
- Constraint Violation Rate (h): when `EnergyBudgetTracker` receives `constraint_violation_count` and `total_constraints_checked` in `coord.constraints`, it emits `info:constraint_violation_rate`. Add a panel to visualize this span-level error rate (origin: Information Structures; see `docs/INFORMATION_METRICS.md`).

---

## 6. Troubleshooting

- **No data**: Ensure experiments were run with `EnergyBudgetTracker.attach(coord)` and `tracker.flush()` after relaxation. Files must exist in `logs/`.
- **Polars schema errors**: The logger already aligns schemas when appending; ensure any manual edits preserve column structure or delete the CSV if corrupted.
- **Streamlit caching stale data**: Click “Rerun” (top right) or reload the browser tab.
- **Performance**: For very large logs (>200k rows), consider pre-filtering by run ID using `pl.scan_csv` and `collect()` only the necessary subset.

---

## 7. References

- `tools/p3_dashboard.py`: Streamlit application source
- `docs/ISO-ENERGY_ORTHOGONAL_NOISE.md`: Noise modes and precision-aware redistribution
- `docs/PRECISION_LAYER.md`: Precision Layer concepts and flags
- `docs/METRIC_AWARE_NOISE_CONTROLLER.md`: Metric-aware projection + precision weighting


