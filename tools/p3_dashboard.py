"""Streamlit dashboard for Phase-2 (Precision Layer) telemetry.

Usage (Windows PowerShell):
    uv run streamlit run tools/p3_dashboard.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import polars as pl
import streamlit as st

LOG_DIR = Path("logs")


def _available_runs(df: pl.DataFrame) -> list[str]:
    if "run_id" not in df.columns:
        return []
    return sorted(df.get_column("run_id").unique().to_list())


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pl.DataFrame:
    path = LOG_DIR / name
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_csv(path)
    except Exception as exc:
        st.warning(f"Failed to read {path}: {exc}")
        return pl.DataFrame()


def filter_by_run(df: pl.DataFrame, run_id: Optional[str]) -> pl.DataFrame:
    if df.is_empty() or run_id is None or "run_id" not in df.columns:
        return df
    return df.filter(pl.col("run_id") == run_id)


def to_pandas(df: pl.DataFrame, columns: Optional[list[str]] = None):
    if df.is_empty():
        return None
    if columns:
        cols = [c for c in columns if c in df.columns]
        if cols:
            return df.select(cols).to_pandas()
    return df.to_pandas()


def draw_energy_panel(df: pl.DataFrame) -> None:
    st.subheader("Energy Timeline")
    if df.is_empty():
        st.info("No energy budget data found in logs/energy_budget.csv")
        return

    pandas_df = to_pandas(
        df,
        columns=[
            "step",
            "energy",
            "delta_energy",
            "contraction_margin",
            "last_backtracks",
        ],
    )
    if pandas_df is None:
        st.warning("Unable to convert energy data to display.")
        return

    st.line_chart(
        pandas_df,
        x="step",
        y=["energy", "delta_energy"],
        height=300,
    )

    col1, col2, col3 = st.columns(3)
    try:
        latest = df.sort("step").tail(1)
        if not latest.is_empty():
            col1.metric("Energy", f"{latest['energy'][0]:.3e}")
            if "contraction_margin" in latest.columns:
                col2.metric("Contraction Margin", f"{latest['contraction_margin'][0]:.3e}")
            if "last_backtracks" in latest.columns:
                col3.metric("Backtracks", int(latest["last_backtracks"][0]))
    except Exception:
        pass


def draw_precision_panel(df: pl.DataFrame) -> None:
    st.subheader("Precision / Stiffness Summary")
    needed = {"precision:min", "precision:mean", "precision:max", "step"}
    if df.is_empty() or not needed.issubset(set(df.columns)):
        st.info("Precision tracking columns not found. Attach EnergyBudgetTracker with precision logging.")
        return
    
    # Summary statistics
    pandas_df = to_pandas(
        df.select(["step", "precision:min", "precision:mean", "precision:max"])
    )
    if pandas_df is None:
        st.warning("Unable to render precision statistics.")
        return
    st.line_chart(
        pandas_df,
        x="step",
        y=["precision:min", "precision:mean", "precision:max"],
        height=250,
    )
    
    # Per-η precision traces (if available)
    per_eta_cols = [c for c in df.columns if c.startswith("precision:") and c.split(":")[-1].isdigit()]
    if per_eta_cols:
        st.subheader("Per-η Precision Traces")
        per_eta_df = to_pandas(df.select(["step"] + per_eta_cols))
        if per_eta_df is not None:
            st.line_chart(per_eta_df, x="step", height=300)
            
            # Distribution at final step
            latest = df.sort("step").tail(1)
            if not latest.is_empty():
                st.subheader("Final Precision Distribution")
                vals = [latest[col][0] for col in per_eta_cols if col in latest.columns]
                st.bar_chart(vals)


def draw_free_energy_panel(df: pl.DataFrame) -> None:
    st.subheader("Free Energy Decomposition (F = U - T·S)")
    needed = {"U_internal_energy", "S_entropy", "F_free_energy", "step"}
    if df.is_empty() or not needed.issubset(set(df.columns)):
        st.info("Free energy decomposition not logged. Enable log_free_energy_decomposition in EnergyBudgetTracker.")
        return
    
    # Plot U, S, F trajectories
    free_energy_df = to_pandas(
        df.select(["step", "U_internal_energy", "S_entropy", "F_free_energy"])
    )
    if free_energy_df is not None:
        st.line_chart(
            free_energy_df,
            x="step",
            y=["U_internal_energy", "S_entropy", "F_free_energy"],
            height=300,
        )
        
        # Latest values
        latest = df.sort("step").tail(1)
        if not latest.is_empty():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("U (Internal)", f"{latest['U_internal_energy'][0]:.3e}")
            col2.metric("S (Entropy)", f"{latest['S_entropy'][0]:.3e}")
            col3.metric("F (Free)", f"{latest['F_free_energy'][0]:.3e}")
            if "T_temperature" in latest.columns:
                col4.metric("T (Temp)", f"{latest['T_temperature'][0]:.2f}")


def draw_confidence_panel(df: pl.DataFrame) -> None:
    st.subheader("Confidence Trajectory")
    if df.is_empty() or "confidence:c" not in df.columns:
        st.info("No confidence data found. Enable enable_confidence_logging in the coordinator.")
        return
    pandas_df = to_pandas(df, columns=["step", "confidence:c"])
    if pandas_df is None:
        st.warning("Unable to render confidence.")
        return
    st.line_chart(pandas_df, x="step", y=["confidence:c"], height=200)


def draw_delta_f_histogram(df: pl.DataFrame) -> None:
    st.subheader("ΔF / ΔEnergy Histogram (Distribution)")
    if df.is_empty() or "delta_energy" not in df.columns:
        st.info("No delta_energy field found. Ensure RelaxationTracker is attached.")
        return
    pdf = to_pandas(df, columns=["delta_energy"])  # type: ignore[assignment]
    if pdf is None:
        st.warning("Unable to render histogram.")
        return
    import numpy as np
    import pandas as pd
    vals = pdf["delta_energy"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        st.info("No finite delta_energy values to plot.")
        return
    counts, bin_edges = np.histogram(vals, bins=40)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_df = pd.DataFrame({"bin": centers, "count": counts})
    st.bar_chart(hist_df, x="bin", y="count", height=200)


def draw_acceptance_panel(df: pl.DataFrame) -> None:
    st.subheader("Acceptance & Backtracking")
    if df.is_empty():
        st.info("No acceptance data available.")
        return
    
    # Backtrack counts over time
    if "last_backtracks" in df.columns:
        backtrack_df = to_pandas(df, columns=["step", "last_backtracks"])
        if backtrack_df is not None:
            st.line_chart(backtrack_df, x="step", y="last_backtracks", height=200)
    
    # Acceptance reason distribution
    if "acceptance_reason" in df.columns:
        reasons = df.get_column("acceptance_reason").value_counts()
        if not reasons.is_empty():
            st.write("**Acceptance Reason Counts:**")
            st.dataframe(reasons.to_pandas())


def draw_escape_events_panel(df: pl.DataFrame) -> None:
    st.subheader("Escape Events (Noise-triggered Transitions)")
    if df.is_empty() or "escape_event_count" not in df.columns:
        st.info("No escape events data found. Enable enable_escape_event_logging in the coordinator.")
        return
    pandas_df = to_pandas(df, columns=["step", "escape_event_count"])
    if pandas_df is None:
        st.warning("Unable to render escape events.")
        return
    st.line_chart(pandas_df, x="step", y=["escape_event_count"], height=200)


def draw_gating_panel(df: pl.DataFrame) -> None:
    st.subheader("Gating Metrics")
    if df.is_empty():
        st.info("No gating metrics found in logs/gating_metrics.csv")
        return
    pandas_df = to_pandas(
        df,
        columns=["step", "hazard", "eta_gate", "redemption"],
    )
    if pandas_df is None:
        st.warning("Unable to render gating metrics.")
        return
    st.line_chart(pandas_df, x="step", y=["hazard", "eta_gate", "redemption"], height=250)


def draw_sensitivity_panel(df: pl.DataFrame) -> None:
    st.subheader("Sensitivity Probes (Dispersion)")
    if df.is_empty() or "sensitivity:dispersion" not in df.columns:
        st.info("No sensitivity probe data found. Enable enable_sensitivity_probes in the coordinator.")
        return
    pandas_df = to_pandas(df, columns=["step", "sensitivity:dispersion"])
    if pandas_df is None:
        st.warning("Unable to render sensitivity data.")
        return
    st.line_chart(pandas_df, x="step", y=["sensitivity:dispersion"], height=250)


def main() -> None:
    st.set_page_config(page_title="Precision Layer Dashboard", layout="wide")
    st.title("Precision Layer Dashboard (Phase 2)")
    st.caption("Reads `logs/*.csv` written by RelaxationTracker / EnergyBudgetTracker / GatingMetricsLogger.")

    with st.sidebar:
        st.header("Run Selection")
        energy_df = load_csv("energy_budget.csv")
        gating_df = load_csv("gating_metrics.csv")
        relaxation_df = load_csv("relaxation_trace.csv")

        run_ids = sorted(set(_available_runs(energy_df) + _available_runs(gating_df) + _available_runs(relaxation_df)))
        run_id = st.selectbox("Run ID", run_ids) if run_ids else None
        st.markdown("**Data sources**")
        st.write(
            {
                "energy_budget": str((LOG_DIR / "energy_budget.csv").resolve()),
                "gating_metrics": str((LOG_DIR / "gating_metrics.csv").resolve()),
                "relaxation_trace": str((LOG_DIR / "relaxation_trace.csv").resolve()),
            }
        )

    filtered_energy = filter_by_run(energy_df, run_id)
    draw_energy_panel(filtered_energy)
    
    st.divider()
    draw_free_energy_panel(filtered_energy)
    
    st.divider()
    draw_precision_panel(filtered_energy)
    
    st.divider()
    draw_sensitivity_panel(filtered_energy)

    st.divider()
    draw_acceptance_panel(filtered_energy)

    st.divider()
    draw_escape_events_panel(filtered_energy)

    st.divider()
    draw_confidence_panel(filtered_energy)

    st.divider()
    draw_delta_f_histogram(filtered_energy)

    st.divider()
    filtered_gating = filter_by_run(gating_df, run_id)
    draw_gating_panel(filtered_gating)

    st.divider()
    st.subheader("Raw Energy Budget Table")
    if filtered_energy.is_empty():
        st.info("No rows to display.")
    else:
        limit = st.slider("Rows", min_value=100, max_value=5000, value=500, step=100)
        st.dataframe(filtered_energy.tail(limit).to_pandas(), use_container_width=True)


if __name__ == "__main__":
    LOG_DIR.mkdir(exist_ok=True)
    main()

