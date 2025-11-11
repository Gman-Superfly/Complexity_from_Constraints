"""Offline analysis for first three experiments from CSV logs."""

from __future__ import annotations

from pathlib import Path
import polars as pl


def summarize_landau() -> str:
    p = Path("logs/landau_sweep.csv")
    if not p.exists():
        return "landau_sweep: no data"
    df = pl.read_csv(p)
    a0 = float(df["a"].min())
    a1 = float(df["a"].max())
    eta_mean = float(df["eta_final"].mean())
    F_mean = float(df["F_final"].mean())
    return f"landau_sweep: a∈[{a0:.2f},{a1:.2f}] η_final≈{eta_mean:.3f} F_final≈{F_mean:.3f}"


def summarize_connectivity() -> str:
    p = Path("logs/non_local_connectivity_threshold_shift.csv")
    if not p.exists():
        return "non_local_connectivity_threshold_shift: no data"
    df = pl.read_csv(p)
    p_no = float(df["p_no_shortcuts"].mean())
    p_yes = float(df["p_with_shortcuts"].mean())
    return f"connectivity_threshold: p_no_shortcuts≈{p_no:.3f}, p_with_shortcuts≈{p_yes:.3f}, Δ≈{(p_no-p_yes):.3f}"


def summarize_redemption() -> str:
    p = Path("logs/sequence_redemption.csv")
    if not p.exists():
        return "sequence_redemption: no data"
    df = pl.read_csv(p)
    red = float(df["redemption_score"].mean())
    return f"sequence_redemption: redemption_score≈{red:.4f}"


def main() -> None:
    print(summarize_landau())
    print(summarize_connectivity())
    print(summarize_redemption())


if __name__ == "__main__":
    main()


