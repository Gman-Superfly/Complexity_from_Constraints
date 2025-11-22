"""Plot per-step energy budget / contraction margin traces from EnergyBudgetTracker logs.

Example:
    uv run python -m experiments.plot_energy_budget --input logs/apc_legendre_budget.csv --metric energy:local:PolynomialEnergyModule --smooth 3

Requires matplotlib (examples extra).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import polars as pl

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required. Install with `uv pip install -e .[examples]`.") from exc


def moving_average(vals: List[float], window: int) -> List[float]:
    if window <= 1:
        return vals
    out = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        segment = [v for v in vals[start : i + 1] if v is not None]
        if not segment:
            out.append(float("nan"))
        else:
            out.append(sum(segment) / len(segment))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("logs/energy_budget.csv"))
    parser.add_argument("--run-id", type=str, default=None, help="Filter by run_id (optional)")
    parser.add_argument("--metric", type=str, default="energy:local:PolynomialEnergyModule", help="Column to plot")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window")
    parser.add_argument("--save", type=Path, default=Path("plots/energy_budget_metric.png"))
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    df = pl.read_csv(args.input)
    if args.run_id:
        df = df.filter(pl.col("run_id") == args.run_id)
    if args.metric not in df.columns:
        raise SystemExit(f"Metric column '{args.metric}' not found in {args.input}")
    if "step" not in df.columns:
        df = df.with_row_index(name="step")

    steps = df.get_column("step").to_list()
    vals = df.get_column(args.metric).to_list()
    smooth_vals = moving_average(vals, max(1, args.smooth))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, smooth_vals, label=f"{args.metric} (smooth={args.smooth})", color="#005bbb")
    ax.set_xlabel("step")
    ax.set_ylabel(args.metric)
    ax.grid(alpha=0.3)
    if args.run_id:
        ax.set_title(f"{args.metric} (run_id={args.run_id})")
    else:
        ax.set_title(args.metric)
    ax.legend()
    fig.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=200)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

