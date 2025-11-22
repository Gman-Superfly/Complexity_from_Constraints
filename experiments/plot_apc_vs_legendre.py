"""Summary plotter for logs/apc_vs_legendre_ood.csv.

Generates side-by-side bar charts for energy_final and total_backtracks by split/basis.

Usage:
    uv run python -m experiments.plot_apc_vs_legendre --input logs/apc_vs_legendre_ood.csv --save plots/apc_vs_legendre_summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "matplotlib is required. Install with `uv pip install -e .[examples]`."
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("logs/apc_vs_legendre_ood.csv"))
    parser.add_argument("--save", type=Path, default=Path("plots/apc_vs_legendre_summary.png"))
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    df = pl.read_csv(args.input)
    # ensure required columns exist
    for col in ("split", "basis", "energy_final", "total_backtracks"):
        if col not in df.columns:
            raise SystemExit(f"Missing column in CSV: {col}")

    # Aggregate (mean) in case multiple runs
    grouped = df.group_by(["split", "basis"]).agg(
        pl.col("energy_final").mean().alias("energy_final_mean"),
        pl.col("total_backtracks").mean().alias("total_backtracks_mean"),
        pl.len().alias("n_runs"),
    ).sort(["split", "basis"])

    # Prepare plot
    splits = grouped.get_column("split").unique().to_list()
    bases = grouped.get_column("basis").unique().to_list()
    # Order bases for readability
    order = [b for b in ("legendre", "apc") if b in bases] + [b for b in bases if b not in ("legendre", "apc")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax_idx, metric in enumerate(("energy_final_mean", "total_backtracks_mean")):
        ax = axes[ax_idx]
        width = 0.35
        x_positions = range(len(splits))
        for i, basis in enumerate(order):
            vals = []
            for split in splits:
                row = grouped.filter((pl.col("split") == split) & (pl.col("basis") == basis))
                if row.height == 0:
                    vals.append(float("nan"))
                else:
                    vals.append(float(row.get_column(metric)[0]))
            offset = (i - (len(order) - 1) / 2) * width
            ax.bar([x + offset for x in x_positions], vals, width=width, label=basis if ax_idx == 0 else "_nolegend_")
        ax.set_xticks(list(x_positions), splits)
        ax.set_title(metric.replace("_mean", "").replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.legend(title="basis")

    fig.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=200)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
*** End Patch ***!

