from __future__ import annotations

import argparse
from typing import List

import polars as pl
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="logs/energy_budget.csv")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.9, help="Warn threshold to draw as a horizontal line")
    args = parser.parse_args()

    df = pl.read_csv(args.input)
    if args.run_id is not None:
        df = df.filter(pl.col("run_id") == args.run_id)

    # Identify polynomial correlation columns
    corr_cols = [c for c in df.columns if c.startswith("poly_corr_max:")]
    warn_cols = [c for c in df.columns if c.startswith("poly_corr_warn:")]
    if not corr_cols:
        print("No poly_corr_max:* columns found; nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in sorted(corr_cols):
        ax.plot(df[c], label=c)
    # Draw threshold
    ax.axhline(y=float(args.threshold), color="tab:red", linestyle="--", alpha=0.5, label=f"threshold={args.threshold}")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("step")
    ax.set_ylabel("poly_corr_max")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()


