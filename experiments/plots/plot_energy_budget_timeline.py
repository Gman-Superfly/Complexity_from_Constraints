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
    parser.add_argument("--prefix", type=str, default="energy:", help="Column prefix to plot (default 'energy:')")
    args = parser.parse_args()

    df = pl.read_csv(args.input)
    if args.run_id is not None:
        df = df.filter(pl.col("run_id") == args.run_id)

    cols = [c for c in df.columns if c.startswith(args.prefix)]
    if not cols:
        print(f"No columns with prefix {args.prefix!r} found in {args.input}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in sorted(cols):
        ax.plot(df[c], label=c)
    ax.set_xlabel("step")
    ax.set_ylabel(args.prefix.rstrip(":"))
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()


