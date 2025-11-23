from __future__ import annotations

import argparse
import polars as pl
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="logs/energy_budget.csv")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = pl.read_csv(args.input)
    if args.run_id is not None:
        df = df.filter(pl.col("run_id") == args.run_id)

    cols = ["agm:rate", "agm:variance", "agm:trend", "agm:oscillation",
            "uncertainty:epistemic", "uncertainty:aleatoric", "uncertainty:total",
            "uncertainty:exploration_boost"]
    present = [c for c in cols if c in df.columns]
    if not present:
        print("No AGM/uncertainty columns present in input.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in present:
        ax.plot(df[c], label=c)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.legend(loc="best")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()


