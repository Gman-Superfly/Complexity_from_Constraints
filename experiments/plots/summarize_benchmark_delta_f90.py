from __future__ import annotations

import argparse
import polars as pl
from typing import List, Optional
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="logs/benchmark_delta_f90.csv")
    parser.add_argument("--run_id_contains", type=str, default=None, help="Filter rows whose run_id contains this substring")
    parser.add_argument("--configs", nargs="*", default=None, help="Optional list of configs to include")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV path to write the summary")
    args = parser.parse_args()

    df = pl.read_csv(args.input)

    exprs: List[pl.Expr] = []
    if args.run_id_contains is not None:
        df = df.filter(pl.col("run_id").str.contains(args.run_id_contains))
    if args.configs:
        df = df.filter(pl.col("config").is_in(args.configs))

    # Include optional sweep parameters if present
    group_cols: List[str] = ["run_id", "config"]
    optional_cols = ["sg_rho", "sg_dw", "adapter"]
    for col in optional_cols:
        if col in df.columns:
            group_cols.append(col)

    # Aggregate KPIs
    summary = (
        df.group_by(group_cols)
        .agg(
            pl.len().alias("n"),
            pl.col("delta_f90_steps").mean().alias("delta_f90_mean"),
            pl.col("delta_f90_steps").median().alias("delta_f90_median"),
            pl.col("wall_time_sec").mean().alias("wall_time_mean_s"),
            pl.col("wall_time_sec").median().alias("wall_time_median_s"),
            pl.col("energy_final").mean().alias("energy_final_mean"),
            pl.col("compute_cost").mean().alias("compute_cost_mean_s"),
            pl.col("redemption_gain").mean().alias("redemption_gain_mean"),
            pl.col("total_backtracks").mean().alias("total_backtracks_mean") if "total_backtracks" in df.columns else pl.lit(None).alias("total_backtracks_mean"),
            pl.col("last_backtracks").mean().alias("last_backtracks_mean") if "last_backtracks" in df.columns else pl.lit(None).alias("last_backtracks_mean"),
        )
        .sort(group_cols)
    )

    if args.out:
        summary.write_csv(args.out)
        print(f"Wrote summary to {args.out}")
    else:
        # Pretty print to stdout with Windows-safe ASCII formatting
        try:
            # Prefer ASCII table format to avoid unicode box-drawing
            with pl.Config(tbl_format="ASCII_MARKDOWN"):
                txt = str(summary)
        except Exception:
            txt = str(summary)
        try:
            print(txt)
        except UnicodeEncodeError:
            # Fallback: write bytes with replacement to bypass cp1252 console errors
            sys.stdout.buffer.write(txt.encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()


