from __future__ import annotations

import argparse
import json
from typing import List

import polars as pl

from modules.polynomial.apc import compute_apc_basis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="RelaxationTracker CSV with eta:<idx> columns")
    parser.add_argument("--module_idx", type=int, required=True, help="Module index to extract eta:<idx>")
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--out", type=str, required=True, help="Output JSON file to write apc_basis (matrix)")
    parser.add_argument("--window", type=int, default=256, help="Use last N rows for APC fit (default 256)")
    args = parser.parse_args()

    col = f"eta:{int(args.module_idx)}"
    df = pl.read_csv(args.input)
    if col not in df.columns:
        raise SystemExit(f"Column {col} not found in {args.input}. Ensure RelaxationTracker(log_per_eta=True).")
    # Use last N samples
    if args.window > 0 and len(df) > args.window:
        df = df.tail(args.window)
    etas = [float(v) for v in df[col].to_list()]
    xis = [2.0 * e - 1.0 for e in etas]
    B = compute_apc_basis(xis, degree=int(args.degree))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"apc_basis": B}, f)
    print(f"Wrote APC basis (degree={args.degree}) to {args.out} using {len(xis)} samples from {col}")


if __name__ == "__main__":
    main()


