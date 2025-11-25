"""Validate ΔF/ΔEnergy histogram negative skewness from CSV logs.

Usage (PowerShell):
    uv run python tools/validate_delta_f.py --file logs/energy_budget.csv
    uv run python tools/validate_delta_f.py --file logs/relaxation_trace.csv --column delta_energy

By default, tries F_free_energy diffs if available; falls back to delta_energy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import polars as pl


def skewness(values: List[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    n = len(vals)
    if n < 3:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = (var if var > 0.0 else 1e-18) ** 0.5
    m3 = sum((v - mean) ** 3 for v in vals) / n
    return float(m3 / (std ** 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to CSV log (energy_budget.csv or relaxation_trace.csv)")
    ap.add_argument("--column", default=None, help="Override column to use (default: F diffs if present else delta_energy)")
    ap.add_argument("--bins", type=int, default=40, help="Histogram bins (unused in validation summary)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"FAIL: file not found: {path}")
        return 2
    try:
        df = pl.read_csv(path)
    except Exception as exc:
        print(f"FAIL: could not read {path}: {exc}")
        return 2

    col = args.column
    if col is None:
        if "F_free_energy" in df.columns:
            # compute diffs
            fvals = df.get_column("F_free_energy").to_list()
            deltas = [float(fvals[i] - fvals[i - 1]) for i in range(1, len(fvals))]
        elif "delta_energy" in df.columns:
            deltas = [float(x) for x in df.get_column("delta_energy").to_list() if x is not None]
        else:
            print("FAIL: neither F_free_energy nor delta_energy column present")
            return 2
    else:
        if col not in df.columns:
            print(f"FAIL: column '{col}' not in CSV")
            return 2
        deltas = [float(x) for x in df.get_column(col).to_list() if x is not None]

    # Basic checks
    if not deltas:
        print("FAIL: no delta values")
        return 2
    non_pos_ratio = sum(1 for d in deltas if d <= 0.0) / max(1, len(deltas))
    sk = skewness(deltas)
    status = "OK" if (non_pos_ratio >= 0.7 and sk <= 0.05) else "WARN"
    print(
        {
            "status": status,
            "count": len(deltas),
            "non_pos_ratio": round(non_pos_ratio, 4),
            "skewness": round(sk, 6),
            "file": str(path),
            "column": col or ("F_free_energy diffs" if "F_free_energy" in df.columns else "delta_energy"),
        }
    )
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
