"""Landau toy experiment: disorder→order sweep over parameter a.

Windows example:
  uv run python -m experiments.landau_sweep --a_min -1.0 --a_max 1.0 --num 21
"""

from __future__ import annotations

import argparse
from typing import List, Dict, Any
import numpy as np

from core.energy import landau_free_energy, descend_free_energy
from logging.metrics_log import log_records


def run(a_min: float, a_max: float, num: int, b: float, eta0: float, lr: float, steps: int) -> None:
    assert num >= 2, "num must be >= 2"
    a_vals = np.linspace(a_min, a_max, num=num)
    rows: List[Dict[str, Any]] = []
    for a in a_vals:
        eta_final, F_final = descend_free_energy(eta0=eta0, a=float(a), b=b, learning_rate=lr, steps=steps)
        rows.append({
            "a": float(a),
            "b": float(b),
            "eta0": float(eta0),
            "eta_final": float(eta_final),
            "F_final": float(F_final),
            "lr": float(lr),
            "steps": int(steps),
        })
    out = log_records("landau_sweep", rows)
    print(f"Wrote {len(rows)} rows to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_min", type=float, default=-1.0)
    parser.add_argument("--a_max", type=float, default=1.0)
    parser.add_argument("--num", type=int, default=21)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--eta0", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    run(a_min=args.a_min, a_max=args.a_max, num=args.num, b=args.b, eta0=args.eta0, lr=args.lr, steps=args.steps)


if __name__ == "__main__":
    main()


