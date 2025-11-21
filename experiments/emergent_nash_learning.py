"""Emergent Nash learning experiment with simple regret metrics."""

from __future__ import annotations

import argparse
from typing import List, Dict, Any, Tuple
import numpy as np

from modules.game.emergent_nash import symmetric_2x2_payoff, strategy_regret, replicator_step
from cf_logging.metrics_log import log_records


def run(trials: int, steps: int, lr: float, seed: int | None) -> None:
    rng = np.random.default_rng(seed)
    A = symmetric_2x2_payoff()
    rows: List[Dict[str, Any]] = []
    for t in range(trials):
        p_row = float(rng.uniform(0, 1))
        p_col = float(rng.uniform(0, 1))
        for s in range(steps):
            # compute payoffs against opponent mix
            u0_row = (1 - p_col) * A[0, 0] + p_col * A[0, 1]
            u1_row = (1 - p_col) * A[1, 0] + p_col * A[1, 1]
            u0_col = (1 - p_row) * A[0, 0] + p_row * A[1, 0]
            u1_col = (1 - p_row) * A[0, 1] + p_row * A[1, 1]
            # replicator-like updates
            p_row = replicator_step(p_row, (u0_row, u1_row), lr=lr)
            p_col = replicator_step(p_col, (u0_col, u1_col), lr=lr)
            reg_row = strategy_regret(A, p_row, p_col)
            reg_col = strategy_regret(A.T, p_col, p_row)  # symmetry
            rows.append({
                "trial": int(t),
                "step": int(s),
                "p_row": float(p_row),
                "p_col": float(p_col),
                "reg_row": float(reg_row),
                "reg_col": float(reg_col),
                "lr": float(lr),
            })
    out = log_records("emergent_nash_learning", rows)
    print(f"Wrote {len(rows)} rows to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    run(trials=args.trials, steps=args.steps, lr=args.lr, seed=args.seed)


if __name__ == "__main__":
    main()


