"""Sparse top-2 branching demo to illustrate coexistence of multiple ends.

At each depth, branches propose two candidate expansions with random gains.
We use a simple Gumbel race over (k * net) to pick the global top-2 to open.
We log ends_count (branches surviving to depth L), branching_rate, and hazard stats.
"""

from __future__ import annotations

import argparse
from typing import List, Dict, Any, Tuple
import math
import numpy as np

from modules.gating.energy_gating import EnergyGatingModule
from cf_logging.metrics_log import log_records
from cf_logging.observability import GatingMetricsLogger


def gumbel_noise(rng: np.random.Generator) -> float:
    u = rng.uniform(0.0, 1.0)
    return -math.log(-math.log(max(1e-12, min(1.0 - 1e-12, u))))


def run(
    depth: int,
    trials: int,
    cost: float,
    k: float,
    gain_mean: float,
    gain_std: float,
    seed: int | None,
    log_gating_metrics: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    ends: List[int] = []
    branching_rates: List[float] = []
    hazards: List[float] = []
    gating_logger = GatingMetricsLogger(run_id=f"branching_cost_{cost}") if log_gating_metrics else None

    for _ in range(trials):
        # start with one branch at depth 0
        branches: List[Tuple[float, float]] = [(0.0, 0.0)]  # (cum_gain, last_hazard)
        opened_total = 0
        for _d in range(depth):
            # propose two candidates per branch
            candidates: List[Tuple[float, float, float, int]] = []  # (score_for_race, gain, hazard, branch_idx)
            for idx, (cum_gain, _hz) in enumerate(branches):
                # gains are random; positive mean encourages occasional coexistence
                g1 = float(rng.normal(gain_mean, gain_std))
                g2 = float(rng.normal(gain_mean, gain_std))
                # set up gating for each candidate
                gate1 = EnergyGatingModule(gain_fn=lambda _x, g=g1: g, cost=cost, k=k, use_hazard=True)
                gate2 = EnergyGatingModule(gain_fn=lambda _x, g=g2: g, cost=cost, k=k, use_hazard=True)
                lam1 = gate1.hazard_rate(None)
                lam2 = gate2.hazard_rate(None)
                hazards.append(lam1)
                hazards.append(lam2)
                if gating_logger is not None:
                    gating_logger.record(hazard=float(lam1), eta_gate=gate1.compute_eta(None), redemption=g1 - cost, good=(g1 - cost > 0))
                    gating_logger.record(hazard=float(lam2), eta_gate=gate2.compute_eta(None), redemption=g2 - cost, good=(g2 - cost > 0))
                # race score approximates argmax over (k * net + gumbel)
                score1 = k * (g1 - cost) + gumbel_noise(rng)
                score2 = k * (g2 - cost) + gumbel_noise(rng)
                candidates.append((score1, g1, lam1, idx))
                candidates.append((score2, g2, lam2, idx))
            # select global top-2 candidates (sparse branching)
            if not candidates:
                branches = []
                break
            candidates.sort(key=lambda t: t[0], reverse=True)
            chosen = candidates[:2]
            opened_total += len(chosen)
            # advance: each chosen candidate becomes a branch at next depth
            new_branches: List[Tuple[float, float]] = []
            for _score, gain, hz, parent_idx in chosen:
                parent_cum_gain, _ = branches[parent_idx]
                new_branches.append((parent_cum_gain + gain, hz))
            branches = new_branches
        ends.append(len(branches))
        branching_rates.append(opened_total / float(max(1, depth)))

    rows.append({
        "depth": int(depth),
        "trials": int(trials),
        "cost": float(cost),
        "k": float(k),
        "gain_mean": float(gain_mean),
        "gain_std": float(gain_std),
        "ends_count_mean": float(np.mean(ends)),
        "branching_rate_mean": float(np.mean(branching_rates)),
        "hazard_mean": float(np.mean(hazards)) if hazards else float("nan"),
    })

    out = log_records("branching_coexistence", rows)
    print(f"Wrote {len(rows)} rows to {out}")
    if gating_logger is not None:
        gating_logger.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--cost", type=float, default=0.1)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--gain_mean", type=float, default=0.05)
    parser.add_argument("--gain_std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log_gating_metrics", action="store_true", help="Log hazard/η for branch gating decisions.")
    args = parser.parse_args()
    run(
        depth=args.depth,
        trials=args.trials,
        cost=args.cost,
        k=args.k,
        gain_mean=args.gain_mean,
        gain_std=args.gain_std,
        seed=args.seed,
        log_gating_metrics=args.log_gating_metrics,
    )


if __name__ == "__main__":
    main()


