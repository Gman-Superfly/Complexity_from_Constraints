"""Energy-gated expansion experiment on sequences with planted mistakes.

Varies expansion cost and measures:
- expansion_rate: fraction of trials where gate opens
- redemption_mean: improvement in full-sequence η when expansion occurs
"""

from __future__ import annotations

import argparse
from typing import List, Dict, Any, Callable
import numpy as np

from modules.sequence.monotonic_eta import sample_monotonicity_score
from modules.gating.energy_gating import EnergyGatingModule
from logging.metrics_log import log_records


def sequence_with_mistake(n: int, pos: int, noise: float, rng: np.random.Generator) -> List[float]:
    seq = np.linspace(0.0, 1.0, num=n).tolist()
    if 0 <= pos < n:
        seq[pos] -= 0.5  # inject dip
    if noise > 0.0:
        seq = (np.asarray(seq) + rng.normal(0.0, noise, size=n)).tolist()
    return [float(x) for x in seq]


def seq_gain_fn_factory(seq: List[float], repair_idx: int, samples: int = 512) -> Callable[[Any], float]:
    base_eta = sample_monotonicity_score(seq, samples=samples)
    def repaired_eta() -> float:
        s = list(seq)
        if 1 <= repair_idx < len(s):
            s[repair_idx] = max(s[repair_idx], s[repair_idx - 1])  # minimal local repair
        return sample_monotonicity_score(s, samples=samples)
    def gain_fn(_: Any) -> float:
        return float(repaired_eta() - base_eta)
    return gain_fn


def run(n: int, mistake_pos: int, trials: int, costs: List[float], noise: float, seed: int | None) -> None:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for c in costs:
        expanded = 0
        total = 0
        red_scores: List[float] = []
        hazards: List[float] = []
        good_exp = 0
        bad_exp = 0
        for t in range(trials):
            seq = sequence_with_mistake(n=n, pos=mistake_pos, noise=noise, rng=rng)
            gain_fn = seq_gain_fn_factory(seq, mistake_pos)
            gate = EnergyGatingModule(gain_fn=gain_fn, cost=c)
            # log hazard prior to decision
            try:
                hazards.append(gate.hazard_rate(None))
            except Exception:
                # fallback if hazard not available
                hazards.append(float("nan"))
            eta_gate = gate.compute_eta(None)
            # baseline full-sequence η
            eta_full = sample_monotonicity_score(seq, samples=512)
            # apply repair only if gate opens
            repaired_seq = list(seq)
            if eta_gate > 0.5 and 1 <= mistake_pos < n:
                repaired_seq[mistake_pos] = max(repaired_seq[mistake_pos], repaired_seq[mistake_pos - 1])
                expanded += 1
            eta_full_after = sample_monotonicity_score(repaired_seq, samples=512)
            redemption = float(eta_full_after - eta_full)
            red_scores.append(redemption)
            if eta_gate > 0.5:
                if redemption > 0.0:
                    good_exp += 1
                else:
                    bad_exp += 1
            total += 1
        rows.append({
            "n": int(n),
            "mistake_pos": int(mistake_pos),
            "cost": float(c),
            "trials": int(trials),
            "noise": float(noise),
            "expansion_rate": float(expanded) / float(total),
            "redemption_mean": float(np.mean(red_scores)),
            "hazard_mean": float(np.nanmean(hazards)) if hazards else float("nan"),
            "mu_hat": float(expanded) / float(sum(max(r, 0.0) for r in red_scores) + 1e-9),
            "good_bad_ratio": float((good_exp + 1e-6) / (bad_exp + 1e-6)),
        })
    out = log_records("energy_gated_expansion", rows)
    print(f"Wrote {len(rows)} rows to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--mistake_pos", type=int, default=20)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--costs", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    run(n=args.n, mistake_pos=args.mistake_pos, trials=args.trials, costs=list(args.costs), noise=args.noise, seed=args.seed)


if __name__ == "__main__":
    main()


