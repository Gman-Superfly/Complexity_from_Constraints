"""Sequence redemption experiment: causal vs non-local surrogate.

We simulate local vs non-local by evaluating prefix-only vs full-sequence
consistency and compute redemption score.
"""

from __future__ import annotations

import argparse
from typing import List, Dict, Any
import numpy as np

from modules.sequence.monotonic_eta import sample_monotonicity_score
from logging.metrics_log import log_records


def sequence_with_mistake(n: int, pos: int, noise: float = 0.0, seed: int | None = None) -> List[float]:
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, 1.0, num=n).tolist()
    if 0 <= pos < n:
        base[pos] = base[pos] - 0.5  # inject a dip ("mistake")
    if noise > 0.0:
        base = (np.asarray(base) + rng.normal(0.0, noise, size=n)).tolist()
    return [float(x) for x in base]


def run(n: int, mistake_pos: int, samples: int, trials: int, seed: int | None) -> None:
    rows: List[Dict[str, Any]] = []
    for t in range(trials):
        seq = sequence_with_mistake(n=n, pos=mistake_pos, noise=0.01, seed=None if seed is None else seed + t)
        # local (prefix-only) vs non-local (full sequence) surrogate
        local_losses = []
        nonlocal_losses = []
        for i in range(2, n + 1):
            prefix = seq[:i]
            eta_local = sample_monotonicity_score(prefix, samples=samples)
            eta_nonlocal = sample_monotonicity_score(seq, samples=samples)
            # define loss as 1 - η
            local_losses.append(1.0 - eta_local)
            nonlocal_losses.append(1.0 - eta_nonlocal)
        redemption = float(np.mean(np.asarray(local_losses) - np.asarray(nonlocal_losses)))
        rows.append({
            "trial": int(t),
            "n": int(n),
            "mistake_pos": int(mistake_pos),
            "samples": int(samples),
            "redemption_score": float(redemption),
        })
    out = log_records("sequence_redemption", rows)
    print(f"Wrote {len(rows)} rows to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--mistake_pos", type=int, default=20)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(n=args.n, mistake_pos=args.mistake_pos, samples=args.samples, trials=args.trials, seed=args.seed)


if __name__ == "__main__":
    main()


