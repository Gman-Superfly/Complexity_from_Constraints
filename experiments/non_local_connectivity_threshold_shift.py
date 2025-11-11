"""Non-Local Connectivity Threshold Shift experiment.

Find apparent critical p with and without sparse shortcuts.
"""

from __future__ import annotations

import argparse
from typing import List, Dict, Any
import numpy as np

from modules.connectivity.nl_threshold_shift import build_grid_bond_graph, largest_component_fraction
from logging.metrics_log import log_records


def find_threshold(n: int, ps: np.ndarray, threshold: float, shortcuts: bool, shortcut_frac: float, trials: int, seed: int | None) -> float:
    """Return min p where mean η >= threshold; if none, return 1.0."""
    assert 0.0 < threshold < 1.0, "threshold must be in (0,1)"
    for p in ps:
        etas: List[float] = []
        for t in range(trials):
            s = None if seed is None else seed + t
            G = build_grid_bond_graph(n=n, p=float(p), add_shortcuts=shortcuts, shortcut_frac=shortcut_frac, seed=s)
            eta = largest_component_fraction(G)
            etas.append(eta)
        if float(np.mean(etas)) >= threshold:
            return float(p)
    return 1.0


def run(n: int, p_min: float, p_max: float, num: int, threshold: float, shortcut_frac: float, trials: int, seed: int | None) -> None:
    ps = np.linspace(p_min, p_max, num=num)
    p_no = find_threshold(n, ps, threshold, shortcuts=False, shortcut_frac=0.0, trials=trials, seed=seed)
    p_yes = find_threshold(n, ps, threshold, shortcuts=True, shortcut_frac=shortcut_frac, trials=trials, seed=seed)
    rows: List[Dict[str, Any]] = [{
        "n": int(n),
        "threshold": float(threshold),
        "p_no_shortcuts": float(p_no),
        "p_with_shortcuts": float(p_yes),
        "shortcut_frac": float(shortcut_frac),
        "trials": int(trials),
    }]
    out = log_records("non_local_connectivity_threshold_shift", rows)
    print(f"Thresholds written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--p_min", type=float, default=0.4)
    parser.add_argument("--p_max", type=float, default=0.7)
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--shortcut_frac", type=float, default=0.02)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    run(
        n=args.n,
        p_min=args.p_min,
        p_max=args.p_max,
        num=args.num,
        threshold=args.threshold,
        shortcut_frac=args.shortcut_frac,
        trials=args.trials,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


