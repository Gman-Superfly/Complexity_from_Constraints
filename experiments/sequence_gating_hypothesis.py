"""Hypothesis experiment: non-local gating redeems a planted mistake versus local baseline."""

from __future__ import annotations

import argparse
from typing import List, Dict, Any
import numpy as np

from modules.sequence.monotonic_eta import sample_monotonicity_score, SequenceConsistencyModule
from modules.gating.energy_gating import EnergyGatingModule
from core.couplings import GateBenefitCoupling
from core.coordinator import EnergyCoordinator
from cf_logging.metrics_log import log_records
from cf_logging.observability import RelaxationTracker, GatingMetricsLogger


def sequence_with_mistake(n: int, pos: int, noise: float, rng: np.random.Generator) -> List[float]:
    seq = np.linspace(0.0, 1.0, num=n).tolist()
    if 0 <= pos < n:
        seq[pos] -= 0.4
    if noise > 0.0:
        seq = (np.asarray(seq) + rng.normal(0.0, noise, size=n)).tolist()
    return [float(x) for x in seq]


def run(
    n: int,
    mistake_pos: int,
    trials: int,
    cost: float,
    noise: float,
    seed: int | None,
    steps: int,
    track_relaxation: bool,
    log_gating_metrics: bool,
) -> None:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    gating_logger = GatingMetricsLogger(run_id="sequence_gating_hypothesis") if log_gating_metrics else None
    for trial in range(trials):
        seq = sequence_with_mistake(n, mistake_pos, noise, rng)
        seq_module = SequenceConsistencyModule(samples=512)
        eta_before = seq_module.compute_eta(seq)
        repaired_seq = list(seq)
        repaired_seq[mistake_pos] = max(repaired_seq[mistake_pos], repaired_seq[mistake_pos - 1])
        eta_repaired = seq_module.compute_eta(repaired_seq)
        delta_eta = max(0.0, eta_repaired - eta_before)
        gain_fn = lambda _x, d=delta_eta: d
        gate_module = EnergyGatingModule(gain_fn=gain_fn, cost=cost, k=10.0, use_hazard=True, a=0.2, b=0.3)
        modules = [seq_module, gate_module]
        couplings = [(1, 0, GateBenefitCoupling(weight=0.9, delta_key="delta_eta_domain"))]
        constraints = {"delta_eta_domain": delta_eta}
        coord = EnergyCoordinator(
            modules=modules,
            couplings=couplings,
            constraints=constraints,
            use_analytic=True,
            line_search=True,
            normalize_grads=True,
            enforce_invariants=True,
        )
        tracker = None
        if track_relaxation:
            tracker = RelaxationTracker(name="sequence_gating_relaxation", run_id=f"trial_{trial}")
            tracker.attach(coord)
        etas0 = coord.compute_etas([seq, None])
        energy0 = coord.energy(etas0)
        etas_final = coord.relax_etas(etas0, steps=steps)
        energy1 = coord.energy(etas_final)
        if tracker is not None:
            tracker.flush()
        eta_gate = etas_final[1]
        seq_after = list(seq)
        if eta_gate > 0.5 and 1 <= mistake_pos < n:
            seq_after[mistake_pos] = max(seq_after[mistake_pos], seq_after[mistake_pos - 1])
        eta_after = seq_module.compute_eta(seq_after)
        redemption = eta_after - eta_before
        if gating_logger is not None:
            hazard = gate_module.hazard_rate(None)
            gating_logger.record(hazard=float(hazard), eta_gate=float(eta_gate), redemption=float(redemption), good=(redemption > 0.0))
        rows.append({
            "trial": int(trial),
            "eta_before": float(eta_before),
            "eta_after": float(eta_after),
            "eta_gate_final": float(eta_gate),
            "energy_before": float(energy0),
            "energy_after": float(energy1),
            "redemption": float(redemption),
        })
    if gating_logger is not None:
        gating_logger.flush()
    out = log_records("sequence_gating_hypothesis", rows)
    print(f"Wrote {len(rows)} rows to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--mistake_pos", type=int, default=20)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--cost", type=float, default=0.05)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--track_relaxation", action="store_true", help="Log ΔF/η traces via RelaxationTracker.")
    parser.add_argument("--log_gating_metrics", action="store_true", help="Log hazard/η/redemption via GatingMetricsLogger.")
    args = parser.parse_args()
    run(
        n=args.n,
        mistake_pos=args.mistake_pos,
        trials=args.trials,
        cost=args.cost,
        noise=args.noise,
        seed=args.seed,
        steps=args.steps,
        track_relaxation=args.track_relaxation,
        log_gating_metrics=args.log_gating_metrics,
    )


if __name__ == "__main__":
    main()


