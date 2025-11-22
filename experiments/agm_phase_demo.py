"""AGM phase-adaptive weighting and uncertainty-gated thresholds demo.

Runs a sequence + gate scenario while:
  - enabling AGMPhaseWeightAdapter to modulate term weights during relaxation
  - optionally adapting the gate cost based on computed uncertainty metrics

Logs per-step weights and energy for inspection via cf_logging.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from cf_logging.metrics_log import log_records
from core.agm_metrics import compute_uncertainty_metrics
from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling, QuadraticCoupling
from core.weight_adapters import AGMPhaseWeightAdapter
from modules.gating.energy_gating import EnergyGatingModule
from modules.sequence.monotonic_eta import SequenceConsistencyModule


def make_sequence(seed: int, noise: float, n: int = 64) -> List[float]:
    rng = np.random.default_rng(seed)
    seq = np.linspace(0.0, 1.0, num=n)
    # planted glitch
    pos = int(n * 0.35)
    seq[pos] -= 0.3
    if noise > 0.0:
        seq += rng.normal(0.0, noise, size=seq.shape)
    return seq.clip(0.0, 1.0).astype(float).tolist()


def redemption_delta(seq: List[float]) -> float:
    mod = SequenceConsistencyModule(samples=512)
    before = mod.compute_eta(seq)
    repaired = list(seq)
    for i in range(1, len(repaired)):
        repaired[i] = max(repaired[i], repaired[i - 1])
    after = mod.compute_eta(repaired)
    return max(0.0, after - before)


def attach_recording(coord: EnergyCoordinator, scenario: str, history: List[Dict[str, Any]]) -> None:
    step = {"i": -1}

    def on_eta(_etas: List[float]) -> None:
        step["i"] += 1
        weights = coord._combined_term_weights()  # noqa: SLF001 - instrumentation
        row = {"scenario": scenario, "step": step["i"], "energy": coord._energy_value(_etas)}  # noqa: SLF001
        for k, v in sorted(weights.items()):
            row[f"w:{k}"] = float(v)
        history.append(row)

    coord.on_eta_updated.append(on_eta)


def attach_uncertainty_gater(
    coord: EnergyCoordinator,
    gate_module: EnergyGatingModule,
) -> None:
    energies: List[float] = []

    def on_energy(F: float) -> None:
        energies.append(float(F))
        u = compute_uncertainty_metrics(energies, recent_performance=None)
        # Adapt cost: high epistemic uncertainty => lower cost (encourage exploration)
        base = gate_module.cost
        multiplier = 0.5 + 1.5 * (1.0 - u.epistemic)
        gate_module.cost = float(base * multiplier)

    coord.on_energy_updated.append(on_energy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--use_uncertainty_gate", action="store_true")
    args = parser.parse_args()

    seq = make_sequence(args.seed, args.noise)
    delta = redemption_delta(seq)

    seq_mod = SequenceConsistencyModule(samples=512)
    gate_mod = EnergyGatingModule(gain_fn=lambda _x, d=delta: d, cost=0.08, k=12.0, a=0.3, b=0.4, use_hazard=True)
    modules = [seq_mod, gate_mod]
    couplings = [
        (1, 0, GateBenefitCoupling(weight=1.0, delta_key="delta_eta_domain")),
        (0, 1, QuadraticCoupling(weight=0.25)),
    ]
    base_tw = {
        "local:SequenceConsistencyModule": 0.4,
        "local:EnergyGatingModule": 1.2,
        "coup:GateBenefitCoupling": 1.0,
        "coup:QuadraticCoupling": 0.25,
    }

    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={"delta_eta_domain": delta, "term_weights": base_tw},
        use_analytic=True,
        line_search=True,
        normalize_grads=True,
        enforce_invariants=True,
        weight_adapter=AGMPhaseWeightAdapter(),
    )

    if args.use_uncertainty_gate:
        attach_uncertainty_gater(coord, gate_mod)

    history: List[Dict[str, Any]] = []
    attach_recording(coord, "agm_phase", history)

    etas0 = coord.compute_etas([seq, None])
    coord.energy(etas0)
    coord.relax_etas(etas0, steps=args.steps)

    out = log_records("agm_phase_demo", history)
    print(f"Wrote {len(history)} rows to {out}")


if __name__ == "__main__":
    main()

