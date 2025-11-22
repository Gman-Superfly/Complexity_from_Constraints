"""Demonstrate GradNorm-style term rebalancing inside EnergyCoordinator.

Runs a simple sequence+gate scenario twice (baseline vs GradNormWeightAdapter)
and logs per-term gradient norms/weights so we can inspect how the adapter
keeps families within the target range.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from cf_logging.metrics_log import log_records
from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling, QuadraticCoupling
from core.weight_adapters import GradNormWeightAdapter
from modules.gating.energy_gating import EnergyGatingModule
from modules.sequence.monotonic_eta import SequenceConsistencyModule


@dataclass
class ScenarioConfig:
    label: str
    use_adapter: bool


def make_sequence(seed: int, noise: float) -> List[float]:
    rng = np.random.default_rng(seed)
    seq = np.linspace(0.0, 1.0, num=64)
    glitch_pos = 20
    seq[glitch_pos] -= 0.3
    if noise > 0.0:
        seq += rng.normal(0.0, noise, size=seq.shape)
    return seq.clip(0.0, 1.0).astype(float).tolist()


def evaluate_redemption(seq: List[float]) -> Tuple[List[float], float]:
    seq_module = SequenceConsistencyModule(samples=512)
    eta_before = seq_module.compute_eta(seq)
    repaired = list(seq)
    for idx in range(1, len(repaired)):
        repaired[idx] = max(repaired[idx], repaired[idx - 1])
    eta_after = seq_module.compute_eta(repaired)
    delta = max(0.0, eta_after - eta_before)
    return repaired, delta


def build_coordinator(use_adapter: bool, delta_eta: float, base_weights: Dict[str, float]) -> EnergyCoordinator:
    seq_module = SequenceConsistencyModule(samples=512)
    gate_module = EnergyGatingModule(
        gain_fn=lambda _x, d=delta_eta: d,
        cost=0.1,
        k=12.0,
        a=0.4,
        b=0.6,
        use_hazard=True,
    )
    modules = [seq_module, gate_module]
    couplings = [
        (1, 0, GateBenefitCoupling(weight=1.2, delta_key="delta_eta_domain")),
        (0, 1, QuadraticCoupling(weight=0.3)),
    ]
    adapter = GradNormWeightAdapter(target_norm=1.0, alpha=1.2, update_rate=0.2) if use_adapter else None
    return EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={
            "delta_eta_domain": delta_eta,
            "term_weights": base_weights,
        },
        use_analytic=True,
        line_search=True,
        normalize_grads=True,
        enforce_invariants=True,
        weight_adapter=adapter,
    )


def attach_history(
    coord: EnergyCoordinator,
    scenario: str,
    history: List[Dict[str, Any]],
) -> None:
    step_counter = {"value": -1}

    def _on_eta(etas: List[float]) -> None:
        step_counter["value"] += 1
        term_norms = coord._term_grad_norms(etas)  # noqa: SLF001 - experiment instrumentation
        weights = coord._combined_term_weights()  # noqa: SLF001 - experiment instrumentation
        row: Dict[str, Any] = {
            "scenario": scenario,
            "step": step_counter["value"],
            "energy": coord._energy_value(etas),  # noqa: SLF001
        }
        for key, value in sorted(term_norms.items()):
            row[f"norm:{key}"] = float(value)
            row[f"weight:{key}"] = float(weights.get(key, 1.0))
        history.append(row)

    coord.on_eta_updated.append(_on_eta)


def run_scenario(
    config: ScenarioConfig,
    seq: List[float],
    base_weights: Dict[str, float],
    steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    repaired, delta_eta = evaluate_redemption(seq)
    coord = build_coordinator(config.use_adapter, delta_eta, base_weights)
    history: List[Dict[str, Any]] = []
    attach_history(coord, config.label, history)
    inputs = [seq, None]
    etas0 = coord.compute_etas(inputs)
    coord.energy(etas0)
    final_etas = coord.relax_etas(etas0, steps=steps)
    # append final summary row
    weights = coord._combined_term_weights()  # noqa: SLF001
    summary = {
        "scenario": config.label,
        "step": steps,
        "energy": coord.energy(final_etas),
    }
    for key, value in sorted(weights.items()):
        summary[f"final_weight:{key}"] = float(value)
    history.append(summary)
    print(
        f"[{config.label}] final weights: "
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(weights.items())),
    )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30, help="Relaxation steps per scenario.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the sequence glitch/noise.")
    parser.add_argument("--noise", type=float, default=0.02, help="Noise injected into the base sequence.")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["baseline", "gradnorm"],
        help="Scenarios to run: choose from {baseline, gradnorm}.",
    )
    args = parser.parse_args()

    seq = make_sequence(seed=args.seed, noise=args.noise)
    base_tw = {
        "local:SequenceConsistencyModule": 0.25,
        "local:EnergyGatingModule": 2.5,
        "coup:GateBenefitCoupling": 1.2,
        "coup:QuadraticCoupling": 0.4,
    }
    options = {
        "baseline": ScenarioConfig(label="baseline", use_adapter=False),
        "gradnorm": ScenarioConfig(label="gradnorm", use_adapter=True),
    }
    histories: List[Dict[str, Any]] = []
    for name in args.scenarios:
        config = options.get(name)
        if config is None:
            raise ValueError(f"Unknown scenario '{name}'. Valid: {sorted(options)}")
        histories.extend(run_scenario(config, seq, base_tw, args.steps, args.seed))
    out = log_records("auto_balance_demo", histories)
    print(f"Wrote {len(histories)} rows to {out}")


if __name__ == "__main__":
    main()

