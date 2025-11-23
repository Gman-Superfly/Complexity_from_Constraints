"""Benchmark helper for compile-time vectorization caches."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from core.interfaces import EnergyModule, OrderParameter


@dataclass
class QuadraticModule(EnergyModule):
    """Simple convex module anchored at a reference value."""

    center: float
    weight: float = 1.0

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(np.clip(x, 0.0, 1.0))

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        diff = float(eta) - float(self.center)
        return float(self.weight * diff * diff)


def build_coordinator(count: int, use_vectorization: bool) -> EnergyCoordinator:
    modules = [QuadraticModule(center=i / max(1, count - 1)) for i in range(count)]
    couplings = [
        (i, i + 1, QuadraticCoupling(weight=0.5 + 0.1 * (i % 3)))
        for i in range(count - 1)
    ]
    return EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={},
        use_vectorized_quadratic=use_vectorization,
        use_vectorized_hinges=use_vectorization,
        use_vectorized_gate_benefits=use_vectorization,
        stability_guard=True,
    )


def run_benchmark(
    num_vars: int,
    steps: int,
    use_vectorization: bool,
) -> dict[str, Any]:
    coord = build_coordinator(num_vars, use_vectorization=use_vectorization)
    inputs = [0.5] * num_vars
    etas0 = coord.compute_etas(inputs)
    start = time.perf_counter()
    final_etas = coord.relax_etas(etas0, steps=steps)
    duration = time.perf_counter() - start
    metrics = coord.last_relaxation_metrics()
    return {
        "num_vars": num_vars,
        "steps": steps,
        "vectorized": use_vectorization,
        "duration_sec": duration,
        "accepted_steps": metrics["accepted_steps"],
        "final_energy": coord.energy(list(final_etas)),
    }


def save_result(row: Mapping[str, Any], path: Path) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorization benchmark helper")
    parser.add_argument("--num-vars", type=int, default=64)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--log-path", type=Path, default=Path("logs/vectorization_benchmark.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = run_benchmark(args.num_vars, args.steps, use_vectorization=False)
    vectorized = run_benchmark(args.num_vars, args.steps, use_vectorization=True)
    save_result(baseline, args.log_path)
    save_result(vectorized, args.log_path)
    print("Baseline duration:", baseline["duration_sec"])
    print("Vectorized duration:", vectorized["duration_sec"])


if __name__ == "__main__":
    main()

