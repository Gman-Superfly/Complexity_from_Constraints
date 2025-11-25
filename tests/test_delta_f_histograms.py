"""Validate that Δ (energy/free-energy) histogram is negatively skewed.

We use RelaxationTracker's delta_energy as a proxy for ΔF when F is not logged.
"""

from __future__ import annotations

import math
from typing import List

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter
from core.couplings import QuadraticCoupling
from cf_logging.observability import RelaxationTracker


class SimpleQ(EnergyModule):
    def __init__(self, a: float = 1.0):
        self.a = float(a)

    def local_energy(self, eta: OrderParameter, constraints: dict) -> float:
        e = float(eta)
        return self.a * e * e


def _skewness(values: List[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    n = len(vals)
    if n < 3:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(max(var, 1e-18))
    m3 = sum((v - mean) ** 3 for v in vals) / n
    return float(m3 / (std ** 3))


def test_delta_energy_histogram_negative_skew():
    mods = [SimpleQ(1.0), SimpleQ(1.2)]
    couplings = [(0, 1, QuadraticCoupling(weight=0.1))]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        line_search=True,
    )
    tracker = RelaxationTracker(name="relaxation_trace", run_id="test")
    tracker.attach(coord)

    etas0 = [0.7, 0.3]
    coord.relax_etas(etas0, steps=20)

    deltas = [row.get("delta_energy") for row in tracker.buffer if "delta_energy" in row]
    deltas = [float(d) for d in deltas if d is not None and math.isfinite(float(d))]
    # Ignore the first (None) delta
    if deltas and deltas[0] == 0.0:
        pass
    # Expect majority non-positive
    non_pos_ratio = sum(1 for d in deltas if d <= 0.0) / max(1, len(deltas))
    assert non_pos_ratio >= 0.8
    # Skewness should be non-positive (negatively skewed or zero)
    sk = _skewness(deltas)
    assert sk <= 0.05  # allow slight numerical noise
