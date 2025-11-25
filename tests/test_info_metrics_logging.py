from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter
from core.couplings import QuadraticCoupling
from cf_logging.observability import EnergyBudgetTracker


class SimpleModule(EnergyModule):
    def __init__(self, a: float = 1.0):
        self.a = float(a)

    def local_energy(self, eta: OrderParameter, constraints: dict) -> float:
        e = float(eta)
        return self.a * e * e


def test_info_metrics_logging_with_reference_and_hallucination():
    mods = [SimpleModule(1.0), SimpleModule(1.2), SimpleModule(0.8)]
    couplings = [(0, 1, QuadraticCoupling(weight=0.1)), (1, 2, QuadraticCoupling(weight=0.1))]

    # Provide reference etas and hallucination counts in constraints
    constraints = {
        "reference_etas": [0.5, 0.5, 0.5],
        "constraint_violation_count": 2,
        "total_constraints_checked": 10,
    }

    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints=constraints,
        step_size=0.05,
    )

    tracker = EnergyBudgetTracker(name="test_info_metrics", run_id="test")
    tracker.attach(coord)

    etas0 = [0.9, 0.3, 0.1]
    coord.relax_etas(etas0, steps=8)

    assert len(tracker.buffer) > 0
    # At least one record contains info metrics
    found_align = any("info:alignment" in rec for rec in tracker.buffer)
    found_drift = any("info:drift" in rec for rec in tracker.buffer)
    found_h = any("info:constraint_violation_rate" in rec for rec in tracker.buffer)
    assert found_align
    assert found_drift
    assert found_h
