"""Tests for confidence trajectory logging."""

from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter
from core.couplings import QuadraticCoupling
from cf_logging.observability import EnergyBudgetTracker


class SimpleE(EnergyModule):
    def __init__(self, a: float = 1.0):
        self.a = float(a)

    def local_energy(self, eta: OrderParameter, constraints: dict) -> float:
        e = float(eta)
        return self.a * e * e


def test_confidence_logging_present_and_bounded():
    mods = [SimpleE(1.0), SimpleE(1.2)]
    couplings = [(0, 1, QuadraticCoupling(weight=0.1))]

    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={"redundancy_rho": 0.2},
        step_size=0.05,
        enable_sensitivity_probes=True,
        sensitivity_probe_window=5,
        enable_confidence_logging=True,
        confidence_a=1.0,
        confidence_b=1.0,
        confidence_rho_max=1.0,
        enable_orthogonal_noise=True,
        noise_magnitude=0.02,
    )

    tracker = EnergyBudgetTracker(name="test_confidence", run_id="test")
    tracker.attach(coord)

    etas0 = [0.6, 0.4]
    coord.relax_etas(etas0, steps=12)

    # At least one logged row
    assert len(tracker.buffer) > 0
    # Confidence present and in [0, 1]
    found = False
    for rec in tracker.buffer:
        if "confidence:c" in rec:
            val = float(rec["confidence:c"])
            assert 0.0 <= val <= 1.0
            found = True
    assert found, "confidence:c not found in logs"
