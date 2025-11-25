"""Tests for sensitivity probes (dispersion) tracking and logging."""

from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter
from core.couplings import QuadraticCoupling
from cf_logging.observability import EnergyBudgetTracker


class QuadModule(EnergyModule):
    def __init__(self, a: float = 1.0):
        self.a = float(a)

    def local_energy(self, eta: OrderParameter, constraints: dict) -> float:
        e = float(eta)
        return self.a * e * e


def test_probe_dispersion_history_and_last_value():
    mods = [QuadModule(1.0), QuadModule(1.2), QuadModule(0.8)]
    couplings = [(0, 1, QuadraticCoupling(weight=0.1)), (1, 2, QuadraticCoupling(weight=0.1))]

    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        enable_sensitivity_probes=True,
        sensitivity_probe_window=5,
        enable_orthogonal_noise=True,
        noise_magnitude=0.01,
    )

    etas0 = [0.6, 0.4, 0.2]
    coord.relax_etas(etas0, steps=10)

    hist = coord.get_probe_dispersion_history()
    assert isinstance(hist, list)
    assert len(hist) > 0
    last = coord.last_probe_dispersion()
    assert last is not None
    assert last >= 0.0


def test_probe_dispersion_logged_in_energy_budget():
    mods = [QuadModule(1.0), QuadModule(1.0)]
    couplings = [(0, 1, QuadraticCoupling(weight=0.2))]

    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        enable_sensitivity_probes=True,
        sensitivity_probe_window=4,
        enable_orthogonal_noise=True,
        noise_magnitude=0.02,
    )

    tracker = EnergyBudgetTracker(name="test_sensitivity", run_id="test")
    tracker.attach(coord)

    etas0 = [0.3, 0.7]
    coord.relax_etas(etas0, steps=12)

    assert len(tracker.buffer) > 0
    # Ensure at least one record contains the sensitivity dispersion field
    found = any("sensitivity:dispersion" in rec for rec in tracker.buffer)
    assert found, "Expected 'sensitivity:dispersion' in energy budget logs"
