"""Tests for escape events detection and logging."""

from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter
from core.couplings import QuadraticCoupling
from cf_logging.observability import EnergyBudgetTracker


class SimpleEnergy(EnergyModule):
    def __init__(self, a: float = 1.0):
        self.a = float(a)

    def local_energy(self, eta: OrderParameter, constraints: dict) -> float:
        e = float(eta)
        return self.a * e * e


def test_escape_events_counter_and_logging():
    mods = [SimpleEnergy(1.0), SimpleEnergy(1.2), SimpleEnergy(0.8)]
    couplings = [(0, 1, QuadraticCoupling(weight=0.05)), (1, 2, QuadraticCoupling(weight=0.05))]

    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.02,
        enable_orthogonal_noise=True,
        auto_noise_controller=False,
        noise_magnitude=0.1,  # encourage significant displacement
        enable_escape_event_logging=True,
        escape_displacement_min_norm=1e-4,
        escape_alignment_max_cosine=0.3,
        escape_min_energy_drop=0.0,
        escape_noise_min_magnitude=1e-4,
    )

    tracker = EnergyBudgetTracker(name="test_escape", run_id="test")
    tracker.attach(coord)

    etas0 = [0.6, 0.4, 0.2]
    coord.relax_etas(etas0, steps=15)

    # Ensure the counter is non-negative and present
    count = coord.get_escape_event_count()
    assert isinstance(count, int)
    assert count >= 0

    # Ensure logs contain escape_event_count
    assert len(tracker.buffer) > 0
    assert any("escape_event_count" in rec for rec in tracker.buffer)
