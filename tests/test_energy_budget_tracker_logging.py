from __future__ import annotations

from typing import Dict, Any

from cf_logging.observability import EnergyBudgetTracker
from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.sequence.monotonic_eta import SequenceConsistencyModule


class _DummyAdapter:
    def __init__(self) -> None:
        self.scores: Dict[str, float] = {"coup:QuadraticCoupling": 1.23}
        self.last_allocations: Dict[str, float] = {"coup:QuadraticCoupling": 0.05}
        self.last_spent_global: float = 0.42


def test_energy_budget_tracker_logs_adapter_fields_in_row() -> None:
    # Minimal coordinator with 2 modules and one coupling
    mods = [SequenceConsistencyModule(), SequenceConsistencyModule()]
    coups = [(0, 1, QuadraticCoupling(weight=1.0))]
    coord = EnergyCoordinator(modules=mods, couplings=coups, constraints={})

    tracker = EnergyBudgetTracker(run_id="test-adapter-logging")
    tracker.attach(coord)

    # Set last etas so per-term energy computation works
    etas = [0.5, 0.7]
    tracker.on_eta(etas)

    # Attach dummy adapter with expected fields
    coord.weight_adapter = _DummyAdapter()  # type: ignore[assignment]

    # Emit a single energy row
    energy = coord.energy(etas)
    tracker.on_energy(energy)

    assert tracker.buffer, "Expected a row in tracker buffer"
    row: Dict[str, Any] = tracker.buffer[-1]
    # Adapter telemetry presence
    assert any(k.startswith("score:") for k in row.keys())
    assert any(k.startswith("alloc:") for k in row.keys())
    assert "spent:global" in row


