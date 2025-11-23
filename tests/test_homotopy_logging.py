from __future__ import annotations

from typing import Any, Dict

from cf_logging.observability import EnergyBudgetTracker
from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.sequence.monotonic_eta import SequenceConsistencyModule


def test_energy_budget_tracker_logs_homotopy_fields_when_present() -> None:
    mods = [SequenceConsistencyModule(), SequenceConsistencyModule()]
    coups = [(0, 1, QuadraticCoupling(weight=1.0))]
    coord = EnergyCoordinator(modules=mods, couplings=coups, constraints={})

    # Simulate homotopy telemetry
    setattr(coord, "_homotopy_scale", 0.3)
    setattr(coord, "_homotopy_backoffs", 2)

    tracker = EnergyBudgetTracker(run_id="test-homotopy-logging")
    tracker.attach(coord)

    etas = [0.5, 0.7]
    tracker.on_eta(etas)
    energy = coord.energy(etas)
    tracker.on_energy(energy)

    assert tracker.buffer, "Expected tracker to have at least one row"
    row: Dict[str, Any] = tracker.buffer[-1]
    assert "homotopy_scale" in row
    assert "homotopy_backoffs" in row
    assert abs(float(row["homotopy_scale"]) - 0.3) < 1e-9
    assert int(row["homotopy_backoffs"]) == 2


