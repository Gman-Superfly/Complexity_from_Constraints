from __future__ import annotations

from typing import Any, Dict

from cf_logging.observability import RelaxationTracker
from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.sequence.monotonic_eta import SequenceConsistencyModule


def test_relaxation_tracker_logs_per_eta_when_enabled() -> None:
    mods = [SequenceConsistencyModule(), SequenceConsistencyModule(), SequenceConsistencyModule()]
    coups = [(0, 1, QuadraticCoupling(weight=1.0))]
    coord = EnergyCoordinator(modules=mods, couplings=coups, constraints={})

    tracker = RelaxationTracker(name="relaxation_trace_test", run_id="rt-eta", log_per_eta=True)
    tracker.attach(coord)

    etas = [0.2, 0.5, 0.8]
    tracker.on_eta(etas)
    energy = coord.energy(etas)
    tracker.on_energy(energy)

    assert tracker.buffer, "Expected at least one row"
    row: Dict[str, Any] = tracker.buffer[-1]
    assert "eta:0" in row and "eta:1" in row and "eta:2" in row
    assert abs(float(row["eta:0"]) - 0.2) < 1e-9
    assert abs(float(row["eta:2"]) - 0.8) < 1e-9


