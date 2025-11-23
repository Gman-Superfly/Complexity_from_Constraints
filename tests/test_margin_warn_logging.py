from __future__ import annotations

from typing import Any, Dict

from cf_logging.observability import EnergyBudgetTracker
from core.coordinator import EnergyCoordinator
from modules.sequence.monotonic_eta import SequenceConsistencyModule


def test_margin_warn_emitted_below_threshold() -> None:
    mods = [SequenceConsistencyModule(), SequenceConsistencyModule()]
    coord = EnergyCoordinator(modules=mods, couplings=[], constraints={})

    tracker = EnergyBudgetTracker(run_id="margin-warn-test")
    tracker.warn_on_margin_shrink = True
    tracker.margin_warn_threshold = 1e-4
    tracker.attach(coord)

    # Provide last_etas so row populates
    etas = [0.3, 0.7]
    tracker.on_eta(etas)

    # Simulate a small contraction margin
    setattr(coord, "_last_contraction_margin", 1e-8)
    energy = coord.energy(etas)
    tracker.on_energy(energy)

    assert tracker.buffer, "Expected at least one logged row"
    row: Dict[str, Any] = tracker.buffer[-1]
    assert "contraction_margin" in row
    assert "margin_warn" in row
    assert int(row["margin_warn"]) == 1


