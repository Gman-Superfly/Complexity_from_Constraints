from __future__ import annotations

from typing import Any, Mapping, Dict

from cf_logging.observability import EnergyBudgetTracker
from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision


class _QuadCurv(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    def __init__(self, k: float, b: float = 0.0) -> None:
        self.k = float(k)
        self.b = float(b)

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        d = float(eta) - self.b
        return 0.5 * self.k * d * d

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return self.k * (float(eta) - self.b)

    def curvature(self, eta: OrderParameter) -> float:
        return self.k


def test_energy_budget_tracker_logs_precision_summary() -> None:
    mods = [_QuadCurv(8.0, 0.0), _QuadCurv(0.2, 0.0)]
    coord = EnergyCoordinator(modules=mods, couplings=[], constraints={}, use_analytic=True)

    tracker = EnergyBudgetTracker(run_id="test-precision-logging")
    tracker.attach(coord)

    etas = [0.5, -0.1]
    tracker.on_eta(etas)

    energy = coord.energy(etas)
    tracker.on_energy(energy)

    assert tracker.buffer, "Expected logged row"
    row: Dict[str, float] = tracker.buffer[-1]  # type: ignore[assignment]
    # Precision summary fields should exist and be positive for our modules
    for key in ("precision:min", "precision:max", "precision:mean"):
        assert key in row, f"Missing {key} in energy budget row"
        assert row[key] >= 0.0, f"{key} should be non-negative"

