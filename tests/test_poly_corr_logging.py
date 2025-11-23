from __future__ import annotations

from typing import Any, Dict

from cf_logging.observability import EnergyBudgetTracker
from modules.polynomial.polynomial_energy import PolynomialEnergyModule
from core.coordinator import EnergyCoordinator


def test_poly_corr_logging_emits_field() -> None:
    # Single polynomial module; correlation over a few steps is defined (even if small)
    mod = PolynomialEnergyModule(degree=3, basis="legendre", normalize_domain=True)
    constraints: Dict[str, Any] = {"poly_coeffs": [0.1, 0.0, 0.2, 0.0]}
    coord = EnergyCoordinator(modules=[mod], couplings=[], constraints=constraints, use_analytic=True)

    tracker = EnergyBudgetTracker(run_id="poly-corr-test")
    tracker.attach(coord)

    # Simulate a few steps
    etas = [0.1]
    for _ in range(6):
        tracker.on_eta(etas)
        energy = coord.energy(etas)
        tracker.on_energy(energy)
        # move eta slightly to produce variation
        etas = [min(1.0, float(etas[0] + 0.05))]

    assert tracker.buffer, "Expected logged rows"
    # Check latest row contains poly_corr_max field
    row = tracker.buffer[-1]
    found = any(k.startswith("poly_corr_max:poly:0") for k in row.keys())
    assert found, f"Expected poly_corr_max field in row, got keys: {list(row.keys())}"


