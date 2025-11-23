from __future__ import annotations

import types

from typing import Any, Dict

from cf_logging.observability import EnergyBudgetTracker
from modules.polynomial.polynomial_energy import PolynomialEnergyModule
from core.coordinator import EnergyCoordinator


def test_poly_corr_warn_threshold_triggers() -> None:
    # Single polynomial module; override basis values to be nearly colinear to trigger high corr
    mod = PolynomialEnergyModule(degree=2, basis="legendre", normalize_domain=True)
    constraints: Dict[str, Any] = {"poly_coeffs": [0.1, 0.0, 0.2]}
    coord = EnergyCoordinator(modules=[mod], couplings=[], constraints=constraints, use_analytic=True)

    t = {"val": 0.0}

    def fake_get_basis_values(self: PolynomialEnergyModule, eta: float, cs: Dict[str, Any]) -> list[float]:
        t["val"] += 0.1
        x = t["val"]
        # Nearly colinear features -> correlation ~ 1
        return [x, x * 1.001, 0.0]

    # Bind method to instance
    mod.get_basis_values = types.MethodType(fake_get_basis_values, mod)  # type: ignore[method-assign]

    tracker = EnergyBudgetTracker(run_id="poly-corr-warn")
    tracker.poly_corr_warn_threshold = 0.8  # make it easy to trigger
    tracker.attach(coord)

    etas = [0.1]
    for _ in range(8):
        tracker.on_eta(etas)
        energy = coord.energy(etas)
        tracker.on_energy(energy)
        etas = [min(1.0, float(etas[0] + 0.05))]

    row = tracker.buffer[-1]
    assert any(k.startswith("poly_corr_max:poly:0") for k in row.keys())
    warn_keys = [k for k in row.keys() if k.startswith("poly_corr_warn:poly:0")]
    assert warn_keys, "Expected poly_corr_warn field"
    assert int(row[warn_keys[0]]) == 1


