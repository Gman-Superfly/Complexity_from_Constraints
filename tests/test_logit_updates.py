from __future__ import annotations

from core.coordinator import EnergyCoordinator
from modules.polynomial.polynomial_energy import PolynomialEnergyModule


def test_logit_updates_preserve_bounds_and_reduce_energy() -> None:
    # Single-module convex energy; gradient descent should reduce energy
    mod = PolynomialEnergyModule(degree=2, basis="legendre", normalize_domain=True)
    constraints = {"poly_coeffs": [0.0, 0.0, 0.5]}  # convex parabola in ξ
    coord = EnergyCoordinator(
        modules=[mod],
        couplings=[],
        constraints=constraints,
        use_analytic=True,
        line_search=False,
        step_size=0.1,
        use_logit_updates=True,
    )
    etas = coord.compute_etas([0.8])  # start near boundary to exercise logit transform
    e0 = coord.energy(etas)
    etas = coord.relax_etas(etas, steps=25)
    e1 = coord.energy(etas)
    assert 0.0 <= etas[0] <= 1.0
    assert e1 <= e0 + 1e-12


