from __future__ import annotations

import math
import random

from modules.polynomial.polynomial_energy import PolynomialEnergyModule


def test_polynomial_energy_grad_matches_fd() -> None:
    mod = PolynomialEnergyModule(degree=4, coeff_key="poly_coeffs")
    constraints = {"poly_coeffs": [0.1, 0.0, 0.3, 0.0, 0.2]}  # even terms emphasize convexity
    for _ in range(10):
        eta = random.random()
        e0 = mod.local_energy(eta, constraints)
        grad = mod.d_local_energy_d_eta(eta, constraints)
        # finite difference
        eps = 1e-6
        e1 = mod.local_energy(min(1.0, eta + eps), constraints)
        fd = (e1 - e0) / eps
        assert math.isfinite(grad) and math.isfinite(fd)
        assert abs(grad - fd) / (abs(fd) + 1e-9) < 5e-3


def test_polynomial_energy_pass_through_eta() -> None:
    mod = PolynomialEnergyModule(degree=2)
    eta = 0.25
    assert mod.compute_eta(eta) == eta

