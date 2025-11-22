from __future__ import annotations

import math
import random
from typing import List

import numpy as np

from modules.polynomial.apc import compute_apc_basis, evaluate_basis, evaluate_basis_derivatives
from modules.polynomial.polynomial_energy import PolynomialEnergyModule


def test_apc_basis_orthonormality() -> None:
    # Draw xi samples biased to make Gram–Schmidt meaningful
    rng = random.Random(7)
    xi_samples: List[float] = [2.0 * rng.random() - 1.0 for _ in range(1024)]
    degree = 4
    B = compute_apc_basis(xi_samples, degree)
    xs = np.asarray(xi_samples, dtype=float)
    # Monte Carlo inner products
    def inner(n: int, m: int) -> float:
        vals_n = np.array([evaluate_basis(B, float(x))[n] for x in xs])
        vals_m = np.array([evaluate_basis(B, float(x))[m] for x in xs])
        return float(np.mean(vals_n * vals_m))
    # Check approximate orthonormality
    for n in range(degree + 1):
        for m in range(degree + 1):
            val = inner(n, m)
            if n == m:
                assert abs(val - 1.0) < 0.15  # rough tolerance
            else:
                assert abs(val) < 0.15


def test_polynomial_module_apc_grad_parity() -> None:
    rng = random.Random(3)
    xi_samples = [2.0 * rng.random() - 1.0 for _ in range(2048)]
    B = compute_apc_basis(xi_samples, degree=3)
    mod = PolynomialEnergyModule(degree=3, basis="apc")
    constraints = {
        "apc_basis": B,
        "poly_coeffs": [0.2, -0.1, 0.3, 0.05],
    }
    for _ in range(10):
        eta = rng.random()
        e0 = mod.local_energy(eta, constraints)
        grad = mod.d_local_energy_d_eta(eta, constraints)
        eps = 1e-6
        e1 = mod.local_energy(min(1.0, eta + eps), constraints)
        fd = (e1 - e0) / eps
        assert math.isfinite(grad) and math.isfinite(fd)
        assert abs(grad - fd) / (abs(fd) + 1e-9) < 1e-3

