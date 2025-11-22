from __future__ import annotations

import math
import numpy as np

from modules.polynomial.polynomial_energy import PolynomialEnergyModule


def _legendre_vals(xi: float, degree: int) -> list[float]:
    P0 = 1.0
    if degree == 0:
        return [P0]
    P1 = xi
    if degree == 1:
        return [P0, P1]
    P2 = 0.5 * (3.0 * xi * xi - 1.0)
    if degree == 2:
        return [P0, P1, P2]
    P3 = 0.5 * (5.0 * xi ** 3 - 3.0 * xi)
    if degree == 3:
        return [P0, P1, P2, P3]
    P4 = (1.0 / 8.0) * (35.0 * xi ** 4 - 30.0 * xi * xi + 3.0)
    return [P0, P1, P2, P3, P4][: degree + 1]


def test_minima_parity_legendre_vs_landau() -> None:
    # Target Landau energy: E_L(η) = a η^2 + b η^4 (h=0)
    a = 0.4
    b = 0.6
    degree = 4
    # Fit Legendre coefficients to approximate Landau over η∈[0,1]
    etas = np.linspace(0.0, 1.0, 201)
    xi = 2.0 * etas - 1.0
    A = np.stack([np.array([_legendre_vals(float(x), degree)[k] for x in xi]) for k in range(degree + 1)], axis=1)
    y = a * (etas ** 2) + b * (etas ** 4)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    coeffs = coeffs.tolist()

    # Compute minima locations
    landau_vals = y
    landau_min_eta = float(etas[int(np.argmin(landau_vals))])

    mod = PolynomialEnergyModule(degree=degree, basis="legendre")
    constraints = {"poly_coeffs": coeffs}
    poly_vals = np.array([mod.local_energy(float(e), constraints) for e in etas])
    poly_min_eta = float(etas[int(np.argmin(poly_vals))])

    # Minima should be very close
    assert abs(poly_min_eta - landau_min_eta) < 1e-2

