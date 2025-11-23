from __future__ import annotations

from typing import List

from modules.polynomial.rhs_probe import fit_rhs_legendre, eval_rhs_legendre


def test_fit_rhs_legendre_matches_quadratic_derivative() -> None:
    # True RHS for E(eta) = a*(eta-0.5)^2 is -dE/deta = -2a(eta-0.5)
    a = 0.7
    def rhs_true(e: float) -> float:
        return -2.0 * a * (float(e) - 0.5)
    xs: List[float] = [i / 20.0 for i in range(21)]
    ys: List[float] = [rhs_true(x) for x in xs]
    coeffs = fit_rhs_legendre(xs, ys, degree=2)
    # Check approximation error small across grid
    max_err = 0.0
    for x in xs:
        est = eval_rhs_legendre(x, coeffs)
        max_err = max(max_err, abs(est - rhs_true(x)))
    assert max_err < 1e-3


