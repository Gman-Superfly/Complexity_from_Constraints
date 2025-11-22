"""Polynomial-basis local energy on η via orthonormal-like Legendre basis on ξ=2η−1.

This module passes through η from input x (expects float in [0,1]) and defines
its local energy as a weighted sum of Legendre basis functions evaluated on
ξ ∈ [-1, 1], where ξ = 2η - 1. Analytic gradient is provided via chain rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, List, Literal
import math

from core.interfaces import EnergyModule, OrderParameter
from .apc import evaluate_basis as apc_eval, evaluate_basis_derivatives as apc_eval_d

__all__ = ["PolynomialEnergyModule"]


def _legendre_values(xi: float, degree: int) -> List[float]:
    """Return Legendre basis values [P0, P1, ..., P_degree] at xi."""
    assert -1.0000001 <= xi <= 1.0000001, "ξ out of bounds"
    # Explicit polynomials up to degree 4 for stability and speed.
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


def _legendre_derivatives(xi: float, degree: int) -> List[float]:
    """dP/dξ for Legendre basis up to degree."""
    # Derivatives of explicit forms used above.
    dP0 = 0.0
    if degree == 0:
        return [dP0]
    dP1 = 1.0
    if degree == 1:
        return [dP0, dP1]
    dP2 = 3.0 * xi
    if degree == 2:
        return [dP0, dP1, dP2]
    dP3 = 0.5 * (15.0 * xi * xi - 3.0)
    if degree == 3:
        return [dP0, dP1, dP2, dP3]
    dP4 = (1.0 / 2.0) * (35.0 * xi ** 3 - 15.0 * xi)
    return [dP0, dP1, dP2, dP3, dP4][: degree + 1]


@dataclass
class PolynomialEnergyModule(EnergyModule):
    """Local energy defined on polynomial basis over ξ = 2η − 1.

    Args:
        degree: Maximum Legendre degree to use (0..4).
        coeff_key: Constraints key where the coefficient list is stored.
                   If not present, defaults to zeros except bias (P0)=0.0.
                   For stability, consider positive coefficients on even orders.
    """

    degree: int = 4
    basis: Literal["legendre", "apc"] = "legendre"
    coeff_key: str = "poly_coeffs"
    apc_basis_key: str = "apc_basis"

    def compute_eta(self, x: Any) -> OrderParameter:
        """Pass-through of η from input x (expects float ∈ [0,1])."""
        try:
            eta = float(x)
        except Exception as exc:
            raise TypeError("PolynomialEnergyModule expects a float η as input") from exc
        assert 0.0 <= eta <= 1.0, "η must be within [0,1]"
        return float(eta)

    def _get_coeffs(self, constraints: Mapping[str, Any]) -> List[float]:
        coeffs = constraints.get(self.coeff_key, None)
        if coeffs is None:
            return [0.0] * (self.degree + 1)
        if not isinstance(coeffs, (list, tuple)):
            raise TypeError(f"{self.coeff_key} must be a list/tuple of floats")
        if len(coeffs) < (self.degree + 1):
            coeffs = list(coeffs) + [0.0] * ((self.degree + 1) - len(coeffs))
        return [float(c) for c in coeffs[: self.degree + 1]]

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        assert 0.0 <= eta <= 1.0, "η must be within [0,1]"
        coeffs = self._get_coeffs(constraints)
        xi = 2.0 * float(eta) - 1.0
        if self.basis == "legendre":
            P = _legendre_values(xi, self.degree)
            energy = 0.0
            for c, p in zip(coeffs, P):
                energy += c * p
        else:
            B = constraints.get(self.apc_basis_key, None)
            if B is None:
                raise ValueError("APC basis requested but no 'apc_basis' provided in constraints")
            vals = apc_eval(B, xi)
            energy = float(sum(c * v for c, v in zip(coeffs, vals)))
        assert math.isfinite(energy), "energy not finite"
        return float(energy)

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        assert 0.0 <= eta <= 1.0, "η must be within [0,1]"
        coeffs = self._get_coeffs(constraints)
        xi = 2.0 * float(eta) - 1.0
        if self.basis == "legendre":
            dPdxi = _legendre_derivatives(xi, self.degree)
            dE_dxi = 0.0
            for c, dp in zip(coeffs, dPdxi):
                dE_dxi += c * dp
        else:
            B = constraints.get(self.apc_basis_key, None)
            if B is None:
                raise ValueError("APC basis requested but no 'apc_basis' provided in constraints")
            dvals = apc_eval_d(B, xi)
            dE_dxi = float(sum(c * dv for c, dv in zip(coeffs, dvals)))
        # dξ/dη = 2
        dE_deta = 2.0 * dE_dxi
        assert math.isfinite(dE_deta), "gradient not finite"
        return float(dE_deta)


