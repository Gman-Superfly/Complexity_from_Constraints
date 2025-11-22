"""APC (arbitrary polynomial chaos) basis fitting on ξ ∈ [-1,1] via Gram–Schmidt.

Given sample points ξ_k, constructs an approximately orthonormal basis
{φ_0, ..., φ_d} expressed in the monomial basis {1, ξ, ξ^2, ...}.

References:
    - Oladyshkin, S., & Nowak, W. (2012). Data-driven uncertainty quantification
      using the arbitrary polynomial chaos expansion. Reliability Engineering & System Safety.
    - Wildt, N., et al. (2025). CODE: A global approach to ODE dynamics learning. arXiv:2511.15619.

Usage:
    B = compute_apc_basis(xi_samples, degree=4)
where B has shape (degree+1, degree+1) and each row contains coefficients for φ_n:
    φ_n(ξ) = sum_{p=0..degree} B[n, p] * ξ^p
"""

from __future__ import annotations

from typing import List
import numpy as np


def _monomial_matrix(xi: np.ndarray, degree: int) -> np.ndarray:
    """Return matrix M with M[k, p] = xi[k]**p for p=0..degree."""
    n = xi.shape[0]
    M = np.empty((n, degree + 1), dtype=float)
    M[:, 0] = 1.0
    for p in range(1, degree + 1):
        M[:, p] = M[:, p - 1] * xi
    return M


def compute_apc_basis(xi_samples: List[float], degree: int) -> List[List[float]]:
    """Compute APC-like orthonormal basis coefficients over monomials via Gram–Schmidt.

    Args:
        xi_samples: empirical samples in [-1, 1] (ξ = 2η−1 transformation of η).
        degree: maximum polynomial degree (0..N).

    Returns:
        B: list of lists, shape (degree+1, degree+1), where B[n] are coefficients for φ_n.
    """
    assert degree >= 0, "degree must be non-negative"
    xi = np.asarray(xi_samples, dtype=float)
    assert xi.ndim == 1 and xi.size >= degree + 1, "not enough samples for target degree"
    assert np.all(np.isfinite(xi)), "non-finite sample(s)"

    # Build monomial matrix and perform empirical Gram–Schmidt
    M = _monomial_matrix(xi, degree)  # shape (N, degree+1)
    # Storage for orthonormal polynomials expressed as monomials (row-wise)
    B = np.zeros((degree + 1, degree + 1), dtype=float)

    # Inner product <f,g> ≈ mean_k f(xi_k) g(xi_k)
    def inner(coeff_f: np.ndarray, coeff_g: np.ndarray) -> float:
        f_vals = M @ coeff_f
        g_vals = M @ coeff_g
        return float(np.mean(f_vals * g_vals))

    # Start with monomials e_p
    for n in range(degree + 1):
        # Candidate = monomial ξ^n
        v = np.zeros(degree + 1, dtype=float)
        v[n] = 1.0
        # Subtract projections onto previous φ_j
        for j in range(n):
            bj = B[j]
            proj = inner(v, bj)
            v = v - proj * bj
        # Normalize
        norm = np.sqrt(inner(v, v))
        if norm <= 1e-12 or not np.isfinite(norm):
            # Fallback: keep as is to avoid blowups
            norm = 1.0
        B[n] = v / norm

    return B.tolist()


def evaluate_basis(B: List[List[float]], xi: float) -> List[float]:
    """Evaluate basis φ at a single ξ using monomial expansion B."""
    degree = len(B) - 1
    # Precompute monomials
    mon = [1.0]
    for _ in range(degree):
        mon.append(mon[-1] * xi)
    # φ_n(ξ) = sum_p B[n,p] ξ^p
    vals = []
    for n in range(degree + 1):
        s = 0.0
        row = B[n]
        for p in range(degree + 1):
            s += float(row[p]) * float(mon[p])
        vals.append(float(s))
    return vals


def evaluate_basis_derivatives(B: List[List[float]], xi: float) -> List[float]:
    """Evaluate dφ/dξ at a single ξ from monomial expansion B."""
    degree = len(B) - 1
    # Precompute monomials up to degree-1 for derivatives
    mon = [1.0]
    for _ in range(degree):
        mon.append(mon[-1] * xi)
    dvals = []
    for n in range(degree + 1):
        s = 0.0
        row = B[n]
        # derivative of ξ^p is p ξ^(p-1)
        for p in range(1, degree + 1):
            s += float(row[p]) * p * float(mon[p - 1])
        dvals.append(float(s))
    return dvals


