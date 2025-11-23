from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable
import math
from .apc import compute_apc_basis, evaluate_basis as apc_eval


def _legendre_values(xi: float, degree: int) -> List[float]:
    assert -1.0000001 <= xi <= 1.0000001, "ξ out of bounds"
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


def fit_rhs_legendre(etas: Iterable[float], rhs_values: Iterable[float], degree: int) -> List[float]:
    """Fit RHS(η) ≈ Σ c_k P_k(ξ) via least squares (Legendre basis on ξ=2η−1)."""
    xs = [float(e) for e in etas]
    ys = [float(r) for r in rhs_values]
    assert len(xs) == len(ys) and len(xs) > 0, "Samples mismatch or empty"
    # Build design matrix
    Phi: List[List[float]] = []
    for e in xs:
        xi = 2.0 * e - 1.0
        Phi.append(_legendre_values(xi, degree))
    # Solve via normal equations (small degree ≤ 4)
    m = degree + 1
    # Compute A = Phi^T Phi, b = Phi^T y
    A = [[0.0 for _ in range(m)] for _ in range(m)]
    b = [0.0 for _ in range(m)]
    for row, y in zip(Phi, ys):
        for i in range(m):
            b[i] += row[i] * y
            for j in range(m):
                A[i][j] += row[i] * row[j]
    # Solve Ax=b (Gaussian elimination, small system)
    # Augment
    for i in range(m):
        A[i].append(b[i])
    # Forward elimination
    for i in range(m):
        pivot = A[i][i] if A[i][i] != 0.0 else 1e-12
        inv_pivot = 1.0 / pivot
        for j in range(i, m + 1):
            A[i][j] *= inv_pivot
        for k in range(i + 1, m):
            factor = A[k][i]
            if factor == 0.0:
                continue
            for j in range(i, m + 1):
                A[k][j] -= factor * A[i][j]
    # Back substitution
    x = [0.0 for _ in range(m)]
    for i in range(m - 1, -1, -1):
        s = A[i][m]
        for j in range(i + 1, m):
            s -= A[i][j] * x[j]
        x[i] = s
    return [float(v) for v in x]


def eval_rhs_legendre(eta: float, coeffs: List[float]) -> float:
    """Evaluate RHS at η using Legendre basis on ξ=2η−1 and given coeffs c_k."""
    xi = 2.0 * float(eta) - 1.0
    vals = _legendre_values(xi, len(coeffs) - 1)
    rhs = 0.0
    for c, v in zip(coeffs, vals):
        rhs += float(c) * float(v)
    return float(rhs)


def fit_rhs_apc(etas: Iterable[float], rhs_values: Iterable[float], degree: int) -> List[float]:
    """Fit RHS(η) ≈ Σ c_k φ_k(ξ) using APC basis computed from ξ samples."""
    xs = [float(e) for e in etas]
    ys = [float(r) for r in rhs_values]
    assert len(xs) == len(ys) and len(xs) > 0, "Samples mismatch or empty"
    xis = [2.0 * e - 1.0 for e in xs]
    B = compute_apc_basis(xis, degree=degree)
    # Build design matrix by evaluating APC basis at sample points
    Phi: List[List[float]] = []
    for xi in xis:
        vals = apc_eval(B, xi)
        Phi.append(vals[: degree + 1])
    # Solve LS via normal equations (small system)
    m = degree + 1
    A = [[0.0 for _ in range(m)] for _ in range(m)]
    b = [0.0 for _ in range(m)]
    for row, y in zip(Phi, ys):
        for i in range(m):
            b[i] += row[i] * y
            for j in range(m):
                A[i][j] += row[i] * row[j]
    for i in range(m):
        A[i].append(b[i])
    for i in range(m):
        pivot = A[i][i] if A[i][i] != 0.0 else 1e-12
        inv_pivot = 1.0 / pivot
        for j in range(i, m + 1):
            A[i][j] *= inv_pivot
        for k in range(i + 1, m):
            factor = A[k][i]
            if factor == 0.0:
                continue
            for j in range(i, m + 1):
                A[k][j] -= factor * A[i][j]
    x = [0.0 for _ in range(m)]
    for i in range(m - 1, -1, -1):
        s = A[i][m]
        for j in range(i + 1, m):
            s -= A[i][j] * x[j]
        x[i] = s
    return [float(v) for v in x]


