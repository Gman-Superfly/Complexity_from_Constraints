from __future__ import annotations

import argparse
from typing import List, Dict, Any
import math

from modules.polynomial.polynomial_energy import PolynomialEnergyModule
from modules.polynomial.rhs_probe import fit_rhs_legendre, eval_rhs_legendre
from modules.polynomial.apc import compute_apc_basis, evaluate_basis as apc_eval
from cf_logging.metrics_log import log_records


def make_energy_module(degree: int) -> tuple[PolynomialEnergyModule, Dict[str, Any]]:
    # Example coefficients for a smooth nontrivial landscape
    coeffs = [0.0] * (degree + 1)
    # Small bias and curvature
    if degree >= 2:
        coeffs[2] = 0.3
    if degree >= 4:
        coeffs[4] = 0.1
    constraints: Dict[str, Any] = {"poly_coeffs": coeffs}
    mod = PolynomialEnergyModule(degree=degree, basis="legendre")
    return mod, constraints


def dE_deta(mod: PolynomialEnergyModule, eta: float, constraints: Dict[str, Any]) -> float:
    return float(mod.d_local_energy_d_eta(float(eta), constraints))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--basis", choices=["legendre", "apc"], default="legendre")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--step_size", type=float, default=0.05)
    parser.add_argument("--eta0", type=float, default=0.3)
    parser.add_argument("--run_id", type=str, default="rhs_probe")
    args = parser.parse_args()

    mod, constraints = make_energy_module(degree=args.degree)

    # Sample RHS on a grid
    xs: List[float] = [i / max(1, args.samples - 1) for i in range(args.samples)]
    rhs_true: List[float] = [-dE_deta(mod, x, constraints) for x in xs]

    # Fit RHS in Legendre basis
    if args.basis == "legendre":
        coeffs = fit_rhs_legendre(xs, rhs_true, degree=args.degree)
        rhs_fn = lambda e: eval_rhs_legendre(e, coeffs)
    else:
        # Compute APC basis on xi samples
        xis = [2.0 * e - 1.0 for e in xs]
        B = compute_apc_basis(xis, degree=args.degree)
        # Build design matrix in APC basis
        Phi = [apc_eval(B, xi)[: args.degree + 1] for xi in xis]
        m = args.degree + 1
        A = [[0.0 for _ in range(m)] for _ in range(m)]
        b = [0.0 for _ in range(m)]
        for row, y in zip(Phi, rhs_true):
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
        coeffs = [0.0 for _ in range(m)]
        for i in range(m - 1, -1, -1):
            s = A[i][m]
            for j in range(i + 1, m):
                s -= A[i][j] * coeffs[j]
            coeffs[i] = s
        def rhs_fn(e: float) -> float:
            xi = 2.0 * float(e) - 1.0
            vals = apc_eval(B, xi)[: args.degree + 1]
            return float(sum(c * v for c, v in zip(coeffs, vals)))

    # Simulate Euler from eta0 with true RHS vs fitted RHS
    def euler_sim(rhs_fn, eta0: float, steps: int, h: float) -> tuple[List[float], List[float]]:
        etas: List[float] = [float(eta0)]
        energies: List[float] = [float(mod.local_energy(float(eta0), constraints))]
        for _ in range(steps):
            eta = float(etas[-1])
            rhs = float(rhs_fn(eta))
            new_eta = float(max(0.0, min(1.0, eta + h * rhs)))
            etas.append(new_eta)
            energies.append(float(mod.local_energy(new_eta, constraints)))
        return etas, energies

    et_true, en_true = euler_sim(lambda e: -dE_deta(mod, e, constraints), args.eta0, args.steps, args.step_size)
    et_fit, en_fit = euler_sim(rhs_fn, args.eta0, args.steps, args.step_size)

    # Compute metrics
    final_error = abs(float(et_true[-1]) - float(et_fit[-1]))
    energy_monotone_true = all(en_true[i + 1] <= en_true[i] + 1e-12 for i in range(len(en_true) - 1))
    energy_monotone_fit = all(en_fit[i + 1] <= en_fit[i] + 1e-12 for i in range(len(en_fit) - 1))

    rows: List[Dict[str, Any]] = [{
        "run_id": args.run_id,
        "degree": int(args.degree),
        "samples": int(args.samples),
        "steps": int(args.steps),
        "step_size": float(args.step_size),
        "eta0": float(args.eta0),
        "basis": str(args.basis),
        "rhs_coeffs": str([round(c, 6) for c in coeffs]),
        "eta_final_true": float(et_true[-1]),
        "eta_final_fit": float(et_fit[-1]),
        "eta_final_abs_error": float(final_error),
        "energy_final_true": float(en_true[-1]),
        "energy_final_fit": float(en_fit[-1]),
        "energy_monotone_true": bool(energy_monotone_true),
        "energy_monotone_fit": bool(energy_monotone_fit),
    }]
    path = log_records("rhs_probe", rows)
    print(f"Wrote RHS probe row to {path}")


if __name__ == "__main__":
    main()


