from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from cf_logging.metrics_log import log_records
from core.coordinator import EnergyCoordinator
from modules.polynomial.polynomial_energy import PolynomialEnergyModule


def make_poly_module(
    degree: int,
    basis: str,
    normalize_domain: bool,
    coeffs: Optional[List[float]] = None,
    apc_basis_path: Optional[str] = None,
) -> tuple[PolynomialEnergyModule, Dict[str, Any]]:
    constraints: Dict[str, Any] = {}
    if coeffs is None:
        coeffs = [0.0] * (degree + 1)
        # small convex curvature
        if degree >= 2:
            coeffs[2] = 0.3
        if degree >= 4:
            coeffs[4] = 0.05
    constraints["poly_coeffs"] = coeffs

    mod = PolynomialEnergyModule(degree=degree, basis="legendre", normalize_domain=normalize_domain)
    if basis.lower() == "apc":
        if not apc_basis_path:
            raise ValueError("APC basis requested but --apc_basis_path not provided")
        with open(apc_basis_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        constraints["apc_basis"] = data.get("apc_basis")
        mod = PolynomialEnergyModule(degree=degree, basis="apc", normalize_domain=normalize_domain)
    return mod, constraints


def run_once(
    basis: str,
    normalize_domain: bool,
    degree: int,
    steps: int,
    step_size: float,
    apc_basis_path: Optional[str],
    run_id: str,
) -> Dict[str, Any]:
    mod, constraints = make_poly_module(
        degree=degree, basis=basis, normalize_domain=normalize_domain, apc_basis_path=apc_basis_path
    )
    coord = EnergyCoordinator(
        modules=[mod],
        couplings=[],
        constraints=constraints,
        use_analytic=True,
        line_search=True,  # allow backtracking to observe conditioning differences
        step_size=step_size,
        log_contraction_margin=True,
    )
    etas = coord.compute_etas([0.8])  # start near a boundary to stress clamps
    e0 = coord.energy(list(etas))
    coord.relax_etas(etas, steps=steps)
    total_backtracks = getattr(coord, "_total_backtracks", None)
    e_final = coord.energy(etas)
    return {
        "run_id": run_id,
        "basis": basis,
        "normalize_domain": bool(normalize_domain),
        "degree": int(degree),
        "steps": int(steps),
        "step_size": float(step_size),
        "energy_initial": float(e0),
        "energy_final": float(e_final),
        "total_backtracks": int(total_backtracks or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--step_size", type=float, default=0.05)
    parser.add_argument("--run_id", type=str, default="poly_cond")
    parser.add_argument("--apc_basis_path", type=str, default=None, help="Path to JSON file containing {'apc_basis': [[...], ...]}")
    parser.add_argument("--out_name", type=str, default="polynomial_basis_conditioning")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    # Compare Legendre normalized vs non-normalized
    rows.append(run_once("legendre", True, args.degree, args.steps, args.step_size, None, args.run_id + "_leg_norm"))
    rows.append(run_once("legendre", False, args.degree, args.steps, args.step_size, None, args.run_id + "_leg_raw"))
    # Optional APC (requires basis file)
    if args.apc_basis_path:
        rows.append(run_once("apc", True, args.degree, args.steps, args.step_size, args.apc_basis_path, args.run_id + "_apc"))

    path = log_records(args.out_name, rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()


