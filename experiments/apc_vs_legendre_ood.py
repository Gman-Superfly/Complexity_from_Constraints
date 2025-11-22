"""Compare Legendre vs APC bases on synthetic OOD splits for stability/backtracks.

Procedure:
  - Generate two ξ-splits (train/test) from different distributions.
  - Fit APC basis on the train split.
  - Configure a PolynomialEnergyModule with fixed coefficients.
  - Run coordinator relaxation with line search once for basis="legendre" and once for basis="apc".
  - Log final energy and total backtracks; repeat on OOD split.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List
import numpy as np

from cf_logging.metrics_log import log_records
from cf_logging.observability import RelaxationTracker, EnergyBudgetTracker
from core.coordinator import EnergyCoordinator
from modules.polynomial.apc import compute_apc_basis
from modules.polynomial.polynomial_energy import PolynomialEnergyModule


def make_splits(n_train: int, n_test: int, seed: int) -> tuple[List[float], List[float]]:
    rng = np.random.default_rng(seed)
    # Train: mildly concentrated near 0
    xi_train = np.clip(rng.normal(0.0, 0.5, size=n_train), -1.0, 1.0).tolist()
    # Test (OOD): bimodal near ±0.8
    modes = rng.choice([-0.8, 0.8], size=n_test)
    xi_test = np.clip(modes + rng.normal(0.0, 0.2, size=n_test), -1.0, 1.0).tolist()
    return xi_train, xi_test


def run_basis_scenario(
    basis: str,
    apc_B: List[List[float]] | None,
    coeffs: List[float],
    eta0: float,
    steps: int,
    track_relaxation: bool,
    track_budget: bool,
    run_id: str,
) -> Dict[str, Any]:
    # Single-module coordinator; inputs are raw η
    mod = PolynomialEnergyModule(degree=len(coeffs) - 1, basis=basis)
    constraints: Dict[str, Any] = {"poly_coeffs": coeffs}
    if basis == "apc":
        assert apc_B is not None, "APC basis required"
        constraints["apc_basis"] = apc_B
    coord = EnergyCoordinator(
        modules=[mod],
        couplings=[],
        constraints=constraints,
        use_analytic=True,
        line_search=True,
        normalize_grads=True,
        enforce_invariants=True,
        max_backtrack=10,
        step_size=0.2,
    )
    tracker = None
    budget_tracker = None
    if track_relaxation:
        tracker = RelaxationTracker(name="apc_legendre_relaxation", run_id=run_id)
        tracker.attach(coord)
    if track_budget:
        budget_tracker = EnergyBudgetTracker(name="apc_legendre_budget", run_id=run_id)
        budget_tracker.attach(coord)
    etas0 = coord.compute_etas([eta0])
    energies: List[float] = []
    coord.on_energy_updated.append(lambda F: energies.append(F))
    final = coord.relax_etas(etas0, steps=steps)
    energy_final = coord.energy(final)
    if tracker is not None:
        tracker.flush()
    if budget_tracker is not None:
        budget_tracker.flush()
    return {
        "basis": basis,
        "energy_final": float(energy_final),
        "total_backtracks": int(coord._total_backtracks),  # instrumentation
        "steps": steps,
        "eta0": float(eta0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n_train", type=int, default=1024)
    parser.add_argument("--n_test", type=int, default=1024)
    parser.add_argument("--track_relaxation", action="store_true")
    parser.add_argument("--track_budget", action="store_true")
    args = parser.parse_args()

    xi_train, xi_test = make_splits(args.n_train, args.n_test, args.seed)
    B_train = compute_apc_basis(xi_train, degree=args.degree)

    # Fixed coefficients for both bases (arbitrary but stable)
    coeffs = [0.2, -0.15, 0.25, 0.0, 0.1][: args.degree + 1]
    # Initial eta
    eta0 = 0.9

    rows: List[Dict[str, Any]] = []
    # Legendre on train/test
    rows.append({**run_basis_scenario("legendre", None, coeffs, eta0, args.steps, args.track_relaxation, args.track_budget, run_id="legendre_train"), "split": "train"})
    rows.append({**run_basis_scenario("legendre", None, coeffs, eta0, args.steps, args.track_relaxation, args.track_budget, run_id="legendre_test"), "split": "test"})
    # APC fitted on train, evaluated on train/test (OOD)
    rows.append({**run_basis_scenario("apc", B_train, coeffs, eta0, args.steps, args.track_relaxation, args.track_budget, run_id="apc_train"), "split": "train"})
    rows.append({**run_basis_scenario("apc", B_train, coeffs, eta0, args.steps, args.track_relaxation, args.track_budget, run_id="apc_test"), "split": "test"})

    out = log_records("apc_vs_legendre_ood", rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

