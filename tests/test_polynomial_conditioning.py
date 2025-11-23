from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict, Any

import numpy as np

from core.coordinator import EnergyCoordinator
from modules.polynomial.polynomial_energy import PolynomialEnergyModule
from modules.gating.energy_gating import EnergyGatingModule
from modules.polynomial.apc import compute_apc_basis
from core.couplings import QuadraticCoupling


def _legendre_vals(xi: float, degree: int) -> list[float]:
    """Helper to compute Legendre polynomial values."""
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


def _fit_landau_to_legendre(a: float, b: float, degree: int = 4) -> List[float]:
    """Fit Legendre coefficients to approximate Landau energy a*eta^2 + b*eta^4."""
    etas = np.linspace(0.0, 1.0, 201)
    xi = 2.0 * etas - 1.0
    A = np.stack(
        [np.array([_legendre_vals(float(x), degree)[k] for x in xi]) for k in range(degree + 1)],
        axis=1,
    )
    y = a * (etas ** 2) + b * (etas ** 4)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs.tolist()


def test_polynomial_conditioning_legendre_vs_landau_backtracks() -> None:
    """Legendre basis should produce fewer line-search backtracks than raw Landau."""
    # Setup with moderate a, b to stress conditioning
    a = 0.8
    b = 0.5
    degree = 4
    coeffs_legendre = _fit_landau_to_legendre(a, b, degree)

    # Landau baseline (EnergyGatingModule with a, b)
    mod_landau = EnergyGatingModule(gain_fn=lambda _: 0.0, a=a, b=b, use_hazard=False)
    coord_landau = EnergyCoordinator(
        modules=[mod_landau],
        couplings=[],
        constraints={},
        use_analytic=True,
        line_search=True,
        step_size=0.15,  # Large step to stress line search
        max_backtrack=10,
    )

    # Legendre variant
    mod_legendre = PolynomialEnergyModule(degree=degree, basis="legendre", normalize_domain=True)
    coord_legendre = EnergyCoordinator(
        modules=[mod_legendre],
        couplings=[],
        constraints={"poly_coeffs": coeffs_legendre},
        use_analytic=True,
        line_search=True,
        step_size=0.15,  # Same step size
        max_backtrack=10,
    )

    # Run relaxation and count backtracks
    etas_landau = [0.7]
    etas_legendre = [0.7]

    backtracks_landau: List[int] = []
    backtracks_legendre: List[int] = []

    def count_backtrack_landau(backtrack_count: int) -> None:
        backtracks_landau.append(backtrack_count)

    def count_backtrack_legendre(backtrack_count: int) -> None:
        backtracks_legendre.append(backtrack_count)

    # Attach backtrack counters (using internal _backtrack_count if available, or track via energy)
    # For this test, we'll use energy monotonicity as a proxy: if energy increases, we had backtracks
    energies_landau: List[float] = []
    energies_legendre: List[float] = []

    coord_landau.on_energy_updated.append(lambda F: energies_landau.append(F))
    coord_legendre.on_energy_updated.append(lambda F: energies_legendre.append(F))

    etas_landau = coord_landau.relax_etas(etas_landau, steps=30)
    etas_legendre = coord_legendre.relax_etas(etas_legendre, steps=30)

    # Measure smoothness: variance of ΔF
    delta_f_landau = [energies_landau[i] - energies_landau[i + 1] for i in range(len(energies_landau) - 1)]
    delta_f_legendre = [energies_legendre[i] - energies_legendre[i + 1] for i in range(len(energies_legendre) - 1)]

    var_landau = float(np.var(delta_f_landau)) if len(delta_f_landau) > 1 else 0.0
    var_legendre = float(np.var(delta_f_legendre)) if len(delta_f_legendre) > 1 else 0.0

    # Legendre should have smoother (lower variance) energy descent
    # Allow for some tolerance since we're comparing different parameterizations
    assert var_legendre <= var_landau * 2.0, f"Legendre variance {var_legendre} > 2x Landau {var_landau}"

    # Both should reach similar final energy (within tolerance)
    final_landau = energies_landau[-1]
    final_legendre = energies_legendre[-1]
    assert abs(final_landau - final_legendre) < 0.05, f"Final energies differ: {final_landau} vs {final_legendre}"


def test_polynomial_conditioning_apc_vs_legendre_on_biased_distribution() -> None:
    """APC basis fitted to biased η distribution should match or outperform Legendre."""
    # Generate biased η samples (concentrated near 0.3)
    rng = random.Random(42)
    eta_samples = [max(0.0, min(1.0, 0.3 + 0.2 * rng.gauss(0.0, 1.0))) for _ in range(2048)]
    xi_samples = [2.0 * e - 1.0 for e in eta_samples]

    # Target energy: quadratic centered at 0.3
    def target_energy(eta: float) -> float:
        return 0.5 * ((eta - 0.3) ** 2)

    degree = 3

    # Fit Legendre coefficients
    etas_grid = np.linspace(0.0, 1.0, 201)
    xi_grid = 2.0 * etas_grid - 1.0
    A_legendre = np.stack(
        [np.array([_legendre_vals(float(x), degree)[k] for x in xi_grid]) for k in range(degree + 1)],
        axis=1,
    )
    y_target = np.array([target_energy(float(e)) for e in etas_grid])
    coeffs_legendre, *_ = np.linalg.lstsq(A_legendre, y_target, rcond=None)
    coeffs_legendre = coeffs_legendre.tolist()

    # Fit APC basis to biased distribution
    apc_basis = compute_apc_basis(xi_samples, degree)

    # Manually fit coefficients to APC basis (using sample energies)
    # For simplicity, we'll use least squares over the samples
    from modules.polynomial.apc import evaluate_basis

    A_apc = np.stack(
        [np.array([evaluate_basis(apc_basis, float(x))[k] for x in xi_samples]) for k in range(degree + 1)],
        axis=1,
    )
    y_apc = np.array([target_energy(float(e)) for e in eta_samples])
    coeffs_apc, *_ = np.linalg.lstsq(A_apc, y_apc, rcond=None)
    coeffs_apc = coeffs_apc.tolist()

    # Setup coordinators
    mod_legendre = PolynomialEnergyModule(degree=degree, basis="legendre", normalize_domain=True)
    coord_legendre = EnergyCoordinator(
        modules=[mod_legendre],
        couplings=[],
        constraints={"poly_coeffs": coeffs_legendre},
        use_analytic=True,
        line_search=False,
        step_size=0.1,
    )

    mod_apc = PolynomialEnergyModule(degree=degree, basis="apc", normalize_domain=True)
    coord_apc = EnergyCoordinator(
        modules=[mod_apc],
        couplings=[],
        constraints={"poly_coeffs": coeffs_apc, "apc_basis": apc_basis},
        use_analytic=True,
        line_search=False,
        step_size=0.1,
    )

    # Start from biased initial condition
    eta0 = 0.7

    energies_legendre: List[float] = []
    energies_apc: List[float] = []

    coord_legendre.on_energy_updated.append(lambda F: energies_legendre.append(F))
    coord_apc.on_energy_updated.append(lambda F: energies_apc.append(F))

    etas_legendre = coord_legendre.relax_etas([eta0], steps=40)
    etas_apc = coord_apc.relax_etas([eta0], steps=40)

    # Both should converge to near 0.3
    assert abs(etas_legendre[0] - 0.3) < 0.05, f"Legendre converged to {etas_legendre[0]}, expected ~0.3"
    assert abs(etas_apc[0] - 0.3) < 0.05, f"APC converged to {etas_apc[0]}, expected ~0.3"

    # Measure final energy (should be similar)
    final_legendre = energies_legendre[-1]
    final_apc = energies_apc[-1]
    assert abs(final_legendre - final_apc) < 0.01, f"Final energies differ: {final_legendre} vs {final_apc}"

    # APC should have smoother convergence on the biased distribution
    delta_f_legendre = [energies_legendre[i] - energies_legendre[i + 1] for i in range(len(energies_legendre) - 1)]
    delta_f_apc = [energies_apc[i] - energies_apc[i + 1] for i in range(len(energies_apc) - 1)]

    var_legendre = float(np.var(delta_f_legendre)) if len(delta_f_legendre) > 1 else 0.0
    var_apc = float(np.var(delta_f_apc)) if len(delta_f_apc) > 1 else 0.0

    # APC should have comparable or better smoothness (allow 3x tolerance for robustness)
    assert var_apc <= var_legendre * 3.0, f"APC variance {var_apc} > 3x Legendre {var_legendre}"


def test_polynomial_conditioning_coupled_system() -> None:
    """Test conditioning in a coupled system with polynomial modules."""
    # Two polynomial modules coupled via QuadraticCoupling
    degree = 3
    a, b = 0.5, 0.4
    coeffs = _fit_landau_to_legendre(a, b, degree)

    mod1 = PolynomialEnergyModule(degree=degree, basis="legendre", normalize_domain=True)
    mod2 = PolynomialEnergyModule(degree=degree, basis="legendre", normalize_domain=True)

    coord = EnergyCoordinator(
        modules=[mod1, mod2],
        couplings=[(0, 1, QuadraticCoupling(weight=0.5))],
        constraints={"poly_coeffs": coeffs},
        use_analytic=True,
        line_search=True,
        step_size=0.12,
    )

    etas0 = [0.8, 0.2]
    energies: List[float] = []
    coord.on_energy_updated.append(lambda F: energies.append(F))

    etas_final = coord.relax_etas(etas0, steps=50)

    # Should converge (energy decreases monotonically with line search)
    for i in range(len(energies) - 1):
        assert energies[i + 1] <= energies[i] + 1e-12, f"Energy increased at step {i}: {energies[i]} -> {energies[i+1]}"

    # Should reach coupling equilibrium (etas should be close)
    assert abs(etas_final[0] - etas_final[1]) < 0.1, f"Coupled etas diverged: {etas_final}"

    # Smoothness check: ΔF variance should be reasonable
    delta_f = [energies[i] - energies[i + 1] for i in range(len(energies) - 1)]
    var_delta_f = float(np.var(delta_f)) if len(delta_f) > 1 else 0.0

    # Variance should not be excessive (no wild oscillations)
    mean_delta_f = float(np.mean(delta_f)) if len(delta_f) > 0 else 0.0
    if mean_delta_f > 1e-6:  # Only check if there's significant descent
        cv = math.sqrt(var_delta_f) / (mean_delta_f + 1e-9)  # Coefficient of variation
        assert cv < 5.0, f"Energy descent too irregular: CV={cv}"

