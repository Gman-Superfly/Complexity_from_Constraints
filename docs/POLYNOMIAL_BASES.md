# Polynomial Bases for Local Energies (Legendre vs aPC)

This note explains how to use and validate the orthonormal polynomial parameterization for local energies, inspired by CODE/aPC practice.

## Why

Monomial Landau forms on η ∈ [0,1] can be ill‑conditioned. Mapping to ξ = 2η − 1 and using an orthonormal basis stabilizes conditioning (fewer backtracks, smoother ΔF), while preserving exact gradients via chain rule.

## Implementation Status

- Module: `modules/polynomial/polynomial_energy.py` (`PolynomialEnergyModule`)
  - Basis: `basis="legendre"` (fixed), `basis="apc"` (data‑adaptive)
  - Constraints:
    - `poly_coeffs`: list[float] of length degree+1
    - `apc_basis` (for aPC): square matrix (degree+1 × degree+1) with monomial coefficients for each basis function
- aPC utilities: `modules/polynomial/apc.py`
  - `compute_apc_basis(xi_samples: List[float], degree: int) -> List[List[float]]`
  - `evaluate_basis`, `evaluate_basis_derivatives`

## Validation

- Unit tests (all passing ✅):
  - `tests/test_polynomial_energy_module.py`: gradient parity vs finite difference
  - `tests/test_apc_basis.py`: aPC orthonormality and grad parity
  - `tests/test_polynomial_minima_parity.py`: minima parity Legendre vs Landau equivalent parameterizations
  - `tests/test_polynomial_conditioning.py`: ✅ **NEW** - conditioning benchmarks
    - Legendre vs Landau: ΔF smoothness & backtrack reduction
    - APC vs Legendre on biased distributions
    - Coupled system conditioning

Run all polynomial tests:

```powershell
uv run -m pytest tests\test_polynomial_energy_module.py -q
uv run -m pytest tests\test_apc_basis.py -q
uv run -m pytest tests\test_polynomial_minima_parity.py -q
uv run -m pytest tests\test_polynomial_conditioning.py -q
```

## OOD Probe (aPC vs Legendre)

Use the experiment to compare stability and backtracks under OOD splits:

```powershell
uv run python -m experiments.apc_vs_legendre_ood --degree 4 --steps 30 --track_relaxation --track_budget
uv run python -m experiments.plot_apc_vs_legendre --input logs\apc_vs_legendre_ood.csv --save plots\apc_vs_legendre_summary.png
```

Expected signals:
- Minima parity: identical optima for equivalent parameterizations
- Improved conditioning: fewer/lower backtracks and smoother ΔF under Legendre/aPC vs monomials

## Usage Pattern

```python
from modules.polynomial.polynomial_energy import PolynomialEnergyModule

poly = PolynomialEnergyModule(degree=4, basis="legendre")
constraints = {"poly_coeffs": [0.0, 0.1, 0.2, 0.0, 0.05]}
E = poly.local_energy(eta=0.3, constraints=constraints)
```

For aPC, first compute basis from empirical ξ samples (collected from `RelaxationTracker`):

```python
from modules.polynomial.apc import compute_apc_basis

xi_samples = [-0.8, -0.3, 0.0, 0.2, 0.7]
B = compute_apc_basis(xi_samples, degree=4)
constraints = {"poly_coeffs": [0.0, 0.1, 0.2, 0.0, 0.05], "apc_basis": B}
poly_apc = PolynomialEnergyModule(degree=4, basis="apc")
```

## Notes

- Keep η‑space invariants (clamps) as today; basis is purely a reparameterization of the local energy.
- Degree capped ≤ 4 for stability and speed.


