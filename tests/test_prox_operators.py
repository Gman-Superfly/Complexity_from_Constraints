from __future__ import annotations

from core.prox_utils import prox_quadratic_pair, prox_asym_hinge_pair, prox_linear_gate


def test_prox_quadratic_pair_symmetry_and_bounds() -> None:
    x, y = prox_quadratic_pair(0.8, 0.2, weight=1.0, tau=0.1)
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    # Should move toward each other
    assert abs(x - y) < abs(0.8 - 0.2)


def test_prox_asym_hinge_pair_inactive_region_identity() -> None:
    # gap <= 0 => identity projection
    x, y = prox_asym_hinge_pair(0.9, 0.1, weight=1.0, alpha=1.0, beta=1.0, tau=0.1)
    assert x == 0.9 and y == 0.1


def test_prox_linear_gate_moves_with_coeff_sign() -> None:
    eta = prox_linear_gate(0.5, coeff=1.0, tau=0.1)
    assert eta > 0.5
    eta2 = prox_linear_gate(0.5, coeff=-1.0, tau=0.1)
    assert eta2 < 0.5


