from __future__ import annotations

from core.couplings import QuadraticCoupling, DirectedHingeCoupling, GateBenefitCoupling


def _fd_grad_i(coup, eta_i: float, eta_j: float, constraints: dict, eps: float = 1e-6) -> float:
    base = coup.coupling_energy(eta_i, eta_j, constraints)
    bi = coup.coupling_energy(eta_i + eps, eta_j, constraints)
    return (bi - base) / eps


def _fd_grad_j(coup, eta_i: float, eta_j: float, constraints: dict, eps: float = 1e-6) -> float:
    base = coup.coupling_energy(eta_i, eta_j, constraints)
    bj = coup.coupling_energy(eta_i, eta_j + eps, constraints)
    return (bj - base) / eps


def test_quadratic_coupling_grads_match_fd():
    c = QuadraticCoupling(weight=0.7)
    eta_i, eta_j = 0.3, 0.8
    gi, gj = c.d_coupling_energy_d_etas(eta_i, eta_j, {})
    gi_num = _fd_grad_i(c, eta_i, eta_j, {})
    gj_num = _fd_grad_j(c, eta_i, eta_j, {})
    assert abs(gi - gi_num) < 1e-4
    assert abs(gj - gj_num) < 1e-4


def test_directed_hinge_coupling_grads_match_fd():
    c = DirectedHingeCoupling(weight=0.9)
    # choose values with positive gap
    eta_i, eta_j = 0.2, 0.6
    gi, gj = c.d_coupling_energy_d_etas(eta_i, eta_j, {})
    gi_num = _fd_grad_i(c, eta_i, eta_j, {})
    gj_num = _fd_grad_j(c, eta_i, eta_j, {})
    assert abs(gi - gi_num) < 1e-4
    assert abs(gj - gj_num) < 1e-4
    # zero gap region: gradients ~ 0
    eta_i2, eta_j2 = 0.7, 0.4
    gi2, gj2 = c.d_coupling_energy_d_etas(eta_i2, eta_j2, {})
    assert abs(gi2) < 1e-12 and abs(gj2) < 1e-12


def test_gate_benefit_coupling_grads_match_fd():
    c = GateBenefitCoupling(weight=1.2, delta_key="delta_eta_domain")
    constraints = {"delta_eta_domain": 0.2}
    eta_i, eta_j = 0.5, 0.1
    gi, gj = c.d_coupling_energy_d_etas(eta_i, eta_j, constraints)
    gi_num = _fd_grad_i(c, eta_i, eta_j, constraints)
    gj_num = _fd_grad_j(c, eta_i, eta_j, constraints)
    assert abs(gi - gi_num) < 1e-4
    assert abs(gj - gj_num) < 1e-12


