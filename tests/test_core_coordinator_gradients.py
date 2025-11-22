from __future__ import annotations

import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import (
    QuadraticCoupling,
    DirectedHingeCoupling,
    AsymmetricHingeCoupling,
    GateBenefitCoupling,
    DampedGateBenefitCoupling,
)
from modules.gating.energy_gating import EnergyGatingModule


def test_analytic_grads_match_finite_diff():
    mods = [
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.1, b=0.4),
    ]
    coups = [(0, 1, QuadraticCoupling(weight=0.5))]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        use_analytic=True,
    )
    etas = [0.2, 0.7]
    ana = coord._analytic_grads(etas)
    base = coord.energy(etas)
    eps = 1e-6
    num = []
    for i in range(len(etas)):
        bumped = etas.copy()
        bumped[i] += eps
        num.append((coord.energy(bumped) - base) / eps)
    for a, n in zip(ana, num):
        assert abs(a - n) < 1e-3


def test_neighbor_gradients_only_matches_full_fd():
    mods = [
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.25, b=0.35),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.4),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.15, b=0.3),
    ]
    coups = [
        (0, 1, QuadraticCoupling(weight=0.7)),
        (1, 2, QuadraticCoupling(weight=0.5)),
    ]
    coord_full = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        neighbor_gradients_only=False,
    )
    coord_neighbor = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        neighbor_gradients_only=True,
    )
    etas = [0.3, 0.6, 0.2]
    full = coord_full._finite_diff_grads(etas)
    neighbor = coord_neighbor._finite_diff_grads(etas)
    np.testing.assert_allclose(neighbor, full, rtol=1e-5, atol=1e-7)


def test_vectorized_hinges_match_reference():
    mods = [
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.25, b=0.35),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.15, b=0.25),
    ]
    coups = [
        (0, 1, DirectedHingeCoupling(weight=0.8)),
        (1, 2, AsymmetricHingeCoupling(weight=0.6, alpha_i=0.8, beta_j=1.2)),
    ]
    etas = [0.2, 0.6, 0.1]
    coord_base = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        use_analytic=True,
    )
    coord_vec = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        use_analytic=True,
        use_vectorized_hinges=True,
    )
    base = coord_base._analytic_grads(list(etas))
    vec = coord_vec._analytic_grads(list(etas))
    np.testing.assert_allclose(base, vec, rtol=1e-6, atol=1e-8)


def test_coordinate_descent_mode_matches_gradient_mode():
    mods = [
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.25, b=0.35),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.15, b=0.3),
    ]
    coups = [
        (0, 1, QuadraticCoupling(weight=0.7)),
        (1, 2, QuadraticCoupling(weight=0.5)),
    ]
    etas = [0.8, 0.3, 0.1]
    coord_grad = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        use_analytic=True,
        line_search=True,
    )
    coord_coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        use_analytic=True,
        line_search=True,
        use_coordinate_descent=True,
        coordinate_steps=80,
    )
    out_grad = coord_grad.relax_etas(list(etas), steps=80)
    out_coord = coord_coord.relax_etas(list(etas), steps=80)
    energy_coord = coord_coord.energy(out_coord)
    energy_start = coord_grad.energy(list(etas))
    assert energy_coord <= energy_start + 1e-9


def test_adaptive_coordinate_descent_warm_start():
    mods = [
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.25, b=0.35),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.15, b=0.3),
    ]
    coups = [
        (0, 1, QuadraticCoupling(weight=0.7)),
        (1, 2, QuadraticCoupling(weight=0.5)),
    ]
    etas = [0.8, 0.3, 0.1]
    coord_grad = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        use_analytic=True,
        line_search=True,
    )
    coord_adaptive = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        use_analytic=True,
        line_search=True,
        adaptive_coordinate_descent=True,
        coordinate_steps=40,
    )
    out_grad = coord_grad.relax_etas(list(etas), steps=80)
    out_adapt = coord_adaptive.relax_etas(list(etas), steps=80)
    energy_grad = coord_grad.energy(out_grad)
    energy_adapt = coord_adaptive.energy(out_adapt)
    assert energy_adapt <= energy_grad + 1e-6


def test_adaptive_switch_triggers_coordinate_refresh():
    mods = [
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.25, b=0.35),
    ]
    coups = [
        (0, 1, QuadraticCoupling(weight=0.5)),
    ]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        use_analytic=True,
        adaptive_coordinate_descent=True,
        adaptive_switch_delta=10.0,
        adaptive_switch_patience=1,
        coordinate_steps=10,
    )
    coord.relax_etas([0.8, 0.2], steps=5)
    assert getattr(coord, "_adaptive_switches", 0) >= 1


def test_vectorized_gate_benefit_matches_reference():
    mods = [
        EnergyGatingModule(gain_fn=lambda _: 0.1, cost=0.05, a=0.2, b=0.3),
        EnergyGatingModule(gain_fn=lambda _: 0.0, cost=0.0, a=0.25, b=0.35),
    ]
    coups = [
        (0, 1, GateBenefitCoupling(weight=0.8, delta_key="delta_eta_domain")),
        (0, 1, DampedGateBenefitCoupling(weight=0.6, damping=0.5, eta_power=1.5, positive_scale=1.2, negative_scale=0.8)),
    ]
    constraints = {"delta_eta_domain": 0.1}
    etas = [0.6, 0.4]
    coord_vec = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints=constraints,
        use_analytic=True,
        use_vectorized_gate_benefits=True,
    )
    coord_ref = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints=constraints,
        use_analytic=True,
        use_vectorized_gate_benefits=False,
    )
    vec = coord_vec._analytic_grads(list(etas))
    ref = coord_ref._analytic_grads(list(etas))
    np.testing.assert_allclose(vec, ref, rtol=1e-6, atol=1e-8)


