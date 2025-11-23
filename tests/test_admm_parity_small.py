from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling, DirectedHingeCoupling, GateBenefitCoupling
from modules.gating.energy_gating import EnergyGatingModule


def _make_small_setup() -> Tuple[List[Any], List[Tuple[int, int, Any]], Dict[str, Any], List[Any]]:
    # Two gating modules; simple quadratic and hinge + gate-benefit couplings
    m0 = EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.2)
    m1 = EnergyGatingModule(gain_fn=lambda _: 0.1, cost=0.05, a=0.25, b=0.35)
    mods = [m0, m1]
    coups: List[Tuple[int, int, Any]] = [
        (0, 1, QuadraticCoupling(weight=0.3)),
        (0, 1, DirectedHingeCoupling(weight=0.2)),
        (0, 1, GateBenefitCoupling(weight=0.4, delta_key="delta_eta_domain")),
    ]
    constraints: Dict[str, Any] = {"delta_eta_domain": 0.05}
    inputs: List[Any] = [None, None]
    return mods, coups, constraints, inputs


def test_admm_parity_with_gradient_on_small_problem() -> None:
    mods, coups, constraints, inputs = _make_small_setup()
    # Gradient baseline
    coord_grad = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints=constraints,
        use_analytic=True,
        line_search=False,
        step_size=0.05,
    )
    etas0 = coord_grad.compute_etas(inputs)
    e0 = coord_grad.energy(list(etas0))
    etas_g = coord_grad.relax_etas(list(etas0), steps=50)
    e_grad = coord_grad.energy(etas_g)

    # ADMM path with gate prox/damping
    mods2, coups2, constraints2, inputs2 = _make_small_setup()
    coord_admm = EnergyCoordinator(
        modules=mods2,
        couplings=coups2,
        constraints=constraints2,
        use_admm=True,
        admm_steps=50,
        admm_rho=1.0,
        admm_step_size=0.05,
        admm_gate_prox=True,
        admm_gate_damping=0.5,
    )
    etas_a = coord_admm.compute_etas(inputs2)
    e0a = coord_admm.energy(list(etas_a))
    etas_a = coord_admm.relax_etas_admm(etas_a, steps=coord_admm.admm_steps, rho=coord_admm.admm_rho, step_size=coord_admm.admm_step_size)
    e_admm = coord_admm.energy(etas_a)

    # Both should reduce energy from start
    assert e_grad <= e0 + 1e-9
    assert e_admm <= e0a + 1e-9
    # Parity within a reasonable tolerance on this small problem
    assert abs(e_admm - e_grad) <= 5e-3


