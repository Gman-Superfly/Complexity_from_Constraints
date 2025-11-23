from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule
from modules.sequence.monotonic_eta import SequenceConsistencyModule


def test_relax_etas_proximal_non_increasing_energy() -> None:
    seq_mod = SequenceConsistencyModule(samples=64)
    # simple constant gain so gate stays small
    gate_mod = EnergyGatingModule(gain_fn=lambda _x: 0.0, cost=0.1, a=0.2, b=0.2)
    modules = [seq_mod, gate_mod]
    couplings = [(0, 1, QuadraticCoupling(weight=0.1))]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={"term_weights": {"coup:QuadraticCoupling": 0.5}},
        operator_splitting=True,
        prox_tau=0.05,
        prox_steps=10,
        enforce_invariants=True,
    )
    etas0 = coord.compute_etas([[0.0, 1.0], None])
    F0 = coord.energy(etas0)
    etas1 = coord.relax_etas(etas0, steps=10)
    F1 = coord.energy(etas1)
    assert F1 <= F0 + 1e-9


def test_relax_etas_proximal_matches_gradient_on_small_case() -> None:
    # On a small convex quadratic toy with hinge, prox and gradient paths both reduce energy
    mods = [SequenceConsistencyModule(), SequenceConsistencyModule()]
    from core.couplings import DirectedHingeCoupling
    coups = [(0, 1, QuadraticCoupling(weight=0.5)), (0, 1, DirectedHingeCoupling(weight=0.2))]

    coord_grad = EnergyCoordinator(mods, coups, constraints={}, use_analytic=True, line_search=True)
    coord_prox = EnergyCoordinator(mods, coups, constraints={}, operator_splitting=True, prox_steps=60, prox_tau=0.05)

    etas0 = [0.3, 0.6]
    e0 = coord_grad.energy(etas0)
    etas_g = coord_grad.relax_etas(etas0, steps=60)
    e_grad = coord_grad.energy(etas_g)

    e1 = coord_prox.energy(etas0)
    etas_p = coord_prox.relax_etas_proximal(etas0, steps=60, tau=0.05)
    e_prox = coord_prox.energy(etas_p)

    assert e_grad <= e0 + 1e-9
    assert e_prox <= e1 + 1e-9
    assert abs(e_prox - e_grad) <= 0.2