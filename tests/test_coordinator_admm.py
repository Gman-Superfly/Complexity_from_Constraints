from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule


def test_relax_etas_admm_non_increasing_energy() -> None:
    # simple 2-var system with quadratic tie
    m0 = EnergyGatingModule(gain_fn=lambda _x: 0.0, a=0.2, b=0.2)
    m1 = EnergyGatingModule(gain_fn=lambda _x: 0.0, a=0.2, b=0.2)
    modules = [m0, m1]
    couplings = [(0, 1, QuadraticCoupling(weight=0.5))]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={"term_weights": {"coup:QuadraticCoupling": 1.0}},
        use_admm=True,
        admm_steps=20,
        admm_rho=1.0,
        admm_step_size=0.05,
        enforce_invariants=True,
    )
    etas0 = coord.compute_etas([None, None])
    F0 = coord.energy(etas0)
    etas1 = coord.relax_etas(etas0, steps=20)
    F1 = coord.energy(etas1)
    assert F1 <= F0 + 1e-9

