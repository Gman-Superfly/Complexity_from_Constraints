from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule


def test_relax_etas_non_increasing_energy():
    mods = [EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3) for _ in range(2)]
    coups = [(0, 1, QuadraticCoupling(weight=0.5))]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        step_size=0.01,
        grad_eps=1e-6,
        use_analytic=True,
        line_search=True,
    )
    etas = [0.8, 0.1]
    energies: list[float] = []
    coord.on_energy_updated.append(lambda F: energies.append(F))
    coord.relax_etas(etas, steps=50)
    assert len(energies) > 1
    assert energies[0] >= energies[-1] - 1e-9


