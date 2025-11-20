from __future__ import annotations

import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule


def test_coordinate_descent_incremental_monotone_and_bounded():
    # Build a chain of modules with quadratic couplings for a smooth convex-ish landscape
    m = 16
    mods = [EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3) for _ in range(m)]
    coups = []
    for i in range(m - 1):
        coups.append((i, i + 1, QuadraticCoupling(weight=0.5)))
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        step_size=0.05,
        use_analytic=True,
        normalize_grads=True,
        use_vectorized_quadratic=True,
    )
    etas0 = list(np.linspace(0.9, 0.1, num=m))
    energies: list[float] = []
    coord.on_energy_updated.append(lambda F: energies.append(F))
    out = coord.relax_etas_coordinate(etas0, steps=200, active_tol=1e-5)
    # energy should have decreased
    assert len(energies) > 1
    assert energies[-1] <= energies[0] + 1e-9
    # etas remain in [0,1]
    assert all(0.0 - 1e-12 <= e <= 1.0 + 1e-12 for e in out)


