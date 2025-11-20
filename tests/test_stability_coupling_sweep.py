from __future__ import annotations

import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import AsymmetricHingeCoupling
from modules.gating.energy_gating import EnergyGatingModule


def test_stability_across_coupling_strength_sweep():
    # small chain with asymmetric hinge favoring j -> i influence
    m = 6
    mods = [EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3) for _ in range(m)]
    etas0 = list(np.linspace(0.9, 0.1, num=m))
    strengths = [0.1, 1.0, 5.0, 10.0]
    for w in strengths:
        coups = []
        for i in range(m - 1):
            coups.append((i, i + 1, AsymmetricHingeCoupling(weight=w, alpha_i=1.0, beta_j=1.0)))
        coord = EnergyCoordinator(
            modules=mods,
            couplings=coups,
            constraints={},
            grad_eps=1e-6,
            step_size=0.05,
            use_analytic=True,
            line_search=True,
            normalize_grads=True,
        )
        energies: list[float] = []
        coord.on_energy_updated.append(lambda F: energies.append(F))
        out = coord.relax_etas(etas0, steps=50)
        # energy should not increase overall
        assert energies[-1] <= energies[0] + 1e-6
        # no NaNs and bounds preserved
        assert all((e == e) and (0.0 <= e <= 1.0) for e in out)


