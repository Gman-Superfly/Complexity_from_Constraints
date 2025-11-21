from __future__ import annotations

import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
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


