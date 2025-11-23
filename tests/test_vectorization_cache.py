from __future__ import annotations

import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling, DirectedHingeCoupling
from core.interfaces import EnergyModule, OrderParameter


class LinearModule(EnergyModule):
    def __init__(self, bias: float) -> None:
        self.bias = bias

    def compute_eta(self, x: float) -> OrderParameter:
        return float(np.clip(x + self.bias, 0.0, 1.0))

    def local_energy(self, eta: OrderParameter, constraints):
        return (float(eta) - self.bias) ** 2


def build_coord(vectorized: bool) -> EnergyCoordinator:
    modules = [LinearModule(bias=0.2), LinearModule(bias=0.6), LinearModule(bias=0.8)]
    couplings = [
        (0, 1, QuadraticCoupling(weight=0.4)),
        (1, 2, DirectedHingeCoupling(weight=0.7)),
    ]
    return EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={},
        use_vectorized_quadratic=vectorized,
        use_vectorized_hinges=vectorized,
        stability_guard=True,
    )


def test_vectorized_gradients_match_naive() -> None:
    etas = [0.1, 0.5, 0.9]
    coord_vec = build_coord(vectorized=True)
    coord_naive = build_coord(vectorized=False)
    grads_vec = coord_vec._analytic_grads(list(etas))
    grads_naive = coord_naive._analytic_grads(list(etas))
    np.testing.assert_allclose(grads_vec, grads_naive, atol=1e-6)


def test_rebuild_vectorization_cache_is_safe() -> None:
    coord = build_coord(vectorized=True)
    coord.rebuild_vectorization_cache()
    grads = coord._analytic_grads([0.2, 0.3, 0.4])
    assert len(grads) == 3

