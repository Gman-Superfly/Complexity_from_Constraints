from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad, SupportsPrecision


class PrecisionTestModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Quadratic module F = 0.5*k*(eta - b)^2 with analytic grad and curvature."""

    def __init__(self, stiffness: float, bias: float = 0.0) -> None:
        self.k = float(stiffness)
        self.b = float(bias)

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        d = float(eta) - self.b
        return 0.5 * self.k * d * d

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return self.k * (float(eta) - self.b)

    def curvature(self, eta: OrderParameter) -> float:
        return self.k


def test_precision_preconditioning_scales_updates_inversely_to_curvature() -> None:
    """With precision preconditioning, stiff dim should update less than loose dim."""
    stiff = PrecisionTestModule(stiffness=10.0, bias=0.0)
    loose = PrecisionTestModule(stiffness=0.5, bias=0.0)

    coord = EnergyCoordinator(
        modules=[stiff, loose],
        couplings=[],
        constraints={},
        use_analytic=True,
        line_search=False,                # direct step to measure update magnitudes
        use_precision_preconditioning=True,
        precision_epsilon=1e-8,
        step_size=0.1,
        enable_orthogonal_noise=False,    # deterministic
    )

    etas0 = [0.8, -0.2]
    before = np.asarray(etas0, dtype=float)
    after = np.asarray(coord.relax_etas(etas0, steps=1), dtype=float)
    delta = np.abs(after - before)

    # The stiff dimension (index 0) should move less than loose (index 1)
    assert delta[0] < delta[1], f"expected stiffer dim to change less, got Δ={delta}"

