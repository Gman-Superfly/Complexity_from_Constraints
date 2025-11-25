from typing import Any, Mapping

import pytest

from core.coordinator import EnergyCoordinator
from core.interfaces import (
    EnergyModule,
    OrderParameter,
    SupportsLocalEnergyGrad,
    SupportsPrecision,
)


class PrecisionTestModule(EnergyModule, SupportsLocalEnergyGrad, SupportsPrecision):
    """Simple quadratic module with analytic gradient and curvature."""

    def __init__(self, stiffness: float, bias: float = 0.0):
        self._stiffness = float(stiffness)
        self._bias = float(bias)

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x) if isinstance(x, (int, float)) else 0.5

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        # F = 0.5 * k * (eta - bias)^2
        delta = float(eta) - self._bias
        return 0.5 * self._stiffness * delta * delta

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        # dF/dη = k * (η - bias)
        return self._stiffness * (float(eta) - self._bias)

    def curvature(self, eta: OrderParameter) -> float:
        # Second derivative is constant and equals stiffness.
        return self._stiffness


def test_precision_cache_updates_from_modules():
    """Coordinator should capture per-module stiffness via SupportsPrecision."""
    stiff = PrecisionTestModule(stiffness=8.0, bias=0.25)
    loose = PrecisionTestModule(stiffness=0.2, bias=0.75)

    coordinator = EnergyCoordinator(
        modules=[stiff, loose],
        couplings=[],
        constraints={},
        step_size=0.01,
    )

    etas = [0.3, 0.8]

    # Cache should be empty prior to relaxation.
    assert coordinator._precision_cache is None

    result = coordinator.relax_etas(etas, steps=1)
    assert len(result) == 2

    precision_diag = coordinator.get_precision_diagonal()
    assert precision_diag == [pytest.approx(8.0), pytest.approx(0.2)]

    # Cache should remain stable if we run another step with different etas.
    coordinator.relax_etas(result, steps=1)
    precision_diag_2 = coordinator.get_precision_diagonal()
    assert precision_diag_2 == [pytest.approx(8.0), pytest.approx(0.2)]


