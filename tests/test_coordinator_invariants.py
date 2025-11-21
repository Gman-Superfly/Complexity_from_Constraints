from __future__ import annotations

import pytest

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from core.interfaces import EnergyModule, OrderParameter


class NaNEnergyModule(EnergyModule):
    def compute_eta(self, x):
        return 0.5

    def local_energy(self, eta: OrderParameter, constraints):
        return float("nan")


class SimpleModule(EnergyModule):
    def compute_eta(self, x):
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints):
        return float(eta * eta)


def test_coordinator_invariants_raise_on_nan_energy():
    coord = EnergyCoordinator(
        modules=[NaNEnergyModule()],
        couplings=[],
        constraints={},
        enforce_invariants=True,
    )
    with pytest.raises(AssertionError):
        coord.relax_etas([0.5], steps=1)


def test_term_weight_calibration_floor_ceiling():
    modules = [SimpleModule(), SimpleModule()]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[],
        constraints={
            "term_weights": {
                "local:SimpleModule": 0.001,
                "local:Other": 2.0,
            }
        },
        term_weight_floor=0.1,
        term_weight_ceiling=0.5,
    )
    weights = coord._combined_term_weights()
    assert weights["local:SimpleModule"] == pytest.approx(0.1)
    assert weights["local:Other"] == pytest.approx(0.5)


def test_auto_balance_term_weights_warns_and_scales():
    modules = [SimpleModule(), SimpleModule()]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[],
        constraints={"term_weights": {"local:SimpleModule": 1.0}},
        auto_balance_term_weights=True,
        term_norm_target=0.1,
        max_term_norm_ratio=2.0,
    )
    with pytest.warns(RuntimeWarning):
        coord.relax_etas([0.9, 0.2], steps=1)
    assert coord._term_weights["local:SimpleModule"] <= 0.1

