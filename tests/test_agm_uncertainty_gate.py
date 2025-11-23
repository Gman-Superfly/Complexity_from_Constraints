from __future__ import annotations

import pytest

from core.coordinator import EnergyCoordinator
from modules.gating.energy_gating import EnergyGatingModule


def _zero_gain(_: float) -> float:
    return 0.0


def test_uncertainty_gate_relaxes_cost_when_rate_high() -> None:
    gate = EnergyGatingModule(gain_fn=_zero_gain, cost=0.2)
    coord = EnergyCoordinator(
        modules=[gate],
        couplings=[],
        constraints={},
        enable_uncertainty_gate=True,
        gate_cost_relax_scale=0.5,
        gate_cost_tighten_scale=2.0,
        gate_cost_smoothing=1.0,
        gate_rate_exploit_threshold=0.2,
        gate_uncertainty_relax_threshold=2.0,
    )
    coord._homotopy_gate_bases = [gate.cost]
    coord._accepted_energy_history = [1.0, 0.8, 0.6, 0.5]
    coord._update_uncertainty_gate_scale()
    assert coord._gate_uncertainty_scale == pytest.approx(0.5)
    coord._apply_gate_costs([gate], homotopy_scale=1.0)
    assert gate.cost == pytest.approx(0.1)


def test_uncertainty_gate_tightens_cost_when_rate_low() -> None:
    gate = EnergyGatingModule(gain_fn=_zero_gain, cost=0.2)
    coord = EnergyCoordinator(
        modules=[gate],
        couplings=[],
        constraints={},
        enable_uncertainty_gate=True,
        gate_cost_relax_scale=0.5,
        gate_cost_tighten_scale=2.0,
        gate_cost_smoothing=1.0,
        gate_rate_explore_threshold=1.0,
        gate_rate_exploit_threshold=2.0,
        gate_uncertainty_relax_threshold=0.0,
    )
    coord._homotopy_gate_bases = [gate.cost]
    coord._accepted_energy_history = [1.0, 1.05, 1.1, 1.2]
    coord._update_uncertainty_gate_scale()
    assert coord._gate_uncertainty_scale == pytest.approx(2.0)
    coord._apply_gate_costs([gate], homotopy_scale=1.0)
    assert gate.cost == pytest.approx(0.4)


def test_gate_cost_floor_respected() -> None:
    gate = EnergyGatingModule(gain_fn=_zero_gain, cost=1e-5)
    coord = EnergyCoordinator(
        modules=[gate],
        couplings=[],
        constraints={},
        enable_uncertainty_gate=True,
        gate_cost_relax_scale=0.1,
        gate_cost_floor=1e-4,
        gate_cost_smoothing=1.0,
    )
    coord._homotopy_gate_bases = [gate.cost]
    coord._accepted_energy_history = [1.0, 0.8]
    coord._update_uncertainty_gate_scale()
    coord._apply_gate_costs([gate], homotopy_scale=1.0)
    assert gate.cost >= 1e-4

