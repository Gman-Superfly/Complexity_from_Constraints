from __future__ import annotations

from modules.gating.energy_gating import EnergyGatingModule


def constant_gain_fn(_: object) -> float:
    return 0.1


def test_gating_eta_decreases_with_cost():
    gate_low = EnergyGatingModule(gain_fn=constant_gain_fn, cost=0.01, k=10.0)
    gate_high = EnergyGatingModule(gain_fn=constant_gain_fn, cost=0.2, k=10.0)
    eta_low = gate_low.compute_eta(None)
    eta_high = gate_high.compute_eta(None)
    assert 0.0 <= eta_low <= 1.0 and 0.0 <= eta_high <= 1.0
    assert eta_low > eta_high


