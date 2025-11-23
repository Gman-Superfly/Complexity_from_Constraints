from __future__ import annotations

from modules.gating.energy_gating import EnergyGatingModule


def test_straight_through_returns_hard_decision() -> None:
    # Gain > cost → eta_soft high; with threshold 0.5 should map to 1.0
    mod = EnergyGatingModule(gain_fn=lambda _: 0.2, cost=0.05, k=10.0, straight_through=True, st_threshold=0.5)
    eta = mod.compute_eta(None)
    assert eta in (0.0, 1.0)
    assert eta == 1.0


def test_straight_through_threshold_respected() -> None:
    # Smaller net → eta_soft low; with a higher threshold should map to 0.0
    mod = EnergyGatingModule(gain_fn=lambda _: 0.02, cost=0.05, k=10.0, straight_through=True, st_threshold=0.6)
    eta = mod.compute_eta(None)
    assert eta in (0.0, 1.0)
    assert eta == 0.0


def test_default_is_soft_without_st() -> None:
    mod = EnergyGatingModule(gain_fn=lambda _: 0.2, cost=0.05, k=10.0, straight_through=False)
    eta = mod.compute_eta(None)
    assert 0.0 <= eta <= 1.0
    assert eta not in (0.0, 1.0)  # should be soft in typical cases


