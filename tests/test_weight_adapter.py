from __future__ import annotations

import math

from core.weight_adapters import GradNormWeightAdapter


def test_gradnorm_adapter_decreases_large_terms() -> None:
    adapter = GradNormWeightAdapter(target_norm=1.0, alpha=1.0, update_rate=0.2)
    updated = adapter.step({"local:Foo": 2.0}, energy=0.0, current={"local:Foo": 1.0})
    assert updated["local:Foo"] < 1.0


def test_gradnorm_adapter_increases_small_terms() -> None:
    adapter = GradNormWeightAdapter(target_norm=1.0, alpha=1.0, update_rate=0.2)
    updated = adapter.step({"coup:Bar": 0.25}, energy=0.0, current={"coup:Bar": 1.0})
    assert updated["coup:Bar"] > 1.0


def test_gradnorm_adapter_respects_clamps_and_defaults() -> None:
    adapter = GradNormWeightAdapter(target_norm=1.0, alpha=5.0, update_rate=1.0, floor=0.5, ceiling=1.5)
    # Missing current weight defaults to 1.0 then gets clamped by ceiling
    updated_high = adapter.step({"local:Foo": 10.0}, energy=0.0, current={})
    assert math.isclose(updated_high["local:Foo"], 0.5)

    updated_low = adapter.step({"coup:Bar": 1e-6}, energy=0.0, current={"coup:Bar": 1.4})
    assert math.isclose(updated_low["coup:Bar"], 1.5)


def test_gradnorm_adapter_handles_invalid_norms() -> None:
    adapter = GradNormWeightAdapter(target_norm=1.0, alpha=1.0, update_rate=0.5)
    norms = {"local:Foo": float("nan"), "coup:Bar": 1.0}
    updated = adapter.step(norms, energy=0.0, current={})
    # NaN falls back to average (=1.0), so both weights remain near 1.0
    assert math.isclose(updated["local:Foo"], 1.0, rel_tol=1e-6)
    assert math.isclose(updated["coup:Bar"], 1.0, rel_tol=1e-6)

