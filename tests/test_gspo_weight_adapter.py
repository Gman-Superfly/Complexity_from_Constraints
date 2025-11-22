from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="GSPO-token adapter requires torch for tests.")

from core.weight_adapters import GSPOTokenWeightAdapter


def _make_adapter() -> GSPOTokenWeightAdapter:
    torch.manual_seed(0)
    return GSPOTokenWeightAdapter(
        target_norm=1.0,
        floor=0.1,
        ceiling=3.0,
        num_buckets=8,
        group_size=2,
        batch_size=1,
        hidden_size=16,
        apply_rate=0.5,
        reference_sync_interval=1,
        use_token_level=True,
        device="cpu",
    )


def test_gspo_adapter_returns_weights_for_all_terms() -> None:
    adapter = _make_adapter()
    norms = {"local:Seq": 0.6, "coup:Gate": 1.5}
    updated = adapter.step(norms, energy=0.0, current={})
    assert set(updated) == set(norms)
    for value in updated.values():
        assert 0.1 <= value <= 3.0


def test_gspo_adapter_records_metrics_and_clamps_weights() -> None:
    adapter = _make_adapter()
    adapter.step({"local:Seq": 1.5, "coup:Gate": 0.2}, energy=0.0, current={})
    assert adapter._last_metrics is not None
    for value in adapter._last_weights.values():
        assert 0.1 <= value <= 3.0


def test_gspo_adapter_throttling_and_logging_hook() -> None:
    captured = []

    def hook(m):
        captured.append(m)

    adapter = GSPOTokenWeightAdapter(
        target_norm=1.0,
        floor=0.1,
        ceiling=3.0,
        num_buckets=8,
        group_size=2,
        batch_size=1,
        hidden_size=16,
        apply_rate=0.5,
        reference_sync_interval=1000,  # avoid full sync
        use_token_level=True,
        device="cpu",
        update_every_n_steps=10,  # throttle: many steps will be skipped
        logging_callback=hook,
    )
    # First step (train due to step%10==0 after increment)
    adapter.step({"local:Seq": 1.0}, energy=0.0, current={})
    # Second step (should skip training)
    adapter.step({"local:Seq": 1.0}, energy=0.0, current={})
    assert len(captured) >= 2
    # Ensure at least one of the hooks reported a skip
    assert any(("skipped" in m) or (m.get("did_train", 0.0) == 0.0) for m in captured)


def test_gspo_adapter_disable_throttling_trains_every_step() -> None:
    captured = []

    def hook(m):
        captured.append(m)

    adapter = GSPOTokenWeightAdapter(
        target_norm=1.0,
        floor=0.1,
        ceiling=3.0,
        num_buckets=8,
        group_size=2,
        batch_size=1,
        hidden_size=16,
        apply_rate=0.5,
        reference_sync_interval=1000,
        use_token_level=True,
        device="cpu",
        update_every_n_steps=10,   # would skip, but we'll disable throttling
        enable_throttling=False,   # ensure we train every step
        logging_callback=hook,
    )
    adapter.step({"local:Seq": 1.0}, energy=0.0, current={})
    adapter.step({"local:Seq": 1.0}, energy=0.0, current={})
    assert len(captured) >= 2
    # No skip markers and did_train should be 1.0 for all captured metrics
    assert not any(("skipped" in m) for m in captured)
    assert all(m.get("did_train", 0.0) == 1.0 for m in captured)


# ------------------------------------------------------------------
# Test coverage notes (Datamutant discipline):
#
# Current tests (MVP):
#   - Smoke tests only: adapter initializes, returns weights, records metrics.
#   - Does NOT assert that policy improves over time (multi-step training).
#
# Tests to add when scaling beyond MVP (see docs/fixes_and__related_todos.md P4):
#   1. Multi-step convergence: run 50+ adapter.step() calls, verify:
#      - last_metrics["mean_reward"] increases over time.
#      - Weights stabilize (variance decreases).
#      - Reference model syncs without drift (KL divergence check).
#   2. Large term count (15-20 families): ensure no OOM, weights clamp correctly.
#   3. Compute profiling: measure wall time vs GradNorm baseline.
#   4. Reference sync validation: 100-step run, verify policy/reference KL < threshold.
#
# Reason for deferral:
#   - Multi-step tests require full coordinator integration (expensive setup).
#   - Current smoke tests catch initialization bugs, import failures, and basic I/O.
#   - Convergence validation will happen in experiments (auto_balance_demo.py --scenarios gspo).
# ------------------------------------------------------------------
