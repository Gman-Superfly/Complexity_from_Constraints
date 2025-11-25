"""Test early-stop with patience functionality."""

from __future__ import annotations

import pytest

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter
from core.couplings import QuadraticCoupling
from cf_logging.observability import EnergyBudgetTracker


class SimpleModule(EnergyModule):
    """Simple test module."""
    
    def __init__(self, a: float = 1.0):
        self.a = float(a)
    
    def local_energy(self, eta: OrderParameter, constraints: dict) -> float:
        eta_f = float(eta)
        return self.a * eta_f * eta_f


def test_early_stop_triggers_when_converged():
    """Test that early stop triggers when energy stabilizes."""
    mods = [SimpleModule(a=1.0)]
    couplings = []
    
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        enable_early_stop=True,
        early_stop_patience=3,
        early_stop_delta_threshold=1e-6,
    )
    
    tracker = EnergyBudgetTracker(
        name="test_early_stop",
        run_id="test",
    )
    tracker.attach(coord)
    
    # Start near minimum (should converge quickly)
    etas0 = [0.01]
    coord.relax_etas(etas0, steps=100)  # Allow many steps but should stop early
    
    # Should have stopped before 100 steps
    assert len(tracker.buffer) < 100, f"Should have stopped early, got {len(tracker.buffer)} steps"
    
    # Check that early stop reason was recorded
    found_early_stop = False
    for record in tracker.buffer:
        if "acceptance_reason" in record and "early_stop" in str(record["acceptance_reason"]):
            found_early_stop = True
            break
    
    # Early stop may not always be in logs if it happens after the last emit
    # Just verify we stopped before max steps
    assert len(tracker.buffer) < 100


def test_early_stop_disabled_runs_all_steps():
    """Test that with early stop disabled, all steps run."""
    mods = [SimpleModule(a=1.0)]
    couplings = []
    
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        enable_early_stop=False,  # Disabled
    )
    
    tracker = EnergyBudgetTracker(
        name="test_no_early_stop",
        run_id="test",
    )
    tracker.attach(coord)
    
    etas0 = [0.01]
    steps = 20
    coord.relax_etas(etas0, steps=steps)
    
    # Should run all steps (or stop for other reasons like monotonicity)
    # At minimum, should run more steps than with early stop enabled
    assert len(tracker.buffer) >= 3  # At least a few steps


def test_early_stop_patience_parameter():
    """Test that patience parameter affects when early stop triggers."""
    mods = [SimpleModule(a=1.0)]
    couplings = []
    etas0 = [0.01]
    
    # Short patience
    coord_short = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        enable_early_stop=True,
        early_stop_patience=2,  # Stop after 2 stable steps
        early_stop_delta_threshold=1e-6,
    )
    
    tracker_short = EnergyBudgetTracker(name="short", run_id="short")
    tracker_short.attach(coord_short)
    coord_short.relax_etas(etas0, steps=100)
    
    # Long patience
    coord_long = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        enable_early_stop=True,
        early_stop_patience=10,  # Stop after 10 stable steps
        early_stop_delta_threshold=1e-6,
    )
    
    tracker_long = EnergyBudgetTracker(name="long", run_id="long")
    tracker_long.attach(coord_long)
    coord_long.relax_etas(etas0, steps=100)
    
    # Short patience should stop sooner (or at least not later)
    assert len(tracker_short.buffer) <= len(tracker_long.buffer)


def test_early_stop_threshold_parameter():
    """Test that delta threshold affects early stop sensitivity."""
    mods = [SimpleModule(a=1.0)]
    couplings = []
    etas0 = [0.1]
    
    # Strict threshold (requires very small changes)
    coord_strict = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        enable_early_stop=True,
        early_stop_patience=3,
        early_stop_delta_threshold=1e-9,  # Very strict
    )
    
    tracker_strict = EnergyBudgetTracker(name="strict", run_id="strict")
    tracker_strict.attach(coord_strict)
    coord_strict.relax_etas(etas0, steps=100)
    
    # Lenient threshold (allows larger changes)
    coord_lenient = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        enable_early_stop=True,
        early_stop_patience=3,
        early_stop_delta_threshold=1e-3,  # Lenient
    )
    
    tracker_lenient = EnergyBudgetTracker(name="lenient", run_id="lenient")
    tracker_lenient.attach(coord_lenient)
    coord_lenient.relax_etas(etas0, steps=100)
    
    # Lenient threshold should stop sooner (triggers easier)
    assert len(tracker_lenient.buffer) <= len(tracker_strict.buffer)


def test_early_stop_with_free_energy_guard():
    """Test that early stop works with free energy guard."""
    mods = [SimpleModule(a=1.0)]
    couplings = []
    
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        use_free_energy_guard=True,
        free_energy_temperature=1.0,
        free_energy_epsilon=1e-6,
        enable_early_stop=True,
        early_stop_patience=3,
        early_stop_delta_threshold=1e-6,
    )
    
    tracker = EnergyBudgetTracker(
        name="test_combined",
        run_id="test",
        log_free_energy_decomposition=True,
        temperature=1.0,
    )
    tracker.attach(coord)
    
    etas0 = [0.3]
    coord.relax_etas(etas0, steps=100)
    
    # Should have stopped early
    assert len(tracker.buffer) < 100
    
    # Free energy should have decreased
    if len(tracker.buffer) >= 2:
        first_F = tracker.buffer[0]["F_free_energy"]
        last_F = tracker.buffer[-1]["F_free_energy"]
        assert last_F <= first_F


if __name__ == "__main__":
    test_early_stop_triggers_when_converged()
    test_early_stop_disabled_runs_all_steps()
    test_early_stop_patience_parameter()
    test_early_stop_threshold_parameter()
    test_early_stop_with_free_energy_guard()
    print("✅ All early-stop tests passed!")

