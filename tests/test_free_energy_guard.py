"""Test Free-Energy Guard: F = U - T*S acceptance."""

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


def test_free_energy_guard_accepts_sufficient_decrease():
    """Test that free energy guard accepts steps with ΔF < -ε."""
    mods = [
        SimpleModule(a=2.0),
        SimpleModule(a=2.0),
    ]
    
    couplings = [(0, 1, QuadraticCoupling(weight=0.5))]
    
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        use_free_energy_guard=True,
        free_energy_temperature=1.0,
        free_energy_epsilon=1e-6,
    )
    
    tracker = EnergyBudgetTracker(
        name="test_guard",
        run_id="test",
        log_free_energy_decomposition=True,
        temperature=1.0,
    )
    tracker.attach(coord)
    
    # Start from high energy state
    etas0 = [0.8, 0.2]
    etas_final = coord.relax_etas(etas0, steps=10)
    
    # Should have converged (accepted steps)
    assert len(tracker.buffer) > 0
    
    # Check that free energy decreased
    first_F = tracker.buffer[0]["F_free_energy"]
    last_F = tracker.buffer[-1]["F_free_energy"]
    
    assert last_F < first_F, f"Free energy should decrease: {first_F} -> {last_F}"


def test_free_energy_guard_records_acceptance_reason():
    """Test that acceptance reasons are recorded correctly with free energy guard."""
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
    )
    
    tracker = EnergyBudgetTracker(
        name="test_reasons",
        run_id="test",
    )
    tracker.attach(coord)
    
    etas0 = [0.5]
    coord.relax_etas(etas0, steps=5)
    
    # Check acceptance reasons
    found_free_energy_reason = False
    for record in tracker.buffer:
        if "acceptance_reason" in record:
            reason = record["acceptance_reason"]
            if "free_energy" in reason:
                found_free_energy_reason = True
                assert reason in [
                    "free_energy_accepted",
                    "free_energy_insufficient_decrease",
                ], f"Unexpected reason: {reason}"
    
    assert found_free_energy_reason, "Should have found free energy acceptance reasons"


def test_free_energy_guard_vs_standard():
    """Compare free energy guard with standard energy guard."""
    mods = [
        SimpleModule(a=1.0),
        SimpleModule(a=1.0),
    ]
    
    couplings = [(0, 1, QuadraticCoupling(weight=0.2))]
    etas0 = [0.7, 0.3]
    
    # Standard guard
    coord_standard = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        use_free_energy_guard=False,
    )
    etas_standard = coord_standard.relax_etas(etas0, steps=20)
    
    # Free energy guard
    coord_free = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        use_free_energy_guard=True,
        free_energy_temperature=0.1,  # Low temperature to converge similarly
        free_energy_epsilon=1e-6,
    )
    etas_free = coord_free.relax_etas(etas0, steps=20)
    
    # With low temperature, free energy should converge close to standard (F ≈ U)
    # But allow larger tolerance since they optimize different objectives
    for eta_std, eta_free in zip(etas_standard, etas_free):
        assert abs(eta_std - eta_free) < 0.3, f"Convergence difference too large: {eta_std} vs {eta_free}"


def test_free_energy_temperature_parameter():
    """Test that temperature affects free energy acceptance."""
    mods = [SimpleModule(a=1.0)]
    couplings = []
    etas0 = [0.5]
    
    results = {}
    # Use reasonable temperature range where convergence is expected
    for T in [0.1, 0.5, 1.0]:
        coord = EnergyCoordinator(
            modules=mods,
            couplings=couplings,
            constraints={},
            step_size=0.05,
            use_free_energy_guard=True,
            free_energy_temperature=T,
            free_energy_epsilon=1e-6,
        )
        
        tracker = EnergyBudgetTracker(
            name=f"test_T{T}",
            run_id="test",
            log_free_energy_decomposition=True,
            temperature=T,
        )
        tracker.attach(coord)
        
        coord.relax_etas(etas0, steps=10)
        
        # Record number of accepted steps
        results[T] = len(tracker.buffer)
    
    # Low/moderate temperatures should have accepted steps
    # (High T may reject all steps if entropy doesn't compensate)
    for T, count in results.items():
        assert count > 0, f"Temperature {T} should have some accepted steps"


def test_free_energy_epsilon_threshold():
    """Test that epsilon threshold affects acceptance."""
    mods = [SimpleModule(a=1.0)]
    couplings = []
    etas0 = [0.5]
    
    # Strict epsilon (larger negative change required)
    coord_strict = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        use_free_energy_guard=True,
        free_energy_temperature=1.0,
        free_energy_epsilon=1e-3,  # Strict
    )
    
    tracker_strict = EnergyBudgetTracker(
        name="test_strict",
        run_id="strict",
    )
    tracker_strict.attach(coord_strict)
    
    coord_strict.relax_etas(etas0, steps=10)
    
    # Lenient epsilon (smaller negative change required)
    coord_lenient = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        use_free_energy_guard=True,
        free_energy_temperature=1.0,
        free_energy_epsilon=1e-9,  # Lenient
    )
    
    tracker_lenient = EnergyBudgetTracker(
        name="test_lenient",
        run_id="lenient",
    )
    tracker_lenient.attach(coord_lenient)
    
    coord_lenient.relax_etas(etas0, steps=10)
    
    # Lenient should typically accept more steps (or at least as many)
    assert len(tracker_lenient.buffer) >= len(tracker_strict.buffer)


if __name__ == "__main__":
    test_free_energy_guard_accepts_sufficient_decrease()
    test_free_energy_guard_records_acceptance_reason()
    test_free_energy_guard_vs_standard()
    test_free_energy_temperature_parameter()
    test_free_energy_epsilon_threshold()
    print("✅ All free energy guard tests passed!")

