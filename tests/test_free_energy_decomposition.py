"""Test free energy decomposition logging (F = U - T*S)."""

from __future__ import annotations

import pytest
import math

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter
from core.couplings import QuadraticCoupling
from cf_logging.observability import EnergyBudgetTracker


class SimpleTestModule(EnergyModule):
    """Simple test module with quadratic energy."""
    
    def __init__(self, a: float = 1.0):
        self.a = float(a)
    
    def local_energy(self, eta: OrderParameter, constraints: dict) -> float:
        """Quadratic energy."""
        eta_f = float(eta)
        return self.a * eta_f * eta_f


def test_free_energy_decomposition_logging():
    """Test that F = U - T*S decomposition is logged correctly."""
    mods = [
        SimpleTestModule(a=1.0),
        SimpleTestModule(a=1.5),
    ]
    
    couplings = [(0, 1, QuadraticCoupling(weight=0.2))]
    
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
    )
    
    # Attach tracker with free energy decomposition enabled
    tracker = EnergyBudgetTracker(
        name="test_free_energy",
        run_id="test",
        log_free_energy_decomposition=True,
        temperature=1.0,
    )
    tracker.attach(coord)
    
    etas0 = [0.3, 0.7]
    coord.relax_etas(etas0, steps=5)
    
    assert len(tracker.buffer) > 0, "Should have logged records"
    
    # Check that free energy components exist
    first_record = tracker.buffer[0]
    assert "U_internal_energy" in first_record
    assert "S_entropy" in first_record
    assert "F_free_energy" in first_record
    assert "T_temperature" in first_record
    
    # Verify F = U - T*S
    U = first_record["U_internal_energy"]
    S = first_record["S_entropy"]
    F = first_record["F_free_energy"]
    T = first_record["T_temperature"]
    
    assert F == pytest.approx(U - T * S, abs=1e-6), f"F={F} should equal U - T*S = {U} - {T}*{S} = {U - T*S}"


def test_entropy_calculation():
    """Test that entropy is calculated correctly for order parameters."""
    mods = [SimpleTestModule(a=1.0)]
    couplings = []
    
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
    )
    
    tracker = EnergyBudgetTracker(
        name="test_entropy",
        run_id="test",
        log_free_energy_decomposition=True,
        temperature=1.0,
    )
    tracker.attach(coord)
    
    # Test with η = 0.5 (maximum entropy for binary variable)
    etas0 = [0.5]
    coord.relax_etas(etas0, steps=1)
    
    assert len(tracker.buffer) > 0
    first_record = tracker.buffer[0]
    
    # For η = 0.5, entropy should be maximal: S = -[0.5*log(0.5) + 0.5*log(0.5)] = log(2)
    expected_S = math.log(2.0)
    actual_S = first_record["S_entropy"]
    
    assert actual_S == pytest.approx(expected_S, rel=0.01), f"Entropy at η=0.5 should be ~{expected_S}, got {actual_S}"


def test_free_energy_disabled_by_default():
    """Test that free energy decomposition is off by default."""
    mods = [SimpleTestModule(a=1.0)]
    couplings = []
    
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
    )
    
    # Default tracker (log_free_energy_decomposition=False)
    tracker = EnergyBudgetTracker(
        name="test_default",
        run_id="test",
    )
    tracker.attach(coord)
    
    etas0 = [0.3]
    coord.relax_etas(etas0, steps=3)
    
    assert len(tracker.buffer) > 0
    
    first_record = tracker.buffer[0]
    # Free energy components should NOT exist
    assert "U_internal_energy" not in first_record
    assert "S_entropy" not in first_record
    assert "F_free_energy" not in first_record


def test_temperature_parameter():
    """Test that temperature parameter affects free energy correctly."""
    mods = [SimpleTestModule(a=1.0)]
    couplings = []
    
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
    )
    
    # Test with different temperatures
    for T in [0.5, 1.0, 2.0]:
        tracker = EnergyBudgetTracker(
            name=f"test_T{T}",
            run_id="test",
            log_free_energy_decomposition=True,
            temperature=T,
        )
        tracker.attach(coord)
        
        etas0 = [0.4]
        coord.relax_etas(etas0, steps=1)
        
        assert len(tracker.buffer) > 0
        record = tracker.buffer[0]
        
        assert record["T_temperature"] == pytest.approx(T, abs=1e-9)
        
        # Verify F = U - T*S
        U = record["U_internal_energy"]
        S = record["S_entropy"]
        F = record["F_free_energy"]
        
        assert F == pytest.approx(U - T * S, abs=1e-6)


if __name__ == "__main__":
    test_free_energy_decomposition_logging()
    test_entropy_calculation()
    test_free_energy_disabled_by_default()
    test_temperature_parameter()
    print("✅ All free energy decomposition tests passed!")

