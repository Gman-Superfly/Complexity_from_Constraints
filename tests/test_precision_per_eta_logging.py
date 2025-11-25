"""Test per-η precision logging and acceptance reason tracking."""

from __future__ import annotations

import pytest

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter, SupportsPrecision
from core.couplings import QuadraticCoupling
from cf_logging.observability import EnergyBudgetTracker


class PrecisionTestModule(EnergyModule, SupportsPrecision):
    """Test module with curvature support."""
    
    def __init__(self, a: float = 1.0, b: float = 0.5, curvature_val: float = 2.0):
        self.a = float(a)
        self.b = float(b)
        self._curvature = float(curvature_val)
    
    def local_energy(self, eta: OrderParameter, constraints: dict) -> float:
        """Quadratic energy with optional quartic."""
        eta_f = float(eta)
        return self.a * eta_f * eta_f + self.b * (eta_f ** 4)
    
    def curvature(self, eta: OrderParameter) -> float:
        """Return constant curvature for testing."""
        return self._curvature


def test_per_eta_precision_logging():
    """Test that per-η precision values are logged when enabled."""
    # Create modules with different curvature values
    mods = [
        PrecisionTestModule(curvature_val=1.0),
        PrecisionTestModule(curvature_val=2.0),
        PrecisionTestModule(curvature_val=5.0),
    ]
    
    # Add some couplings
    couplings = [
        (0, 1, QuadraticCoupling(weight=0.1)),
        (1, 2, QuadraticCoupling(weight=0.1)),
    ]
    
    # Create coordinator
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
        use_analytic=True,
    )
    
    # Attach tracker with per-η precision logging enabled
    tracker = EnergyBudgetTracker(
        name="test_precision",
        run_id="test",
        log_per_eta_precision=True,
    )
    tracker.attach(coord)
    
    # Relax
    etas0 = [0.3, 0.5, 0.7]
    coord.relax_etas(etas0, steps=5)
    
    # Check that per-η precision was logged
    assert len(tracker.buffer) > 0, "Should have logged records"
    
    # Check that precision columns exist
    first_record = tracker.buffer[0]
    assert "precision:min" in first_record
    assert "precision:max" in first_record
    assert "precision:mean" in first_record
    
    # Check per-η columns
    assert "precision:0" in first_record
    assert "precision:1" in first_record
    assert "precision:2" in first_record
    
    # Verify values correspond to module curvatures
    assert first_record["precision:0"] == pytest.approx(1.0, abs=0.01)
    assert first_record["precision:1"] == pytest.approx(2.0, abs=0.01)
    assert first_record["precision:2"] == pytest.approx(5.0, abs=0.01)


def test_precision_logging_disabled_by_default():
    """Test that per-η precision logging is off by default."""
    mods = [
        PrecisionTestModule(curvature_val=1.0),
        PrecisionTestModule(curvature_val=2.0),
    ]
    
    couplings = [(0, 1, QuadraticCoupling(weight=0.1))]
    
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.05,
    )
    
    # Default tracker (log_per_eta_precision=False)
    tracker = EnergyBudgetTracker(
        name="test_default",
        run_id="test",
    )
    tracker.attach(coord)
    
    etas0 = [0.3, 0.5]
    coord.relax_etas(etas0, steps=3)
    
    assert len(tracker.buffer) > 0
    
    first_record = tracker.buffer[0]
    # Summary stats should exist
    assert "precision:min" in first_record
    assert "precision:max" in first_record
    assert "precision:mean" in first_record
    
    # Per-η columns should NOT exist
    assert "precision:0" not in first_record
    assert "precision:1" not in first_record


def test_acceptance_reason_tracking():
    """Test that acceptance reasons are tracked and logged."""
    mods = [
        PrecisionTestModule(curvature_val=1.0),
        PrecisionTestModule(curvature_val=2.0),
    ]
    
    couplings = [(0, 1, QuadraticCoupling(weight=0.1))]
    
    # Test with line search
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.1,
        line_search=True,
    )
    
    tracker = EnergyBudgetTracker(
        name="test_acceptance",
        run_id="test",
    )
    tracker.attach(coord)
    
    etas0 = [0.3, 0.5]
    coord.relax_etas(etas0, steps=5)
    
    assert len(tracker.buffer) > 0
    
    # Check that acceptance reason is logged
    for record in tracker.buffer:
        if "acceptance_reason" in record:
            reason = record["acceptance_reason"]
            assert reason in [
                "armijo_accepted",
                "armijo_failed_fallback",
                "monotone_decrease",
                "initial_step",
            ], f"Unexpected acceptance reason: {reason}"


def test_backtrack_counts_logged():
    """Test that backtrack counts are logged."""
    mods = [
        PrecisionTestModule(curvature_val=1.0),
        PrecisionTestModule(curvature_val=2.0),
    ]
    
    couplings = [(0, 1, QuadraticCoupling(weight=0.5))]
    
    # Use line search to trigger backtracks
    coord = EnergyCoordinator(
        modules=mods,
        couplings=couplings,
        constraints={},
        step_size=0.5,  # Large step to encourage backtracks
        line_search=True,
        max_backtrack=3,
    )
    
    tracker = EnergyBudgetTracker(
        name="test_backtracks",
        run_id="test",
    )
    tracker.attach(coord)
    
    etas0 = [0.2, 0.8]
    coord.relax_etas(etas0, steps=10)
    
    assert len(tracker.buffer) > 0
    
    # Check that backtrack counts exist
    found_backtracks = False
    for record in tracker.buffer:
        if "last_backtracks" in record:
            found_backtracks = True
            assert isinstance(record["last_backtracks"], int)
            assert record["last_backtracks"] >= 0
    
    assert found_backtracks, "Should have logged backtrack counts"


if __name__ == "__main__":
    test_per_eta_precision_logging()
    test_precision_logging_disabled_by_default()
    test_acceptance_reason_tracking()
    test_backtrack_counts_logged()
    print("✅ All per-η precision logging tests passed!")

