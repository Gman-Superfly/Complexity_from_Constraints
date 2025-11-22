"""Tests for monotonic energy assertion feature."""

from __future__ import annotations

import pytest
import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule
from modules.sequence.monotonic_eta import SequenceConsistencyModule


def _dummy_gain_fn(x: any) -> float:
    """Simple gain function for testing."""
    return 0.5


def test_monotonic_energy_assertion_passes_in_deterministic_mode() -> None:
    """Verify assertion passes when energy decreases monotonically."""
    gate_mod = EnergyGatingModule(gain_fn=_dummy_gain_fn, cost=0.5, a=1.0, b=1.0)
    seq_mod = SequenceConsistencyModule(alpha=1.0, beta=1.0)
    
    coord = EnergyCoordinator(
        modules=[gate_mod, seq_mod],
        couplings=[(0, 1, QuadraticCoupling(weight=1.0))],
        constraints={},
        assert_monotonic_energy=True,
        monotonic_energy_tol=1e-10,
        noise_magnitude=0.0,  # Deterministic
        line_search=False,
        step_size=0.01,  # Conservative for testing
        enforce_invariants=True,
    )
    
    etas0 = [0.0, 0.5]
    etas_final = coord.relax_etas(etas0, steps=50)
    
    # If we get here, assertion never triggered
    assert len(etas_final) == 2
    assert all(0.0 <= eta <= 1.0 for eta in etas_final)


def test_monotonic_energy_assertion_skipped_with_noise() -> None:
    """Verify assertion is skipped when exploration noise is active."""
    gate_mod = EnergyGatingModule(gain_fn=_dummy_gain_fn, cost=0.5, a=1.0, b=1.0)
    seq_mod = SequenceConsistencyModule(alpha=1.0, beta=1.0)
    
    coord = EnergyCoordinator(
        modules=[gate_mod, seq_mod],
        couplings=[(0, 1, QuadraticCoupling(weight=1.0))],
        constraints={},
        assert_monotonic_energy=True,  # Enabled but will be skipped
        monotonic_energy_tol=1e-10,
        enable_orthogonal_noise=True,
        noise_magnitude=0.05,  # Non-zero noise triggers guard
        step_size=0.05,
    )
    
    etas0 = [0.0, 0.5]
    # Should not raise even if noise causes transient energy increases
    etas_final = coord.relax_etas(etas0, steps=50)
    
    assert len(etas_final) == 2


def test_monotonic_energy_assertion_skipped_with_line_search() -> None:
    """Verify assertion is skipped when line search is active."""
    gate_mod = EnergyGatingModule(gain_fn=_dummy_gain_fn, cost=0.5, a=1.0, b=1.0)
    seq_mod = SequenceConsistencyModule(alpha=1.0, beta=1.0)
    
    coord = EnergyCoordinator(
        modules=[gate_mod, seq_mod],
        couplings=[(0, 1, QuadraticCoupling(weight=1.0))],
        constraints={},
        assert_monotonic_energy=True,  # Enabled but will be skipped
        monotonic_energy_tol=1e-10,
        noise_magnitude=0.0,
        line_search=True,  # Triggers guard
        backtrack_factor=0.5,
        max_backtrack=5,
    )
    
    etas0 = [0.0, 0.5]
    etas_final = coord.relax_etas(etas0, steps=50)
    
    assert len(etas_final) == 2


def test_monotonic_energy_assertion_triggers_on_large_step() -> None:
    """Verify assertion catches energy increase from too-large step size."""
    gate_mod = EnergyGatingModule(gain_fn=_dummy_gain_fn, cost=0.5, a=1.0, b=1.0)
    seq_mod = SequenceConsistencyModule(alpha=1.0, beta=1.0)
    
    coord = EnergyCoordinator(
        modules=[gate_mod, seq_mod],
        couplings=[(0, 1, QuadraticCoupling(weight=10.0))],  # Strong coupling
        constraints={},
        assert_monotonic_energy=True,
        monotonic_energy_tol=1e-10,
        noise_magnitude=0.0,
        line_search=False,
        step_size=1.0,  # Too large, will overshoot
        enforce_invariants=True,
    )
    
    etas0 = [0.0, 0.5]
    
    # Should raise AssertionError due to energy increase
    with pytest.raises(AssertionError, match="Energy increased"):
        coord.relax_etas(etas0, steps=50)


def test_monotonic_energy_tolerance_allows_small_jitter() -> None:
    """Verify tolerance parameter allows small numeric jitter."""
    gate_mod = EnergyGatingModule(gain_fn=_dummy_gain_fn, cost=0.5, a=1.0, b=1.0)
    
    coord = EnergyCoordinator(
        modules=[gate_mod],
        couplings=[],
        constraints={},
        assert_monotonic_energy=True,
        monotonic_energy_tol=1e-8,  # Relaxed tolerance
        noise_magnitude=0.0,
        line_search=False,
        step_size=0.05,
    )
    
    etas0 = [0.5]
    # Should complete without assertion even with minor floating-point jitter
    etas_final = coord.relax_etas(etas0, steps=100)
    
    assert len(etas_final) == 1


def test_monotonic_energy_can_be_disabled() -> None:
    """Verify assertion can be disabled and doesn't interfere when off."""
    gate_mod = EnergyGatingModule(gain_fn=_dummy_gain_fn, cost=0.5, a=1.0, b=1.0)
    seq_mod = SequenceConsistencyModule(alpha=1.0, beta=1.0)
    
    coord = EnergyCoordinator(
        modules=[gate_mod, seq_mod],
        couplings=[(0, 1, QuadraticCoupling(weight=10.0))],
        constraints={},
        assert_monotonic_energy=False,  # explicitly disabled
        noise_magnitude=0.0,
        line_search=False,
        step_size=1.0,  # Large step that would trigger assertion if enabled
    )
    
    etas0 = [0.0, 0.5]
    # Should not raise even with large step size
    etas_final = coord.relax_etas(etas0, steps=10)
    
    assert len(etas_final) == 2

