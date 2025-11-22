from __future__ import annotations

import numpy as np
from core.coordinator import EnergyCoordinator, project_noise_orthogonal
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule


def test_project_noise_orthogonal() -> None:
    """Verify noise projection is truly orthogonal to gradient."""
    # Case 1: Simple 2D
    grad = np.array([1.0, 0.0])
    noise = np.array([1.0, 1.0])
    # Should remove x component
    proj = project_noise_orthogonal(noise, grad)
    assert abs(proj[0]) < 1e-9
    assert abs(proj[1] - 1.0) < 1e-9
    assert abs(np.dot(proj, grad)) < 1e-9

    # Case 2: Zero gradient (should preserve noise)
    grad_zero = np.array([0.0, 0.0])
    proj_zero = project_noise_orthogonal(noise, grad_zero)
    assert np.allclose(proj_zero, noise)

    # Case 3: Noise parallel to gradient (should be zero)
    noise_para = np.array([2.0, 0.0])
    proj_para = project_noise_orthogonal(noise_para, grad)
    assert np.allclose(proj_para, 0.0)


def test_coordinator_orthogonal_noise_integration() -> None:
    """Verify coordinator injects orthogonal noise in a non-degenerate (2D) case."""
    # Use 2D setup where a null space exists (orthogonal component is non-trivial).
    mods2 = [
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.5, b=0.5),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.5, b=0.5)
    ]
    # No coupling, just two independent wells.
    coord2 = EnergyCoordinator(
        modules=mods2,
        couplings=[],
        constraints={},
        enable_orthogonal_noise=True,
        noise_magnitude=0.5,
        noise_schedule_decay=1.0,
        step_size=1e-6, # Tiny descent (must be >0 per validation), effectively only noise visible
        line_search=False
    )
    
    etas2 = [0.5, 0.5]
    # Gradients will be [g1, g2] (positive).
    # Noise will be projected.
    
    np.random.seed(42)
    final_etas = coord2.relax_etas(etas2, steps=1)
    
    # Check that etas changed (noise was added)
    # If they changed, noise was non-zero.
    diff = np.linalg.norm(np.array(final_etas) - np.array(etas2))
    assert diff > 1e-9, "Noise should move etas in 2D"
    
    # Verify the move was roughly orthogonal to gradient?
    # Hard to check inside black box without mocking, but `project_noise_orthogonal` unit test covers math.
    # This test just ensures the flag triggers the logic.

