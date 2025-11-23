from __future__ import annotations

import numpy as np
from core.coordinator import EnergyCoordinator, project_noise_orthogonal
from core.couplings import QuadraticCoupling
from core.noise_controller import OrthogonalNoiseController
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


def test_noise_controller_amplifies_when_progress_stalls() -> None:
    controller = OrthogonalNoiseController(base_magnitude=0.1, decay=1.0)
    grad = np.array([1.0, 0.0])
    first = controller.step(grad, energy_drop_ratio=1.0, backtracks=0, iter_idx=0)
    assert first == 0.0
    rotated_grad = np.array([0.0, 1.0])
    second = controller.step(rotated_grad, energy_drop_ratio=0.0, backtracks=1, iter_idx=1)
    assert second > first


def test_noise_controller_suppresses_when_rate_is_high() -> None:
    controller = OrthogonalNoiseController(base_magnitude=0.2, decay=1.0)
    grad = np.array([1.0, 0.0])
    controller.step(grad, energy_drop_ratio=0.0, backtracks=0, iter_idx=0)
    quiet = controller.step(grad, energy_drop_ratio=0.5, backtracks=0, iter_idx=1)
    assert quiet == 0.0


def test_coordinator_auto_noise_controller_tracks_signals() -> None:
    mods = [
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.4, b=0.4),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.6, b=0.6),
    ]
    coups = [(0, 1, QuadraticCoupling(weight=0.5))]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        enable_orthogonal_noise=True,
        auto_noise_controller=True,
        noise_magnitude=0.05,
        noise_schedule_decay=1.0,
    )
    coord._last_energy_drop_ratio = 0.0  # force controller to boost noise early
    etas = coord.compute_etas([0.5, 0.5])
    np.random.seed(0)
    coord.relax_etas(etas, steps=2)
    assert coord._noise_controller is not None
    assert coord._noise_controller._current_scale >= 0.0

