from __future__ import annotations

import numpy as np

from core.energy import project_noise_metric_orthogonal
from core.coordinator import EnergyCoordinator
from modules.gating.energy_gating import EnergyGatingModule


def test_project_noise_metric_orthogonal_is_M_orthogonal() -> None:
    # Gradient and random noise in R^3
    g = np.array([1.0, 2.0, -1.0], dtype=float)
    z = np.array([0.5, -0.3, 0.7], dtype=float)
    # Simple SPD metric (diagonal)
    M = np.diag([2.0, 1.0, 3.0])

    z_perp = project_noise_metric_orthogonal(z, g, M=M)

    # Check M-orthogonality: g^T M z_perp ≈ 0
    Mg = M @ g
    dot = float(np.dot(g, M @ z_perp))
    assert abs(dot) < 1e-9, f"M-orthogonality failed: g^T M z_perp = {dot}"


def test_coordinator_metric_vector_product_is_used_when_enabled() -> None:
    # Two gating modules to produce non-zero gradients
    mods = [
        EnergyGatingModule(gain_fn=lambda _x: 0.0, a=0.4, b=0.4),
        EnergyGatingModule(gain_fn=lambda _x: 0.0, a=0.6, b=0.6),
    ]
    coups: list[tuple[int, int, object]] = []

    calls = {"count": 0}

    def mv(v: np.ndarray) -> np.ndarray:
        # Record that the metric-vector product is being used and apply a diagonal metric
        calls["count"] += 1
        D = np.diag([1.5, 0.75])  # SPD metric in 2D
        return D @ v

    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        enable_orthogonal_noise=True,
        auto_noise_controller=True,
        noise_magnitude=0.1,
        noise_schedule_decay=1.0,
        step_size=1e-6,  # tiny step so noise path is engaged by controller
        metric_aware_noise_controller=True,
        metric_vector_product=mv,
        line_search=False,
    )
    # Encourage the controller to produce non-zero noise magnitude on the first step
    coord._last_energy_drop_ratio = 0.0  # type: ignore[attr-defined]

    etas0 = coord.compute_etas([0.5, 0.5])
    np.random.seed(123)
    coord.relax_etas(etas0, steps=1)

    # Metric-vector product should have been invoked at least once
    assert calls["count"] >= 1, "metric_vector_product was not used by coordinator"


