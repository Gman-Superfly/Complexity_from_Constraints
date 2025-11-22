from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule


def test_homotopy_gate_cost_scales_then_restores() -> None:
    gate = EnergyGatingModule(gain_fn=lambda _x: 0.0, cost=0.3, a=0.2, b=0.2)
    modules = [gate]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=[],
        constraints={},
        homotopy_gate_cost_scale_start=2.0,
        homotopy_steps=4,
        use_analytic=True,
    )
    costs = []
    coord.on_energy_updated.append(lambda _F: costs.append(gate.cost))
    coord.relax_etas([0.5], steps=5)
    assert costs, "expected recorded costs"
    assert abs(costs[0] - 0.6) < 1e-6
    assert abs(costs[-1] - 0.3) < 1e-3
    assert abs(gate.cost - 0.3) < 1e-6  # restored


def test_homotopy_term_scale_starts() -> None:
    m0 = EnergyGatingModule(gain_fn=lambda _x: 0.0, cost=0.1, a=0.2, b=0.2)
    m1 = EnergyGatingModule(gain_fn=lambda _x: 0.0, cost=0.1, a=0.2, b=0.2)
    modules = [m0, m1]
    couplings = [(0, 1, QuadraticCoupling(weight=1.0))]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={},
        homotopy_term_scale_starts={"coup:QuadraticCoupling": 0.1},
        homotopy_steps=5,
        use_analytic=True,
    )
    scales = []
    coord.on_energy_updated.append(lambda _F: scales.append(coord._homotopy_term_scales.get("coup:QuadraticCoupling") if coord._homotopy_term_scales else None))
    coord.relax_etas([0.5, 0.4], steps=6)
    assert scales[0] is not None and abs(scales[0] - 0.1) < 1e-6
    assert scales[-1] is None or scales[-1] > 0.9

