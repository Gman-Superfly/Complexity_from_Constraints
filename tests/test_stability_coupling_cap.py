from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule


def test_stability_coupling_scale_applied() -> None:
    m0 = EnergyGatingModule(gain_fn=lambda _x: 0.0, a=0.5, b=0.5)
    m1 = EnergyGatingModule(gain_fn=lambda _x: 0.0, a=0.5, b=0.5)
    modules = [m0, m1]
    couplings = [(0, 1, QuadraticCoupling(weight=5.0))]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={},
        use_analytic=True,
        stability_coupling_auto_cap=True,
        stability_coupling_target=1.0,
        stability_guard=False,
        line_search=False,
    )
    scales = []
    coord.on_energy_updated.append(lambda _F: scales.append(coord._stability_coupling_scale))
    coord.relax_etas([0.5, 0.4], steps=2)
    assert scales, "expected at least one energy callback"
    assert scales[0] is None or scales[0] < 1.0

