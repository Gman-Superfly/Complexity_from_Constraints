from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule


def test_homotopy_scale_increases_to_one() -> None:
    m0 = EnergyGatingModule(gain_fn=lambda _x: 0.0, cost=0.1, a=0.2, b=0.2)
    m1 = EnergyGatingModule(gain_fn=lambda _x: 0.0, cost=0.1, a=0.2, b=0.2)
    modules = [m0, m1]
    couplings = [(0, 1, QuadraticCoupling(weight=1.0))]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={},
        line_search=False,
        homotopy_coupling_scale_start=0.2,
        homotopy_steps=4,
        use_analytic=True,
    )
    scales = []
    coord.on_energy_updated.append(lambda _F: scales.append(coord._homotopy_scale))
    coord.relax_etas([0.5, 0.4], steps=5)
    assert scales[0] is not None and abs(scales[0] - 0.2) < 1e-6
    assert scales[-1] is not None and scales[-1] <= 1.0 and scales[-1] > 0.9
    assert all(scales[i] <= scales[i + 1] + 1e-6 for i in range(len(scales) - 1))

