from __future__ import annotations

from typing import Mapping

from core.coordinator import EnergyCoordinator
from core.interfaces import WeightAdapter
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule


class DummyAdapter:
    def step(self, term_grad_norms: Mapping[str, float], energy: float, current: Mapping[str, float]) -> Mapping[str, float]:
        # Increase weight for any local term observed; cap at 2.0
        updated = dict(current)
        for key, val in term_grad_norms.items():
            if key.startswith("local:"):
                updated[key] = min(2.0, float(updated.get(key, 1.0)) + 0.1)
        return updated


def test_weight_adapter_updates_term_weights_and_affects_energy():
    mods = [EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.3, b=0.3) for _ in range(2)]
    coups = [(0, 1, QuadraticCoupling(weight=0.5))]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        step_size=0.01,
        grad_eps=1e-6,
        use_analytic=True,
        weight_adapter=DummyAdapter(),  # type: ignore[arg-type]
    )
    etas = [0.7, 0.2]
    F0 = coord.energy(etas)
    coord.relax_etas(etas, steps=5)
    # After a few steps, adapter should have updated local weights
    keys = [f"local:{mods[0].__class__.__name__}", f"local:{mods[1].__class__.__name__}"]
    # Access protected field for validation only in test
    tw = getattr(coord, "_term_weights")
    assert all(k in tw for k in keys)
    # With higher local weights, weighted energy should reflect increased local contribution
    F1 = coord.energy(etas)
    assert F1 != F0


