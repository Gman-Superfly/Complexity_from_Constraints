from __future__ import annotations

from core.energy import total_energy
from core.couplings import QuadraticCoupling
from core.coordinator import EnergyCoordinator
from modules.gating.energy_gating import EnergyGatingModule


def test_total_energy_respects_term_weights():
    m = [EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.5, b=0.5)]
    etas = [0.6]
    coups = [(0, 0, QuadraticCoupling(weight=1.0))]
    base_F = total_energy(etas, m, coups, constraints={})
    # Double local gate weight; halve coupling weight
    constraints = {
        "term_weights": {
            f"local:{m[0].__class__.name if hasattr(m[0].__class__, 'name') else m[0].__class__.__name__}": 2.0,
            f"coup:{QuadraticCoupling.__name__}": 0.5,
        }
    }
    F_w = total_energy(etas, m, coups, constraints=constraints)
    # Should not equal base unless specific values cancel; just assert change
    assert abs(F_w - base_F) > 1e-9


def test_coordinator_energy_uses_calibrated_weights():
    mods = [EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.5, b=0.5)]
    etas = [0.4]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=[],
        constraints={"term_weights": {f"local:{mods[0].__class__.__name__}": 0.0}},
        term_weight_floor=0.3,
    )
    energy = coord.energy(etas)
    expected = total_energy(
        etas,
        mods,
        [],
        constraints={"term_weights": {f"local:{mods[0].__class__.__name__}": 0.3}},
    )
    assert abs(energy - expected) <= 1e-9


