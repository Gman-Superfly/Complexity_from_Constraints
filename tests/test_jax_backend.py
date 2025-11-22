from __future__ import annotations

import pytest

from modules.gating.energy_gating import EnergyGatingModule
from core.couplings import QuadraticCoupling, GateBenefitCoupling
from core.energy import total_energy


def test_jax_backend_supports_gate_benefit():
    pytest.importorskip("jax")

    from core.jax_backend import JaxEnergyRunner

    mods = [
        EnergyGatingModule(gain_fn=lambda _: 0.15, cost=0.05, a=0.2, b=0.3),
        EnergyGatingModule(gain_fn=lambda _: 0.0, cost=0.0, a=0.2, b=0.3),
    ]
    coups = [
        (0, 1, QuadraticCoupling(weight=0.4)),
        (0, 1, GateBenefitCoupling(weight=0.7, delta_key="delta_eta_domain")),
    ]
    constraints = {"delta_eta_domain": 0.1}
    runner = JaxEnergyRunner(mods, coups, constraints, steps=25, step_size=0.05)
    inputs = [None, None]
    etas0 = runner._compute_initial_etas(inputs)
    energy0 = total_energy(etas0, mods, coups, constraints)
    etas_final = runner.relax(inputs=inputs)
    energy_final = total_energy(etas_final, mods, coups, constraints)
    assert energy_final <= energy0 + 1e-6
    assert all(0.0 <= e <= 1.0 for e in etas_final)

