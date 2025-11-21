from __future__ import annotations

import pytest

from modules.gating.energy_gating import EnergyGatingModule
from core.couplings import QuadraticCoupling


def test_torch_backend_relaxation_decreases_energy():
    pytest.importorskip("torch")

    from core.torch_backend import TorchEnergyRunner
    from core.energy import total_energy

    mods = [EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3) for _ in range(3)]
    coups = [
        (0, 1, QuadraticCoupling(weight=0.5)),
        (1, 2, QuadraticCoupling(weight=0.5)),
    ]
    inputs = [None, None, None]
    constraints = {}
    runner = TorchEnergyRunner(mods, coups, constraints)
    etas0 = runner._compute_initial_etas(inputs)
    energy0 = total_energy(etas0, mods, coups, constraints)
    etas_final = runner.relax(inputs=inputs, steps=50, lr=0.05)
    energy_final = total_energy(etas_final, mods, coups, constraints)
    assert energy_final <= energy0 + 1e-6
    assert all(0.0 <= e <= 1.0 for e in etas_final)

