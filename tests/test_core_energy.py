from __future__ import annotations

import numpy as np

from core.energy import landau_free_energy, descend_free_energy


def test_landau_descend_reduces_energy_below_tc():
    a = -1.0  # ordered phase
    b = 1.0
    eta0 = 0.1
    F0 = float(landau_free_energy(eta0, a, b))
    eta_final, F_final = descend_free_energy(eta0=eta0, a=a, b=b, learning_rate=0.05, steps=300)
    assert F_final <= F0 + 1e-9
    assert abs(eta_final) > abs(eta0) or F_final < F0


