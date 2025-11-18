"""Production-style tests for EnergyGatingModule lifecycle and invariants.

Tests validate:
- η_gate ∈ [0,1] and increases with net benefit (gain - cost)
- hazard λ(net) ≥ 0 and increases with net
- local energy non-negative and derivative matches analytic form
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from modules.gating.energy_gating import EnergyGatingModule


def test_eta_range_and_monotonicity() -> None:
    # gain_fn returns x directly; vary x to change net benefit
    gate = EnergyGatingModule(gain_fn=lambda x: float(x), cost=0.1, k=8.0, use_hazard=True)
    eta_low = gate.compute_eta(0.0)   # net = -0.1
    eta_high = gate.compute_eta(1.0)  # net = 0.9
    assert 0.0 <= eta_low <= 1.0
    assert 0.0 <= eta_high <= 1.0
    assert eta_high > eta_low, "η should increase with net benefit"


def test_hazard_non_negative_and_monotonic() -> None:
    gate = EnergyGatingModule(gain_fn=lambda x: float(x), cost=0.0, k=10.0, use_hazard=True)
    lam0 = gate.hazard_rate(-1.0)
    lam1 = gate.hazard_rate(0.0)
    lam2 = gate.hazard_rate(1.0)
    assert lam0 >= 0.0 and lam1 >= 0.0 and lam2 >= 0.0
    assert lam2 > lam1 >= lam0, "hazard should increase with net"


def test_local_energy_non_negative_and_derivative() -> None:
    gate = EnergyGatingModule(gain_fn=lambda _: 0.0, cost=0.0, a=0.2, b=0.3)
    for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        f = gate.local_energy(eta, constraints={})
        assert f >= 0.0
        # check analytic derivative vs finite difference
        d_analytic = gate.d_local_energy_d_eta(eta, constraints={})
        eps = 1e-5
        f_plus = gate.local_energy(min(1.0, eta + eps), constraints={})
        f_minus = gate.local_energy(max(0.0, eta - eps), constraints={})
        # central difference; use simple forward/backward near boundaries
        if 0.0 < eta < 1.0:
            d_num = (f_plus - f_minus) / (2.0 * eps)
        elif eta == 0.0:
            d_num = (f_plus - f) / eps
        else:
            d_num = (f - f_minus) / eps
        assert math.isfinite(d_analytic)
        assert abs(d_analytic - d_num) < 1e-3



