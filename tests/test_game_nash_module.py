from __future__ import annotations

import math

from modules.game.emergent_nash import NashModule


def test_nash_module_derivative_matches_finite_difference():
    module = NashModule()
    constraints = {}
    for eta in [0.05, 0.5, 0.95]:
        d_analytic = module.d_local_energy_d_eta(eta, constraints)
        eps = 1e-5
        upper = min(1.0, eta + eps)
        lower = max(0.0, eta - eps)
        f_plus = module.local_energy(upper, constraints)
        f_minus = module.local_energy(lower, constraints)
        if 0.0 < eta < 1.0:
            d_numeric = (f_plus - f_minus) / (2.0 * eps)
        elif eta <= 0.0:
            d_numeric = (f_plus - module.local_energy(eta, constraints)) / eps
        else:
            d_numeric = (module.local_energy(eta, constraints) - f_minus) / eps
        assert math.isfinite(d_analytic)
        assert abs(d_analytic - d_numeric) < 1e-3

