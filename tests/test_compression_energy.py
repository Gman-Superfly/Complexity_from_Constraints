from __future__ import annotations

import numpy as np

from modules.compression.compression_energy import CompressionEnergyModule


def test_compression_energy_minimum_at_target() -> None:
    mod = CompressionEnergyModule(a=1.0, b=0.5, target_default=0.6)
    constraints = {}
    etas = np.linspace(0.0, 1.0, 51)
    vals = [mod.local_energy(float(e), constraints) for e in etas]
    e_min = etas[int(np.argmin(vals))]
    assert abs(e_min - 0.6) <= 0.04  # coarse grid tolerance


def test_compression_energy_derivative_matches_fd() -> None:
    mod = CompressionEnergyModule(a=1.2, b=0.3, target_default=0.7)
    constraints = {}
    eta = 0.8
    fd_eps = 1e-5
    base = mod.local_energy(eta, constraints)
    bump = mod.local_energy(min(1.0, eta + fd_eps), constraints)
    fd = (bump - base) / fd_eps
    der = mod.d_local_energy_d_eta(eta, constraints)
    assert abs(fd - der) <= 1e-3


