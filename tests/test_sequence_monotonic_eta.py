from __future__ import annotations

import numpy as np

from modules.sequence.monotonic_eta import sample_monotonicity_score, SequenceConsistencyModule


def test_monotonicity_score_bounds_and_behavior():
    inc = list(range(20))
    dec = list(range(20, 0, -1))
    eta_inc = sample_monotonicity_score(inc, samples=1000, seed=123)
    eta_dec = sample_monotonicity_score(dec, samples=1000, seed=123)
    assert 0.0 <= eta_inc <= 1.0
    assert 0.0 <= eta_dec <= 1.0
    assert eta_inc > 0.9
    assert eta_dec < 0.2


def test_sequence_module_local_energy_decreases_with_higher_eta():
    mod = SequenceConsistencyModule(alpha=1.0, beta=1.0, samples=256, seed=7)
    inc = list(range(10))
    noisy = [0, 2, 1, 3, 4, 2, 5, 6, 4, 7]
    eta_inc = mod.compute_eta(inc)
    eta_noisy = mod.compute_eta(noisy)
    F_inc = mod.local_energy(eta_inc, constraints={})
    F_noisy = mod.local_energy(eta_noisy, constraints={})
    assert F_inc <= F_noisy + 1e-9


