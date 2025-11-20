from __future__ import annotations

import numpy as np

from modules.sequence.monotonic_eta import sample_monotonicity_score


def test_soft_gating_convex_blend_improves_eta_more_with_higher_blend():
    n = 20
    mistake_pos = 8
    seq = np.linspace(0.0, 1.0, num=n).tolist()
    seq[mistake_pos] -= 0.5
    # baseline
    eta_before = sample_monotonicity_score(seq, samples=256, seed=7)
    # define soft repair
    def apply_soft(seq_in, blend: float):
        s = list(seq_in)
        orig = s[mistake_pos]
        target = max(s[mistake_pos], s[mistake_pos - 1])
        s[mistake_pos] = (1.0 - blend) * orig + blend * target
        return s
    # low vs high blend
    seq_low = apply_soft(seq, 0.1)
    seq_high = apply_soft(seq, 0.8)
    eta_low = sample_monotonicity_score(seq_low, samples=256, seed=7)
    eta_high = sample_monotonicity_score(seq_high, samples=256, seed=7)
    # both should be >= baseline; high blend should improve more or equal
    assert eta_low >= eta_before - 1e-9
    assert eta_high >= eta_low - 1e-9


