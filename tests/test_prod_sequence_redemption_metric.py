from __future__ import annotations

import numpy as np

from modules.sequence.monotonic_eta import sample_monotonicity_score


def test_prod_sequence_redemption_metric_positive_for_planted_mistake():
    rng = np.random.default_rng(123)
    n = 40
    mistake_pos = 15
    trials = 5
    reds = []
    for t in range(trials):
        seq = np.linspace(0.0, 1.0, num=n).tolist()
        seq[mistake_pos] -= 0.5
        seq = (np.asarray(seq) + rng.normal(0.0, 0.0, size=n)).tolist()
        local_losses = []
        nonlocal_losses = []
        for i in range(2, n + 1):
            prefix = seq[:i]
            eta_local = sample_monotonicity_score(prefix, samples=256, seed=42 + t)
            eta_nonlocal = sample_monotonicity_score(seq, samples=256, seed=42 + t)
            local_losses.append(1.0 - eta_local)
            nonlocal_losses.append(1.0 - eta_nonlocal)
        red = float(np.mean(np.asarray(local_losses) - np.asarray(nonlocal_losses)))
        reds.append(red)
    # average redemption should be non-negative for constructed dip
    assert float(np.mean(reds)) >= -1e-6


