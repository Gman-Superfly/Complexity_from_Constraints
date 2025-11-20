from __future__ import annotations

import polars as pl

from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from logging.observability import RelaxationTracker
from modules.gating.energy_gating import EnergyGatingModule


def test_observability_relaxation_trace_non_increasing_and_bounded():
    # small chain to stabilize behavior
    mods = [EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3) for _ in range(4)]
    coups = [(0, 1, QuadraticCoupling(weight=0.5)),
             (1, 2, QuadraticCoupling(weight=0.5)),
             (2, 3, QuadraticCoupling(weight=0.5))]
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        grad_eps=1e-6,
        step_size=0.05,
        use_analytic=True,
        line_search=True,
        normalize_grads=True,
        use_vectorized_quadratic=True,
    )
    tracker = RelaxationTracker(name="relaxation_trace_test", run_id="run1")
    tracker.attach(coord)
    out = coord.relax_etas([0.9, 0.6, 0.3, 0.1], steps=30)
    tracker.flush()
    # validate file existence and monotonic energy (non-increasing deltas <= 1e-9)
    p = pl.read_csv("logs/relaxation_trace_test.csv")
    df = p.filter(pl.col("run_id") == "run1").sort("step")
    assert df.height > 0
    # delta_energy is NaN for first step; for subsequent steps should be <= tiny epsilon
    if df.height > 1:
        deltas = df["delta_energy"].to_list()[1:]
        assert all((d is not None) and (d <= 1e-6) for d in deltas if d == d)  # ignore NaNs
    # eta bounds
    assert all(0.0 <= v <= 1.0 for v in out)


