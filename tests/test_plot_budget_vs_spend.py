from __future__ import annotations

import polars as pl

from experiments.plots.plot_budget_vs_spend import compute_alloc_cost_totals


def test_compute_alloc_cost_totals_rowwise_sum() -> None:
    # Build a tiny synthetic per-step dataframe with alloc:/cost: columns
    df = pl.DataFrame(
        {
            "run_id": ["r"] * 3,
            "step": [1, 2, 3],
            "spent:global": [0.1, 0.2, 0.3],
            "contraction_margin": [0.5, 0.4, 0.6],
            "alloc:coup:A": [0.01, 0.02, 0.03],
            "alloc:coup:B": [0.04, 0.05, 0.06],
            "cost:coup:A": [0.10, 0.20, 0.30],
            "cost:coup:B": [0.40, 0.50, 0.60],
        }
    )
    alloc_total, cost_total = compute_alloc_cost_totals(df)
    assert alloc_total is not None and cost_total is not None
    # Expected row-wise sums
    expected_alloc = [0.05, 0.07, 0.09]
    expected_cost = [0.50, 0.70, 0.90]
    # Compare with a small tolerance
    for a, ea in zip(alloc_total, expected_alloc):
        assert abs(float(a) - float(ea)) < 1e-9
    for c, ec in zip(cost_total, expected_cost):
        assert abs(float(c) - float(ec)) < 1e-9


