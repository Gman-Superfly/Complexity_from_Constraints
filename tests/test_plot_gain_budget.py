from __future__ import annotations

import polars as pl

from experiments.plots.plot_gain_budget import compute_gain_budget_timeseries


def test_compute_gain_budget_timeseries_totals_and_ratio() -> None:
    df = pl.DataFrame(
        {
            "run_id": ["r"] * 3,
            "step": [0, 1, 2],
            "alloc:coup:QuadraticCoupling": [0.2, 0.4, 0.6],
            "alloc:coup:DirectedHingeCoupling": [0.3, 0.1, 0.1],
            "cost:coup:QuadraticCoupling": [1.0, 1.0, 1.0],
            "cost:coup:DirectedHingeCoupling": [0.5, 0.5, 0.5],
        }
    )
    out, alloc_cols, cost_cols = compute_gain_budget_timeseries(df)
    assert "alloc:total" in out.columns
    assert "cost:total" in out.columns
    assert "alloc_to_cost_ratio" in out.columns
    # Check totals row 0
    a0 = float(out.row(0, named=True)["alloc:total"])
    c0 = float(out.row(0, named=True)["cost:total"])
    assert abs(a0 - (0.2 + 0.3)) < 1e-12
    assert abs(c0 - (1.0 + 0.5)) < 1e-12
    r0 = float(out.row(0, named=True)["alloc_to_cost_ratio"])
    assert abs(r0 - (a0 / c0)) < 1e-12


