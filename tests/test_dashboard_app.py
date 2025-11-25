from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("streamlit")

from tools import p3_dashboard as dashboard


def test_load_csv_returns_empty_for_missing_file(tmp_path, monkeypatch):
    # Point dashboard to a temporary logs directory with no files
    monkeypatch.setattr(dashboard, "LOG_DIR", tmp_path)
    result = dashboard.load_csv("missing.csv")
    assert result.is_empty()


def test_load_csv_reads_existing_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "LOG_DIR", tmp_path)
    data = "run_id,step,energy\nrunA,0,1.23\nrunA,1,1.00\n"
    csv_path = tmp_path / "energy_budget.csv"
    csv_path.write_text(data, encoding="utf-8")

    df = dashboard.load_csv("energy_budget.csv")
    assert not df.is_empty()
    assert df.height == 2
    assert df.get_column("energy")[0] == pytest.approx(1.23)


def test_filter_by_run_filters_rows():
    df = pl.DataFrame({"run_id": ["alpha", "beta"], "value": [10, 20]})
    filtered = dashboard.filter_by_run(df, "beta")
    assert filtered.height == 1
    assert filtered.get_column("value")[0] == 20
    # None run_id should return unfiltered df
    assert dashboard.filter_by_run(df, None).height == 2

