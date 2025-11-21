from __future__ import annotations

import polars as pl

from cf_logging.observability import GatingMetricsLogger


def test_gating_metrics_logger_writes_csv(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    logger = GatingMetricsLogger(run_id="test_gate")
    logger.record(hazard=0.2, eta_gate=0.7, redemption=0.05, good=True)
    logger.record(hazard=0.1, eta_gate=0.1, redemption=-0.02, good=False)
    logger.flush()
    df = pl.read_csv("logs/gating_metrics.csv")
    assert df.height == 2
    assert df["run_id"].to_list() == ["test_gate", "test_gate"]
    assert bool(df["is_good"][0])

