"""Lightweight metrics logging using Polars."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import polars as pl

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def log_records(name: str, records: List[Dict[str, Any]]) -> Path:
    """Append records to a CSV file under logs/.
    
    Args:
        name: Base filename without extension.
        records: List of dict rows.
    Returns:
        Path to the written CSV file.
    """
    assert isinstance(name, str) and len(name) > 0, "Invalid log name"
    assert isinstance(records, list), "records must be a list"
    if not records:
        return LOG_DIR / f"{name}.csv"
    df = pl.DataFrame(records)
    out = LOG_DIR / f"{name}.csv"
    if out.exists():
        # read, vstack, and overwrite
        prev = pl.read_csv(out)
        df = pl.concat([prev, df], how="vertical_relaxed")
    df.write_csv(out)
    return out


def log_record(name: str, record: Dict[str, Any]) -> Path:
    """Append a single record to a CSV file."""
    return log_records(name, [record])


