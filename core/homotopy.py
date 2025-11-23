from __future__ import annotations

from typing import Dict, Mapping


def linear_scale(start: float, total_steps: int, iter_idx: int) -> float:
    """Linear schedule from start→1.0 over total_steps (clamped)."""
    if total_steps <= 0:
        return 1.0
    t = min(1.0, max(0.0, float(iter_idx) / float(total_steps)))
    return max(0.0, float(start) + (1.0 - float(start)) * t)


def term_scales_from_starts(
    starts: Mapping[str, float],
    total_steps: int,
    iter_idx: int,
) -> Dict[str, float]:
    """Return per-term scale factors from individual start scales using linear schedule."""
    result: Dict[str, float] = {}
    for key, start in starts.items():
        result[str(key)] = linear_scale(float(start), total_steps, iter_idx)
    return result


