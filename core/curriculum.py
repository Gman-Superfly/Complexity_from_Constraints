from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.agm_metrics import compute_agm_phase_metrics


@dataclass
class CurriculumDecision:
    """Decision output from the curriculum scheduler."""
    new_level: int
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CurriculumScheduler:
    """Rule-based curriculum scheduler driven by AGM phase metrics and stability hints.
    
    Progress when:
      - improvement rate is positive and oscillation is low for `patience` consecutive updates
    Regress when:
      - margin warnings are present or oscillation is high for `patience` consecutive updates
    
    All transitions are clamped to [min_level, max_level].
    """
    min_level: int = 0
    max_level: int = 5
    level: int = 0
    # Thresholds
    progress_rate_threshold: float = 0.0       # positive improvement
    oscillation_max_for_progress: float = 0.25 # low oscillation target
    regress_on_oscillation_threshold: float = 0.6
    # Controls
    patience: int = 2
    # Internal streak counters
    _progress_streak: int = 0
    _regress_streak: int = 0

    def reset(self) -> None:
        self._progress_streak = 0
        self._regress_streak = 0

    def update(
        self,
        recent_energy_history: List[float],
        margin_warn: Optional[bool] = None,
    ) -> CurriculumDecision:
        assert self.min_level <= self.level <= self.max_level, "Level out of bounds"
        metrics = compute_agm_phase_metrics(list(recent_energy_history or []))
        rate = float(metrics.get("rate", 0.0))
        oscillation = float(metrics.get("oscillation", 0.0))
        # Evaluate progress/regress predicates
        can_progress = (rate > self.progress_rate_threshold) and (oscillation <= self.oscillation_max_for_progress) and not bool(margin_warn)
        should_regress = bool(margin_warn) or (oscillation >= self.regress_on_oscillation_threshold)
        reason = "hold"
        # Update streaks
        if can_progress and not should_regress:
            self._progress_streak += 1
            self._regress_streak = 0
        elif should_regress and not can_progress:
            self._regress_streak += 1
            self._progress_streak = 0
        else:
            # conflicting or neutral signals -> reset both
            self._progress_streak = 0
            self._regress_streak = 0
        # Apply transitions
        if self._progress_streak >= self.patience and self.level < self.max_level:
            self.level += 1
            self._progress_streak = 0
            reason = "progress"
        elif self._regress_streak >= self.patience and self.level > self.min_level:
            self.level -= 1
            self._regress_streak = 0
            reason = "regress"
        return CurriculumDecision(new_level=int(self.level), reason=reason, metrics={"rate": rate, "oscillation": oscillation})


