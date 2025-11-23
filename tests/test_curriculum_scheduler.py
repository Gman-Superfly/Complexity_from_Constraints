from __future__ import annotations

from core.curriculum import CurriculumScheduler


def test_curriculum_progress_and_regress() -> None:
    sched = CurriculumScheduler(min_level=0, max_level=3, level=1, patience=2, progress_rate_threshold=0.0, oscillation_max_for_progress=0.3, regress_on_oscillation_threshold=0.7)
    # Simulate improving energy (monotone drop) -> progress after patience
    history = [1.0, 0.9, 0.82, 0.75, 0.70]
    d1 = sched.update(history, margin_warn=False)
    d2 = sched.update(history + [0.66], margin_warn=False)
    assert d2.new_level >= 2  # should progress at least once
    # Now simulate oscillation or margin warn -> regress after patience
    history2 = [0.70, 0.72, 0.69, 0.73, 0.71]  # oscillatory
    d3 = sched.update(history2, margin_warn=True)   # strong signal to regress
    d4 = sched.update(history2, margin_warn=True)   # second hit -> regress
    assert d4.new_level <= 2  # moved back from peak


