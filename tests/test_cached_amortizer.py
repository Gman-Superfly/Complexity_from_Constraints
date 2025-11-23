from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.amortizer import CachedActiveSetAmortizer, SimpleHeuristicAmortizer
from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter


@dataclass
class CacheModule(EnergyModule):
    base: float

    def compute_eta(self, x: Mapping[str, float]) -> OrderParameter:
        return float(x.get("normalized_count", self.base))

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return (float(eta) - self.base) ** 2


def build_modules(count: int) -> list[CacheModule]:
    return [CacheModule(base=0.2 + 0.1 * idx) for idx in range(count)]


def test_cached_amortizer_reuses_eta_vectors() -> None:
    modules = build_modules(3)
    inputs = [{"normalized_count": 0.1 * i} for i in range(3)]
    amortizer = CachedActiveSetAmortizer(default_eta=0.4, cache_size=2, similarity_threshold=0.01)
    first = amortizer.propose_initial_etas(modules, inputs)
    summary = amortizer.cache_summary()
    assert summary["entries"] == 1
    second = amortizer.propose_initial_etas(modules, inputs)
    assert second == first


def test_stage_planning_emits_budget_fraction() -> None:
    modules = build_modules(2)
    coord = EnergyCoordinator(modules=modules, couplings=[], constraints={}, stability_guard=False)
    amortizer = SimpleHeuristicAmortizer()
    etas = amortizer.propose_initial_etas(modules, [{"normalized_count": 0.0}, {"normalized_count": 0.5}])
    plans = amortizer.plan_stage_execution(coord, etas, stages=[("stage0", 1), ("stage1", 1)])
    assert len(plans) == 2
    assert abs(sum(plan.budget_fraction for plan in plans) - 1.0) < 1e-6

