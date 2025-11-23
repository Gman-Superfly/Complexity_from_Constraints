from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.amortizer import SimpleHeuristicAmortizer
from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling, GateBenefitCoupling
from modules.gating.energy_gating import EnergyGatingModule


def _make_small_ring(n: int = 4) -> Tuple[List[Any], List[Tuple[int, int, Any]], Dict[str, Any], List[Any]]:
    modules: List[EnergyGatingModule] = []
    inputs: List[Any] = []
    for idx in range(n):
        modules.append(EnergyGatingModule(gain_fn=lambda _, i=idx: 0.05 * (i % 3), a=0.2, b=0.2))
        inputs.append(None)
    couplings: List[Tuple[int, int, Any]] = []
    for i in range(n):
        j = (i + 1) % n
        couplings.append((i, j, QuadraticCoupling(weight=0.4)))
        couplings.append((j, i, GateBenefitCoupling(weight=0.3, delta_key="delta_eta_domain")))
    constraints: Dict[str, Any] = {"delta_eta_domain": 0.04}
    return modules, couplings, constraints, inputs


def test_simple_heuristic_amortizer_proposal_and_active_set() -> None:
    mods, coups, constraints, inputs = _make_small_ring(4)
    coord = EnergyCoordinator(modules=mods, couplings=coups, constraints=constraints, use_analytic=True, line_search=True)
    amort = SimpleHeuristicAmortizer(default_eta=0.5)
    etas0 = amort.propose_initial_etas(mods, inputs)
    assert len(etas0) == len(mods)
    assert all(0.0 <= e <= 1.0 for e in etas0)
    # Pick k=1 and ensure we get at least that many plus neighbors
    active = amort.select_active_set(coord, etas0, k=1, include_neighbors=True)
    assert len(active) >= 1
    # With a ring, picking one index should include at least two neighbors total (left+right)
    assert len(active) >= 3


