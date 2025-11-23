from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence, Set, Tuple

import math

from core.coordinator import EnergyCoordinator


class AmortizedProposal(Protocol):
    """Interface for hierarchical/amortized inference scaffolding."""

    def propose_initial_etas(self, modules: Sequence[Any], inputs: Sequence[object]) -> List[float]:
        """Return an initial eta vector in [0,1] for the given modules/inputs."""
        ...

    def select_active_set(
        self,
        coord: EnergyCoordinator,
        etas: Sequence[float],
        k: int,
        include_neighbors: bool = True,
        fd_eps: float = 1e-5,
    ) -> Set[int]:
        """Return indices of an active set to refine (top-|grad| + optional neighbors)."""
        ...


def _build_adjacency(couplings: Sequence[Tuple[int, int, object]], num_nodes: int) -> List[Set[int]]:
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    for i, j, _ in couplings:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            adj[i].add(j)
            adj[j].add(i)
    return adj


def _numeric_gradients(coord: EnergyCoordinator, etas: Sequence[float], eps: float = 1e-5) -> List[float]:
    """Finite-difference estimate of dF/deta_i with simple forward differences and clamping."""
    base = float(coord.energy(list(etas)))
    grads: List[float] = []
    for i in range(len(etas)):
        step_vec = list(etas)
        step_vec[i] = float(min(1.0, max(0.0, step_vec[i] + eps)))
        val = float(coord.energy(step_vec))
        grads.append((val - base) / eps)
    return grads


@dataclass
class SimpleHeuristicAmortizer(AmortizedProposal):
    """Heuristic amortizer:
    - proposes initial etas from module.compute_eta if available (else 0.5)
    - selects active set as top-K |grad| plus their neighbors
    """

    default_eta: float = 0.5

    def propose_initial_etas(self, modules: Sequence[Any], inputs: Sequence[object]) -> List[float]:
        assert len(modules) == len(inputs), "modules/inputs length mismatch"
        etas: List[float] = []
        for m, x in zip(modules, inputs):
            try:
                eta = float(m.compute_eta(x))  # type: ignore[attr-defined]
            except Exception:
                eta = float(self.default_eta)
            eta = float(min(1.0, max(0.0, eta)))
            etas.append(eta)
        return etas

    def select_active_set(
        self,
        coord: EnergyCoordinator,
        etas: Sequence[float],
        k: int,
        include_neighbors: bool = True,
        fd_eps: float = 1e-5,
    ) -> Set[int]:
        assert k > 0, "k must be positive"
        n = len(etas)
        assert n == len(coord.modules), "etas length must match number of modules"
        grads = _numeric_gradients(coord, list(etas), eps=fd_eps)
        order = sorted(range(n), key=lambda i: abs(float(grads[i])), reverse=True)
        chosen: Set[int] = set(order[: min(k, n)])
        if include_neighbors and coord.couplings:
            adj = _build_adjacency(coord.couplings, n)
            extras: Set[int] = set()
            for i in chosen:
                extras.update(adj[i])
            chosen.update(extras)
        return chosen


