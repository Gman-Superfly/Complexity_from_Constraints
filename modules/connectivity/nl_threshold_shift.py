"""Non-Local Connectivity Threshold Shift utilities.

Computes connectivity order parameter on a 2D grid with bond probability p.
Optionally adds sparse non-local shortcuts that lower the apparent critical p.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple
import random

import networkx as nx

from core.interfaces import EnergyModule, OrderParameter

__all__ = ["build_grid_bond_graph", "largest_component_fraction", "ConnectivityModule"]


def build_grid_bond_graph(n: int, p: float, add_shortcuts: bool = False, shortcut_frac: float = 0.0, seed: int | None = None) -> nx.Graph:
    assert n >= 2, "n must be >= 2"
    assert 0.0 <= p <= 1.0, "p must be within [0,1]"
    assert 0.0 <= shortcut_frac <= 1.0, "shortcut fraction must be within [0,1]"
    rng = random.Random(seed)
    G = nx.grid_2d_graph(n, n)
    # Remove edges with probability (1-p)
    to_remove = []
    for u, v in G.edges():
        if rng.random() > p:
            to_remove.append((u, v))
    if to_remove:
        G.remove_edges_from(to_remove)
    if add_shortcuts and shortcut_frac > 0.0:
        # add shortcuts between random pairs of nodes
        nodes = list(G.nodes())
        m = int(shortcut_frac * G.number_of_edges())
        for _ in range(m):
            a = rng.choice(nodes)
            b = rng.choice(nodes)
            if a != b:
                G.add_edge(a, b)
    return G


def largest_component_fraction(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    comps = list(nx.connected_components(G))
    if not comps:
        return 0.0
    largest = max(comps, key=len)
    return float(len(largest)) / float(n)


@dataclass
class ConnectivityModule(EnergyModule):
    """Connectivity module: η is largest component fraction; local energy penalizes fragmentation."""
    alpha: float = 1.0
    beta: float = 1.0

    def compute_eta(self, x: Any) -> OrderParameter:
        assert isinstance(x, nx.Graph), "Expected a networkx Graph as input"
        eta = largest_component_fraction(x)
        assert 0.0 <= eta <= 1.0, "η must be within [0,1]"
        return float(eta)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        assert 0.0 <= eta <= 1.0, "η must be within [0,1]"
        a = float(constraints.get("conn_alpha", self.alpha))
        b = float(constraints.get("conn_beta", self.beta))
        assert a >= 0.0 and b >= 0.0, "alpha/beta must be non-negative"
        delta = (1.0 - float(eta))
        return float(a * (delta ** 2) + b * (delta ** 4))


