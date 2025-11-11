"""Energy-gated expansion module aligned with non-local, free-energy coordination.

η_gate reflects the normalized net benefit of expansion vs. its cost and rises
only when expansion is likely to reduce total free energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Callable
import math

from core.interfaces import EnergyModule, OrderParameter

GainFn = Callable[[Any], float]

__all__ = ["EnergyGatingModule"]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class EnergyGatingModule(EnergyModule):
    """Order parameter is a sigmoid of (gain - cost); local energy discourages casual expansion.
    
    - gain_fn should return a positive value when expansion improves order (e.g., η_after - η_before).
    - cost models external constraint/penalty for expansion.
    """
    gain_fn: GainFn
    cost: float = 0.1
    k: float = 10.0  # sharpness of decision boundary
    a: float = 0.1   # local energy weights
    b: float = 0.1

    def compute_eta(self, x: Any) -> OrderParameter:
        gain = float(self.gain_fn(x))
        net = gain - float(self.cost)
        eta = _sigmoid(self.k * net)
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        return float(eta)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        a = float(constraints.get("gate_alpha", self.a))
        b = float(constraints.get("gate_beta", self.b))
        assert a >= 0.0 and b >= 0.0, "alpha/beta must be non-negative"
        # Landau-like around zero: small η preferred unless justified by coupling/benefit
        return float(a * (eta ** 2) + b * (eta ** 4))


