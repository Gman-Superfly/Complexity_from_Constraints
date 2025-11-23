from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.interfaces import EnergyModule, OrderParameter


@dataclass
class CompressionEnergyModule(EnergyModule):
    """Penalize deviation from a target compression ratio.

    Interpret η as observed compression ratio (in [0,1]). Local energy:
        F(η) = a * (η - target)^2 + b * (η - target)^4
    Defaults: target from constraints['compression_target'] or 1.0.
    """

    a: float = 1.0
    b: float = 0.0
    target_default: float = 1.0

    def compute_eta(self, x: Any) -> OrderParameter:
        """Treat x as observed compression ratio in [0,1]."""
        try:
            eta = float(x)
        except Exception:
            eta = 0.0
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        return float(eta)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        a = float(constraints.get("compression_alpha", self.a))
        b = float(constraints.get("compression_beta", self.b))
        target = float(constraints.get("compression_target", self.target_default))
        delta = float(eta) - target
        return float(a * (delta ** 2) + b * (delta ** 4))

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
        a = float(constraints.get("compression_alpha", self.a))
        b = float(constraints.get("compression_beta", self.b))
        target = float(constraints.get("compression_target", self.target_default))
        delta = float(eta) - target
        # d/dη [a (η-t)^2 + b (η-t)^4] = 2a(η-t) + 4b(η-t)^3
        return float(2.0 * a * delta + 4.0 * b * (delta ** 3))


