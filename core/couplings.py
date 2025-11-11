"""Standard coupling energies between order parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .interfaces import EnergyCoupling, OrderParameter

__all__ = ["QuadraticCoupling", "DirectedHingeCoupling", "GateBenefitCoupling"]


@dataclass(frozen=True)
class QuadraticCoupling(EnergyCoupling):
    """Symmetric quadratic coupling: w * (η_i - η_j)^2."""
    weight: float = 1.0

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        assert self.weight >= 0.0, "weight must be non-negative"
        diff = float(eta_i) - float(eta_j)
        return float(self.weight * (diff * diff))


@dataclass(frozen=True)
class DirectedHingeCoupling(EnergyCoupling):
    """Directed hinge-like coupling: w * max(0, η_j - η_i)^2.
    
    Encourages η_i to not fall below η_j.
    """
    weight: float = 1.0

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        assert self.weight >= 0.0, "weight must be non-negative"
        gap = max(0.0, float(eta_j) - float(eta_i))
        return float(self.weight * (gap * gap))


@dataclass(frozen=True)
class GateBenefitCoupling(EnergyCoupling):
    """Coupling that rewards opening a gate when domain improvement exists.
    
    Energy: F = - w * η_gate * Δη_domain
    Where:
      - η_gate in [0,1]
      - Δη_domain provided via constraints under key `delta_eta_domain`
    
    Note: Caller is responsible for computing Δη_domain (e.g., η_after - η_before)
    based on current inputs. This keeps the coupling generic and non-invasive.
    """
    weight: float = 1.0
    delta_key: str = "delta_eta_domain"

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        assert self.weight >= 0.0, "weight must be non-negative"
        # interpret eta_i as gate, eta_j as domain eta (unused directly)
        delta = float(constraints.get(self.delta_key, 0.0))
        eta_gate = float(eta_i)
        return float(-self.weight * eta_gate * delta)


