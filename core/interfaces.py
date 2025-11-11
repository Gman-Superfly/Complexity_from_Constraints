"""Interfaces for energy-coordinated micro-modules.

Exposes strict typed Protocols for modules and couplings.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable, Tuple, List

__all__ = [
    "OrderParameter",
    "EnergyModule",
    "EnergyCoupling",
]

OrderParameter = float


@runtime_checkable
class EnergyModule(Protocol):
    """A small module exposing an order parameter and a local energy."""

    def compute_eta(self, x: Any) -> OrderParameter:
        """Compute the order parameter η for the given input."""
        ...

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        """Compute the module's local free energy F(η; c)."""
        ...


@runtime_checkable
class EnergyCoupling(Protocol):
    """Sparse coupling between two modules' order parameters."""

    def coupling_energy(
        self,
        eta_i: OrderParameter,
        eta_j: OrderParameter,
        constraints: Mapping[str, Any],
    ) -> float:
        """Energy contribution from coupling two order parameters."""
        ...


