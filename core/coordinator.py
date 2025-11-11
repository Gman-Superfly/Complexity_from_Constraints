"""Coordinator for total energy evaluation and optional eta relaxation.

This coordinator can:
- compute etas from inputs via modules
- compute total energy with couplings
- optionally relax etas by gradient steps on F_total (finite-difference)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Tuple

import math

from .interfaces import EnergyModule, EnergyCoupling, OrderParameter
from .energy import total_energy

EtaUpdateCallback = Callable[[List[OrderParameter]], None]
EnergyUpdateCallback = Callable[[float], None]


@dataclass
class EnergyCoordinator:
    """Energy coordinator with simple event hooks."""

    modules: List[EnergyModule]
    couplings: List[tuple[int, int, EnergyCoupling]]
    constraints: Mapping[str, Any]
    grad_eps: float = 1e-4
    step_size: float = 0.05

    on_eta_updated: List[EtaUpdateCallback] = field(default_factory=list)
    on_energy_updated: List[EnergyUpdateCallback] = field(default_factory=list)
    use_analytic: bool = False

    def compute_etas(self, inputs: List[Any]) -> List[OrderParameter]:
        assert len(inputs) == len(self.modules), "inputs/modules length mismatch"
        etas: List[OrderParameter] = []
        for module, x in zip(self.modules, inputs):
            eta = float(module.compute_eta(x))
            etas.append(eta)
        self._emit_eta(etas)
        return etas

    def energy(self, etas: List[OrderParameter]) -> float:
        F = total_energy(etas, self.modules, self.couplings, self.constraints)
        self._emit_energy(F)
        return F

    def relax_etas(self, etas0: List[OrderParameter], steps: int = 50) -> List[OrderParameter]:
        """Finite-difference gradient steps on F_total w.r.t. etas."""
        etas = [float(e) for e in etas0]
        for _ in range(steps):
            grads = self._grads(etas)
            # gradient descent
            for i in range(len(etas)):
                etas[i] -= self.step_size * grads[i]
            self._emit_eta(etas)
            self._emit_energy(self.energy(etas))
        return etas

    def _finite_diff_grads(self, etas: List[OrderParameter]) -> List[float]:
        base = self.energy(etas)
        grads: List[float] = []
        for i in range(len(etas)):
            bumped = list(etas)
            bumped[i] += self.grad_eps
            Fb = self.energy(bumped)
            grad_i = (Fb - base) / self.grad_eps
            grads.append(float(grad_i))
        return grads

    def _analytic_grads(self, etas: List[OrderParameter]) -> List[float]:
        """Attempt analytic grads using optional d_local_energy_d_eta on modules; fallback per component."""
        grads: List[float] = [0.0 for _ in etas]
        # local energy derivatives if available
        for idx, (m, eta) in enumerate(zip(self.modules, etas)):
            d = None
            # use duck-typing to check for derivative
            if hasattr(m, "d_local_energy_d_eta"):
                try:
                    d = float(getattr(m, "d_local_energy_d_eta")(float(eta), self.constraints))  # type: ignore[attr-defined]
                except Exception:
                    d = None
            if d is None:
                # finite-diff for this component
                base = float(m.local_energy(eta, self.constraints))
                b = float(m.local_energy(eta + self.grad_eps, self.constraints))
                d = (b - base) / self.grad_eps
            grads[idx] += d
        # couplings: always finite-diff per eta since generic
        base_total = self.energy(etas)
        for i in range(len(etas)):
            bumped = list(etas)
            bumped[i] += self.grad_eps
            Fb = self.energy(bumped)
            grads[i] += (Fb - base_total) / self.grad_eps
        return grads

    def _grads(self, etas: List[OrderParameter]) -> List[float]:
        if self.use_analytic:
            try:
                return self._analytic_grads(etas)
            except Exception:
                return self._finite_diff_grads(etas)
        return self._finite_diff_grads(etas)

    def _emit_eta(self, etas: List[OrderParameter]) -> None:
        for cb in self.on_eta_updated:
            cb(etas)

    def _emit_energy(self, F: float) -> None:
        for cb in self.on_energy_updated:
            cb(F)


