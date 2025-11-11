"""Lightweight entity adapter for modules with event-style hooks.

Does not require external Abstractions; provides ecs_id, version, and basic
emit semantics to integrate with the coordinator callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Callable
from uuid import UUID, uuid4
from datetime import datetime, timezone

from .interfaces import EnergyModule, OrderParameter

EmitFn = Callable[[str, dict], None]


@dataclass
class ModuleEntity:
    """Wraps an EnergyModule with ecs-style identity and versioning."""

    module: EnergyModule
    ecs_id: UUID = field(default_factory=uuid4)
    version: int = 0

    def compute_eta(self, x: Any) -> OrderParameter:
        eta = self.module.compute_eta(x)
        return float(eta)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return float(self.module.local_energy(eta, constraints))

    def bump_version(self) -> None:
        self.version += 1

    def emit(self, emit_fn: EmitFn, name: str, payload: dict) -> None:
        event = {
            "event": name,
            "ecs_id": str(self.ecs_id),
            "version": int(self.version),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        emit_fn(name, event)


