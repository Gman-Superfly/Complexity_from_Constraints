from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.coordinator import EnergyCoordinator
from logging.metrics_log import log_records


@dataclass
class RelaxationTracker:
    """Attach to EnergyCoordinator callbacks and log energy/η traces to Polars CSV.
    
    Usage:
        tracker = RelaxationTracker(name="relaxation_trace", run_id="demo")
        tracker.attach(coord)
        coord.relax_etas(...)
        tracker.flush()
    """
    name: str
    run_id: str
    buffer: List[Dict[str, Any]] = field(default_factory=list)
    step: int = 0
    prev_energy: Optional[float] = None
    last_etas: Optional[List[float]] = None

    def attach(self, coord: EnergyCoordinator) -> None:
        coord.on_eta_updated.append(self.on_eta)
        coord.on_energy_updated.append(self.on_energy)

    def on_eta(self, etas: List[float]) -> None:
        self.last_etas = [float(e) for e in etas]

    def on_energy(self, energy: float) -> None:
        self.step += 1
        delta = None if self.prev_energy is None else float(energy - self.prev_energy)
        self.prev_energy = float(energy)
        if self.last_etas:
            min_eta = float(min(self.last_etas))
            max_eta = float(max(self.last_etas))
            mean_eta = float(sum(self.last_etas) / float(len(self.last_etas)))
        else:
            min_eta = float("nan")
            max_eta = float("nan")
            mean_eta = float("nan")
        self.buffer.append({
            "run_id": self.run_id,
            "step": int(self.step),
            "energy": float(energy),
            "delta_energy": float("nan") if delta is None else float(delta),
            "min_eta": min_eta,
            "max_eta": max_eta,
            "mean_eta": mean_eta,
        })

    def flush(self) -> None:
        if self.buffer:
            log_records(self.name, self.buffer)
            self.buffer.clear()


