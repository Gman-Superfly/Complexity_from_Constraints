from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.coordinator import EnergyCoordinator
from cf_logging.metrics_log import log_records


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


@dataclass
class EnergyBudgetTracker:
    """Per-step energy budget and gradient norms with optional stability metrics."""

    name: str = "energy_budget"
    run_id: str = "default"
    buffer: List[Dict[str, Any]] = field(default_factory=list)
    coord: Optional[EnergyCoordinator] = None
    last_etas: Optional[List[float]] = None

    def attach(self, coord: EnergyCoordinator) -> None:
        self.coord = coord
        coord.on_eta_updated.append(self.on_eta)
        coord.on_energy_updated.append(self.on_energy)

    def on_eta(self, etas: List[float]) -> None:
        self.last_etas = [float(e) for e in etas]

    def on_energy(self, energy: float) -> None:
        if self.coord is None or self.last_etas is None:
            return
        coord = self.coord
        etas = self.last_etas
        row: Dict[str, Any] = {"run_id": self.run_id, "energy": float(energy)}
        # Per-term energies
        cw = coord._combined_term_weights()  # instrumentation
        for idx, m in enumerate(coord.modules):
            key = f"local:{m.__class__.__name__}"
            w = float(cw.get(key, 1.0))
            e = float(m.local_energy(float(etas[idx]), coord.constraints)) * w
            row[f"energy:{key}"] = e
        for i, j, coup in coord.couplings:
            key = f"coup:{coup.__class__.__name__}"
            w = float(cw.get(key, 1.0))
            e = float(coup.coupling_energy(float(etas[i]), float(etas[j]), coord.constraints)) * w
            row[f"energy:{key}"] = float(row.get(f"energy:{key}", 0.0) + e)
        # Gradient norms per term
        norms = coord._term_grad_norms(etas)  # instrumentation
        for k, v in norms.items():
            row[f"grad_norm:{k}"] = float(v)
        # Stability/backtracks (if available)
        margin = getattr(coord, "_last_contraction_margin", None)
        last_bk = getattr(coord, "_last_step_backtracks", None)
        total_bk = getattr(coord, "_total_backtracks", None)
        if margin is not None:
            row["contraction_margin"] = float(margin)
        # Lipschitz allocator details (if available)
        details = getattr(coord, "_last_lipschitz_details", None)
        if isinstance(details, dict):
            gm = details.get("global_margin", None)
            if gm is not None:
                try:
                    row["margin:global"] = float(gm)
                except Exception:
                    pass
            row_m = details.get("row_margins", None)
            if isinstance(row_m, dict):
                for r, val in row_m.items():
                    try:
                        row[f"margin:row:{int(r)}"] = float(val)
                    except Exception:
                        continue
            fam_costs = details.get("family_costs", None)
            if isinstance(fam_costs, dict):
                for k, val in fam_costs.items():
                    try:
                        row[f"cost:{str(k)}"] = float(val)
                    except Exception:
                        continue
        # Adapter allocations/scores if present
        wa = getattr(coord, "weight_adapter", None)
        if wa is not None:
            scores = getattr(wa, "scores", None)
            if isinstance(scores, dict):
                for k, val in scores.items():
                    try:
                        row[f"score:{str(k)}"] = float(val)
                    except Exception:
                        continue
            allocs = getattr(wa, "last_allocations", None)
            if isinstance(allocs, dict):
                for k, val in allocs.items():
                    try:
                        row[f"alloc:{str(k)}"] = float(val)
                    except Exception:
                        continue
            spent_g = getattr(wa, "last_spent_global", None)
            if isinstance(spent_g, (int, float)):
                row["spent:global"] = float(spent_g)
        if last_bk is not None:
            row["last_backtracks"] = int(last_bk)
        if total_bk is not None:
            row["total_backtracks"] = int(total_bk)
        self.buffer.append(row)

    def flush(self) -> None:
        if not self.buffer:
            return
        log_records(self.name, self.buffer)
        self.buffer.clear()

@dataclass
class GatingMetricsLogger:
    """Lightweight logger for gating decisions (η_gate, hazard, redemption, etc.)."""

    name: str = "gating_metrics"
    run_id: str = "default"
    buffer: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, *, hazard: float, eta_gate: float, redemption: float, good: bool) -> None:
        self.buffer.append({
            "run_id": self.run_id,
            "hazard": float(hazard),
            "eta_gate": float(eta_gate),
            "redemption": float(redemption),
            "is_good": bool(good),
        })

    def flush(self) -> None:
        if not self.buffer:
            return
        log_records(self.name, self.buffer)
        self.buffer.clear()



