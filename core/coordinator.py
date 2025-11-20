"""Coordinator for total energy evaluation and optional eta relaxation.

This coordinator can:
- compute etas from inputs via modules
- compute total energy with couplings
- optionally relax etas by gradient steps on F_total (finite-difference)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Tuple, Optional

import math
import numpy as np

from .interfaces import (
    EnergyModule,
    EnergyCoupling,
    OrderParameter,
    SupportsLocalEnergyGrad,
    SupportsCouplingGrads,
    WeightAdapter,
)
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
    # Gradient/optimization controls
    use_analytic: bool = False
    normalize_grads: bool = False
    max_grad_norm: Optional[float] = None
    line_search: bool = False
    backtrack_factor: float = 0.5
    max_backtrack: int = 5
    armijo_c: float = 1e-6
    use_vectorized_quadratic: bool = False
    # Optional term-weight adapter
    weight_adapter: Optional[WeightAdapter] = None

    on_eta_updated: List[EtaUpdateCallback] = field(default_factory=list)
    on_energy_updated: List[EnergyUpdateCallback] = field(default_factory=list)

    _adjacency: Optional[List[List[Tuple[int, EnergyCoupling]]]] = field(default=None, init=False, repr=False)
    _term_weights: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def compute_etas(self, inputs: List[Any]) -> List[OrderParameter]:
        assert len(inputs) == len(self.modules), "inputs/modules length mismatch"
        etas: List[OrderParameter] = []
        for module, x in zip(self.modules, inputs):
            eta = float(module.compute_eta(x))
            etas.append(eta)
        self._emit_eta(etas)
        return etas

    def energy(self, etas: List[OrderParameter]) -> float:
        # Merge term weights (constraints.term_weights overridden by adapter-maintained _term_weights)
        merged_constraints: dict[str, Any] = dict(self.constraints)
        base_tw = {}
        tw = self.constraints.get("term_weights", None)
        if isinstance(tw, dict):
            for k, v in tw.items():
                try:
                    base_tw[str(k)] = float(v)  # type: ignore[arg-type]
                except Exception:
                    continue
        if self._term_weights:
            base_tw.update({str(k): float(v) for k, v in self._term_weights.items()})
        if base_tw:
            merged_constraints["term_weights"] = base_tw
        F = total_energy(etas, self.modules, self.couplings, merged_constraints)
        self._emit_energy(F)
        return F

    def relax_etas(self, etas0: List[OrderParameter], steps: int = 50) -> List[OrderParameter]:
        """Finite-difference gradient steps on F_total w.r.t. etas."""
        etas = [float(e) for e in etas0]
        for _ in range(steps):
            grads = self._grads(etas)
            # optional normalization/clipping
            if self.normalize_grads:
                norm = float(np.linalg.norm(np.asarray(grads, dtype=float)))
                if norm > 0.0:
                    grads = [g / norm for g in grads]
            if self.max_grad_norm is not None:
                norm = float(np.linalg.norm(np.asarray(grads, dtype=float)))
                if norm > self.max_grad_norm and norm > 0.0:
                    scale = self.max_grad_norm / norm
                    grads = [g * scale for g in grads]
            # step
            if self.line_search:
                etas = self._step_with_backtracking(etas, grads, self.step_size)
            else:
                for i in range(len(etas)):
                    etas[i] = float(max(0.0, min(1.0, etas[i] - self.step_size * grads[i])))
            self._emit_eta(etas)
            # Optional adaptive term weights
            if self.weight_adapter is not None:
                term_norms = self._term_grad_norms(etas)
                E = self.energy(etas)
                updated = self.weight_adapter.step(term_norms, E, dict(self._term_weights))
                # keep only float-like
                self._term_weights = {
                    str(k): float(v) for k, v in updated.items() if isinstance(k, str)
                }
            else:
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
        """Analytic grads using optional module/coupling derivatives; finite-diff per term as fallback (no double-count)."""
        n = len(etas)
        grads: List[float] = [0.0 for _ in range(n)]
        # Local terms (apply term weights)
        cw = self._combined_term_weights()
        for idx, (m, eta) in enumerate(zip(self.modules, etas)):
            w = float(cw.get(f"local:{m.__class__.__name__}", 1.0))
            if isinstance(m, SupportsLocalEnergyGrad):
                grads[idx] += w * float(m.d_local_energy_d_eta(float(eta), self.constraints))
            else:
                base = float(m.local_energy(eta, self.constraints))
                b = float(m.local_energy(eta + self.grad_eps, self.constraints))
                grads[idx] += w * ((b - base) / self.grad_eps)
        # Coupling terms
        if self.use_vectorized_quadratic and self._all_quadratic():
            q_grads = self._quadratic_coupling_gradients_vectorized(etas, cw)
            grads = [gi + gq for gi, gq in zip(grads, q_grads)]
        else:
            for i, j, coup in self.couplings:
                w = float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
                if isinstance(coup, SupportsCouplingGrads):
                    gi, gj = coup.d_coupling_energy_d_etas(etas[i], etas[j], self.constraints)
                    grads[i] += w * float(gi)
                    grads[j] += w * float(gj)
                else:
                    base = float(coup.coupling_energy(etas[i], etas[j], self.constraints))
                    bi = float(coup.coupling_energy(etas[i] + self.grad_eps, etas[j], self.constraints))
                    bj = float(coup.coupling_energy(etas[i], etas[j] + self.grad_eps, self.constraints))
                    grads[i] += w * ((bi - base) / self.grad_eps)
                    grads[j] += w * ((bj - base) / self.grad_eps)
        return grads

    def _grads(self, etas: List[OrderParameter]) -> List[float]:
        if self.use_analytic:
            try:
                return self._analytic_grads(etas)
            except Exception:
                return self._finite_diff_grads(etas)
        return self._finite_diff_grads(etas)

    def relax_etas_coordinate(
        self,
        etas0: List[OrderParameter],
        steps: int = 200,
        active_tol: float = 1e-4,
    ) -> List[OrderParameter]:
        """Coordinate descent: update the index with largest |grad| each iteration."""
        etas = [float(e) for e in etas0]
        # Build adjacency
        self._ensure_adjacency(len(etas))
        # Initialize gradients and energy once
        grads = self._grads(etas)
        F = self.energy(etas)
        for _ in range(steps):
            # pick active coordinate
            idx = int(np.argmax(np.abs(np.asarray(grads, dtype=float))))
            g_i = float(grads[idx])
            if abs(g_i) < active_tol:
                break
            # choose step length
            step = float(self.step_size)
            if self.normalize_grads:
                gabs = abs(g_i)
                if gabs > 0.0:
                    step = step / gabs
            eta_i_old = float(etas[idx])
            eta_i_new = float(max(0.0, min(1.0, eta_i_old - step * g_i)))
            if eta_i_new == eta_i_old:
                # nothing to update
                break
            # local gradient delta
            d_local_old = self._local_grad(idx, eta_i_old)
            d_local_new = self._local_grad(idx, eta_i_new)
            delta_gi = d_local_new - d_local_old
            # local energy delta
            f_local_old = self._local_energy(idx, eta_i_old)
            f_local_new = self._local_energy(idx, eta_i_new)
            delta_F = f_local_new - f_local_old
            # coupling deltas on neighbors
            for (j, coup) in self._adjacency[idx]:  # type: ignore[union-attr]
                eta_j = float(etas[j])
                gi_old, gj_old = self._pair_coupling_grads(coup, idx, j, eta_i_old, eta_j)
                gi_new, gj_new = self._pair_coupling_grads(coup, idx, j, eta_i_new, eta_j)
                delta_gi += (gi_new - gi_old)
                grads[j] = float(grads[j] + (gj_new - gj_old))
                # energy delta for this edge
                f_ij_old = self._pair_coupling_energy(coup, idx, j, eta_i_old, eta_j)
                f_ij_new = self._pair_coupling_energy(coup, idx, j, eta_i_new, eta_j)
                delta_F += (f_ij_new - f_ij_old)
            # commit update
            etas[idx] = eta_i_new
            grads[idx] = float(g_i + delta_gi)
            F = float(F + delta_F)
            self._emit_eta(etas)
            self._emit_energy(F)
        return etas

    def _all_quadratic(self) -> bool:
        if not self.couplings:
            return False
        return all(getattr(c, "__class__", type(None)).__name__ == "QuadraticCoupling" for _i, _j, c in self.couplings)

    def _quadratic_coupling_gradients_vectorized(self, etas: List[OrderParameter], cw: dict[str, float]) -> List[float]:
        """Vectorized accumulation of gradients for quadratic couplings."""
        n = len(etas)
        grads = np.zeros(n, dtype=float)
        if not self.couplings:
            return grads.tolist()
        idx_i = np.fromiter((i for i, _j, _c in self.couplings), dtype=int)
        idx_j = np.fromiter((j for _i, j, _c in self.couplings), dtype=int)
        base_w = np.fromiter((float(getattr(c, "weight", 0.0)) for _i, _j, c in self.couplings), dtype=float)
        term_w = np.fromiter((float(cw.get(f"coup:{c.__class__.__name__}", 1.0)) for _i, _j, c in self.couplings), dtype=float)
        weights = base_w * term_w
        eta_arr = np.asarray(etas, dtype=float)
        diff = eta_arr[idx_i] - eta_arr[idx_j]
        gi = 2.0 * weights * diff
        gj = -2.0 * weights * diff
        np.add.at(grads, idx_i, gi)
        np.add.at(grads, idx_j, gj)
        return grads.tolist()

    def _step_with_backtracking(self, etas: List[OrderParameter], grads: List[float], step_init: float) -> List[float]:
        F0 = self.energy(etas)
        step = float(step_init)
        gvec = np.asarray(grads, dtype=float)
        g2 = float(np.dot(gvec, gvec))
        for _ in range(self.max_backtrack + 1):
            trial = [float(max(0.0, min(1.0, e - step * g))) for e, g in zip(etas, grads)]
            F1 = self.energy(trial)
            if F1 <= F0 - self.armijo_c * step * g2:
                return trial
            step *= self.backtrack_factor
        return [float(max(0.0, min(1.0, e - step_init * g))) for e, g in zip(etas, grads)]

    def _coordinate_backtracking(self, etas: List[OrderParameter], idx: int, grad_i: float, step_init: float) -> List[OrderParameter]:
        F0 = self.energy(etas)
        step = float(step_init)
        g2 = float(grad_i * grad_i)
        for _ in range(self.max_backtrack + 1):
            trial = list(etas)
            trial[idx] = float(max(0.0, min(1.0, trial[idx] - step * grad_i)))
            F1 = self.energy(trial)
            if F1 <= F0 - self.armijo_c * step * g2:
                return trial
            step *= self.backtrack_factor
        trial = list(etas)
        trial[idx] = float(max(0.0, min(1.0, trial[idx] - step_init * grad_i)))
        return trial

    def _emit_eta(self, etas: List[OrderParameter]) -> None:
        for cb in self.on_eta_updated:
            cb(etas)

    def _emit_energy(self, F: float) -> None:
        for cb in self.on_energy_updated:
            cb(F)

    # --- Helpers for adjacency and local/edge terms ---
    def _ensure_adjacency(self, n: int) -> None:
        if self._adjacency is not None:
            return
        adj: List[List[Tuple[int, EnergyCoupling]]] = [[] for _ in range(n)]
        for i, j, coup in self.couplings:
            adj[i].append((j, coup))
            adj[j].append((i, coup))
        self._adjacency = adj

    def _local_energy(self, idx: int, eta_i: float) -> float:
        m = self.modules[idx]
        w = float(self._combined_term_weights().get(f"local:{m.__class__.__name__}", 1.0))
        return float(w * m.local_energy(float(eta_i), self.constraints))

    def _local_grad(self, idx: int, eta_i: float) -> float:
        m = self.modules[idx]
        w = float(self._combined_term_weights().get(f"local:{m.__class__.__name__}", 1.0))
        if isinstance(m, SupportsLocalEnergyGrad):
            return float(w * m.d_local_energy_d_eta(float(eta_i), self.constraints))
        base = float(m.local_energy(eta_i, self.constraints))
        b = float(m.local_energy(min(1.0, eta_i + self.grad_eps), self.constraints))
        return float(w * ((b - base) / self.grad_eps))

    def _pair_coupling_energy(self, coup: EnergyCoupling, i: int, j: int, eta_i: float, eta_j: float) -> float:
        w = float(self._combined_term_weights().get(f"coup:{coup.__class__.__name__}", 1.0))
        return float(w * coup.coupling_energy(float(eta_i), float(eta_j), self.constraints))

    def _pair_coupling_grads(
        self,
        coup: EnergyCoupling,
        i: int,
        j: int,
        eta_i: float,
        eta_j: float,
    ) -> Tuple[float, float]:
        if isinstance(coup, SupportsCouplingGrads):
            gi, gj = coup.d_coupling_energy_d_etas(float(eta_i), float(eta_j), self.constraints)
            w = float(self._combined_term_weights().get(f"coup:{coup.__class__.__name__}", 1.0))
            return float(w * gi), float(w * gj)
        base = float(coup.coupling_energy(eta_i, eta_j, self.constraints))
        bi = float(coup.coupling_energy(min(1.0, eta_i + self.grad_eps), eta_j, self.constraints))
        bj = float(coup.coupling_energy(eta_i, min(1.0, eta_j + self.grad_eps), self.constraints))
        w = float(self._combined_term_weights().get(f"coup:{coup.__class__.__name__}", 1.0))
        gi = (bi - base) / self.grad_eps
        gj = (bj - base) / self.grad_eps
        return float(w * gi), float(w * gj)

    def _combined_term_weights(self) -> dict[str, float]:
        base_tw: dict[str, float] = {}
        tw = self.constraints.get("term_weights", None)
        if isinstance(tw, dict):
            for k, v in tw.items():
                try:
                    base_tw[str(k)] = float(v)  # type: ignore[arg-type]
                except Exception:
                    continue
        if self._term_weights:
            base_tw.update({str(k): float(v) for k, v in self._term_weights.items()})
        return base_tw

    def _term_grad_norms(self, etas: List[OrderParameter]) -> dict[str, float]:
        """Compute L2 norms of term-specific gradient contributions (weighted)."""
        norms_sq: dict[str, float] = {}
        cw = self._combined_term_weights()
        # local
        for idx, (m, eta) in enumerate(zip(self.modules, etas)):
            key = f"local:{m.__class__.__name__}"
            w = float(cw.get(key, 1.0))
            if isinstance(m, SupportsLocalEnergyGrad):
                g = w * float(m.d_local_energy_d_eta(float(eta), self.constraints))
            else:
                base = float(m.local_energy(eta, self.constraints))
                b = float(m.local_energy(eta + self.grad_eps, self.constraints))
                g = w * ((b - base) / self.grad_eps)
            norms_sq[key] = float(norms_sq.get(key, 0.0) + g * g)
        # couplings
        for i, j, coup in self.couplings:
            key = f"coup:{coup.__class__.__name__}"
            w = float(cw.get(key, 1.0))
            if isinstance(coup, SupportsCouplingGrads):
                gi, gj = coup.d_coupling_energy_d_etas(etas[i], etas[j], self.constraints)
                gi = w * float(gi)
                gj = w * float(gj)
            else:
                base = float(coup.coupling_energy(etas[i], etas[j], self.constraints))
                bi = float(coup.coupling_energy(etas[i] + self.grad_eps, etas[j], self.constraints))
                bj = float(coup.coupling_energy(etas[i], etas[j] + self.grad_eps, self.constraints))
                gi = w * ((bi - base) / self.grad_eps)
                gj = w * ((bj - base) / self.grad_eps)
            norms_sq[key] = float(norms_sq.get(key, 0.0) + gi * gi + gj * gj)
        # sqrt
        return {k: float(math.sqrt(v)) for k, v in norms_sq.items()}


