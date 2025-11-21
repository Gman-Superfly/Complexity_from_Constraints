"""Coordinator for total energy evaluation and optional eta relaxation.

This coordinator can:
- compute etas from inputs via modules
- compute total energy with couplings
- optionally relax etas by gradient steps on F_total (finite-difference)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Tuple, Optional, Iterable

import math
import warnings
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
    neighbor_gradients_only: bool = False
    enforce_invariants: bool = True
    # Term-weight calibration
    term_weight_floor: float = 0.0
    term_weight_ceiling: Optional[float] = None
    auto_balance_term_weights: bool = False
    term_norm_target: float = 1.0
    max_term_norm_ratio: float = 10.0
    # Optional term-weight adapter
    weight_adapter: Optional[WeightAdapter] = None

    on_eta_updated: List[EtaUpdateCallback] = field(default_factory=list)
    on_energy_updated: List[EnergyUpdateCallback] = field(default_factory=list)

    _adjacency: Optional[List[List[Tuple[int, EnergyCoupling]]]] = field(default=None, init=False, repr=False)
    _term_weights: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_configuration()

    def compute_etas(self, inputs: List[Any]) -> List[OrderParameter]:
        assert len(inputs) == len(self.modules), "inputs/modules length mismatch"
        etas: List[OrderParameter] = []
        for module, x in zip(self.modules, inputs):
            eta = float(module.compute_eta(x))
            etas.append(eta)
        self._emit_eta(etas)
        return etas

    def energy(self, etas: List[OrderParameter]) -> float:
        F = self._energy_value(etas)
        self._emit_energy(F)
        return F

    def _energy_value(self, etas: List[OrderParameter]) -> float:
        # Merge term weights (constraints.term_weights overridden by adapter-maintained _term_weights)
        merged_constraints: dict[str, Any] = dict(self.constraints)
        calibrated_weights = self._combined_term_weights()
        if calibrated_weights:
            merged_constraints["term_weights"] = calibrated_weights
        return total_energy(etas, self.modules, self.couplings, merged_constraints)

    def relax_etas(self, etas0: List[OrderParameter], steps: int = 50) -> List[OrderParameter]:
        """Finite-difference gradient steps on F_total w.r.t. etas."""
        etas = [float(e) for e in etas0]
        prev_energy_value: Optional[float] = None
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
            energy_value = self._energy_value(etas)
            if self.enforce_invariants:
                self._check_invariants(etas, energy_value)
            # Early stop on non-monotonic energy (guard against oscillations)
            if prev_energy_value is not None and energy_value > prev_energy_value + 1e-12:
                break
            # Emit only after acceptance
            self._emit_energy(energy_value)
            prev_energy_value = energy_value
            term_norms = self._term_grad_norms(etas)
            if self.auto_balance_term_weights:
                self._auto_balance_term_weights(term_norms)
            if self.weight_adapter is not None:
                updated = self.weight_adapter.step(term_norms, energy_value, dict(self._term_weights))
                self._term_weights = {
                    str(k): float(v) for k, v in updated.items() if isinstance(k, str)
                }
        return etas

    def _finite_diff_grads(self, etas: List[OrderParameter]) -> List[float]:
        base = self._energy_value(etas)
        grads: List[float] = [0.0 for _ in etas]
        indices: Iterable[int]
        if self.neighbor_gradients_only:
            self._ensure_adjacency(len(etas))
            indices = self._active_indices(etas)
        else:
            indices = range(len(etas))
        for i in indices:
            bumped = list(etas)
            bumped[i] += self.grad_eps
            Fb = self._energy_value(bumped)
            grad_i = (Fb - base) / self.grad_eps
            grads[i] = float(grad_i)
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
                grads = self._analytic_grads(etas)
            except Exception:
                grads = self._finite_diff_grads(etas)
        else:
            grads = self._finite_diff_grads(etas)
        return grads

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
        F = self._energy_value(etas)
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
            energy_value = self.energy(etas)
            if self.enforce_invariants:
                self._check_invariants(etas, energy_value)
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

    def _quadratic_energy_vectorized(self, etas: List[OrderParameter], cw: dict[str, float]) -> float:
        if not self.couplings:
            return 0.0
        idx_i = np.fromiter((i for i, _j, c in self.couplings if isinstance(c, QuadraticCoupling)), dtype=int)
        idx_j = np.fromiter((j for _i, j, c in self.couplings if isinstance(c, QuadraticCoupling)), dtype=int)
        weights = np.fromiter(
            (
                float(getattr(c, "weight", 0.0)) * float(cw.get(f"coup:{c.__class__.__name__}", 1.0))
                for _i, _j, c in self.couplings
                if isinstance(c, QuadraticCoupling)
            ),
            dtype=float,
        )
        if len(idx_i) == 0:
            return 0.0
        eta_arr = np.asarray(etas, dtype=float)
        diff = eta_arr[idx_i] - eta_arr[idx_j]
        return float(np.sum(weights * diff * diff))

    def _step_with_backtracking(self, etas: List[OrderParameter], grads: List[float], step_init: float) -> List[float]:
        F0 = self._energy_value(etas)
        step = float(step_init)
        gvec = np.asarray(grads, dtype=float)
        g2 = float(np.dot(gvec, gvec))
        for _ in range(self.max_backtrack + 1):
            trial = [float(max(0.0, min(1.0, e - step * g))) for e, g in zip(etas, grads)]
            F1 = self._energy_value(trial)
            if F1 <= F0 - self.armijo_c * step * g2:
                return trial
            step *= self.backtrack_factor
        return [float(max(0.0, min(1.0, e - step_init * g))) for e, g in zip(etas, grads)]

    def _coordinate_backtracking(self, etas: List[OrderParameter], idx: int, grad_i: float, step_init: float) -> List[OrderParameter]:
        F0 = self._energy_value(etas)
        step = float(step_init)
        g2 = float(grad_i * grad_i)
        for _ in range(self.max_backtrack + 1):
            trial = list(etas)
            trial[idx] = float(max(0.0, min(1.0, trial[idx] - step * grad_i)))
            F1 = self._energy_value(trial)
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

    def _active_indices(self, etas: List[OrderParameter]) -> Iterable[int]:
        """Return indices participating in any coupling (plus their neighbors)."""
        if self._adjacency is None:
            return range(len(etas))
        active: set[int] = set()
        for idx, neighbors in enumerate(self._adjacency):
            if neighbors:
                active.add(idx)
            for j, _coup in neighbors:
                if 0 <= j < len(etas):
                    active.add(j)
        if not active:
            return range(len(etas))
        return tuple(sorted(active))

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
        floor = float(self.term_weight_floor)
        ceiling = None if self.term_weight_ceiling is None else float(self.term_weight_ceiling)
        if floor < 0.0:
            raise ValueError("term_weight_floor must be >= 0")
        if ceiling is not None and ceiling < floor:
            raise ValueError("term_weight_ceiling must be >= floor")
        calibrated: dict[str, float] = {}
        for key, value in base_tw.items():
            v = float(value)
            if floor:
                v = max(v, floor)
            if ceiling is not None:
                v = min(v, ceiling)
            calibrated[key] = v
        return calibrated

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

    def _auto_balance_term_weights(self, term_norms: Mapping[str, float]) -> None:
        if not term_norms:
            return
        target = max(float(self.term_norm_target), 1e-9)
        ratio_cap = max(float(self.max_term_norm_ratio), 1.0)
        for key, norm in term_norms.items():
            norm = float(norm)
            if not math.isfinite(norm) or norm <= 0.0:
                continue
            ratio = norm / target
            if ratio <= ratio_cap:
                continue
            current = float(self._term_weights.get(key, 1.0))
            scale = target / norm
            new_weight = current * scale
            self._term_weights[key] = new_weight
            warnings.warn(
                f"Term '{key}' gradient norm {norm:.3f} exceeded target {target:.3f}; "
                f"auto-balancing weight from {current:.3f} to {new_weight:.3f}.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _validate_configuration(self) -> None:
        assert isinstance(self.modules, list) and len(self.modules) > 0, "at least one module required"
        assert self.grad_eps > 0.0, "grad_eps must be > 0"
        assert self.step_size > 0.0, "step_size must be > 0"
        assert 0.0 < self.armijo_c < 1.0, "armijo_c must be between 0 and 1"
        assert 0.0 < self.backtrack_factor < 1.0, "backtrack_factor must be in (0,1)"
        assert self.max_backtrack >= 0, "max_backtrack must be non-negative"
        if self.term_weight_ceiling is not None:
            assert self.term_weight_ceiling >= self.term_weight_floor >= 0.0
        for i, j, _ in self.couplings:
            assert 0 <= i < len(self.modules), "coupling index out of range"
            assert 0 <= j < len(self.modules), "coupling index out of range"

    def _check_invariants(self, etas: List[OrderParameter], energy_value: Optional[float] = None) -> None:
        tol = 1e-9
        for eta in etas:
            assert math.isfinite(eta), "η must be finite"
            assert -tol <= eta <= 1.0 + tol, "η out of bounds"
        if energy_value is not None:
            assert math.isfinite(energy_value), "Energy must be finite"


