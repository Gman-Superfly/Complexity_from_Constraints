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
from .couplings import (
    QuadraticCoupling,
    DirectedHingeCoupling,
    AsymmetricHingeCoupling,
    GateBenefitCoupling,
    DampedGateBenefitCoupling,
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
    use_analytic: bool = True
    normalize_grads: bool = False
    max_grad_norm: Optional[float] = None
    line_search: bool = False
    backtrack_factor: float = 0.5
    max_backtrack: int = 5
    armijo_c: float = 1e-6
    use_vectorized_quadratic: bool = False
    use_vectorized_hinges: bool = False
    use_vectorized_gate_benefits: bool = True
    neighbor_gradients_only: bool = True
    use_coordinate_descent: bool = False
    adaptive_coordinate_descent: bool = False
    coordinate_steps: int = 100
    coordinate_active_tol: float = 1e-4
    adaptive_switch_delta: float = 1e-5
    adaptive_switch_patience: int = 5
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
    _grad_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _trial_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _local_energy_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _local_grad_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _adaptive_switches: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_configuration()
        self._ensure_adjacency(len(self.modules))

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
        if self.use_coordinate_descent:
            return self.relax_etas_coordinate(
                etas,
                steps=self.coordinate_steps,
                active_tol=self.coordinate_active_tol,
            )
        stalled_steps = 0
        energy_value = self._energy_value(etas)
        prev_energy_value: Optional[float] = energy_value
        if self.adaptive_coordinate_descent:
            etas = self.relax_etas_coordinate(
                etas,
                steps=self.coordinate_steps,
                active_tol=self.coordinate_active_tol,
            )
            energy_value = self._energy_value(etas)
            prev_energy_value = energy_value
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
            if self.adaptive_coordinate_descent and prev_energy_value is not None:
                drop = prev_energy_value - energy_value
                if drop < self.adaptive_switch_delta:
                    stalled_steps += 1
                else:
                    stalled_steps = 0
                if stalled_steps >= self.adaptive_switch_patience:
                    etas = self.relax_etas_coordinate(
                        etas,
                        steps=self.coordinate_steps,
                        active_tol=self.coordinate_active_tol,
                    )
                    self._adaptive_switches += 1
                    stalled_steps = 0
                    prev_energy_value = self._energy_value(etas)
                    continue
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
        grad_arr = self._grad_buffer_for(n)
        # Local terms (apply term weights)
        cw = self._combined_term_weights()
        for idx, (m, eta) in enumerate(zip(self.modules, etas)):
            w = float(cw.get(f"local:{m.__class__.__name__}", 1.0))
            if isinstance(m, SupportsLocalEnergyGrad):
                grad_arr[idx] += w * float(m.d_local_energy_d_eta(float(eta), self.constraints))
            else:
                base = float(m.local_energy(eta, self.constraints))
                b = float(m.local_energy(eta + self.grad_eps, self.constraints))
                grad_arr[idx] += w * ((b - base) / self.grad_eps)
        if self.use_vectorized_quadratic:
            q_grads = self._quadratic_coupling_gradients_vectorized(etas, cw)
            grad_arr += np.asarray(q_grads, dtype=float)
        if self.use_vectorized_hinges:
            hinge_grads = self._hinge_coupling_gradients_vectorized(etas, cw)
            grad_arr += np.asarray(hinge_grads, dtype=float)
        if self.use_vectorized_gate_benefits:
            grad_arr += self._gate_benefit_gradients_vectorized(etas, cw)
        for i, j, coup in self.couplings:
            if self.use_vectorized_quadratic and isinstance(coup, QuadraticCoupling):
                continue
            if self.use_vectorized_hinges and isinstance(coup, (DirectedHingeCoupling, AsymmetricHingeCoupling)):
                continue
            if self.use_vectorized_gate_benefits and isinstance(coup, (GateBenefitCoupling, DampedGateBenefitCoupling)):
                continue
            w = float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
            if isinstance(coup, SupportsCouplingGrads):
                gi, gj = coup.d_coupling_energy_d_etas(etas[i], etas[j], self.constraints)
                grad_arr[i] += w * float(gi)
                grad_arr[j] += w * float(gj)
            else:
                base = float(coup.coupling_energy(etas[i], etas[j], self.constraints))
                bi = float(coup.coupling_energy(etas[i] + self.grad_eps, etas[j], self.constraints))
                bj = float(coup.coupling_energy(etas[i], etas[j] + self.grad_eps, self.constraints))
                grad_arr[i] += w * ((bi - base) / self.grad_eps)
                grad_arr[j] += w * ((bj - base) / self.grad_eps)
        return grad_arr.tolist()

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

    def _hinge_coupling_gradients_vectorized(self, etas: List[OrderParameter], cw: dict[str, float]) -> List[float]:
        """Vectorized gradients for directed/asymmetric hinge couplings."""
        n = len(etas)
        grads = np.zeros(n, dtype=float)
        if not self.couplings:
            return grads.tolist()
        eta_arr = np.asarray(etas, dtype=float)

        # Directed hinge: w * max(0, eta_j - eta_i)^2
        dir_i = []
        dir_j = []
        dir_w = []
        for i, j, coup in self.couplings:
            if isinstance(coup, DirectedHingeCoupling):
                dir_i.append(i)
                dir_j.append(j)
                key = f"coup:{coup.__class__.__name__}"
                dir_w.append(float(getattr(coup, "weight", 0.0)) * float(cw.get(key, 1.0)))
        if dir_i:
            i_idx = np.asarray(dir_i, dtype=int)
            j_idx = np.asarray(dir_j, dtype=int)
            weights = np.asarray(dir_w, dtype=float)
            gap = eta_arr[j_idx] - eta_arr[i_idx]
            mask = gap > 0.0
            contrib = 2.0 * weights * gap * mask
            np.add.at(grads, i_idx, -contrib)
            np.add.at(grads, j_idx, contrib)

        # Asymmetric hinge: w * max(0, beta*eta_j - alpha*eta_i)^2
        asym_i = []
        asym_j = []
        asym_w = []
        alphas = []
        betas = []
        for i, j, coup in self.couplings:
            if isinstance(coup, AsymmetricHingeCoupling):
                asym_i.append(i)
                asym_j.append(j)
                alphas.append(float(coup.alpha_i))
                betas.append(float(coup.beta_j))
                key = f"coup:{coup.__class__.__name__}"
                asym_w.append(float(getattr(coup, "weight", 0.0)) * float(cw.get(key, 1.0)))
        if asym_i:
            i_idx = np.asarray(asym_i, dtype=int)
            j_idx = np.asarray(asym_j, dtype=int)
            weights = np.asarray(asym_w, dtype=float)
            alpha_arr = np.asarray(alphas, dtype=float)
            beta_arr = np.asarray(betas, dtype=float)
            gap = beta_arr * eta_arr[j_idx] - alpha_arr * eta_arr[i_idx]
            mask = gap > 0.0
            gi = -2.0 * weights * gap * alpha_arr * mask
            gj = 2.0 * weights * gap * beta_arr * mask
            np.add.at(grads, i_idx, gi)
            np.add.at(grads, j_idx, gj)
        return grads.tolist()

    def _gate_benefit_gradients_vectorized(self, etas: List[OrderParameter], cw: dict[str, float]) -> np.ndarray:
        n = len(etas)
        grads = np.zeros(n, dtype=float)
        if not self.couplings:
            return grads
        eta_arr = np.asarray(etas, dtype=float)

        # GateBenefitCoupling
        gb_idx = []
        gb_weights = []
        gb_delta = []
        for i, _j, coup in self.couplings:
            if isinstance(coup, GateBenefitCoupling):
                key = f"coup:{coup.__class__.__name__}"
                weight = float(coup.weight) * float(cw.get(key, 1.0))
                delta = float(self.constraints.get(coup.delta_key, 0.0))
                gb_idx.append(i)
                gb_weights.append(weight)
                gb_delta.append(delta)
        if gb_idx:
            idx = np.asarray(gb_idx, dtype=int)
            weights = np.asarray(gb_weights, dtype=float)
            delta = np.asarray(gb_delta, dtype=float)
            contrib = -weights * delta
            np.add.at(grads, idx, contrib)

        # DampedGateBenefitCoupling
        dg_idx = []
        dg_weights = []
        dg_damping = []
        dg_eta_power = []
        dg_scaled_delta = []
        for i, _j, coup in self.couplings:
            if isinstance(coup, DampedGateBenefitCoupling):
                key = f"coup:{coup.__class__.__name__}"
                weight = float(coup.weight) * float(cw.get(key, 1.0))
                delta = float(self.constraints.get(coup.delta_key, 0.0))
                if delta >= 0.0:
                    scaled = float(coup.positive_scale) * delta
                else:
                    scaled = float(coup.negative_scale) * delta
                dg_idx.append(i)
                dg_weights.append(weight)
                dg_damping.append(float(coup.damping))
                dg_eta_power.append(float(coup.eta_power))
                dg_scaled_delta.append(scaled)
        if dg_idx:
            idx = np.asarray(dg_idx, dtype=int)
            weights = np.asarray(dg_weights, dtype=float)
            damping = np.asarray(dg_damping, dtype=float)
            eta_power = np.asarray(dg_eta_power, dtype=float)
            scaled_delta = np.asarray(dg_scaled_delta, dtype=float)
            gate_vals = eta_arr[idx]
            grad_vals = np.zeros_like(gate_vals)
            # When scaled_delta == 0 or damping/weights 0, contrib is 0
            mask_nonzero = (scaled_delta != 0.0) & (weights != 0.0) & (damping != 0.0)
            if np.any(mask_nonzero):
                mask_one = mask_nonzero & (eta_power == 1.0)
                grad_vals[mask_one] = -weights[mask_one] * damping[mask_one] * scaled_delta[mask_one]
                mask_pow = mask_nonzero & (eta_power != 1.0) & (gate_vals > 0.0)
                if np.any(mask_pow):
                    grad_vals[mask_pow] = -weights[mask_pow] * damping[mask_pow] * scaled_delta[mask_pow] * eta_power[mask_pow] * (
                        gate_vals[mask_pow] ** (eta_power[mask_pow] - 1.0)
                    )
            np.add.at(grads, idx, grad_vals)
        return grads

    def _local_energy_grad_batch(self, etas: List[OrderParameter]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(etas)
        cw = self._combined_term_weights()
        energy_buf = self._local_energy_buffer_for(n)
        grad_buf = self._local_grad_buffer_for(n)
        for idx, (m, eta) in enumerate(zip(self.modules, etas)):
            w = float(cw.get(f"local:{m.__class__.__name__}", 1.0))
            energy = w * float(m.local_energy(float(eta), self.constraints))
            energy_buf[idx] = energy
            if isinstance(m, SupportsLocalEnergyGrad):
                grad_buf[idx] = w * float(m.d_local_energy_d_eta(float(eta), self.constraints))
            else:
                base = float(m.local_energy(eta, self.constraints))
                bumped = float(m.local_energy(min(1.0, eta + self.grad_eps), self.constraints))
                grad_buf[idx] = w * ((bumped - base) / self.grad_eps)
        return energy_buf, grad_buf

    def _grad_buffer_for(self, n: int) -> np.ndarray:
        buf = self._grad_buffer
        if buf is None or buf.shape[0] != n:
            buf = np.zeros(n, dtype=float)
            self._grad_buffer = buf
        else:
            buf.fill(0.0)
        return buf

    def _local_energy_buffer_for(self, n: int) -> np.ndarray:
        buf = self._local_energy_buffer
        if buf is None or buf.shape[0] != n:
            buf = np.zeros(n, dtype=float)
            self._local_energy_buffer = buf
        else:
            buf.fill(0.0)
        return buf

    def _local_grad_buffer_for(self, n: int) -> np.ndarray:
        buf = self._local_grad_buffer
        if buf is None or buf.shape[0] != n:
            buf = np.zeros(n, dtype=float)
            self._local_grad_buffer = buf
        else:
            buf.fill(0.0)
        return buf

    def _trial_array_for(self, etas: List[OrderParameter]) -> np.ndarray:
        n = len(etas)
        buf = self._trial_buffer
        if buf is None or buf.shape[0] != n:
            buf = np.asarray(etas, dtype=float).copy()
            self._trial_buffer = buf
        else:
            buf[:] = np.asarray(etas, dtype=float)
        return buf

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
            trial_arr = self._trial_array_for(etas)
            trial_arr -= step * gvec
            np.clip(trial_arr, 0.0, 1.0, out=trial_arr)
            trial = trial_arr.tolist()
            F1 = self._energy_value(trial)
            if F1 <= F0 - self.armijo_c * step * g2:
                return trial
            step *= self.backtrack_factor
        trial_arr = self._trial_array_for(etas)
        trial_arr -= step_init * gvec
        np.clip(trial_arr, 0.0, 1.0, out=trial_arr)
        return trial_arr.tolist()

    def _coordinate_backtracking(self, etas: List[OrderParameter], idx: int, grad_i: float, step_init: float) -> List[OrderParameter]:
        F0 = self._energy_value(etas)
        step = float(step_init)
        g2 = float(grad_i * grad_i)
        for _ in range(self.max_backtrack + 1):
            trial_arr = self._trial_array_for(etas)
            trial_arr[idx] = float(max(0.0, min(1.0, trial_arr[idx] - step * grad_i)))
            trial = trial_arr.tolist()
            F1 = self._energy_value(trial)
            if F1 <= F0 - self.armijo_c * step * g2:
                return trial
            step *= self.backtrack_factor
        trial_arr = self._trial_array_for(etas)
        trial_arr[idx] = float(max(0.0, min(1.0, trial_arr[idx] - step_init * grad_i)))
        return trial_arr.tolist()

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
        local_grads = self._local_energy_grad_batch(etas)[1]
        for idx, m in enumerate(self.modules):
            key = f"local:{m.__class__.__name__}"
            g = float(local_grads[idx])
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
        if self.adaptive_coordinate_descent:
            assert self.adaptive_switch_delta > 0.0, "adaptive_switch_delta must be > 0 when adaptive coordinate descent is enabled"
            assert self.adaptive_switch_patience >= 1, "adaptive_switch_patience must be >= 1"
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


