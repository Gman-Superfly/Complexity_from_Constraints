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
from .energy import total_energy, project_noise_orthogonal

EtaUpdateCallback = Callable[[List[OrderParameter]], None]
EnergyUpdateCallback = Callable[[float], None]


def _is_gate_module(module: EnergyModule) -> bool:
    return hasattr(module, "cost") and module.__class__.__name__ == "EnergyGatingModule"


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
    # Operator-splitting / proximal mode
    operator_splitting: bool = False
    prox_tau: float = 0.05
    prox_steps: int = 50
    # ADMM (experimental, quadratic couplings focus)
    use_admm: bool = False
    admm_rho: float = 1.0
    admm_steps: int = 50
    admm_step_size: float = 0.05
    # Homotopy / continuation
    homotopy_coupling_scale_start: Optional[float] = None  # scale applied to all coupling term weights
    homotopy_term_scale_starts: Optional[Mapping[str, float]] = None  # individual term keys -> start scale
    homotopy_gate_cost_scale_start: Optional[float] = None
    homotopy_steps: int = 0
    # Term-weight calibration
    term_weight_floor: float = 0.0
    term_weight_ceiling: Optional[float] = None
    auto_balance_term_weights: bool = False
    term_norm_target: float = 1.0
    max_term_norm_ratio: float = 10.0
    # Optional term-weight adapter
    weight_adapter: Optional[WeightAdapter] = None
    # Stability / small-gain guard (optional)
    stability_guard: bool = False
    stability_cap_fraction: float = 0.9  # cap step to this fraction of 2/L estimate
    log_contraction_margin: bool = False
    stability_coupling_auto_cap: bool = False
    stability_coupling_target: Optional[float] = None  # desired max Lipschitz (if None, auto)
    # Lipschitz/allocator details (instrumentation for adapters/telemetry)
    expose_lipschitz_details: bool = False
    # Noise / Exploration controls
    enable_orthogonal_noise: bool = True  # Inject noise orthogonal to gradient (structure-preserving)
    # Note: default magnitude is 0.0 to preserve determinism unless explicitly enabled in experiments.
    noise_magnitude: float = 0.0
    noise_schedule_decay: float = 0.99  # Simple exponential decay for noise magnitude

    on_eta_updated: List[EtaUpdateCallback] = field(default_factory=list)
    on_energy_updated: List[EnergyUpdateCallback] = field(default_factory=list)

    _adjacency: Optional[List[List[Tuple[int, EnergyCoupling]]]] = field(default=None, init=False, repr=False)
    _term_weights: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _grad_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _trial_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _local_energy_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _local_grad_buffer: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _adaptive_switches: int = field(default=0, init=False, repr=False)
    _total_backtracks: int = field(default=0, init=False, repr=False)
    _last_step_backtracks: int = field(default=0, init=False, repr=False)
    _last_contraction_margin: Optional[float] = field(default=None, init=False, repr=False)
    _stability_coupling_scale: Optional[float] = field(default=None, init=False, repr=False)
    _homotopy_term_scales: Optional[dict[str, float]] = field(default=None, init=False, repr=False)
    _homotopy_gate_bases: Optional[List[float]] = field(default=None, init=False, repr=False)
    _last_lipschitz_details: Optional[dict] = field(default=None, init=False, repr=False)

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
        # Optional operator-splitting path
        if self.operator_splitting:
            return self.relax_etas_proximal(etas0, steps=self.prox_steps, tau=self.prox_tau)
        if self.use_admm:
            return self.relax_etas_admm(etas0, steps=self.admm_steps, rho=self.admm_rho, step_size=self.admm_step_size)
        etas = [float(e) for e in etas0]
        if self.use_coordinate_descent:
            return self.relax_etas_coordinate(
                etas,
                steps=self.coordinate_steps,
                active_tol=self.coordinate_active_tol,
            )
        self._homotopy_scale = None
        self._stability_coupling_scale = None
        self._homotopy_term_scales = None
        self._homotopy_gate_bases = None
        homotopy_active = (
            self.homotopy_coupling_scale_start is not None
            and self.homotopy_coupling_scale_start >= 0.0
            and self.homotopy_steps > 0
        )
        gate_modules = [m for m in self.modules if _is_gate_module(m)]
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
        for iter_idx in range(steps):
            L_est = None
            if homotopy_active:
                t = min(1.0, iter_idx / float(self.homotopy_steps))
                start = float(self.homotopy_coupling_scale_start)
                scale = start + (1.0 - start) * t
                self._homotopy_scale = max(0.0, scale)
            if self.homotopy_term_scale_starts and self.homotopy_steps > 0:
                t = min(1.0, iter_idx / float(self.homotopy_steps))
                term_scales = {}
                for key, start in self.homotopy_term_scale_starts.items():
                    start = float(start)
                    term_scales[str(key)] = max(0.0, start + (1.0 - start) * t)
                self._homotopy_term_scales = term_scales
            if self.homotopy_gate_cost_scale_start is not None and self.homotopy_steps > 0 and gate_modules:
                if self._homotopy_gate_bases is None:
                    self._homotopy_gate_bases = [float(m.cost) for m in gate_modules]
                t = min(1.0, iter_idx / float(self.homotopy_steps))
                start = float(self.homotopy_gate_cost_scale_start)
                scale = max(0.0, start + (1.0 - start) * t)
                for m, base in zip(gate_modules, self._homotopy_gate_bases):
                    m.cost = float(base * scale)
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
            # Stability guard: cap step size if enabled
            step_to_use = self.step_size
            need_L = self.stability_guard or self.stability_coupling_auto_cap
            if need_L:
                L_est = self._estimate_lipschitz_bound(etas)
            if self.stability_guard and L_est and L_est > 0.0 and math.isfinite(L_est):
                cap = self.stability_cap_fraction * (2.0 / L_est)
                if cap > 0.0:
                    step_to_use = min(step_to_use, cap)
                    if self.log_contraction_margin:
                        self._last_contraction_margin = (2.0 / L_est) - step_to_use
            elif self.stability_guard and self.log_contraction_margin:
                self._last_contraction_margin = None
            
            # Inject orthogonal noise if enabled (structure-preserving exploration)
            grad_vector = np.array(grads, dtype=float)
            noise_vector = np.zeros_like(grad_vector)
            if self.enable_orthogonal_noise and self.noise_magnitude > 1e-9:
                # Decay magnitude
                current_noise_mag = self.noise_magnitude * (self.noise_schedule_decay ** iter_idx)
                if current_noise_mag > 1e-9:
                    # Generate raw noise
                    raw_noise = np.random.normal(0, 1, size=grad_vector.shape)
                    # Project onto null space of gradient (level set exploration)
                    noise_vector = project_noise_orthogonal(raw_noise, grad_vector)
                    # Scale
                    noise_norm = np.linalg.norm(noise_vector)
                    if noise_norm > 1e-9:
                        noise_vector = noise_vector * (current_noise_mag / noise_norm)
            
            # step
            if self.line_search:
                # Line search applies to the gradient direction
                # Noise is added *after* the descent step decision (Langevin-style) or *to* the direction?
                # Standard practice: add noise to the update. 
                # But line search needs a direction. Let's define direction d = -grad + noise
                # Then line search along d.
                direction = -grad_vector
                # If noise enabled, perturb the search direction
                if self.enable_orthogonal_noise:
                     # Orthogonal noise doesn't affect dF/dalpha at alpha=0 (by definition dF/deta * noise = 0)
                     # So it preserves the descent property locally!
                     direction = direction + noise_vector
                
                # _step_with_backtracking expects grads, but we want to move in `direction`.
                # The current helper assumes direction = -grads. We need to patch it or just apply update manually if noise is on.
                # For safety/simplicity in this "nugget" phase: if noise is on, skip line search or force step.
                # BETTER: Just add noise to the etas *after* the gradient step? 
                # No, Langevin adds it to the update.
                # Orthogonal noise is safe because grad dot noise = 0.
                # So F(eta - eps*grad + sigma*noise) approx F(eta) - eps*|grad|^2 + 0 (first order)
                # Descent is preserved.
                
                # Let's modify the update logic below to support explicit direction.
                # For now, to keep diff small: if noise is active, we'll bypass the standard line search 
                # or just add the noise to the final update if using fixed step.
                # If line search is on, we really should search along -g + noise.
                pass # Logic continues below

            # Coupling auto-cap
            coupling_scale = None
            if self.stability_coupling_auto_cap and L_est and L_est > 0.0 and math.isfinite(L_est):
                target = self.stability_coupling_target
                if target is None:
                    target = L_est
                if target > 0.0 and L_est > 0.0:
                    coupling_scale = min(1.0, max(0.0, target / L_est))
                    if coupling_scale >= 0.999:
                        coupling_scale = None
            self._stability_coupling_scale = coupling_scale
            # Optional: prepare Lipschitz details for allocator/telemetry
            self._last_lipschitz_details = None
            need_details = (
                self.expose_lipschitz_details
                or (self.weight_adapter is not None and any(
                    hasattr(self.weight_adapter, attr) for attr in ("edge_costs", "row_margins", "global_margin")
                ))
            )
            if need_details:
                target_L = self.stability_coupling_target if (self.stability_coupling_target and self.stability_coupling_target > 0.0) else L_est
                self._last_lipschitz_details = self._estimate_lipschitz_details(
                    etas, smoothing_epsilon=max(self.grad_eps * 0.5, 1e-6), target_L=target_L
                )
            # step
            if self.line_search:
                # Standard Armijo along -grad direction
                # NOTE: If orthogonal noise is enabled, we currently apply it *after* or ignore it in line search.
                # For strict correctness, line search should support custom directions.
                # MVP: If noise is ON, we skip line search or apply noise separately?
                # Decision: Apply line search to the gradient part, then add noise.
                # This is "Split Langevin": deterministic descent + stochastic diffusion.
                etas = self._step_with_backtracking(etas, grads, step_to_use)
                if self.enable_orthogonal_noise and np.any(noise_vector):
                    # Add noise (orthogonal to gradient, so doesn't fight the descent step to first order)
                    for i in range(len(etas)):
                        etas[i] = float(max(0.0, min(1.0, etas[i] + noise_vector[i])))
            else:
                for i in range(len(etas)):
                    # Update: eta - step*grad + noise
                    # noise_vector is already scaled by noise_magnitude
                    update = -step_to_use * grads[i]
                    if self.enable_orthogonal_noise:
                        update += noise_vector[i]
                    etas[i] = float(max(0.0, min(1.0, etas[i] + update)))
            
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
                # If adapter supports allocator fields, inject details snapshot
                if self._last_lipschitz_details is not None:
                    if hasattr(self.weight_adapter, "edge_costs"):
                        edge_costs = self._last_lipschitz_details.get("family_costs", {})
                        try:
                            # type: ignore[attr-defined]
                            self.weight_adapter.edge_costs = {str(k): float(v) for k, v in edge_costs.items()}
                        except Exception:
                            pass
                    if hasattr(self.weight_adapter, "row_margins"):
                        row_margins = self._last_lipschitz_details.get("row_margins", {})
                        try:
                            # type: ignore[attr-defined]
                            self.weight_adapter.row_margins = {int(k): float(v) for k, v in row_margins.items()}
                        except Exception:
                            pass
                    if hasattr(self.weight_adapter, "global_margin"):
                        gm = float(self._last_lipschitz_details.get("global_margin", 0.0))
                        try:
                            # type: ignore[attr-defined]
                            self.weight_adapter.global_margin = gm
                        except Exception:
                            pass
                updated = self.weight_adapter.step(term_norms, energy_value, dict(self._term_weights))
                self._term_weights = {
                    str(k): float(v) for k, v in updated.items() if isinstance(k, str)
                }
        self._homotopy_scale = None
        self._homotopy_term_scales = None
        if self._homotopy_gate_bases is not None:
            for m, base in zip(gate_modules, self._homotopy_gate_bases):
                m.cost = float(base)
        self._homotopy_gate_bases = None
        self._stability_coupling_scale = None
        return etas

    # --- Proximal / operator-splitting mode ---
    def relax_etas_proximal(self, etas0: List[OrderParameter], steps: int = 50, tau: float = 0.05) -> List[OrderParameter]:
        """Block-coordinate proximal updates on locals + incident couplings.

        Strategy (conservative):
          - For each iteration:
            1) Local proximal gradient for every module (project to [0,1]).
            2) Pairwise prox on couplings (closed-form for quadratic/hinge family; projected if needed).
          - Emit accepted energies only; stop if energy increases.
        """
        assert tau > 0.0, "prox tau must be positive"
        etas = [float(e) for e in etas0]
        prev_energy = self._energy_value(etas)
        self._emit_eta(etas)
        self._emit_energy(prev_energy)
        for _ in range(steps):
            # 1) Local proximal gradient step
            for idx, m in enumerate(self.modules):
                # gradient of local (analytic if available)
                if isinstance(m, SupportsLocalEnergyGrad):
                    g = float(m.d_local_energy_d_eta(etas[idx], self.constraints))
                else:
                    base = float(m.local_energy(etas[idx], self.constraints))
                    bumped = float(m.local_energy(min(1.0, etas[idx] + self.grad_eps), self.constraints))
                    g = (bumped - base) / self.grad_eps
                # apply term weight
                w = float(self._combined_term_weights().get(f"local:{m.__class__.__name__}", 1.0))
                # proximal gradient with projection to [0,1]
                etas[idx] = float(max(0.0, min(1.0, etas[idx] - tau * w * g)))
            # 2) Pairwise coupling prox/updates
            for i, j, coup in self.couplings:
                key = f"coup:{coup.__class__.__name__}"
                w_c = float(self._combined_term_weights().get(key, 1.0))
                if isinstance(coup, QuadraticCoupling):
                    etas[i], etas[j] = self._prox_quadratic_pair(etas[i], etas[j], coup.weight * w_c, tau)
                elif isinstance(coup, DirectedHingeCoupling):
                    etas[i], etas[j] = self._prox_asym_hinge_pair(
                        etas[i], etas[j], weight=coup.weight * w_c, alpha=1.0, beta=1.0, tau=tau
                    )
                elif isinstance(coup, AsymmetricHingeCoupling):
                    etas[i], etas[j] = self._prox_asym_hinge_pair(
                        etas[i], etas[j], weight=coup.weight * w_c, alpha=coup.alpha_i, beta=coup.beta_j, tau=tau
                    )
                elif isinstance(coup, (GateBenefitCoupling, DampedGateBenefitCoupling)):
                    # Linearized projected step on gate variable (i) only
                    gi, _ = coup.d_coupling_energy_d_etas(etas[i], etas[j], self.constraints)
                    etas[i] = float(max(0.0, min(1.0, etas[i] - tau * w_c * float(gi))))
                # else: no-op for unknown types
            # Check invariants and acceptance
            if self.enforce_invariants:
                self._check_invariants(etas)
            F = self._energy_value(etas)
            if F > prev_energy + 1e-12:
                break
            self._emit_eta(etas)
            self._emit_energy(F)
            prev_energy = F
        return etas

    def _prox_quadratic_pair(self, x0: float, y0: float, weight: float, tau: float) -> tuple[float, float]:
        """Closed-form prox for w*(x-y)^2 + (1/(2τ))||[x;y]-[x0;y0]||^2 with box projection."""
        a = 2.0 * weight + (1.0 / tau)
        b = -2.0 * weight
        c = -2.0 * weight
        d = 2.0 * weight + (1.0 / tau)
        # Solve linear system [[a,b],[c,d]][x;y] = [x0/tau; y0/tau]
        det = a * d - b * c
        if det == 0.0:
            return float(max(0.0, min(1.0, x0))), float(max(0.0, min(1.0, y0)))
        inv_a = d / det
        inv_b = -b / det
        inv_c = -c / det
        inv_d = a / det
        rhs_x = x0 / tau
        rhs_y = y0 / tau
        x = inv_a * rhs_x + inv_b * rhs_y
        y = inv_c * rhs_x + inv_d * rhs_y
        return float(max(0.0, min(1.0, x))), float(max(0.0, min(1.0, y)))

    def _prox_asym_hinge_pair(self, x0: float, y0: float, weight: float, alpha: float, beta: float, tau: float) -> tuple[float, float]:
        """Closed-form prox for w*max(0, β y - α x)^2 + (1/(2τ))||[x;y]-[x0;y0]||^2."""
        assert alpha >= 0.0 and beta >= 0.0
        # Check gap at current point
        gap0 = beta * y0 - alpha * x0
        if gap0 <= 0.0 or weight == 0.0:
            # No penalty region => identity prox (then project to box)
            return float(max(0.0, min(1.0, x0))), float(max(0.0, min(1.0, y0)))
        # Use λ_eff = 2w per gradient derivation
        lam = 2.0 * weight
        denom = 1.0 + tau * lam * (beta * beta + alpha * alpha)
        g_star = (beta * y0 - alpha * x0) / denom
        x = x0 + tau * lam * alpha * g_star
        y = y0 - tau * lam * beta * g_star
        return float(max(0.0, min(1.0, x))), float(max(0.0, min(1.0, y)))

    # --- ADMM (experimental) for quadratic couplings ---
    def relax_etas_admm(
        self,
        etas0: List[OrderParameter],
        steps: int = 50,
        rho: float = 1.0,
        step_size: float = 0.05,
    ) -> List[OrderParameter]:
        """ADMM-like splitting primarily for quadratic couplings.

        - Introduces auxiliary variables s_k for each quadratic edge k = (i,j) to represent (η_i - η_j).
        - Alternates:
            s-update:  minimize w*s^2 + (ρ/2)*(s - (d_ij - u))^2   (closed form)
            η-update:  gradient step on locals + augmented term (ρ/2)*||s - (η_i-η_j) + u||^2
            u-update:  u ← u + s - (η_i - η_j)
        - Hinge-family couplings use s ≥ 0 on gaps g = β η_j − α η_i with the same augmented residual r = s − g + u.
        - Other couplings (e.g., gate-benefit) contribute via their gradients during η-update.
        """
        assert rho > 0.0 and step_size > 0.0
        n = len(etas0)
        etas = [float(e) for e in etas0]
        # Track quadratic couplings' indices for s/u
        quad_edges: list[tuple[int, int, float, int]] = []
        # Track hinge-family couplings (directed/asymmetric) for s/u with nonnegativity
        hinge_edges: list[tuple[int, int, float, float, float, int]] = []  # (i,j,weight,alpha,beta,edge_idx)
        cw = self._combined_term_weights()
        for idx, (i, j, coup) in enumerate(self.couplings):
            if isinstance(coup, QuadraticCoupling):
                w = float(coup.weight) * float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
                quad_edges.append((i, j, w, idx))
            elif isinstance(coup, DirectedHingeCoupling):
                w = float(coup.weight) * float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
                hinge_edges.append((i, j, w, 1.0, 1.0, idx))
            elif isinstance(coup, AsymmetricHingeCoupling):
                w = float(coup.weight) * float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
                hinge_edges.append((i, j, w, float(coup.alpha_i), float(coup.beta_j), idx))
        m = len(quad_edges)
        s = [0.0] * m
        u = [0.0] * m
        mh = len(hinge_edges)
        sh = [0.0] * mh
        uh = [0.0] * mh
        prev_energy = self._energy_value(etas)
        self._emit_eta(etas)
        self._emit_energy(prev_energy)
        for _ in range(steps):
            # s-update (closed form): s = ρ*(d - u) / (ρ + 2w)
            for k, (i, j, w, _edge_idx) in enumerate(quad_edges):
                d_ij = float(etas[i] - etas[j])
                denom = rho + 2.0 * w
                s[k] = (rho * (d_ij - u[k])) / (denom if denom > 0.0 else rho)
            # hinge s-update with nonnegativity: sh = max(0, ρ*(gap - uh)/(ρ+2w))
            for k, (i, j, w, alpha, beta, _edge_idx) in enumerate(hinge_edges):
                gap = beta * float(etas[j]) - alpha * float(etas[i])
                denom = rho + 2.0 * w
                sh_k = (rho * (gap - uh[k])) / (denom if denom > 0.0 else rho)
                sh[k] = float(max(0.0, sh_k))
            # η-update: gradient step using locals + augmented quadratic terms (and non-quadratic couplings)
            grads = [0.0] * n
            # local grads
            for idx_m, (m_module, eta_val) in enumerate(zip(self.modules, etas)):
                w_loc = float(cw.get(f"local:{m_module.__class__.__name__}", 1.0))
                if isinstance(m_module, SupportsLocalEnergyGrad):
                    grads[idx_m] += w_loc * float(m_module.d_local_energy_d_eta(float(eta_val), self.constraints))
                else:
                    base = float(m_module.local_energy(float(eta_val), self.constraints))
                    bumped = float(m_module.local_energy(min(1.0, float(eta_val) + self.grad_eps), self.constraints))
                    grads[idx_m] += w_loc * ((bumped - base) / self.grad_eps)
            # augmented terms from quadratic edges
            for k, (i, j, _w, _edge_idx) in enumerate(quad_edges):
                r = s[k] - (float(etas[i]) - float(etas[j])) + u[k]
                grads[i] += -rho * r
                grads[j] += rho * r
            # augmented terms from hinge edges
            for k, (i, j, _w, alpha, beta, _edge_idx) in enumerate(hinge_edges):
                r = sh[k] - (beta * float(etas[j]) - alpha * float(etas[i])) + uh[k]
                # ∂r/∂η_i = +alpha, ∂r/∂η_j = -beta
                grads[i] += rho * r * alpha
                grads[j] += -rho * r * beta
            # non-quadratic couplings via gradients
            for i, j, coup in self.couplings:
                if isinstance(coup, (QuadraticCoupling, DirectedHingeCoupling, AsymmetricHingeCoupling)):
                    continue
                w_c = float(cw.get(f"coup:{coup.__class__.__name__}", 1.0))
                if isinstance(coup, SupportsCouplingGrads):
                    gi, gj = coup.d_coupling_energy_d_etas(etas[i], etas[j], self.constraints)
                    grads[i] += w_c * float(gi)
                    grads[j] += w_c * float(gj)
                else:
                    base = float(coup.coupling_energy(etas[i], etas[j], self.constraints))
                    bi = float(coup.coupling_energy(min(1.0, etas[i] + self.grad_eps), etas[j], self.constraints))
                    bj = float(coup.coupling_energy(etas[i], min(1.0, etas[j] + self.grad_eps), self.constraints))
                    grads[i] += w_c * ((bi - base) / self.grad_eps)
                    grads[j] += w_c * ((bj - base) / self.grad_eps)
            # gradient step + projection
            for i in range(n):
                etas[i] = float(max(0.0, min(1.0, float(etas[i]) - step_size * float(grads[i]))))
            # u-update
            for k, (i, j, _w, _edge_idx) in enumerate(quad_edges):
                u[k] += s[k] - (float(etas[i]) - float(etas[j]))
            for k, (i, j, _w, alpha, beta, _edge_idx) in enumerate(hinge_edges):
                uh[k] += sh[k] - (beta * float(etas[j]) - alpha * float(etas[i]))
            # acceptance guard
            if self.enforce_invariants:
                self._check_invariants(etas)
            F = self._energy_value(etas)
            if F > prev_energy + 1e-12:
                break
            self._emit_eta(etas)
            self._emit_energy(F)
            prev_energy = F
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
        local_bk = 0
        for _ in range(self.max_backtrack + 1):
            trial_arr = self._trial_array_for(etas)
            trial_arr -= step * gvec
            np.clip(trial_arr, 0.0, 1.0, out=trial_arr)
            trial = trial_arr.tolist()
            F1 = self._energy_value(trial)
            if F1 <= F0 - self.armijo_c * step * g2:
                self._last_step_backtracks = local_bk
                self._total_backtracks += local_bk
                return trial
            step *= self.backtrack_factor
            local_bk += 1
        trial_arr = self._trial_array_for(etas)
        trial_arr -= step_init * gvec
        np.clip(trial_arr, 0.0, 1.0, out=trial_arr)
        self._last_step_backtracks = local_bk
        self._total_backtracks += local_bk
        return trial_arr.tolist()

    def _coordinate_backtracking(self, etas: List[OrderParameter], idx: int, grad_i: float, step_init: float) -> List[OrderParameter]:
        F0 = self._energy_value(etas)
        step = float(step_init)
        g2 = float(grad_i * grad_i)
        local_bk = 0
        for _ in range(self.max_backtrack + 1):
            trial_arr = self._trial_array_for(etas)
            trial_arr[idx] = float(max(0.0, min(1.0, trial_arr[idx] - step * grad_i)))
            trial = trial_arr.tolist()
            F1 = self._energy_value(trial)
            if F1 <= F0 - self.armijo_c * step * g2:
                self._last_step_backtracks = local_bk
                self._total_backtracks += local_bk
                return trial
            step *= self.backtrack_factor
            local_bk += 1
        trial_arr = self._trial_array_for(etas)
        trial_arr[idx] = float(max(0.0, min(1.0, trial_arr[idx] - step_init * grad_i)))
        self._last_step_backtracks = local_bk
        self._total_backtracks += local_bk
        return trial_arr.tolist()

    def _estimate_lipschitz_bound(self, etas: List[OrderParameter]) -> float:
        """Conservative Gershgorin-style bound on gradient Lipschitz constant.

        Approximates diagonal (local curvature) via finite differences of local gradient
        and adds coupling curvature contributions for quadratic/hinge families.
        """
        n = len(etas)
        if n == 0:
            return 0.0
        diag = np.zeros(n, dtype=float)
        offsum = np.zeros(n, dtype=float)
        eps = max(self.grad_eps * 0.5, 1e-6)
        # Local curvature (finite-diff on local gradient)
        for i in range(n):
            eta_i = float(etas[i])
            g_m = self._local_grad(i, max(0.0, min(1.0, eta_i - eps)))
            g_p = self._local_grad(i, max(0.0, min(1.0, eta_i + eps)))
            curv = (g_p - g_m) / (2.0 * eps)
            if math.isfinite(curv) and curv > 0.0:
                diag[i] += float(curv)
        # Coupling curvature
        for i, j, coup in self.couplings:
            key = f"coup:{coup.__class__.__name__}"
            w_eff = float(self._combined_term_weights().get(key, 1.0))
            if isinstance(coup, QuadraticCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                diag[i] += 2.0 * w
                diag[j] += 2.0 * w
                offsum[i] += 2.0 * w
                offsum[j] += 2.0 * w
            elif isinstance(coup, DirectedHingeCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                gap = float(etas[j]) - float(etas[i])
                if gap > 0.0:
                    diag[i] += 2.0 * w
                    diag[j] += 2.0 * w
                    offsum[i] += 2.0 * w
                    offsum[j] += 2.0 * w
            elif isinstance(coup, AsymmetricHingeCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                alpha = float(getattr(coup, "alpha_i", 1.0))
                beta = float(getattr(coup, "beta_j", 1.0))
                gap = beta * float(etas[j]) - alpha * float(etas[i])
                if gap > 0.0:
                    diag[i] += 2.0 * w * (alpha * alpha)
                    diag[j] += 2.0 * w * (beta * beta)
                    offsum[i] += 2.0 * w * abs(alpha * beta)
                    offsum[j] += 2.0 * w * abs(alpha * beta)
            else:
                # GateBenefit are linear (no curvature); ignore others
                continue
        L_est = float(np.max(diag + offsum))
        if not math.isfinite(L_est) or L_est <= 0.0:
            return 0.0
        return L_est

    def _estimate_lipschitz_details(
        self,
        etas: List[OrderParameter],
        smoothing_epsilon: float = 1e-3,
        target_L: Optional[float] = None,
    ) -> dict:
        """Return detailed Gershgorin-like bound components and family costs.

        Produces:
          - L_est: current Lipschitz estimate (float)
          - row_sums: dict[row_index -> row_sum]
          - row_targets: dict[row_index -> target_row_sum] (proportional scaling if target_L < L_est)
          - row_margins: dict[row_index -> max(0, target - current)]
          - global_margin: max(0, target_L - L_est)
          - family_costs: dict['coup:ClassName' -> ΔL per unit relative scaling (max over rows)]

        Notes:
          - Hinge contributions near activation are smoothed with a simple linear ramp in [-ε, 0].
          - Family cost aggregates the maximum row contribution attributable to a family; this
            approximates impact on the max row sum that defines L_est.
        """
        n = len(etas)
        if n == 0:
            return {
                "L_est": 0.0,
                "row_sums": {},
                "row_targets": {},
                "row_margins": {},
                "global_margin": 0.0,
                "family_costs": {},
            }
        diag = np.zeros(n, dtype=float)
        offsum = np.zeros(n, dtype=float)
        # Per-row, per-family contributions to row sum
        per_row_family = {}  # row -> {family_key: contrib}
        for r in range(n):
            per_row_family[r] = {}

        def _add_row_family(row: int, fam: str, amount: float) -> None:
            if amount == 0.0:
                return
            d = per_row_family[row]
            d[fam] = float(d.get(fam, 0.0) + amount)

        for i, j, coup in self.couplings:
            key = f"coup:{coup.__class__.__name__}"
            w_eff = float(self._combined_term_weights().get(key, 1.0))
            if isinstance(coup, QuadraticCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                add = 2.0 * w
                if add != 0.0:
                    diag[i] += add
                    diag[j] += add
                    offsum[i] += add
                    offsum[j] += add
                    _add_row_family(i, key, add + add)  # diag+offsum contribution to row i
                    _add_row_family(j, key, add + add)  # row j
            elif isinstance(coup, DirectedHingeCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                gap = float(etas[j]) - float(etas[i])
                # Smoothed activity in [-ε, 0] → [0,1]
                if gap > 0.0:
                    s = 1.0
                elif -smoothing_epsilon < gap <= 0.0:
                    s = (gap + smoothing_epsilon) / smoothing_epsilon
                else:
                    s = 0.0
                if s > 0.0 and w != 0.0:
                    add = 2.0 * w * s
                    diag[i] += add
                    diag[j] += add
                    offsum[i] += add
                    offsum[j] += add
                    _add_row_family(i, key, add + add)
                    _add_row_family(j, key, add + add)
            elif isinstance(coup, AsymmetricHingeCoupling):
                w = float(getattr(coup, "weight", 0.0)) * w_eff
                alpha = float(getattr(coup, "alpha_i", 1.0))
                beta = float(getattr(coup, "beta_j", 1.0))
                gap = beta * float(etas[j]) - alpha * float(etas[i])
                if gap > 0.0:
                    s = 1.0
                elif -smoothing_epsilon < gap <= 0.0:
                    s = (gap + smoothing_epsilon) / smoothing_epsilon
                else:
                    s = 0.0
                if s > 0.0 and w != 0.0:
                    add_i = 2.0 * w * (alpha * alpha) * s
                    add_j = 2.0 * w * (beta * beta) * s
                    add_off = 2.0 * w * abs(alpha * beta) * s
                    diag[i] += add_i
                    diag[j] += add_j
                    offsum[i] += add_off
                    offsum[j] += add_off
                    _add_row_family(i, key, add_i + add_off)
                    _add_row_family(j, key, add_j + add_off)
            else:
                # GateBenefit (linear) and others: no curvature
                continue

        row_sums = (diag + offsum)
        L_est = float(np.max(row_sums)) if row_sums.size > 0 else 0.0
        # Targets and margins
        if not target_L or not math.isfinite(target_L) or target_L <= 0.0:
            target_L = L_est
        row_targets = {}
        row_margins = {}
        if L_est > 0.0 and target_L < L_est:
            scale = target_L / L_est
            for r in range(n):
                target_row = float(row_sums[r] * scale)
                row_targets[r] = target_row
                row_margins[r] = max(0.0, target_row - float(row_sums[r]))
        else:
            for r in range(n):
                row_targets[r] = float(row_sums[r])
                row_margins[r] = 0.0
        global_margin = max(0.0, float(target_L - L_est))

        # Family costs: max row contribution for each family
        family_costs: dict[str, float] = {}
        for r, fam_map in per_row_family.items():
            for fam, amount in fam_map.items():
                current = family_costs.get(fam, 0.0)
                if amount > current:
                    family_costs[fam] = float(amount)

        # Build compact row_sums dict
        row_sums_dict = {i: float(row_sums[i]) for i in range(n)}

        return {
            "L_est": L_est,
            "row_sums": row_sums_dict,
            "row_targets": row_targets,
            "row_margins": row_margins,
            "global_margin": global_margin,
            "family_costs": family_costs,
        }

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
        homotopy_scale = getattr(self, "_homotopy_scale", None)
        term_scales = getattr(self, "_homotopy_term_scales", None)
        coupling_scale = getattr(self, "_stability_coupling_scale", None)
        floor = float(self.term_weight_floor)
        ceiling = None if self.term_weight_ceiling is None else float(self.term_weight_ceiling)
        if floor < 0.0:
            raise ValueError("term_weight_floor must be >= 0")
        if ceiling is not None and ceiling < floor:
            raise ValueError("term_weight_ceiling must be >= floor")
        calibrated: dict[str, float] = {}
        for key, value in base_tw.items():
            v = float(value)
            if homotopy_scale is not None and key.startswith("coup:"):
                v *= homotopy_scale
            if term_scales and key in term_scales:
                v *= float(term_scales[key])
            if coupling_scale is not None and key.startswith("coup:"):
                v *= coupling_scale
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


