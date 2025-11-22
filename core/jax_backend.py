"""Optional JAX-based relaxation path for Landau-style modules + supported couplings.

Usage:
    runner = JaxEnergyRunner(modules, couplings, constraints, device="cpu")
    etas = runner.relax(inputs=[...], steps=100, lr=0.05)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

from modules.gating.energy_gating import EnergyGatingModule
from modules.sequence.monotonic_eta import SequenceConsistencyModule
from modules.connectivity.nl_threshold_shift import ConnectivityModule
from modules.game.emergent_nash import NashModule
from core.couplings import (
    QuadraticCoupling,
    DirectedHingeCoupling,
    AsymmetricHingeCoupling,
    GateBenefitCoupling,
    DampedGateBenefitCoupling,
)
from core.interfaces import EnergyModule, EnergyCoupling, OrderParameter

def _require_jax():
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "JAX is required for jax backend. Install with `uv pip install -e .[jax]` (or custom jax install)."
        ) from exc


@dataclass
class JaxEnergyRunner:
    modules: List[EnergyModule]
    couplings: List[tuple[int, int, EnergyCoupling]]
    constraints: Mapping[str, Any]
    device: str = "cpu"
    step_size: float = 0.05
    steps: int = 100

    def __post_init__(self) -> None:
        _require_jax()
        import jax
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp
        self._validate_support()
        self._term_weights = self._extract_term_weights()
        self._local_terms: List[Tuple[float, float, float]] = []
        for module in self.modules:
            self._local_terms.append(self._resolve_local_term(module))
        self._quadratic_weights: List[Tuple[int, int, float]] = []
        self._directed_hinges: List[Tuple[int, int, float]] = []
        self._asymmetric_hinges: List[Tuple[int, int, float, float, float]] = []
        self._gate_benefits: List[Tuple[int, float]] = []
        self._damped_gate_benefits: List[Tuple[int, float, float, float, float]] = []
        for i, j, coup in self.couplings:
            self._register_coupling(i, j, coup)

    def relax(self, inputs: Sequence[Any]) -> List[float]:
        etas0 = self._compute_initial_etas(inputs)
        jax = self.jax
        jnp = self.jnp
        eta = jnp.array(etas0, dtype=jnp.float32)

        def energy_fn(vec):
            total = jnp.array(0.0, dtype=jnp.float32)
            for idx, val in enumerate(vec):
                target, a, b = self._local_terms[idx]
                delta = val - target
                total = total + a * (delta ** 2) + b * (delta ** 4)
            for i, j, w in self._quadratic_weights:
                diff = vec[i] - vec[j]
                total = total + w * (diff * diff)
            for i, j, w in self._directed_hinges:
                gap = jnp.maximum(0.0, vec[j] - vec[i])
                total = total + w * (gap ** 2)
            for i, j, w, alpha, beta in self._asymmetric_hinges:
                gap = jnp.maximum(0.0, beta * vec[j] - alpha * vec[i])
                total = total + w * (gap ** 2)
            for i, w, delta in self._gate_benefits:
                total = total - w * vec[i] * delta
            for i, w, damping, eta_power, scaled_delta in self._damped_gate_benefits:
                eta_gate = jnp.maximum(0.0, vec[i])
                if eta_power != 1.0:
                    eta_gate = eta_gate ** eta_power
                total = total - w * damping * eta_gate * scaled_delta
            return total

        grad_fn = jax.grad(energy_fn)
        for _ in range(self.steps):
            grads = grad_fn(eta)
            eta = eta - self.step_size * grads
            eta = jnp.clip(eta, 0.0, 1.0)
        return [float(v) for v in eta.tolist()]

    def _compute_initial_etas(self, inputs: Sequence[Any]) -> List[OrderParameter]:
        assert len(inputs) == len(self.modules), "inputs/modules length mismatch"
        etas: List[OrderParameter] = []
        for module, x in zip(self.modules, inputs):
            eta = float(module.compute_eta(x))
            etas.append(eta)
        return etas

    def _extract_term_weights(self) -> dict[str, float]:
        result: dict[str, float] = {}
        raw = self.constraints.get("term_weights")
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                try:
                    result[str(key)] = float(value)  # type: ignore[arg-type]
                except Exception:
                    continue
        return result

    def _term_weight(self, key: str) -> float:
        return float(self._term_weights.get(key, 1.0))

    def _resolve_local_term(self, module: EnergyModule) -> Tuple[float, float, float]:
        if isinstance(module, EnergyGatingModule):
            a = float(self.constraints.get("gate_alpha", module.a))
            b = float(self.constraints.get("gate_beta", module.b))
            target = 0.0
        elif isinstance(module, SequenceConsistencyModule):
            a = float(self.constraints.get("seq_alpha", module.alpha))
            b = float(self.constraints.get("seq_beta", module.beta))
            target = 1.0
        elif isinstance(module, ConnectivityModule):
            a = float(self.constraints.get("conn_alpha", module.alpha))
            b = float(self.constraints.get("conn_beta", module.beta))
            target = 1.0
        elif isinstance(module, NashModule):
            a = float(self.constraints.get("nash_alpha", 1.0))
            b = float(self.constraints.get("nash_beta", 1.0))
            target = 1.0
        else:
            raise TypeError(f"JAX backend does not support module type: {module!r}")
        assert a >= 0.0 and b >= 0.0, "local energy coefficients must be non-negative"
        return float(target), float(a), float(b)

    def _register_coupling(self, i: int, j: int, coup: EnergyCoupling) -> None:
        key = f"coup:{coup.__class__.__name__}"
        tw = self._term_weight(key)
        if isinstance(coup, QuadraticCoupling):
            self._quadratic_weights.append((i, j, float(coup.weight) * tw))
        elif isinstance(coup, DirectedHingeCoupling):
            self._directed_hinges.append((i, j, float(coup.weight) * tw))
        elif isinstance(coup, AsymmetricHingeCoupling):
            self._asymmetric_hinges.append(
                (
                    i,
                    j,
                    float(coup.weight) * tw,
                    float(coup.alpha_i),
                    float(coup.beta_j),
                )
            )
        elif isinstance(coup, GateBenefitCoupling):
            delta = float(self.constraints.get(coup.delta_key, 0.0))
            self._gate_benefits.append((i, float(coup.weight) * tw, delta))
        elif isinstance(coup, DampedGateBenefitCoupling):
            delta = float(self.constraints.get(coup.delta_key, 0.0))
            if delta >= 0.0:
                scaled = float(coup.positive_scale) * delta
            else:
                scaled = float(coup.negative_scale) * delta
            self._damped_gate_benefits.append(
                (
                    i,
                    float(coup.weight) * tw,
                    float(coup.damping),
                    float(coup.eta_power),
                    scaled,
                )
            )
        else:
            raise TypeError(f"JAX backend does not support coupling type: {coup!r}")

    def _validate_support(self) -> None:
        if not self.modules:
            raise ValueError("No modules provided.")
        for module in self.modules:
            if not isinstance(
                module,
                (
                    EnergyGatingModule,
                    SequenceConsistencyModule,
                    ConnectivityModule,
                    NashModule,
                ),
            ):
                raise TypeError(f"Unsupported module type for JAX backend: {module!r}")
        for _i, _j, coup in self.couplings:
            if not isinstance(
                coup,
                (
                    QuadraticCoupling,
                    DirectedHingeCoupling,
                    AsymmetricHingeCoupling,
                    GateBenefitCoupling,
                    DampedGateBenefitCoupling,
                ),
            ):
                raise TypeError("JAX backend supports Quadratic, hinge, and gate-benefit couplings only.")


