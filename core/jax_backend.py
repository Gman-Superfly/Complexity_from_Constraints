"""Optional JAX-based relaxation path for gating + quadratic couplings.

Usage:
    runner = JaxEnergyRunner(modules, couplings, constraints, device="cpu")
    etas = runner.relax(inputs=[...], steps=100, lr=0.05)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

from modules.gating.energy_gating import EnergyGatingModule
from core.couplings import QuadraticCoupling
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
        self._gate_params: List[Tuple[float, float]] = []
        for module in self.modules:
            assert isinstance(module, EnergyGatingModule), "JAX runner currently supports EnergyGatingModule only."
            a = float(self.constraints.get("gate_alpha", module.a))
            b = float(self.constraints.get("gate_beta", module.b))
            self._gate_params.append((a, b))
        self._quadratic_weights: List[Tuple[int, int, float]] = []
        for i, j, coup in self.couplings:
            assert isinstance(coup, QuadraticCoupling), "JAX runner currently supports QuadraticCoupling only."
            self._quadratic_weights.append((i, j, float(coup.weight)))

    def relax(self, inputs: Sequence[Any]) -> List[float]:
        etas0 = self._compute_initial_etas(inputs)
        jax = self.jax
        jnp = self.jnp
        eta = jnp.array(etas0, dtype=jnp.float32)

        def energy_fn(vec):
            total = jnp.array(0.0, dtype=jnp.float32)
            for idx, val in enumerate(vec):
                a, b = self._gate_params[idx]
                total = total + a * (val ** 2) + b * (val ** 4)
            for i, j, w in self._quadratic_weights:
                diff = vec[i] - vec[j]
                total = total + w * (diff * diff)
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

    def _validate_support(self) -> None:
        if not self.modules:
            raise ValueError("No modules provided.")
        for module in self.modules:
            if not isinstance(module, EnergyGatingModule):
                raise TypeError(f"Unsupported module type for JAX backend: {module!r}")
        for _i, _j, coup in self.couplings:
            if not isinstance(coup, QuadraticCoupling):
                raise TypeError("JAX backend currently supports QuadraticCoupling only.")


