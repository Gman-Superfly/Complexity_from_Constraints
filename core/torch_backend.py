"""Optional Torch-based relaxation path for quadratic couplings + gating modules.

This is a prototype backend (Roadmap P4). It relies on PyTorch autograd to run
gradient steps on GPU/CPU. Currently supports:
    - Local energies with Landau form (EnergyGatingModule, SequenceConsistencyModule,
      ConnectivityModule, NashModule)
    - Couplings: Quadratic, Directed/Asymmetric hinge, GateBenefit, DampedGateBenefit

Usage:
    from core.torch_backend import TorchEnergyRunner
    runner = TorchEnergyRunner(modules, couplings, constraints, device="cuda")
    etas = runner.relax(inputs=[...], steps=100, lr=0.05)

Torch is optional; install with `uv pip install torch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

from core.interfaces import EnergyModule, EnergyCoupling, OrderParameter
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


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for torch backend. Install with `uv pip install torch`."
        ) from exc


@dataclass
class TorchEnergyRunner:
    """Torch autograd relaxation for gating modules with quadratic couplings."""

    modules: List[EnergyModule]
    couplings: List[tuple[int, int, EnergyCoupling]]
    constraints: Mapping[str, Any]
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        _require_torch()
        import torch

        self._torch = torch
        self._validate_support()
        self._torch = torch
        self._term_weights = self._extract_term_weights()
        self._local_terms: List[Tuple[float, float, float]] = []  # (target, a, b)
        for module in self.modules:
            self._local_terms.append(self._resolve_local_term(module))

        self._quadratic_weights: List[Tuple[int, int, float]] = []
        self._directed_hinges: List[Tuple[int, int, float]] = []
        self._asymmetric_hinges: List[Tuple[int, int, float, float, float]] = []
        self._gate_benefits: List[Tuple[int, float]] = []
        self._damped_gate_benefits: List[Tuple[int, float, float, float, float]] = []
        for i, j, coup in self.couplings:
            self._register_coupling(i, j, coup)

    def relax(
        self,
        inputs: Sequence[Any],
        steps: int = 100,
        lr: float = 0.05,
        clamp: Tuple[float, float] = (0.0, 1.0),
    ) -> List[float]:
        """Run SGD on η using torch autograd."""
        assert steps > 0, "steps must be positive"
        assert 0.0 < lr < 1.0, "learning rate out of bounds"
        etas0 = self._compute_initial_etas(inputs)
        torch = self._torch
        device = torch.device(self.device)
        tensor = torch.tensor
        dtype = getattr(torch, self.dtype)
        etas = tensor(etas0, dtype=dtype, device=device, requires_grad=True)
        opt = torch.optim.SGD([etas], lr=lr)
        history: List[float] = []
        for _ in range(steps):
            opt.zero_grad()
            energy = self._total_energy(etas)
            history.append(float(energy.detach().cpu()))
            energy.backward()
            opt.step()
            with torch.no_grad():
                etas.clamp_(clamp[0], clamp[1])
        # final energy recording
        history.append(float(self._total_energy(etas).detach().cpu()))
        return [float(v) for v in etas.detach().cpu().tolist()]

    def _compute_initial_etas(self, inputs: Sequence[Any]) -> List[OrderParameter]:
        assert len(inputs) == len(self.modules), "inputs/modules length mismatch"
        etas: List[OrderParameter] = []
        for module, x in zip(self.modules, inputs):
            eta = float(module.compute_eta(x))
            assert 0.0 <= eta <= 1.0, "η must be within [0, 1]"
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
        # returns (target, a, b)
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
            raise TypeError(f"Torch backend does not support module type: {module!r}")
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
            raise TypeError(f"Torch backend does not support coupling type: {coup!r}")

    def _total_energy(self, etas):
        torch = self._torch
        total = torch.tensor(0.0, dtype=getattr(torch, self.dtype), device=etas.device)
        for idx, eta in enumerate(etas):
            target, a, b = self._local_terms[idx]
            delta = eta - target
            total = total + a * (delta ** 2) + b * (delta ** 4)
        for i, j, w in self._quadratic_weights:
            diff = etas[i] - etas[j]
            total = total + w * (diff * diff)
        for i, j, w in self._directed_hinges:
            gap = torch.relu(etas[j] - etas[i])
            total = total + w * (gap ** 2)
        for i, j, w, alpha, beta in self._asymmetric_hinges:
            gap = torch.relu(beta * etas[j] - alpha * etas[i])
            total = total + w * (gap ** 2)
        for i, w, delta in self._gate_benefits:
            total = total - w * etas[i] * delta
        for i, w, damping, eta_power, scaled_delta in self._damped_gate_benefits:
            eta_gate = torch.relu(etas[i])
            if eta_power != 1.0:
                eta_gate = eta_gate ** eta_power
            total = total - w * damping * eta_gate * scaled_delta
        return total

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
                raise TypeError(
                    "Torch backend supports EnergyGatingModule, "
                    "SequenceConsistencyModule, ConnectivityModule, and NashModule only."
                )
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
                raise TypeError(
                    "Torch backend supports Quadratic, hinge, and gate-benefit couplings only."
                )


