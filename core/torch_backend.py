"""Optional Torch-based relaxation path for quadratic couplings + gating modules.

This is a prototype backend (Roadmap P4). It relies on PyTorch autograd to run
gradient steps on GPU/CPU. Currently supports:
    - EnergyGatingModule local energies (a η^2 + b η^4)
    - QuadraticCoupling between order parameters

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
from core.couplings import QuadraticCoupling


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
        self._gate_params: List[Tuple[float, float]] = []
        for module in self.modules:
            if isinstance(module, EnergyGatingModule):
                a = float(self.constraints.get("gate_alpha", module.a))
                b = float(self.constraints.get("gate_beta", module.b))
                assert a >= 0.0 and b >= 0.0, "gate_alpha/beta must be non-negative"
                self._gate_params.append((a, b))
            else:
                raise TypeError(
                    "TorchEnergyRunner currently supports EnergyGatingModule only."
                )
        self._quadratic_weights: List[Tuple[int, int, float]] = []
        for i, j, coup in self.couplings:
            if not isinstance(coup, QuadraticCoupling):
                raise TypeError("Only QuadraticCoupling is supported in torch backend.")
            self._quadratic_weights.append((i, j, float(coup.weight)))

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

    def _total_energy(self, etas):
        torch = self._torch
        total = torch.tensor(0.0, dtype=getattr(torch, self.dtype), device=etas.device)
        for idx, eta in enumerate(etas):
            a, b = self._gate_params[idx]
            total = total + a * (eta ** 2) + b * (eta ** 4)
        for i, j, w in self._quadratic_weights:
            diff = etas[i] - etas[j]
            total = total + w * (diff * diff)
        return total

    def _validate_support(self) -> None:
        # ensures every module is supported (currently gating only)
        if not self.modules:
            raise ValueError("No modules provided.")
        for module in self.modules:
            if not isinstance(module, EnergyGatingModule):
                raise TypeError(
                    f"Torch backend currently supports gating modules only; got {module!r}"
                )


