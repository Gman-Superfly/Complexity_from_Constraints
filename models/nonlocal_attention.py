"""Energy-regularized non-local attention (PyTorch, optional).

If torch is unavailable, import will fail with an informative error at runtime.
"""

from __future__ import annotations

from typing import Tuple

try:
    import torch
    import torch.nn as nn
except Exception as e:  # noqa: BLE001
    torch = None
    nn = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

__all__ = ["EnergyRegularizedAttention", "torch_available"]


def torch_available() -> bool:
    return _IMPORT_ERROR is None and (torch is not None) and (nn is not None)


class EnergyRegularizedAttention(nn.Module):  # type: ignore[misc]
    """Non-local attention with a simple variance-based energy penalty."""

    def __init__(self, d_model: int, lambda_energy: float = 0.1):
        if not torch_available():
            raise ImportError(f"PyTorch is required for EnergyRegularizedAttention: {_IMPORT_ERROR}")
        super().__init__()
        assert d_model > 0, "d_model must be positive"
        assert 0.0 <= lambda_energy <= 1.0, "lambda_energy must be in [0,1]"
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.lambda_energy = lambda_energy

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        assert x.dim() == 3, "Expected [batch, seq, d_model]"
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)  # non-local
        y = torch.matmul(attn, v)
        # Energy surrogate encourages coherent attention (lower variance)
        energy = self.lambda_energy * attn.var(dim=(-2, -1))
        return y, energy.mean()


