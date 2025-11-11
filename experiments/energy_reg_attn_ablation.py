"""Energy-regularized attention ablation (requires PyTorch)."""

from __future__ import annotations

import argparse
from typing import List, Dict, Any

from logging.metrics_log import log_records
from models.nonlocal_attention import EnergyRegularizedAttention, torch_available


def run(batch: int, seq: int, d_model: int, lambdas: list[float]) -> None:
    if not torch_available():
        print("PyTorch not available; skipping attention ablation.")
        return
    import torch  # type: ignore[reportMissingImports]

    rows: List[Dict[str, Any]] = []
    x = torch.randn(batch, seq, d_model)
    for lam in lambdas:
        attn = EnergyRegularizedAttention(d_model=d_model, lambda_energy=lam)
        y, energy = attn(x)
        # Simple surrogate loss: reconstruction error to identity + energy
        loss = (y - x).pow(2).mean() + energy
        rows.append({
            "batch": int(batch),
            "seq": int(seq),
            "d_model": int(d_model),
            "lambda": float(lam),
            "loss": float(loss.item()),
            "energy": float(energy.item()),
        })
    out = log_records("energy_reg_attn_ablation", rows)
    print(f"Wrote {len(rows)} rows to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.05, 0.1])
    args = parser.parse_args()
    run(batch=args.batch, seq=args.seq, d_model=args.d_model, lambdas=list(args.lambdas))


if __name__ == "__main__":
    main()


