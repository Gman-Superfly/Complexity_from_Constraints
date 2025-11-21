"""Quick Landau free-energy plot for η ∈ [-1, 1].

Usage (Windows PowerShell):
    uv run python examples.landau_plot --a -0.5 --b 1.0 --save plots/landau.png

If --save is omitted we open an interactive window (matplotlib). Install extras:
    uv pip install -e .[examples]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "matplotlib is required for examples. Install with `uv pip install -e .[examples]`."
    ) from exc

from core.energy import landau_free_energy


def compute_curve(a: float, b: float, eta_min: float, eta_max: float, num: int) -> Tuple[np.ndarray, np.ndarray]:
    assert num >= 16, "num must be >= 16 for a smooth curve"
    eta = np.linspace(eta_min, eta_max, num=num, dtype=float)
    F = landau_free_energy(eta, a=a, b=b)
    return eta, np.asarray(F, dtype=float)


def plot_landau(a: float, b: float, eta_bounds: Tuple[float, float], num: int, out: Path | None) -> None:
    eta, F = compute_curve(a, b, eta_bounds[0], eta_bounds[1], num)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eta, F, label=f"F(η; a={a:.2f}, b={b:.2f})", color="#005bbb", linewidth=2.0)
    ax.set_xlabel("η")
    ax.set_ylabel("Free energy F(η)")
    ax.set_title("Landau Free Energy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Saved plot to {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=-0.5, help="Quadratic coefficient (phase control)")
    parser.add_argument("--b", type=float, default=1.0, help="Quartic coefficient (>0 for stability)")
    parser.add_argument("--eta-min", type=float, default=-1.0)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--num", type=int, default=400, help="Samples along η axis")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save PNG instead of showing")
    args = parser.parse_args()
    if args.b <= 0.0:
        raise SystemExit("Parameter b must be > 0 for stability.")
    if args.eta_min >= args.eta_max:
        raise SystemExit("eta-min must be less than eta-max.")
    plot_landau(args.a, args.b, (args.eta_min, args.eta_max), args.num, args.save)


if __name__ == "__main__":
    main()


