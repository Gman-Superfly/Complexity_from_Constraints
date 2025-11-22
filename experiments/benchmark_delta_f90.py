"""Benchmark harness for ΔF90 comparisons across coordinator configs.

Usage:
    uv run python -m experiments.benchmark_delta_f90
    uv run python -m experiments.benchmark_delta_f90 --configs default analytic vect hinge coord

Outputs a CSV under logs/benchmark_delta_f90.csv with:
    run_id, config_name, steps, wall_time_sec, delta_f90_steps
"""

from __future__ import annotations

import argparse
import time
from typing import List, Dict, Any, Tuple

from modules.gating.energy_gating import EnergyGatingModule
from core.couplings import QuadraticCoupling, GateBenefitCoupling
from core.coordinator import EnergyCoordinator
from cf_logging.metrics_log import log_records
from core.weight_adapters import GradNormWeightAdapter, AGMPhaseWeightAdapter
try:
    from core.weight_adapters import SmallGainWeightAdapter  # type: ignore
except Exception:
    SmallGainWeightAdapter = None  # fallback if not present


def make_modules_and_couplings() -> Tuple[List[Any], List[Tuple[int, int, Any]], Dict[str, Any]]:
    seq = [i / 63 for i in range(64)]
    seq_mod = EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.3)
    gate_mod = EnergyGatingModule(gain_fn=lambda _: 0.1, cost=0.05, a=0.25, b=0.35)
    mods = [seq_mod, gate_mod]
    coups = [
        (0, 1, QuadraticCoupling(weight=0.7)),
        (1, 0, GateBenefitCoupling(weight=0.8, delta_key="delta_eta_domain")),
    ]
    constraints = {"delta_eta_domain": 0.08}
    return mods, coups, constraints, [seq, None]


def delta_f90(energies: List[float]) -> int:
    if len(energies) < 2:
        return len(energies)
    total_drop = energies[0] - min(energies)
    if total_drop <= 0.0:
        return len(energies)
    threshold = energies[0] - 0.9 * total_drop
    for idx, val in enumerate(energies):
        if val <= threshold:
            return idx + 1
    return len(energies)


def _per_term_breakdown(coord: EnergyCoordinator, etas: List[float]) -> Dict[str, float]:
    """Return per-term energy contributions and gradient norms keyed by term names."""
    out: Dict[str, float] = {}
    cw = coord._combined_term_weights()  # instrumentation
    # Local energies
    for idx, m in enumerate(coord.modules):
        key = f"local:{m.__class__.__name__}"
        w = float(cw.get(key, 1.0))
        e = float(m.local_energy(float(etas[idx]), coord.constraints)) * w
        out[f"energy:{key}"] = e
    # Coupling energies
    for i, j, coup in coord.couplings:
        key = f"coup:{coup.__class__.__name__}"
        w = float(cw.get(key, 1.0))
        e = float(coup.coupling_energy(float(etas[i]), float(etas[j]), coord.constraints)) * w
        out[f"energy:{key}"] = out.get(f"energy:{key}", 0.0) + e
    # Gradient norms per term
    norms = coord._term_grad_norms(etas)  # instrumentation
    for k, v in norms.items():
        out[f"grad_norm:{k}"] = float(v)
    return out


def run_config(name: str, coord_kwargs: Dict[str, Any], steps: int) -> Dict[str, Any]:
    mods, coups, constraints, inputs = make_modules_and_couplings()
    # Materialize adapter if requested via a sentinel in coord_kwargs
    adapter_name = coord_kwargs.pop("_adapter", None)
    adapter = None
    if adapter_name == "gradnorm":
        adapter = GradNormWeightAdapter()
    elif adapter_name == "agm":
        adapter = AGMPhaseWeightAdapter()
    elif adapter_name == "smallgain" and SmallGainWeightAdapter is not None:
        adapter = SmallGainWeightAdapter()
        # ensure coordinator exposes details needed by the adapter
        coord_kwargs = dict(coord_kwargs)
        coord_kwargs["expose_lipschitz_details"] = True
        # keep stability guard on for a meaningful contraction margin
        coord_kwargs.setdefault("stability_guard", True)
        coord_kwargs.setdefault("log_contraction_margin", True)
    coord = EnergyCoordinator(mods, coups, constraints, weight_adapter=adapter, **coord_kwargs)
    etas = coord.compute_etas(inputs)
    energies: List[float] = []
    coord.on_energy_updated.append(lambda F: energies.append(F))
    start = time.perf_counter()
    coord.relax_etas(etas, steps=steps)
    duration = time.perf_counter() - start
    if not energies:
        energies = [coord.energy(etas)]
    row: Dict[str, Any] = {
        "config": name,
        "steps": steps,
        "wall_time_sec": duration,
        "delta_f90_steps": delta_f90(energies),
        "energy_final": energies[-1],
    }
    # Append per-term breakdown at the end of relaxation
    breakdown = _per_term_breakdown(coord, etas)
    row.update(breakdown)
    # Mark mode flags for analysis
    row["operator_splitting"] = bool(getattr(coord, "operator_splitting", False))
    row["adapter"] = str(adapter_name or "none")
    return row


PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "use_analytic": False,
        "use_vectorized_quadratic": False,
        "use_vectorized_hinges": False,
        "neighbor_gradients_only": False,
    },
    "analytic": {
        "use_analytic": True,
        "use_vectorized_quadratic": False,
        "use_vectorized_hinges": False,
        "neighbor_gradients_only": False,
    },
    "vect": {
        "use_analytic": True,
        "use_vectorized_quadratic": True,
        "use_vectorized_hinges": True,
        "neighbor_gradients_only": True,
        "line_search": True,
        "normalize_grads": True,
    },
    "coord": {
        "use_analytic": True,
        "use_vectorized_quadratic": True,
        "neighbor_gradients_only": True,
        "use_coordinate_descent": True,
        "coordinate_steps": 60,
    },
    "adaptive": {
        "use_analytic": True,
        "use_vectorized_quadratic": True,
        "use_vectorized_hinges": True,
        "neighbor_gradients_only": True,
        "adaptive_coordinate_descent": True,
        "coordinate_steps": 30,
        "line_search": True,
    },
    # Operator-splitting / proximal mode
    "prox": {
        "use_analytic": True,
        "operator_splitting": True,
        "prox_steps": 60,
        "prox_tau": 0.05,
    },
    # Adapter comparisons (gradient-based relaxation path)
    "gradnorm": {
        "use_analytic": True,
        "_adapter": "gradnorm",
        "line_search": True,
        "normalize_grads": True,
    },
    "agm": {
        "use_analytic": True,
        "_adapter": "agm",
        "line_search": True,
        "normalize_grads": True,
    },
    "smallgain": {
        "use_analytic": True,
        "_adapter": "smallgain",
        "line_search": True,
        "normalize_grads": True,
        # helpful toggles
        "use_vectorized_quadratic": True,
        "use_vectorized_hinges": True,
        "neighbor_gradients_only": True,
        # stability and telemetry
        "stability_guard": True,
        "log_contraction_margin": True,
        "expose_lipschitz_details": True,
    },
    "admm": {
        "use_analytic": True,
        "use_admm": True,
        "admm_steps": 60,
        "admm_rho": 1.0,
        "admm_step_size": 0.05,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", choices=list(PRESETS.keys()), default=list(PRESETS.keys()))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--run_id", type=str, default="benchmark_delta_f90")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for cfg in args.configs:
        result = run_config(cfg, PRESETS[cfg], args.steps)
        result["run_id"] = args.run_id
        rows.append(result)
        print(
            f"[{cfg}] dF90 steps={result['delta_f90_steps']}, "
            f"wall_time={result['wall_time_sec']:.4f}s, energy_final={result['energy_final']:.6f}"
        )
    path = log_records("benchmark_delta_f90", rows)
    print(f"Logged {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()

