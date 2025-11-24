"""Repeatable IEON (orthogonal-noise) benchmark with optional adaptive controller.

Usage (Windows PowerShell):
    uv run python -m experiments.ieon_repeats --configs vect gradnorm agm `
        --repeats 5 --steps 80 --scenario dense --dense_size 32 `
        --noise_magnitude 0.05 --auto_noise_controller --log_budget --run_id_prefix ieon_noise

Outputs:
    - Appends rows to logs/ieon_noise_benchmark.csv via cf_logging.metrics_log
      with per-run dF90, runtime, final energy, and (if available) backtracks.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from cf_logging.metrics_log import log_records
from experiments.benchmark_delta_f90 import PRESETS, run_config


def _validate_configs(configs: List[str]) -> List[str]:
    assert len(configs) > 0, "At least one config is required"
    for c in configs:
        assert c in PRESETS, f"Unknown config '{c}'. Valid: {sorted(PRESETS)}"
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(PRESETS.keys()),
        default=["vect", "gradnorm", "agm"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--scenario", choices=["baseline", "dense"], default="dense")
    parser.add_argument("--dense_size", type=int, default=32)
    parser.add_argument("--noise_magnitude", type=float, default=0.05)
    parser.add_argument("--noise_schedule_decay", type=float, default=0.99)
    parser.add_argument("--auto_noise_controller", action="store_true")
    parser.add_argument("--run_id_prefix", type=str, default="ieon_noise")
    parser.add_argument("--log_budget", action="store_true")
    parser.add_argument("--budget_name", type=str, default="ieon_noise_budget")
    # Optional: warnings on contraction margin shrink (if budget logging is enabled)
    parser.add_argument("--warn_on_margin_shrink", action="store_true")
    parser.add_argument("--margin_warn_threshold", type=float, default=None)
    # Optional: mirror/logit updates for gradient mode
    parser.add_argument("--use_logit_updates", action="store_true")
    args = parser.parse_args()

    configs = _validate_configs(list(args.configs))
    assert args.repeats >= 1, "repeats must be >= 1"
    assert args.steps > 0, "steps must be > 0"
    assert args.dense_size >= 3, "dense_size must be >= 3"
    assert 0.0 <= args.noise_magnitude <= 1.0, "noise_magnitude must be in [0,1]"
    assert 0.0 < args.noise_schedule_decay <= 1.0, "decay must be in (0,1]"

    rows: List[Dict[str, Any]] = []
    for rep in range(1, int(args.repeats) + 1):
        rid = f"{args.run_id_prefix}_r{rep}"
        for cfg in configs:
            # Start with preset; inject IEON orthogonal-noise controls
            coord_kwargs: Dict[str, Any] = dict(PRESETS[cfg])
            coord_kwargs["enable_orthogonal_noise"] = True
            coord_kwargs["noise_magnitude"] = float(args.noise_magnitude)
            coord_kwargs["noise_schedule_decay"] = float(args.noise_schedule_decay)
            if args.auto_noise_controller:
                coord_kwargs["auto_noise_controller"] = True
            if args.use_logit_updates and cfg in (
                "default",
                "analytic",
                "vect",
                "coord",
                "adaptive",
                "gradnorm",
                "agm",
                "smallgain",
            ):
                coord_kwargs["use_logit_updates"] = True

            result = run_config(
                name=cfg,
                coord_kwargs=coord_kwargs,
                steps=int(args.steps),
                scenario=str(args.scenario),
                dense_size=int(args.dense_size),
                log_budget=bool(args.log_budget),
                budget_name=str(args.budget_name),
                run_id=rid,
                warn_on_margin_shrink=bool(args.warn_on_margin_shrink),
                margin_warn_threshold=(
                    float(args.margin_warn_threshold)
                    if args.margin_warn_threshold is not None
                    else None
                ),
            )
            result["run_id"] = rid
            rows.append(result)
            print(
                f"[{rid}][{cfg}] dF90 steps={result['delta_f90_steps']}, "
                f"wall_time={result['wall_time_sec']:.4f}s, energy_final={result['energy_final']:.6f}"
            )

    out = log_records("ieon_noise_benchmark", rows)
    print(f"Logged {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()



