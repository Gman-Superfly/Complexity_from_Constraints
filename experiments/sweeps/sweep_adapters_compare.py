from __future__ import annotations

import argparse
import subprocess
from itertools import product
from typing import Iterable, List, Tuple

import polars as pl


ADAPTER_CONFIGS = ("analytic", "gradnorm", "smallgain")


def build_jobs(
    scenarios: Iterable[str],
    configs: Iterable[str],
    steps: int,
    dense_size: int,
) -> List[Tuple[str, str, int, int]]:
    """
    Build sweep jobs as tuples: (scenario, config, steps, dense_size).
    """
    jobs: List[Tuple[str, str, int, int]] = []
    for scenario, config in product(scenarios, configs):
        jobs.append((str(scenario), str(config), int(steps), int(dense_size)))
    return jobs


def run_job(
    scenario: str,
    config: str,
    steps: int,
    dense_size: int,
    run_id_prefix: str,
    log_budget: bool,
    warn_on_margin_shrink: bool,
    margin_warn_threshold: float | None,
) -> None:
    run_id = f"{run_id_prefix}_{scenario}_{config}"
    cmd = [
        "python",
        "-m",
        "experiments.benchmark_delta_f90",
        "--configs",
        config,
        "--scenario",
        scenario,
        "--steps",
        str(int(steps)),
        "--run_id",
        run_id,
    ]
    if scenario == "dense":
        cmd += ["--dense_size", str(int(dense_size))]
    if log_budget:
        cmd += ["--log_budget"]
        if warn_on_margin_shrink:
            cmd += ["--warn_on_margin_shrink"]
        if margin_warn_threshold is not None:
            cmd += ["--margin_warn_threshold", str(float(margin_warn_threshold))]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def summarize(run_id_prefix: str, out_csv: str) -> str:
    path = "logs/benchmark_delta_f90.csv"
    df = pl.read_csv(path)
    df = df.filter(pl.col("run_id").str.starts_with(run_id_prefix))
    # Add a scenario column inferred from run_id suffix
    scen = (
        pl.when(pl.col("run_id").str.contains("_dense_"))
        .then(pl.lit("dense"))
        .otherwise(pl.lit("baseline"))
        .alias("scenario")
    )
    cols = [
        "run_id",
        "config",
        "delta_f90_steps",
        "wall_time_sec",
        "energy_final",
        "total_backtracks",
        "redemption_gain",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df.with_columns(scen).select(["scenario"] + cols)
    out.write_csv(f"plots/{out_csv}")
    return f"plots/{out_csv}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=str, nargs="+", choices=["baseline", "dense"], default=["baseline", "dense"])
    parser.add_argument("--configs", type=str, nargs="+", choices=list(ADAPTER_CONFIGS), default=list(ADAPTER_CONFIGS))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--dense_size", type=int, default=16)
    parser.add_argument("--run_id_prefix", type=str, default="adapter_compare")
    parser.add_argument("--log_budget", action="store_true")
    parser.add_argument("--warn_on_margin_shrink", action="store_true")
    parser.add_argument("--margin_warn_threshold", type=float, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick mode: steps=40, dense_size=8")
    parser.add_argument("--summary_out", type=str, default="df90_adapters_compare_summary.csv")
    args = parser.parse_args()

    steps = 40 if args.quick else int(args.steps)
    dense_size = 8 if args.quick else int(args.dense_size)

    jobs = build_jobs(args.scenarios, args.configs, steps, dense_size)
    for scenario, config, st, ds in jobs:
        run_job(
            scenario,
            config,
            st,
            ds,
            args.run_id_prefix,
            log_budget=args.log_budget,
            warn_on_margin_shrink=args.warn_on_margin_shrink,
            margin_warn_threshold=(args.margin_warn_threshold if args.margin_warn_threshold is not None else None),
        )
    path = summarize(args.run_id_prefix, args.summary_out)
    print(f"Wrote summary to {path}")


if __name__ == "__main__":
    main()


