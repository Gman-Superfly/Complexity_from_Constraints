from __future__ import annotations

from experiments.sweeps.sweep_adapters_compare import build_jobs, ADAPTER_CONFIGS


def test_build_jobs_adapters_compare() -> None:
    scenarios = ["baseline", "dense"]
    configs = list(ADAPTER_CONFIGS)
    jobs = build_jobs(scenarios, configs, steps=80, dense_size=16)
    assert len(jobs) == len(scenarios) * len(configs)
    scen, cfg, steps, dense = jobs[0]
    assert scen in scenarios
    assert cfg in configs
    assert isinstance(steps, int) and isinstance(dense, int)


