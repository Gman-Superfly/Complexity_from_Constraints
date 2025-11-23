from __future__ import annotations

from experiments.sweeps.sweep_smallgain import build_jobs


def test_build_jobs_cartesian() -> None:
    jobs = build_jobs(rhos=[0.5, 0.7], dws=[0.05, 0.10], scenarios=["baseline", "dense"], steps=80, dense_size=16)
    # 2 scenarios * 2 rhos * 2 dws = 8 jobs
    assert len(jobs) == 8
    # Check one exemplar tuple contents and types
    scen, rho, dw, steps, dense = jobs[0]
    assert scen in ("baseline", "dense")
    assert isinstance(rho, float) and isinstance(dw, float)
    assert isinstance(steps, int) and isinstance(dense, int)


