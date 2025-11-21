from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl
import numpy as np

from core.coordinator import EnergyCoordinator
from core.couplings import GateBenefitCoupling
from cf_logging.observability import RelaxationTracker, GatingMetricsLogger
from modules.sequence.monotonic_eta import SequenceConsistencyModule, sample_monotonicity_score
from modules.gating.energy_gating import EnergyGatingModule


def make_sequence_with_dip(n: int, dip_idx: int, dip: float = 0.4) -> List[float]:
    seq = np.linspace(0.0, 1.0, num=n).tolist()
    seq[dip_idx] -= dip
    return seq


def test_prod_sequence_gating_flow_reduces_energy_and_logs_metrics() -> None:
    n = 48
    dip_idx = 18
    seq = make_sequence_with_dip(n, dip_idx, dip=0.35)
    seq_module = SequenceConsistencyModule(samples=512)
    eta_before = sample_monotonicity_score(seq, samples=512, seed=7)
    repaired = seq.copy()
    repaired[dip_idx] = max(repaired[dip_idx], repaired[dip_idx - 1])
    eta_after = sample_monotonicity_score(repaired, samples=512, seed=7)
    delta_eta = float(eta_after - eta_before)
    assert delta_eta > 0.0

    gain_fn = lambda _: delta_eta  # constant gain derived from observed improvement
    gate_module = EnergyGatingModule(gain_fn=gain_fn, cost=0.05, k=10.0, use_hazard=True, a=0.15, b=0.25)

    modules = [seq_module, gate_module]
    couplings = [(1, 0, GateBenefitCoupling(weight=0.8, delta_key="delta_eta_domain"))]
    constraints = {
        "delta_eta_domain": delta_eta,
        "gate_alpha": 0.05,
        "gate_beta": 0.10,
        "term_weights": {
            "local:EnergyGatingModule": 0.1,
            "coup:GateBenefitCoupling": 3.5,
        },
    }
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints=constraints,
        grad_eps=1e-6,
        step_size=0.05,
        use_analytic=True,
        normalize_grads=True,
        line_search=True,
        armijo_c=1e-12,
        backtrack_factor=0.5,
        max_backtrack=8,
    )
    inputs = [seq, None]
    etas0 = coord.compute_etas(inputs)
    tracker_path = Path("logs") / "prod_sequence_gating_trace.csv"
    tracker_path.parent.mkdir(exist_ok=True)
    # Ensure a clean log file per test invocation to avoid cross-run contamination
    tracker_path.unlink(missing_ok=True)
    tracker = RelaxationTracker(name=tracker_path.stem, run_id="flow_seq_gate")
    tracker.attach(coord)
    gating_logger = GatingMetricsLogger(run_id="flow_seq_gate")
    etas_final = coord.relax_etas(etas0, steps=40)
    tracker.flush()
    gating_logger.record(
        hazard=gate_module.hazard_rate(None),
        eta_gate=etas_final[1],
        redemption=delta_eta,
        good=(delta_eta > 0),
    )
    gating_logger.flush()

    assert etas_final[1] >= etas0[1], "gate η should not decrease when delta_eta > 0"
    assert 0.0 <= etas_final[1] <= 1.0
    assert 0.0 <= etas_final[0] <= 1.0

    df = pl.read_csv(tracker_path)
    df = df.filter(pl.col("run_id") == "flow_seq_gate").sort("step")
    assert df.height > 1
    energies = df["energy"].to_list()
    assert energies[-1] <= energies[0] + 1e-9
    # delta energy after first step should be <= 0 (within tolerance)
    deltas = [d for d in df["delta_energy"].to_list()[1:] if d == d]
    assert all(d <= 1e-6 for d in deltas)
    assert df["min_eta"].min() >= -1e-9
    assert df["max_eta"].max() <= 1.0 + 1e-9

    tracker_path.unlink(missing_ok=True)


