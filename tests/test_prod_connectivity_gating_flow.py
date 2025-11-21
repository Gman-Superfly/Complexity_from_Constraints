from __future__ import annotations

from core.coordinator import EnergyCoordinator
from core.couplings import DampedGateBenefitCoupling
from modules.connectivity.nl_threshold_shift import (
    ConnectivityModule,
    build_grid_bond_graph,
)
from modules.gating.energy_gating import EnergyGatingModule
from core.energy import total_energy
from cf_logging.observability import GatingMetricsLogger


def test_prod_connectivity_gating_flow_energy_drops_and_gate_opens():
    base = build_grid_bond_graph(n=6, p=0.45, add_shortcuts=False, seed=7)
    shortcut = build_grid_bond_graph(n=6, p=0.45, add_shortcuts=True, shortcut_frac=0.2, seed=7)
    conn = ConnectivityModule()
    eta_base = conn.compute_eta(base)
    eta_short = conn.compute_eta(shortcut)
    delta = float(max(0.0, eta_short - eta_base))
    assert delta > 0.0

    gate = EnergyGatingModule(gain_fn=lambda _: delta, cost=0.05, k=8.0, use_hazard=True, a=0.15, b=0.25)
    modules = [conn, gate]
    couplings = [
        (1, 0, DampedGateBenefitCoupling(weight=0.8, delta_key="delta_eta_domain", damping=0.6, eta_power=1.0))
    ]
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints={
            "delta_eta_domain": delta,
            "gate_alpha": 0.1,
            "gate_beta": 0.15,
            "term_weights": {
                "local:EnergyGatingModule": 0.2,
                "coup:DampedGateBenefitCoupling": 2.5,
            },
        },
        grad_eps=1e-6,
        step_size=0.05,
        use_analytic=True,
        enforce_invariants=True,
        normalize_grads=True,
    )
    inputs = [base, None]
    etas0 = coord.compute_etas(inputs)
    F0 = total_energy(etas0, modules, couplings, coord.constraints)
    etas_final = coord.relax_etas(etas0, steps=40)
    F1 = total_energy(etas_final, modules, couplings, coord.constraints)
    assert F1 <= F0 + 1e-6
    assert etas_final[1] >= etas0[1]
    assert all(0.0 - 1e-9 <= eta <= 1.0 + 1e-9 for eta in etas_final)
    logger = GatingMetricsLogger(run_id="flow_conn_gate")
    logger.record(hazard=gate.hazard_rate(None), eta_gate=etas_final[1], redemption=delta, good=(delta > 0))
    logger.flush()

