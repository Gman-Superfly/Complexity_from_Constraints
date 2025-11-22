from __future__ import annotations

from core.agm_metrics import compute_agm_phase_metrics, compute_uncertainty_metrics
from core.weight_adapters import AGMPhaseWeightAdapter


def test_compute_agm_phase_metrics_shapes() -> None:
    # Not enough points
    m0 = compute_agm_phase_metrics([1.0])
    assert set(m0.keys()) == {"rate", "variance", "trend", "oscillation"}

    # Simple decreasing energy => high rate, positive trend on rates
    vals = [1.0, 0.8, 0.7, 0.65, 0.62, 0.60]
    m1 = compute_agm_phase_metrics(vals)
    assert 0.0 <= m1["rate"] <= 1.0


def test_uncertainty_metrics_bounds() -> None:
    energy = [1.0, 0.9, 0.85, 0.84, 0.83]
    u = compute_uncertainty_metrics(energy, recent_performance=[0.6, 0.7, 0.65])
    assert 0.0 <= u.epistemic <= 1.0
    assert 0.0 <= u.aleatoric <= 1.0
    assert isinstance(u.total, float)


def test_agm_phase_weight_adapter_updates() -> None:
    adapter = AGMPhaseWeightAdapter(increase_factor=1.1, decrease_factor=0.9)
    # Seed a stable/improving regime
    adapter.energy_history.extend([1.0, 0.8, 0.7, 0.65, 0.62, 0.60])
    current = {
        "coup:GateBenefitCoupling": 1.0,
        "local:EnergyGatingModule": 1.0,
    }
    updated = adapter.step({}, energy=0.59, current=current)
    # Expect couplings to increase slightly, gate local to decrease slightly
    assert updated["coup:GateBenefitCoupling"] > 1.0
    assert updated["local:EnergyGatingModule"] < 1.0

