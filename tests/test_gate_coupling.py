from __future__ import annotations

from core.couplings import GateBenefitCoupling


def test_gate_benefit_coupling_energy_sign():
    coup = GateBenefitCoupling(weight=1.0, delta_key="delta_eta_domain")
    constraints = {"delta_eta_domain": 0.2}
    # eta_i = gate; higher gate with positive delta reduces energy
    F_low = coup.coupling_energy(eta_i=0.1, eta_j=0.0, constraints=constraints)
    F_high = coup.coupling_energy(eta_i=0.9, eta_j=0.0, constraints=constraints)
    assert F_high < F_low


