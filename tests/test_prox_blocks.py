from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter, SupportsLocalEnergyGrad
from core.couplings import QuadraticCoupling


@dataclass
class QuadraticWell(EnergyModule, SupportsLocalEnergyGrad):
    target: float = 0.0

    def compute_eta(self, x: Any) -> OrderParameter:
        return float(x)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return float((eta - self.target) ** 2)

    def d_local_energy_d_eta(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return float(2.0 * (eta - self.target))


def _chain_modules():
    mods = [
        QuadraticWell(target=0.0),
        QuadraticWell(target=0.5),
        QuadraticWell(target=1.0),
    ]
    coups = [
        (0, 1, QuadraticCoupling(weight=0.8)),
        (1, 2, QuadraticCoupling(weight=0.8)),
    ]
    return mods, coups


def test_prox_star_matches_sequential_on_chain() -> None:
    mods, coups = _chain_modules()
    coord_seq = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        operator_splitting=True,
        prox_steps=40,
        prox_tau=0.1,
    )
    coord_star = EnergyCoordinator(
        modules=[QuadraticWell(0.0), QuadraticWell(0.5), QuadraticWell(1.0)],
        couplings=[
            (0, 1, QuadraticCoupling(weight=0.8)),
            (1, 2, QuadraticCoupling(weight=0.8)),
        ],
        constraints={},
        operator_splitting=True,
        prox_steps=40,
        prox_tau=0.1,
        prox_block_mode="star",
    )
    etas0 = [0.2, 0.2, 0.2]
    seq = coord_seq.relax_etas_proximal(etas0, steps=coord_seq.prox_steps, tau=coord_seq.prox_tau)
    star = coord_star.relax_etas_proximal(etas0, steps=coord_star.prox_steps, tau=coord_star.prox_tau)
    energy_seq = coord_seq._energy_value(seq)
    energy_star = coord_star._energy_value(star)
    assert abs(energy_seq - energy_star) < 5e-3


def test_prox_star_decreases_energy() -> None:
    mods, coups = _chain_modules()
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints={},
        operator_splitting=True,
        prox_steps=20,
        prox_tau=0.1,
        prox_block_mode="star",
    )
    etas0 = [1.0, 0.0, 0.0]
    etas = coord.relax_etas_proximal(etas0, steps=coord.prox_steps, tau=coord.prox_tau)
    initial_energy = coord._energy_value([float(e) for e in etas0])
    final_energy = coord._energy_value(etas)
    assert final_energy <= initial_energy + 1e-6

