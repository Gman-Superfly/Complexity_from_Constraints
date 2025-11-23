from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, List

from core.amortizer import MLPWarmStartProposer, run_warm_start_relaxation
from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter


@dataclass
class DummyModule(EnergyModule):
    target: float = 0.5

    def compute_eta(self, x: Mapping[str, float]) -> OrderParameter:
        return float(x.get("eta", 0.5))

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        target = float(constraints.get("target", self.target))
        diff = float(eta) - target
        return diff * diff


def _build_coord(count: int) -> EnergyCoordinator:
    modules: List[DummyModule] = [DummyModule(target=0.3 + 0.1 * i) for i in range(count)]
    return EnergyCoordinator(modules=modules, couplings=[], constraints={"target": 0.5}, stability_guard=True)


def test_mlp_warm_start_serialization_roundtrip() -> None:
    modules = [DummyModule()]
    proposer = MLPWarmStartProposer(input_dim=2, hidden_dim=8, activation="tanh")
    proposal = proposer.propose(modules, [{"eta": 0.7, "extra": 0.2}], hints={"bias": 0.1})
    assert len(proposal.etas) == 1
    assert 0.0 <= proposal.etas[0] <= 1.0
    state = proposer.state_dict()
    cloned = MLPWarmStartProposer(input_dim=2, hidden_dim=8, activation="tanh")
    cloned.load_state_dict(state)
    proposal_clone = cloned.propose(modules, [{"eta": 0.2, "extra": 0.0}])
    assert len(proposal_clone.etas) == 1


def test_run_warm_start_relaxation_emits_metrics() -> None:
    coord = _build_coord(3)
    proposer = MLPWarmStartProposer(input_dim=2, hidden_dim=4)
    inputs = [{"eta": 0.9, "extra": 0.1} for _ in range(3)]
    result = run_warm_start_relaxation(coord, proposer=proposer, inputs=inputs, relax_steps=6)
    assert result.final_energy <= result.initial_energy + 1e-6
    metrics = result.relaxation_metrics
    assert "accepted_steps" in metrics
    assert isinstance(metrics["accepted_steps"], int)
    assert isinstance(metrics.get("contraction_margins"), list)

