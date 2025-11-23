"""Minimal System-2 reasoning demo: LLM draft → adapter → relaxation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.amortizer import MLPWarmStartProposer, run_warm_start_relaxation
from core.coordinator import EnergyCoordinator
from core.llm_adapter import StructuredTextLLMAdapter
from core.interfaces import EnergyModule, OrderParameter
from core.couplings import QuadraticCoupling


@dataclass
class TokenEnergyModule(EnergyModule):
    """Tiny module whose energy prefers confident tokens close to a target density."""

    token: str
    weight: float = 1.0

    def compute_eta(self, x: Mapping[str, float]) -> OrderParameter:
        return float(x.get("normalized_count", 0.0))

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        target = float(constraints.get("target_density", 0.5))
        diff = float(eta) - target
        return float(self.weight * diff * diff)


def build_modules(vocabulary: Sequence[str]) -> list[TokenEnergyModule]:
    return [TokenEnergyModule(token=token) for token in vocabulary]


def main() -> None:
    vocabulary = ("def", "return", "if", "else")
    modules = build_modules(vocabulary)
    couplings = [
        (0, 1, QuadraticCoupling(weight=0.4)),
        (2, 3, QuadraticCoupling(weight=0.6)),
    ]
    constraints = {"target_density": 0.4}
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints=constraints,
        stability_guard=True,
        log_contraction_margin=True,
    )

    warm_start = MLPWarmStartProposer(input_dim=3, hidden_dim=16, activation="relu")
    adapter = StructuredTextLLMAdapter(
        warm_start=warm_start,
        vocabulary=vocabulary,
        temperature=0.7,
    )

    llm_draft = "def foo(x):\n    if x > 0:\n        return x\n    else:\n        return -x"
    adapter_result = adapter.build(llm_draft, modules, hints={"schema": "python"})
    warm_result = run_warm_start_relaxation(
        coord,
        proposer=None,
        inputs=None,
        proposal=adapter_result.proposal,
        constraint_overrides=adapter_result.constraint_overrides,
        relax_steps=8,
    )
    print("Initial energy:", warm_result.initial_energy)
    print("Final energy:", warm_result.final_energy)
    print("Contraction margins:", warm_result.relaxation_metrics["contraction_margins"])


if __name__ == "__main__":
    main()

