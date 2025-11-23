from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.amortizer import MLPWarmStartProposer
from core.llm_adapter import StructuredTextLLMAdapter
from core.interfaces import EnergyModule, OrderParameter


@dataclass
class TextModule(EnergyModule):
    token: str

    def compute_eta(self, x: Mapping[str, float]) -> OrderParameter:
        return float(x.get("normalized_count", 0.0))

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        return (float(eta) - 0.5) ** 2


def test_structured_text_adapter_produces_constraints() -> None:
    modules = [TextModule(token="if"), TextModule(token="return")]
    proposer = MLPWarmStartProposer(input_dim=3, hidden_dim=8)
    adapter = StructuredTextLLMAdapter(
        warm_start=proposer,
        vocabulary=("if", "return"),
        temperature=0.5,
    )
    result = adapter.build("if x: return x", modules, hints={"schema": "python"})
    assert "llm_token_counts" in result.constraint_overrides
    assert result.metadata["adapter"] == "StructuredTextLLMAdapter"
    assert len(result.proposal.etas) == len(modules)

