"""LLM adapter utilities bridging System-1 drafts to System-2 relaxation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Protocol, List, Dict

from .amortizer import WarmStartProposal, WarmStartProposer, _clamp01


@dataclass
class LLMAdapterResult:
    """Output of an LLM adapter: warm-start proposal + constraint overrides."""

    proposal: WarmStartProposal
    constraint_overrides: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class LLMAdapter(Protocol):
    """Protocol for adapters that translate LLM outputs into η₀ + constraints."""

    def build(
        self,
        llm_output: Any,
        modules: Sequence[Any],
        hints: Optional[Mapping[str, Any]] = None,
    ) -> LLMAdapterResult:
        ...


def _tokenize_llm_output(output: Any) -> List[str]:
    if output is None:
        return []
    if isinstance(output, str):
        return output.replace("\n", " ").split()
    if isinstance(output, Mapping) and "text" in output:
        return _tokenize_llm_output(output["text"])
    if isinstance(output, Sequence):
        tokens: List[str] = []
        for chunk in output:
            tokens.extend(_tokenize_llm_output(chunk))
        return tokens
    return [str(output)]


@dataclass
class StructuredTextLLMAdapter(LLMAdapter):
    """Simple adapter that maps token counts into warm-start hints."""

    warm_start: WarmStartProposer
    vocabulary: Sequence[str]
    temperature: float = 1.0
    confidence_floor: float = 0.2

    def build(
        self,
        llm_output: Any,
        modules: Sequence[Any],
        hints: Optional[Mapping[str, Any]] = None,
    ) -> LLMAdapterResult:
        assert len(modules) == len(self.vocabulary), "modules/vocabulary length mismatch"
        hints = dict(hints or {})
        tokens = _tokenize_llm_output(llm_output)
        counts = Counter(token.lower() for token in tokens)
        max_count = max(counts.values()) if counts else 1
        per_module_inputs: List[Dict[str, float]] = []
        for token_key in self.vocabulary:
            token = token_key.lower()
            count = float(counts.get(token, 0))
            normalized = count / max(1.0, float(max_count))
            confidence = _clamp01(normalized)
            per_module_inputs.append(
                {
                    "count": count,
                    "normalized_count": normalized,
                    "token_length": float(len(token)),
                }
            )
        proposal = self.warm_start.propose(modules, per_module_inputs, hints=hints)
        confidence = max(self.confidence_floor, proposal.confidence)
        constraint_overrides = dict(hints)
        constraint_overrides["llm_token_counts"] = dict(counts)
        constraint_overrides["llm_token_confidence"] = float(confidence)
        metadata = {
            "adapter": "StructuredTextLLMAdapter",
            "vocabulary": list(self.vocabulary),
            "confidence": confidence,
            "token_count": len(tokens),
        }
        return LLMAdapterResult(
            proposal=proposal,
            constraint_overrides=constraint_overrides,
            metadata=metadata,
        )

