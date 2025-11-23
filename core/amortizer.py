from __future__ import annotations

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from core.coordinator import EnergyCoordinator

OrderParameter = float


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


class AmortizedProposal(Protocol):
    """Interface for hierarchical/amortized inference scaffolding."""

    def propose_initial_etas(self, modules: Sequence[Any], inputs: Sequence[object]) -> List[float]:
        """Return an initial eta vector in [0,1] for the given modules/inputs."""
        ...

    def select_active_set(
        self,
        coord: EnergyCoordinator,
        etas: Sequence[float],
        k: int,
        include_neighbors: bool = True,
        fd_eps: float = 1e-5,
    ) -> Set[int]:
        """Return indices of an active set to refine (top-|grad| + optional neighbors)."""
        ...

    def plan_stage_execution(
        self,
        coord: EnergyCoordinator,
        etas: Sequence[float],
        stages: Sequence[Tuple[str, int]],
        include_neighbors: bool = True,
        fd_eps: float = 1e-5,
    ) -> List["ActiveStagePlan"]:
        """Return ordered stage plans with explicit budgets."""
        ...


@dataclass
class WarmStartProposal:
    """Container for warm-start η₀ plus observability metadata."""

    etas: List[float]
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WarmStartProposer(Protocol):
    """Protocol for learned or heuristic warm-start proposers."""

    def propose(
        self,
        modules: Sequence[Any],
        inputs: Sequence[object],
        hints: Optional[Mapping[str, Any]] = None,
    ) -> WarmStartProposal:
        ...

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


@dataclass
class WarmStartResult:
    """Summary of a warm-start + truncated relaxation run."""

    proposal: WarmStartProposal
    final_etas: List[float]
    initial_energy: float
    final_energy: float
    steps_requested: int
    relaxation_metrics: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _ensure_numpy(array_like: Iterable[float], dtype: np.dtype = float) -> np.ndarray:
    arr = np.asarray(list(array_like), dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return default
    if isinstance(value, Mapping) and "value" in value:
        return _safe_float(value["value"], default=default)
    return default


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class MLPWarmStartProposer(WarmStartProposer):
    """Tiny numpy MLP proposer with optional custom feature extractor."""

    input_dim: int
    hidden_dim: int = 32
    activation: str = "relu"
    feature_extractor: Optional[Callable[[Any, Mapping[str, Any]], Sequence[float]]] = None
    seed: Optional[int] = None

    _w1: np.ndarray = field(init=False, repr=False)
    _b1: np.ndarray = field(init=False, repr=False)
    _w2: np.ndarray = field(init=False, repr=False)
    _b2: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        rng = np.random.default_rng(self.seed)
        limit1 = 1.0 / math.sqrt(self.input_dim)
        self._w1 = rng.uniform(-limit1, limit1, size=(self.hidden_dim, self.input_dim))
        self._b1 = np.zeros(self.hidden_dim, dtype=float)
        limit2 = 1.0 / math.sqrt(self.hidden_dim)
        self._w2 = rng.uniform(-limit2, limit2, size=(1, self.hidden_dim))
        self._b2 = np.zeros(1, dtype=float)

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "gelu":
            return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))
        # default relu
        return np.maximum(0.0, x)

    def _features(self, raw_input: Any, hints: Mapping[str, Any]) -> np.ndarray:
        if self.feature_extractor is not None:
            feats = list(self.feature_extractor(raw_input, hints))
        else:
            feats = []
            if isinstance(raw_input, Mapping):
                for key in sorted(raw_input.keys()):
                    feats.append(_safe_float(raw_input[key], default=0.0))
            elif isinstance(raw_input, Sequence) and not isinstance(raw_input, (str, bytes)):
                feats = [_safe_float(v) for v in raw_input]
            else:
                feats = [_safe_float(raw_input)]
        if len(feats) < self.input_dim:
            feats = feats + [0.0] * (self.input_dim - len(feats))
        return _ensure_numpy(feats[: self.input_dim])

    def propose(
        self,
        modules: Sequence[Any],
        inputs: Sequence[object],
        hints: Optional[Mapping[str, Any]] = None,
    ) -> WarmStartProposal:
        assert len(modules) == len(inputs), "modules/inputs length mismatch"
        hints = hints or {}
        etas: List[float] = []
        confidences: List[float] = []
        for raw_input in inputs:
            feats = self._features(raw_input, hints)
            hidden = self._act(self._w1 @ feats + self._b1)
            logits_arr = (self._w2 @ hidden) + self._b2
            logits = float(logits_arr.item()) if hasattr(logits_arr, 'item') else float(logits_arr)
            eta = _clamp01(float(_sigmoid(np.asarray([logits]))[0]))
            etas.append(eta)
            confidences.append(1.0 - abs(eta - 0.5) * 2.0)
        confidence = float(np.clip(np.mean(confidences), 0.0, 1.0))
        metadata = {
            "proposer": "MLPWarmStartProposer",
            "activation": self.activation,
            "hidden_dim": self.hidden_dim,
            "confidence_trace": confidences,
        }
        return WarmStartProposal(etas=etas, confidence=confidence, metadata=metadata)

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "activation": self.activation,
            },
            "w1": self._w1.tolist(),
            "b1": self._b1.tolist(),
            "w2": self._w2.tolist(),
            "b2": self._b2.tolist(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assert "w1" in state and "w2" in state, "state missing weights"
        self._w1 = np.asarray(state["w1"], dtype=float)
        self._b1 = np.asarray(state.get("b1", [0.0] * self._w1.shape[0]), dtype=float)
        self._w2 = np.asarray(state["w2"], dtype=float)
        self._b2 = np.asarray(state.get("b2", [0.0]), dtype=float)

    def save(self, path: str | Path) -> None:
        payload = dict(self.state_dict())
        path = Path(path)
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MLPWarmStartProposer":
        path = Path(path)
        state = json.loads(path.read_text())
        config = state.get("config", {})
        proposer = cls(
            input_dim=int(config.get("input_dim", state["w1"] and len(state["w1"][0]) or 1)),
            hidden_dim=int(config.get("hidden_dim", len(state["w1"]))),
            activation=str(config.get("activation", "relu")),
        )
        proposer.load_state_dict(state)
        return proposer


@dataclass
class ActiveStagePlan:
    """Structured execution plan for active-set refinement."""

    name: str
    active_indices: List[int]
    budget_fraction: float


@dataclass
class _ActiveSetCacheEntry:
    signature: np.ndarray
    etas: List[float]
    active_indices: Set[int]


@dataclass
class CachedActiveSetAmortizer(AmortizedProposal):
    """Amortizer with similarity-based cache and stage planning."""

    default_eta: float = 0.5
    cache_size: int = 32
    similarity_threshold: float = 0.05
    feature_extractor: Optional[
        Callable[[Sequence[Any], Sequence[object], Mapping[str, Any]], Sequence[float]]
    ] = None

    _cache: Deque[_ActiveSetCacheEntry] = field(default_factory=deque, init=False, repr=False)
    _last_signature: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def _signature(
        self,
        modules: Sequence[Any],
        inputs: Sequence[object],
        hints: Mapping[str, Any],
    ) -> np.ndarray:
        if self.feature_extractor is not None:
            feats = list(self.feature_extractor(modules, inputs, hints))
        else:
            feats: List[float] = []
            for module, raw_input in zip(modules, inputs):
                feats.append(_safe_float(getattr(module, "default_eta", self.default_eta)))
                if isinstance(raw_input, Mapping):
                    feats.append(_safe_float(raw_input.get("normalized_count", 0.0)))
                    feats.append(_safe_float(raw_input.get("count", 0.0)))
                else:
                    feats.append(_safe_float(raw_input))
            feats.append(_safe_float(hints.get("temperature", 1.0)))
        arr = _ensure_numpy(feats)
        norm = np.linalg.norm(arr)
        if norm > 0.0:
            arr = arr / norm
        return arr

    def _lookup(self, signature: np.ndarray) -> Optional[_ActiveSetCacheEntry]:
        for entry in self._cache:
            if entry.signature.shape == signature.shape:
                dist = float(np.linalg.norm(entry.signature - signature))
                if dist <= self.similarity_threshold:
                    return entry
        return None

    def _remember(self, entry: _ActiveSetCacheEntry) -> None:
        self._cache.appendleft(entry)
        while len(self._cache) > max(1, self.cache_size):
            self._cache.pop()

    def propose_initial_etas(self, modules: Sequence[Any], inputs: Sequence[object]) -> List[float]:
        hints: Mapping[str, Any] = {}
        signature = self._signature(modules, inputs, hints)
        self._last_signature = signature
        cached = self._lookup(signature)
        if cached is not None:
            return list(cached.etas)
        etas: List[float] = []
        for module, raw_input in zip(modules, inputs):
            try:
                eta = float(module.compute_eta(raw_input))  # type: ignore[attr-defined]
            except Exception:
                eta = float(self.default_eta)
            etas.append(_clamp01(eta))
        self._remember(_ActiveSetCacheEntry(signature=signature, etas=list(etas), active_indices=set()))
        return etas

    def select_active_set(
        self,
        coord: EnergyCoordinator,
        etas: Sequence[float],
        k: int,
        include_neighbors: bool = True,
        fd_eps: float = 1e-5,
    ) -> Set[int]:
        assert k > 0, "k must be positive"
        grads = _numeric_gradients(coord, list(etas), eps=fd_eps)
        order = sorted(range(len(etas)), key=lambda idx: abs(float(grads[idx])), reverse=True)
        chosen: Set[int] = set(order[: min(k, len(order))])
        if include_neighbors and coord.couplings:
            adj = _build_adjacency(coord.couplings, len(etas))
            for idx in list(chosen):
                chosen.update(adj[idx])
        if self._last_signature is not None:
            cached = self._lookup(self._last_signature)
            if cached is not None:
                cached.active_indices = set(chosen)
        return chosen

    def plan_stage_execution(
        self,
        coord: EnergyCoordinator,
        etas: Sequence[float],
        stages: Sequence[Tuple[str, int]],
        include_neighbors: bool = True,
        fd_eps: float = 1e-5,
    ) -> List[ActiveStagePlan]:
        plans: List[ActiveStagePlan] = []
        total_budget = sum(max(1, k) for _name, k in stages)
        for name, k in stages:
            k_eff = max(1, k)
            active = self.select_active_set(coord, etas, k=k_eff, include_neighbors=include_neighbors, fd_eps=fd_eps)
            plans.append(
                ActiveStagePlan(
                    name=name,
                    active_indices=sorted(active),
                    budget_fraction=float(k_eff / total_budget) if total_budget > 0 else 1.0,
                )
            )
        return plans

    def cache_summary(self) -> Mapping[str, Any]:
        return {"entries": len(self._cache)}

    def clear_cache(self) -> None:
        self._cache.clear()
        self._last_signature = None


@dataclass
class SimpleHeuristicAmortizer(AmortizedProposal):
    """Baseline amortizer used when caching is unnecessary."""

    default_eta: float = 0.5

    def propose_initial_etas(self, modules: Sequence[Any], inputs: Sequence[object]) -> List[float]:
        assert len(modules) == len(inputs), "modules/inputs length mismatch"
        etas: List[float] = []
        for module, raw_input in zip(modules, inputs):
            try:
                eta = float(module.compute_eta(raw_input))  # type: ignore[attr-defined]
            except Exception:
                eta = float(self.default_eta)
            etas.append(_clamp01(eta))
        return etas

    def select_active_set(
        self,
        coord: EnergyCoordinator,
        etas: Sequence[float],
        k: int,
        include_neighbors: bool = True,
        fd_eps: float = 1e-5,
    ) -> Set[int]:
        assert k > 0, "k must be positive"
        n = len(etas)
        grads = _numeric_gradients(coord, list(etas), eps=fd_eps)
        ordering = sorted(range(n), key=lambda idx: abs(float(grads[idx])), reverse=True)
        chosen: Set[int] = set(ordering[: min(k, n)])
        if include_neighbors and coord.couplings:
            adjacency = _build_adjacency(coord.couplings, n)
            for idx in list(chosen):
                chosen.update(adjacency[idx])
        return chosen

    def plan_stage_execution(
        self,
        coord: EnergyCoordinator,
        etas: Sequence[float],
        stages: Sequence[Tuple[str, int]],
        include_neighbors: bool = True,
        fd_eps: float = 1e-5,
    ) -> List[ActiveStagePlan]:
        plans: List[ActiveStagePlan] = []
        total = sum(max(1, k) for _name, k in stages)
        if total <= 0:
            total = 1
        for name, k in stages:
            k_eff = max(1, k)
            indices = sorted(
                self.select_active_set(
                    coord,
                    etas,
                    k=k_eff,
                    include_neighbors=include_neighbors,
                    fd_eps=fd_eps,
                )
            )
            plans.append(
                ActiveStagePlan(
                    name=name,
                    active_indices=indices,
                    budget_fraction=float(k_eff / total),
                )
            )
        return plans


def run_warm_start_relaxation(
    coord: EnergyCoordinator,
    *,
    proposer: Optional[WarmStartProposer] = None,
    inputs: Optional[Sequence[object]] = None,
    hints: Optional[Mapping[str, Any]] = None,
    proposal: Optional[WarmStartProposal] = None,
    constraint_overrides: Optional[Mapping[str, Any]] = None,
    relax_steps: int = 10,
    enforce_stability_guard: bool = True,
) -> WarmStartResult:
    """Execute a warm-started truncated relaxation with optional constraint overrides."""

    assert (
        proposal is not None or (proposer is not None and inputs is not None)
    ), "Provide either an existing proposal or (proposer, inputs)"
    hints = hints or {}
    original_constraints = coord.constraints
    merged_constraints = dict(coord.constraints)
    if constraint_overrides:
        merged_constraints.update(constraint_overrides)
    coord.constraints = merged_constraints
    try:
        if proposal is None:
            assert proposer is not None and inputs is not None
            proposal = proposer.propose(coord.modules, inputs, hints)
        assert proposal is not None
        etas0 = list(proposal.etas)
        initial_energy = float(coord.energy(etas0))
        prev_guard = coord.stability_guard
        prev_log = coord.log_contraction_margin
        if enforce_stability_guard:
            coord.stability_guard = True
            coord.log_contraction_margin = True
        coord._contraction_margin_history = []  # type: ignore[attr-defined]
        final_etas = coord.relax_etas(list(etas0), steps=relax_steps)
        final_energy = float(coord.energy(list(final_etas)))
        metrics = coord.last_relaxation_metrics()
        result = WarmStartResult(
            proposal=proposal,
            final_etas=list(final_etas),
            initial_energy=initial_energy,
            final_energy=final_energy,
            steps_requested=int(relax_steps),
            relaxation_metrics=metrics,
            metadata={
                "constraint_overrides": list((constraint_overrides or {}).keys()),
                "enforce_stability_guard": bool(enforce_stability_guard),
            },
        )
        return result
    finally:
        coord.constraints = original_constraints
        if enforce_stability_guard:
            coord.stability_guard = prev_guard
            coord.log_contraction_margin = prev_log


def _build_adjacency(couplings: Sequence[Tuple[int, int, object]], num_nodes: int) -> List[Set[int]]:
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    for i, j, _ in couplings:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            adj[i].add(j)
            adj[j].add(i)
    return adj


def _numeric_gradients(coord: EnergyCoordinator, etas: Sequence[float], eps: float = 1e-5) -> List[float]:
    """Finite-difference estimate of dF/deta_i with simple forward differences and clamping."""
    base = float(coord.energy(list(etas)))
    grads: List[float] = []
    for i in range(len(etas)):
        step_vec = list(etas)
        step_vec[i] = float(min(1.0, max(0.0, step_vec[i] + eps)))
        val = float(coord.energy(step_vec))
        grads.append((val - base) / eps)
    return grads


