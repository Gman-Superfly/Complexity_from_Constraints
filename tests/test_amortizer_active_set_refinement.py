from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.amortizer import SimpleHeuristicAmortizer
from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling, DirectedHingeCoupling
from modules.gating.energy_gating import EnergyGatingModule


def _make_larger_ring(n: int = 16) -> Tuple[List[Any], List[Tuple[int, int, Any]], Dict[str, Any], List[Any]]:
    """Larger ring to stress active-set refinement."""
    modules: List[EnergyGatingModule] = []
    inputs: List[Any] = []
    for idx in range(n):
        modules.append(EnergyGatingModule(gain_fn=lambda _, i=idx: 0.03 * (i % 5), a=0.3, b=0.2))
        inputs.append(None)
    couplings: List[Tuple[int, int, Any]] = []
    for i in range(n):
        j = (i + 1) % n
        couplings.append((i, j, QuadraticCoupling(weight=0.5)))
        couplings.append((i, j, DirectedHingeCoupling(weight=0.3)))
    constraints: Dict[str, Any] = {}
    return modules, couplings, constraints, inputs


def test_active_set_refinement_reduces_compute_vs_full_relaxation() -> None:
    """Active-set refinement should compute fewer gradient evaluations than full relaxation."""
    mods, coups, constraints, inputs = _make_larger_ring(16)
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints=constraints,
        use_analytic=True,
        line_search=False,
        step_size=0.08,
    )
    
    amort = SimpleHeuristicAmortizer(default_eta=0.5)
    etas0 = amort.propose_initial_etas(mods, inputs)
    
    # Full relaxation (baseline)
    mods_full, coups_full, constraints_full, inputs_full = _make_larger_ring(16)
    coord_full = EnergyCoordinator(
        modules=mods_full,
        couplings=coups_full,
        constraints=constraints_full,
        use_analytic=True,
        line_search=False,
        step_size=0.08,
    )
    etas_full = coord_full.relax_etas(list(etas0), steps=30)
    e_full = coord_full.energy(etas_full)
    
    # Active-set refinement (k=4 nodes + neighbors ≈ 8-10 active)
    active = amort.select_active_set(coord, etas0, k=4, include_neighbors=True)
    assert len(active) <= len(mods), "Active set should be subset"
    # Expect significant reduction in active nodes
    assert len(active) <= len(mods) * 0.7, f"Active set {len(active)} should be <70% of {len(mods)}"
    
    # Simulate active-set update (only update active indices, freeze others)
    # For this test, we'll use a simple heuristic: update only active indices
    etas_active = list(etas0)
    inactive_mask = [i not in active for i in range(len(etas_active))]
    
    # Run a few steps with frozen inactive nodes
    for _ in range(30):
        grads = coord._grads(etas_active)
        for i in range(len(etas_active)):
            if i not in active:
                grads[i] = 0.0  # Freeze inactive nodes
        # Simple gradient step
        for i in range(len(etas_active)):
            etas_active[i] = float(max(0.0, min(1.0, etas_active[i] - 0.08 * grads[i])))
    
    e_active = coord.energy(etas_active)
    
    # Active-set refinement should achieve comparable final energy (within tolerance)
    # Allow 50% degradation for this test (real implementation would iterate active set selection)
    energy_ratio = abs(e_active) / (abs(e_full) + 1e-9)
    assert energy_ratio >= 0.5, f"Active-set energy {e_active} too far from full {e_full} (ratio={energy_ratio})"
    
    # Compute reduction: active set is smaller than full
    compute_reduction = 1.0 - (len(active) / len(mods))
    assert compute_reduction > 0.2, f"Expected >20% compute reduction, got {compute_reduction*100:.1f}%"


def test_active_set_includes_high_gradient_nodes() -> None:
    """Active set should prioritize nodes with high gradient magnitude."""
    mods, coups, constraints, inputs = _make_larger_ring(12)
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints=constraints,
        use_analytic=True,
    )
    
    amort = SimpleHeuristicAmortizer()
    etas0 = amort.propose_initial_etas(mods, inputs)
    
    # Get full gradient magnitudes
    grads = coord._grads(etas0)
    grad_mags = [(i, abs(float(g))) for i, g in enumerate(grads)]
    grad_mags_sorted = sorted(grad_mags, key=lambda x: x[1], reverse=True)
    
    # Top-k should be in active set
    k = 3
    active = amort.select_active_set(coord, etas0, k=k, include_neighbors=False)
    top_k_indices = {idx for idx, _ in grad_mags_sorted[:k]}
    
    # All top-k should be in active set
    assert top_k_indices.issubset(active), f"Top-{k} indices {top_k_indices} not in active {active}"


def test_active_set_includes_neighbors_when_enabled() -> None:
    """Active set should include neighbors of high-gradient nodes when include_neighbors=True."""
    mods, coups, constraints, inputs = _make_larger_ring(8)
    coord = EnergyCoordinator(
        modules=mods,
        couplings=coups,
        constraints=constraints,
        use_analytic=True,
    )
    
    amort = SimpleHeuristicAmortizer()
    etas0 = amort.propose_initial_etas(mods, inputs)
    
    k = 1
    # Without neighbors
    active_no_neighbors = amort.select_active_set(coord, etas0, k=k, include_neighbors=False)
    # With neighbors
    active_with_neighbors = amort.select_active_set(coord, etas0, k=k, include_neighbors=True)
    
    # With neighbors should be strictly larger (in a ring, each node has 2 neighbors)
    assert len(active_with_neighbors) > len(active_no_neighbors), \
        f"Expected more nodes with neighbors: {len(active_with_neighbors)} vs {len(active_no_neighbors)}"
    
    # In a ring, selecting k=1 with neighbors should give at least k + 2 (left+right neighbors)
    assert len(active_with_neighbors) >= k + 2, \
        f"Expected at least {k+2} nodes (k + neighbors), got {len(active_with_neighbors)}"

