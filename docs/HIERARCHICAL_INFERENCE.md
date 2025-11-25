# Hierarchical Inference: From "Gist" to Detail

The **Hierarchical Inference Scaffold** enables the Homeostat to solve large-scale problems efficiently by first solving a simplified "coarse" version of the problem (families of modules) before committing computational resources to the detailed "fine" version (individual modules).

## Why We Need This

As the number of modules ($N$) grows, the cost of energy minimization increases significantly:
- **Coupling Complexity:** Pairwise interactions grow as $O(N^2)$.
- **Search Space:** The dimensionality of the energy landscape ($\eta \in \mathbb{R}^N$) creates vast flat regions and many local minima.
- **Cognitive Bottleneck:** Just as humans don't analyze every leaf on a tree when planning a route through a forest, the Homeostat shouldn't optimize every minute detail until it knows which "forest" (functional area) matters.

### The Cognitive Analogy: "Gist" First
This architecture mimics the **Global-to-Local** processing seen in biological vision and planning:
1.  **Gist (Coarse):** Rapidly identify active regions ("It's a forest scene", "Use math and logic modules").
2.  **Focus (Selection):** Direct attention to relevant details ("Look for a path", "Activate specific arithmetic operators").
3.  **Detail (Fine):** Process high-resolution information only where needed ("Step over this root", "Solve 2+2").

## Architectural Layers

### 1. The Coarse Level (Families)
Instead of optimizing $N$ individual $\eta$ parameters, we optimize $K$ **family masses** ($K \ll N$).
- **Families:** Logical groupings of modules (e.g., "Arithmetic", "Logic", "Retrieval").
- **Family Mass ($M_k$):** An aggregate activation level for the family (e.g., mean or sum of member $\eta$).
- **Coarse Energy:** A simplified energy function operating on masses:
  $$ E_{\text{coarse}} = \sum_k w_k M_k^2 + \lambda_{\text{consistency}} \cdot \text{Var}(M) $$
  This encourages the system to decide which *types* of processing are needed without getting bogged down in specifics.

### 2. The Selection Stage
Once the coarse energy is minimized, we use the resulting family masses to gate the fine-grained model.
- **Thresholding:** Activate families with mass $M_k > \tau$.
- **Top-K:** Select the top $k$ most promising modules within active families.
- **Result:** A reduced set of active indices $I_{\text{active}} \subset \{1, \dots, N\}$ where $|I_{\text{active}}| \ll N$.

### 3. The Fine Level (Full Physics)
We run the full `EnergyCoordinator` physics engine (couplings, constraints, noise) *only* on the selected modules $I_{\text{active}}$.
- This restores full precision and constraint satisfaction.
- The computational cost is now dominated by the size of the active set, not the full library.

## Usage Guide

### Grouping Modules
Create a `FamilyGrouping` to map modules to families.

```python
from core.hierarchy import FamilyGrouping, compute_coarse_energy, select_modules_by_families

# Define families for 100 modules
mapping = {i: "Math" if i < 50 else "Logic" for i in range(100)}
grouping = FamilyGrouping.from_mapping(mapping)
```

### Coarse Optimization
Use a lightweight optimizer (or the coordinator itself on aggregated proxies) to find optimal masses.

```python
# Hypothetical coarse optimization loop
etas = [0.5] * 100  # Initial state
# ... optimize etas to minimize coarse energy ...
total_E, breakdown = compute_coarse_energy(etas, grouping)
```

### Selection & Refinement
Prune the inactive modules and run the fine-grained solve.

```python
# Select active modules (e.g., families with mean activation > 0.4)
active_indices = select_modules_by_families(
    etas, 
    grouping, 
    family_mass_threshold=0.4
)

# Configure coordinator for just the active subset
fine_coordinator = EnergyCoordinator(
    modules=[all_modules[i] for i in active_indices],
    couplings=filter_couplings(all_couplings, active_indices),
    # ...
)
final_etas = fine_coordinator.relax_etas(...)
```

## Implementation Status
- **Core Scaffold:** `core/hierarchy.py` (Done)
  - `FamilyGrouping` class
  - `compute_coarse_energy`
  - `select_modules_by_families`
- **Tests:** `tests/test_hierarchical_scaffold.py` (Done)
- **Integration:** Ready for use in experiments (Next Phase).

