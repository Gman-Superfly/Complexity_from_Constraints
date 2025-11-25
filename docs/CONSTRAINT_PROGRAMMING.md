# Constraint Programming: The Engine of Complexity

In the **Neuro-Symbolic Homeostat**, constraints are not merely checks to be passed; they are the **generative force** that drives the system. By defining what is *forbidden* or *costly*, we carve out the landscape in which intelligence and structure emerge.

This document serves as a comprehensive guide to programming, configuring, and understanding constraints within our framework. It is divided into two parts:

1.  **The Paradigm:** The general philosophy and "standard library" of energy-based constraints.
2.  **The Implementation:** Technical details, code examples, and debugging guides for this specific framework.

---

# Part 1: The Paradigm

## 1. The Shift: Declarative vs. Imperative

Traditional programming is **Imperative**: you tell the computer *how* to get to the solution ("Move left, then check X, then calculate Y").

Constraint Programming is **Declarative**: you define *what the solution looks like* (the constraints), and a solver finds the path.

In this framework, we take a specific approach: **Physics-Based Constraint Satisfaction**.
-   **Logic becomes Geometry**: A logical rule ("A implies B") becomes a shape in a high-dimensional energy landscape.
-   **Inference becomes Relaxation**: Finding a valid state is equivalent to a ball rolling downhill to the lowest energy point.
-   **Reasoning becomes Dynamics**: The path the system takes to the solution represents the "thought process."

### 1.1 Energy as the Common Language
Instead of writing imperative code, we define an **energy landscape**.
-   **Low Energy** = High Satisfaction of Constraints.
-   **High Energy** = Constraint Violation or Stress.
-   **Dynamics** = The system "relaxing" (optimizing) to find a configuration ($\eta$) that minimizes total energy.

$$ F_{\text{total}} = \sum F_{\text{local}} + \sum F_{\text{coupling}} + \sum F_{\text{global}} $$

### 1.2 Types of Constraints
In this framework (and we could argue unversally), constraints manifest in three primary forms:

1.  **Hard Constraints (Bounds & Invariants):**
    -   Absolute rules that *must* be obeyed at every step.
    -   Example: Order parameters must stay in range $\eta \in [0, 1]$.
    -   Enforced by: Projection, clipping, coordinate geometry.

2.  **Soft Constraints (Energy Penalties):**
    -   Preferences or rules that *should* be obeyed but can be traded off against others.
    -   Example: "Module A should align with Module B" or "Minimize complexity".
    -   Enforced by: Adding terms to the energy function (e.g., quadratic cost $k(\eta_A - \eta_B)^2$). Violations increase energy, creating a "force" to correct them.

3.  **Information Constraints (Entropy & Information Structure):**
    -   Constraints on the *distribution* or *quality* of the state.
    -   Example: "Maintain sufficient entropy" (exploration) or "Minimize drift from reference" (stability).
    -   Enforced by: Free-energy functionals ($F = U - TS$), meta-learning adapters.

### 1.3 Why "Complexity from Constraints"?
Complex behavior often looks like magic, but it usually emerges from simple parts obeying strict local rules.
-   **Biology**: Cells don't know the "plan" for a human; they just obey local chemical constraints. The human emerges.
-   **Physics**: Atoms don't know thermodynamics; they obey conservation laws. Temperature emerges.
-   **This Framework**: Modules don't know the "task"; they obey local energy constraints. Intelligence emerges.

---

## 2. The "Standard Library" of Constraints

While you can invent any constraint, we have identified five recurring patterns (the "Five Equations") that act as the fundamental building blocks of neuro-symbolic energy landscapes.

### 2.1 Local Identity (The Landau Potential)
*Constraint: "Be yourself, but be decisive."*

A module should usually be in a clear state (0 or 1, active or inactive), not stuck in indecision.
-   **Equation**: $F(\eta) = a\eta^2 + b\eta^4$
-   **Effect**: Creates "bins" or stable states. Forces decision-making.

### 2.2 Connection & Consistency (The Quadratic/Hinge)
*Constraint: "Agree with your neighbors."*

-   **Equality**: "State A should match State B".
    -   $E = w(\eta_A - \eta_B)^2$
-   **Implication**: "If A is true, B must be true" (Soft Logic).
    -   $E = w \cdot \max(0, \alpha \eta_A - \beta \eta_B)$

### 2.3 Redemption (The "Wormhole")
*Constraint: "The future matters more than the past."*

This is a special non-local constraint. It connects a "gate" (a decision to act) with the *benefit* of that action, even if the action hasn't happened yet.
-   **Effect**: Teleports gradient information from the future (benefit) to the present (gate), solving the "Zero-Gradient Problem." It allows the system to make choices based on potential outcomes.

### 2.4 Parsimony (Complexity Penalty)
*Constraint: "Do less."*

-   **Equation**: $E = \gamma \cdot \sum \text{ActiveModules}$
-   **Effect**: Forces the system to solve the problem using the minimum necessary structure. This is Occam's Razor encoded as a force.

### 2.5 Stability (The Control Limit)
*Constraint: "Don't explode."*

-   **Mechanism**: Bounded spectral radius (Small-Gain Theorem).
-   **Effect**: Ensures that feedback loops between constraints dampen out rather than amplifying into chaos.

---

## 3. Designing Constraint Landscapes

Effective constraint programming requires designing landscapes that are **solvable** and **informative**. This is the art of "Feng Shui for Optimizers."

### 3.1 Convexity vs. Richness
-   **Convex Landscape**: Bowl-shaped. Easy to solve (single global minimum), but limited expressivity. Good for simple regulation.
-   **Non-Convex Landscape**: Rugged, with many hills and valleys. Harder to solve (local minima), but required for **memory** (stable states are memories) and **choice** (choosing a path).
    -   *Strategy:* Use **noise** (Orthogonal Exploration) and **annealing** (Temperature) to navigate non-convexity.

### 3.2 Frustration
**Frustration** occurs when not all constraints can be satisfied simultaneously. This is a *feature*, not a bug.
-   Frustration drives the system to find compromise solutions.
-   High energy (unresolved frustration) signals "uncertainty" or "anomaly," which can trigger:
    -   **Expansion:** Add more modules to resolve the conflict.
    -   **Re-weighting:** Change priorities (Meta-learning).
    -   **Attention:** Focus resources on the high-energy area.

### 3.3 Hierarchy of Constraints
Constraints should be layered:
1.  **Hard Safety Bounds:** (e.g., $0 \le \eta \le 1$). Never violated.
2.  **Strong Functional Rules:** (e.g., "Output must match input schema"). High energy penalty.
3.  **Weak Heuristics:** (e.g., "Prefer sparse activations"). Low energy penalty.

---

# Part 2: The Implementation

This section covers how to implement constraints specifically using the `Complexity_from_Constraints` library.

## 4. Technical Implementation

The `EnergyCoordinator` acts as the central constraint solver (or "homeostat"). It orchestrates the relaxation process where all constraints compete and cooperate.

### 4.1 The Constraint Dictionary
Every module and coupling receives a `constraints` dictionary during energy computation. This is the primary mechanism for passing dynamic parameters and global context.

```python
# Example constraints dictionary passed to relax_etas
constraints = {
    "target_value": 0.8,              # Data constraint for a specific task
    "complexity_penalty": 0.1,        # Global soft constraint weight
    "reference_etas": [0.5, 0.5],     # Reference trajectory for drift/alignment
    "constraint_violation_rate": 0,   # External validator feedback
}
```

### 4.2 Local Constraints (The "Self")
Defined by `EnergyModule.local_energy(eta, constraints)`.
These constraints dictate the internal preferences of a module.

-   **Example:** A "Gating" module might have a bistable energy function ($F = \eta^2(1-\eta)^2$) that constrains it to be either "open" (1) or "closed" (0), resisting ambiguous states (0.5).
-   **Example:** A "Polynomial" module might constrain its state to match a target value derived from its inputs.

### 4.3 Coupling Constraints (The "Society")
Defined by `EnergyCoupling.coupling_energy(eta_i, eta_j, constraints)`.
These constraints dictate relationships *between* modules.

-   **Quadratic Coupling:** Enforces similarity ($\eta_i \approx \eta_j$).
    -   $E = w \cdot (\eta_i - \eta_j)^2$
-   **Hinge Coupling:** Enforces directional relationships (e.g., "If $i$ is high, $j$ must be high").
    -   $E = w \cdot \max(0, \alpha \eta_i - \beta \eta_j)$
-   **Gate Benefit Coupling:** Enforces economic relationships ("Open the gate only if it yields a benefit").
    -   $E = -w \cdot \eta_{\text{gate}} \cdot \text{Gain}_{\text{domain}}$

### 4.4 Global Constraints & Meta-Constraints
Handled by the Coordinator and Adapters.

-   **Budget Constraints:** `EnergyBudgetTracker` monitors total "spend".
-   **Stability Constraints:** `SmallGainWeightAdapter` dynamically adjusts coupling weights to prevent runaway feedback loops (instability).
-   **Monotonicity Constraints:** The `Free-Energy Guard` enforces that the system trajectory improves over time ($\Delta F < 0$).

---

## 5. Practical Guide: How to Add a New Constraint

### Scenario: Enforcing "Sparsity" (Only a few modules should be active)

**Method A: Local Energy Modification**
Add a term to `local_energy` in your module that penalizes non-zero activation (L1 regularization).
```python
def local_energy(self, eta, constraints):
    # ... existing energy ...
    sparsity_weight = constraints.get("sparsity_weight", 0.1)
    return base_energy + sparsity_weight * abs(eta)
```

**Method B: Global Constraint via Meta-Learning**
Use a `WeightAdapter` to dynamically increase penalties on modules that are chronically active but contribute little to the task.

**Method C: Hard Selection (Top-K)**
Use the **Hierarchical Scaffold** (`core/hierarchy.py`) to strictly limit activity to the top-K modules before running fine-grained relaxation. This transforms a soft energy pressure into a hard structural constraint.

---

## 6. Debugging Constraints

When the system misbehaves, it is usually a constraint conflict.

1.  **Energy Explosion:** Energies go to infinity or NaN.
    -   *Cause:* Incompatible hard constraints or runaway positive feedback.
    -   *Fix:* Check `SmallGain` stability limits; enable `normalize_grads`.
2.  **Frozen State:** System refuses to change despite inputs.
    -   *Cause:* "Stiffness" (Curvature) is too high; constraints are too rigid.
    -   *Fix:* Increase Temperature (T) or Noise; reduce coupling weights.
3.  **Constraint Violation:** System settles in a low-energy state that violates external truth.
    -   *Cause:* The "truth" wasn't encoded as a constraint (or was weighted too low).
    -   *Fix:* Add a specific energy term for the violated rule; increase its weight.

---

## 7. Constraint Violation Rate (h): Universal pattern and extensibility

It’s a universal pattern, not a universal truth. “Constraint violation rate (h)” is intentionally neutral (violations/checked) so it fits any domain, but the meaning comes from your task’s constraint checker.

Why it works here:
- Minimal contract: expose `constraint_violation_count` and `total_constraints_checked`; we just log `info:constraint_violation_rate`.
- Optional and modular: if you don’t supply counts, nothing is logged; there’s no coupling to your domain.
- Provenance-ready: IDs/offsets/KG triples let any domain define “violation” precisely. See 7.1 Origin and provenance or: `docs/INFORMATION_METRICS.md`.

When to program your own:
- Always do so if your domain has richer constraints. You can add typed counters (e.g., `constraint_violation_count:policy`, `:schema`, `:factual`) and log multiple rates side-by-side.
- Weight or window as needed (severity weights, rolling windows). The logger will carry any extra columns you include.
- Keep our h as a basic KPI; add your domain metrics alongside it.

Practical guidance (event-driven, reactive system):
1) Define constraints and a verifier for your task; compute per-step violation counts and totals as your run evolves.
2) Emit in `coord.constraints` before (or during) relaxation:
   - `constraint_violation_count`, `total_constraints_checked`
   - Optionally: `constraint_violation_count:<type>`, `total_constraints_checked:<type>`
3) The system is event-driven/reactive: loggers observe the accepted-step stream and record what you provide without imposing domain assumptions. The dashboard and CSVs will include your columns immediately; add panels if desired.

In short: h is a good default pattern (count/total + provenance). The system is designed so you can program your own constraints and log additional, domain-specific metrics next to h.

### 7.1 Origin and provenance (Information Structures)
- This metric originates from the Information Structures research where five metrics are instrumented together:
  - ρ (redundancy), a (alignment), Δ (drift), E (entropy), and h (constraint violation rate).
- In that context, h was introduced as a span-level error rate for LLM evaluation:
  - “Unsupported or contradicted factual spans, measured with provenance (document IDs/offsets or KG triples).”
- Span-level calculation (for a window or episode):
  - Count every answer span that the verifier marks as unsupported or contradicted (with concrete provenance),
  - Divide by the total checked spans.
  - This yields a “channel quality” indicator complementary to ρ and a; it is distinct from thermodynamic entropy.
- Provenance primitives (for auditable evidence):
  - Document IDs: unique identifiers for each evidence source (e.g., `trial_2021_phase3.pdf`).
  - Offsets: byte/character/token ranges pointing to the exact supporting span (e.g., `2315–2420`).
  - KG Triples: subject–predicate–object facts `(DrugX, reduces, HbA1c)` with optional provenance metadata.

## 8. Summary

In the Homeostat, **"Programming" = "Defining Energy Functions."**

You do not write the solution. You write the **constraints** that define the *shape* of the problem, and the physics engine (Coordinator) finds the solution. This shift from imperative to declarative/physical programming is the key to scalable, robust, neuro-symbolic intelligence.

---

## 9. Epilogue: Defining the Unknown

> *"When people want to describe complex things, they find it easier to describe what something isn't rather than what something is."*

This framework embraces that reality. In scientific discovery and complex problem solving, we often cannot fully define the solution (the "what is"). We haven't internalized it yet; we don't have the names for it.

But we can almost always define what the solution **isn't**.
-   It isn't high energy.
-   It isn't contradictory (Logic).
-   It isn't unstable (Control).
-   It isn't random (Information).

By rigorously programming these "isn'ts" as constraints, we create a mold. We then pour energy into the system, and the solution—the "what is"—emerges as the shape that fills the void we created. We give names to it only *after* it has been defined by its boundaries.
