# The Constraint Dictionary: Runtime Control Plane

The **Constraint Dictionary** (`constraints`) is the primary mechanism for injecting dynamic context, parameters, and rules into the `EnergyCoordinator` and its modules. It transforms the system from a static solver into a dynamic, controllable machine.

---

## 1. What is it?

It is a simple Python dictionary (`Dict[str, Any]`) passed to `relax_etas`. It acts as a **shared blackboard** that every module, coupling, and the coordinator itself can read from during the energy evaluation loop.

```python
constraints = {
    "term_weights": {"local:MyModule": 1.5},  # Coordinator reads this
    "target_value": 0.8,                      # Modules read this
    "reference_etas": [0.5, 0.5],             # Metrics logger reads this
}
coord.relax_etas(etas0, constraints=constraints)
```

## 2. Why use it?

### 2.1 Decoupling Logic from Configuration
You don't need to bake parameters into your module classes.
-   **Bad:** `Module(target=0.5)` (Hardcoded)
-   **Good:** `Module()` reads `constraints["target"]` (Dynamic)

### 2.2 Runtime Adaptation
You can change the rules of the universe *between steps* without rebuilding the graph.
-   **Curriculum Learning:** Start with weak constraints, tighten them over time.
-   **Meta-Learning:** Let an outer loop (RL, Gradient) update weights in the dictionary.
-   **Context Injection:** Pass data-dependent targets (e.g., from a user prompt) at inference time.

---

## 3. Standard Keys & Schema

While you can define any key, the framework reserves specific keys for core functionality.

### 3.1 Global Control (Coordinator)
| Key | Type | Description |
| :--- | :--- | :--- |
| `term_weights` | `Dict[str, float]` | Multipliers for specific energy terms. Overrides base weights. <br> Format: `{"local:ClassName": w, "coup:ClassName": w}` |
| `redundancy_rho` | `float` | Used by confidence logging to compute certainty. |

### 3.2 Observability & Metrics
| Key | Type | Description |
| :--- | :--- | :--- |
| `reference_etas` | `List[float]` | Ground-truth trajectory. If present, logger computes `info:alignment` and `info:drift`. |
| `constraint_violation_count` | `int` | Number of violated constraints (for `info:constraint_violation_rate`). |
| `total_constraints_checked` | `int` | Total constraints checked. |

### 3.3 Module-Specific (Conventions)
Modules define their own keys. Check `docs/README_MODULES.md` for details on specific modules.

-   **Gating:** `gate_alpha`, `gate_beta` (softness control).
-   **Compression:** `compression_target` (desired ratio).
-   **Sequence:** `monotonicity_tolerance`.

---

## 4. Usage Patterns

### Pattern A: Static Configuration
Set it once for a specific task.

```python
task_constraints = {
    "target_value": 0.9,
    "term_weights": {"coup:QuadraticCoupling": 2.0} # Strong coupling for this task
}
coord.relax_etas(etas0, constraints=task_constraints)
```

### Pattern B: Dynamic Schedule (Curriculum)
Tighten constraints over time.

```python
for step in range(10):
    # Increase penalty over time
    constraints = {"complexity_penalty": 0.1 * step}
    coord.relax_etas(etas, steps=10, constraints=constraints)
```

### Pattern C: Data-Driven Context
Inject external data (e.g., from a sensor or LLM) as a constraint.

```python
def solve_for_input(user_input: float):
    # The input defines the target
    constraints = {"target_value": user_input}
    return coord.relax_etas(etas0, constraints=constraints)
```

---

## 5. Advanced: Interactions with Adapters

If you use a `WeightAdapter` (like `GradNorm` or `SmallGain`), it effectively **writes** to the `term_weights` section of the constraint dictionary internally.

1.  Coordinator initializes with your `constraints`.
2.  Adapter observes gradients.
3.  Adapter computes new weights.
4.  Coordinator updates its internal view of `term_weights`.

*Note: Explicitly passed constraints usually take precedence or act as baselines depending on the adapter implementation.*

---

## 6. Best Practices

1.  **Namespace your keys:** If writing a custom module, prefix keys to avoid collisions (e.g., `mymod_target` instead of just `target`).
2.  **Provide defaults:** Your `local_energy` method should handle missing keys gracefully.
    ```python
    def local_energy(self, eta, constraints):
        target = constraints.get("mymod_target", 0.5) # Default to 0.5
        # ...
    ```
3.  **Log your constraints:** When debugging, print the `constraints` dict to ensure the system is solving the problem you *think* it is.

