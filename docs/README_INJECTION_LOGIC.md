# Injection Logic & Orthogonal Noise

> **Ideally suited for structure-preserving exploration on custom manifolds.**

The "Injection Logic" refers to a specific, compact interception point within the `EnergyCoordinator`'s optimization loop. It allows for the injection of auxiliary signals (typically noise) that are **mathematically orthogonal** to the primary gradient descent direction.

This mechanism decouples **Exploration** (finding new configurations) from **Optimization** (minimizing energy), ensuring that exploration moves *along* iso-energy contours rather than fighting the gradient descent process.

## The Core Concept

In a standard gradient descent step:
$$ \eta_{t+1} = \eta_t - \alpha \nabla F(\eta_t) $$

If we simply add random noise $\xi$:
$$ \eta_{t+1} = \eta_t - \alpha \nabla F(\eta_t) + \xi $$
...we risk increasing the energy arbitrarily, effectively fighting the optimization we just performed.

**Orthogonal Injection** ensures the noise $\xi_{\perp}$ satisfies:
$$ \xi_{\perp} \cdot \nabla F = 0 $$

This means the noise moves the system sideways along the "walls" of the energy valley, exploring alternative configurations that have the *same* energy cost, without climbing uphill.

---

## The "Clean 20 Lines"

The logic resides in `core/coordinator.py`, inside the `relax_etas` loop. It intercepts the gradient calculation before the step is taken.

```python
# 1. Capture the current gradient direction
grad_vector = np.array(grads, dtype=float)

# 2. Determine Noise Magnitude (via Controller)
current_noise_mag = 0.0
if self.enable_orthogonal_noise:
    if controller is not None:
        # Controller decides "how much" based on energy drops/stagnation
        current_noise_mag = controller.step(grad_vector, ...)
    else:
        # Simple decay schedule
        current_noise_mag = self.noise_magnitude * (self.noise_schedule_decay ** iter_idx)

# 3. Generate and Project Noise (The Injection)
if current_noise_mag > 1e-9:
    raw_noise = np.random.normal(0, 1, size=grad_vector.shape)
    
    # Custom Manifold Logic happens here:
    if self.metric_aware_noise_controller and (self.metric_vector_product or self.metric_matrix):
        # Riemannian / Metric-aware Projection
        noise_vector = project_noise_metric_orthogonal(
            raw_noise,
            grad_vector,
            M=self.metric_matrix,
            Mv=self.metric_vector_product,
        )
    else:
        # Standard Euclidean Projection
        noise_vector = project_noise_orthogonal(raw_noise, grad_vector)
    
    # Scale to desired magnitude
    noise_norm = np.linalg.norm(noise_vector)
    if noise_norm > 1e-9:
        noise_vector = noise_vector * (current_noise_mag / noise_norm)
```

This compact block (approx 20 lines in source) is the entire interface for structural exploration.

---

## Adaptation for Custom Manifolds

The most powerful feature of this logic is the `project_noise_metric_orthogonal` function. It allows you to redefine what "orthogonal" means for your specific problem space.

Standard Euclidean orthogonality is:
$$ z \cdot g = 0 $$

For a **Riemannian Manifold** or a **Constraint Surface** defined by a metric tensor $M$, orthogonality is defined as:
$$ \langle z, g \rangle_M = z^T M g = 0 $$

### How to Tweak It

You do not need to rewrite the Coordinator. You simply provide the metric definition when initializing the `EnergyCoordinator`.

#### 1. Explicit Metric Matrix ($M$)
If your manifold has a constant or slowly changing metric tensor $M$:

```python
# M defines the curvature or constraints of your space
my_metric_matrix = np.array([[2.0, 0.5], [0.5, 1.0]])

coordinator = EnergyCoordinator(
    modules=...,
    couplings=...,
    metric_aware_noise_controller=True,
    metric_matrix=my_metric_matrix
)
```

#### 2. Metric-Vector Product ($M_v$)
If $M$ is too large to materialize (e.g., in high-dimensional systems) or changes dynamically, you can provide a callable that computes $M \cdot v$:

```python
def apply_manifold_metric(vector):
    # Apply custom geometry logic
    # e.g., stiff in dimension 0, loose in dimension 1
    result = vector.copy()
    result[0] *= 10.0 
    return result

coordinator = EnergyCoordinator(
    ...,
    metric_aware_noise_controller=True,
    metric_vector_product=apply_manifold_metric
)
```

The injection logic automatically uses this to ensure exploration respects the geometry of your problem.

---

## Does it ALWAYS have to be Orthogonal?

**No.** While the current implementation defaults to orthogonal noise to preserve monotonicity (preventing the system from climbing energy hills), the injection point is generic.

You can modify the logic to support **Thermal / Langevin Noise** (which allows hill-climbing for annealing) by simply skipping the projection step.

### Modifying for Thermal Noise (Langevin Dynamics)
If you want standard thermal noise:
$$ \eta_{t+1} = \eta_t - \nabla F + \sqrt{2T}\xi $$

You would adjust the injection block to skip projection:

```python
# Modified injection logic
if current_noise_mag > 1e-9:
    raw_noise = np.random.normal(0, 1, size=grad_vector.shape)
    
    if self.use_thermal_noise:  # <--- New Flag
        # Don't project! Just scale.
        noise_vector = raw_noise * current_noise_mag
    else:
        # Standard Orthogonal Logic
        noise_vector = project_noise_orthogonal(raw_noise, grad_vector)
        noise_vector = noise_vector * (current_noise_mag / np.linalg.norm(noise_vector))
```

### Why Default to Orthogonal?
1.  **Monotonicity Guarantee**: Orthogonal noise keeps $F(\eta_{t+1}) \approx F(\eta_t)$. It doesn't fight the optimization.
2.  **Efficient Search**: It forces the system to search the "width" of the solution space (the null space of the gradient) rather than just jittering up and down the energy walls.

---

## Separation of Concerns: Controller vs. Projector

The design separates **Magnitude** from **Direction**.

### 1. The Controller (Magnitude)
*   **Responsibility:** Decides *when* and *how much* to explore.
*   **Logic:** Uses `EnergyDropRatio` and `Backtrack` counts. If the system is stuck (low energy drop), it increases noise. If it is descending fast, it decreases noise.
*   **Component:** `OrthogonalNoiseController` (or your custom subclass).

### 2. The Projector (Direction)
*   **Responsibility:** Decides *where* exploration is legally allowed to go.
*   **Logic:** Uses `project_noise_metric_orthogonal`. It subtracts the gradient component from the random noise.
*   **Component:** `core/energy.py` projection functions.

---

Suppose your order parameters $\eta$ represent probabilities. The 
natural geometry is not Euclidean, but defined by the Fisher 
Information Matrix (Fisher-Rao metric).

To explore this space efficiently without violating the probability 
structure:

1.  Define $M$ as the Fisher Information Matrix for your distribution.
2.  Enable `metric_aware_noise_controller`.
3.  The injection logic will now generate noise that is orthogonal to 
the gradient in the **statistical manifold**, not just in the 
parameter array.

## Advanced Applications

Beyond standard optimization, this injection pattern supports several advanced use cases often missed in standard gradient descent frameworks.

### 1. Multi-Objective "Pareto Navigation"
If you have a primary objective $F_{primary}$ and a secondary objective $F_{secondary}$:
*   Use $F_{primary}$ for the main gradient descent.
*   Instead of random noise, **inject the negative gradient of $F_{secondary}$**.
*   **Crucial Step**: Project $-\nabla F_{secondary}$ to be **orthogonal** to $\nabla F_{primary}$.

**Result**: The system optimizes the secondary objective *strictly within the null space* of the primary objective. You "slide" along the iso-energy contours of $F_{primary}$ to find the best possible $F_{secondary}$ without degrading the primary goal.

### 2. Robotics: Null-Space Control
In robotics with redundant degrees of freedom (e.g., a 7-DOF arm reaching a point in 3D space):
*   **Primary Gradient**: Minimize distance of end-effector to target.
*   **Injection**: "Stay away from joint limits" or "Avoid obstacle".
*   **Projection**: Orthogonal to the primary Jacobian.

This allows the robot to reconfigure its "elbows" to avoid collisions while keeping the hand perfectly still on the target.

### 3. Symmetry Breaking & Saddle Points
In highly symmetric potentials (like a "Mexican Hat" potential), the gradient at the top is zero (or near zero).
*   Standard gradient descent stalls.
*   **Injection**: A specific symmetry-breaking vector (e.g., derived from an eigenvector of the Hessian) can be injected.
*   Even if the gradient is non-zero but small, orthogonal injection helps explore the "flat" directions around the saddle to find the steepest descent path more quickly.

### 4. Robustness & Sharpness-Awareness
You can inject "Adversarial Noise"—noise specifically calculated to maximize the energy increase (move toward the walls).
*   By continuously injecting orthogonal noise that pushes *towards* instability, you force the optimizer to find "wide" valleys where the energy doesn't rise sharply in orthogonal directions.
*   This is akin to finding "flat minima" which generalize better in machine learning contexts.

## Summary

The injection logic is a "plugin slot" for geometry. By intercepting the gradient step and applying a metric-aware projection, it allows the `EnergyCoordinator` to support:

*   **Riemannian Optimization**
*   **Constrained Exploration**
*   **Iso-Energy Sampling**
*   **Multi-Objective Prioritization**
*   **Null-Space Control**

All while maintaining the monotonicity guarantees of the primary gradient descent backbone.
