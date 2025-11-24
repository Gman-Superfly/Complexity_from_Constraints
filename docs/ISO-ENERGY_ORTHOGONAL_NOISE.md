# ISO-ENERGY_ORTHOGONAL_NOISE: Structure‑Preserving Noise Exploration for Energy Minimization, inspired by Normalized Dynamics OPT algo

**NOTE the ND OPT version is contains an optimized version of noise injection with extras for a specialist manifold learning project, a spiritual inspiration, check disambiguation and the repo link for details**

Terminology:
- We use the name Iso‑Energy Orthogonal Noise for explanations and abbreviate it as IEON when helpful.
- In the code, we refer to this simply as “orthogonal noise” for brevity (see `enable_orthogonal_noise`, `project_noise_orthogonal`, `OrthogonalNoiseController`).

This document presents Iso‑Energy Orthogonal Noise (IEON): a simple, testable recipe that combines unit‑gradient descent with tangent‑plane (orthogonal) noise to achieve iso‑energy exploration to first order. We place IEON within the broader free‑energy minimization view used in this repository, describe how it integrates with our gating and adaptive weighting (AGM), and provide guidance for practice and evaluation.

If you use this repository in your research, please cite it as below.

Authors:
- Oscar Goldman - Shogu research Group @ Datamutant.ai subsidiary of 温心重工業

## 1. Motivation
Gradient descent reduces energy efficiently when gradients are reliable and well‑conditioned. However, in flat yet anisotropic regions, near saddles, or under uncertain curvature, naive noise can increase energy and destabilize line search. IEON addresses this by:
- Using unit‑norm gradients (direction‑only flow) to define a local normal to the level set.
- Injecting noise in the orthogonal complement (tangent plane), which preserves energy to first order.

This pairing encourages exploration along iso‑energy contours while retaining descent directionality—useful for escaping poor local structure without paying first‑order energy penalties.

## 2. Method (Core Idea)
Let \(F(\eta)\) be the total energy and \(g = \nabla_\eta F(\eta)\). IEON uses:
1) Normalized descent direction \( \hat{g} = g / \lVert g \rVert \) (if \( \lVert g \rVert > 0 \)).
2) Tangent‑plane noise \( z_\perp = z - \frac{z^\top g}{\lVert g \rVert^2} g \) (if \( \lVert g \rVert > 0 \)).

Update (with optional line search and invariants):
\[
\eta_{t+1} \leftarrow \operatorname{Proj}_{[0,1]}\!\Big(\eta_t - \alpha \,\hat{g} + \sigma_t\, z_\perp \Big).
\]

Key property: \(z_\perp^\top g = 0\), so to first order the injected noise does not increase energy. This “structure‑preserving” exploration complements line search, homotopy, and adaptive weight schedules.

## 3. Algorithm (Pseudo‑Code)
```python
def orthogonal_noise_step(eta, grad, step_size, noise_std, project, line_search=None):
    # 1) Normalize gradient direction
    g_norm = np.linalg.norm(grad)
    if g_norm > 0.0:
        g_hat = grad / g_norm
    else:
        g_hat = np.zeros_like(grad)

    # 2) Orthogonal noise
    z = np.random.normal(size=grad.shape)
    if g_norm > 0.0:
        z = z - (np.dot(z, grad) / (g_norm ** 2)) * grad  # tangent-plane noise

    # 3) Trial update
    eta_trial = project(eta - step_size * g_hat + noise_std * z)

    # 4) Optional line search / acceptance
    if line_search is not None:
        return line_search(eta, eta_trial)
    return eta_trial
```

Practical notes:
- Use small but non‑zero `step_size`; `line_search=True` can still be helpful with normalized gradients.
- Noise magnitude `noise_std` can be adapted by a simple controller (increase when progress stalls, decrease when rate is high).
- Always clamp η to [0,1] (mirror/logit updates are also supported in this repo).

## 4. Theory and Intuition
- First‑order iso‑energy noise: Since \(z_\perp^\top g = 0\), the noise component does no work against the gradient to first order. Energy changes from noise arise only at higher order or through curvature.
- Level‑set geometry: The gradient defines the normal to the iso‑energy surface. Orthogonal noise explores within that surface (locally), aiding escape from narrow valleys and saddle neighborhoods without triggering line‑search backtracks as aggressively.
- Relation to natural gradient and manifold methods: Normalization by \(\lVert g \rVert\) loosely echoes conditioning ideas; in more formal setups, the Fisher metric yields natural gradients, and manifold MCMC (e.g., Riemann Manifold Langevin/HMC) injects geometry‑aware noise. This is why one of our algos is called "Normalized" Dynamics, here IEON is a lightweight, metric‑free specialization of this noise injection technique compatible with our energy modules.

## 5. Implementation in This Repository
IEON is available via simple toggles in `EnergyCoordinator`:
- Gradient normalization: `normalize_grads=True`
- Orthogonal noise injection: `enable_orthogonal_noise=True`
- Optional controller: `auto_noise_controller=True` (adapts noise magnitude when progress stalls)
- Guards: Works with `line_search=True`, stability caps, homotopy, and weight adapters.

### 5.1 Auto noise controller (signals and update rule)
- Goal: scale the orthogonal-noise magnitude based on stability/progress signals.
- Signals:
  - Descent rate: small relative energy drop → increase exploration
  - Backtracks: any line-search backtrack → increase exploration
  - Gradient rotation: large angle between successive gradients → increase exploration
- Update rule (conceptual):
  ```python
  # Inputs at step t:
  #   grad_t, grad_{t-1}, energy_drop_ratio, backtracks, iter_idx
  # Config:
  #   base_magnitude, decay, weights w_rate, w_backtrack, w_rotation
  rotation_signal = 0.5 * (1.0 - cos(grad_t, grad_{t-1}))      # in [0,1]
  rate_signal     = max(0.0, 1.0 - 10.0 * energy_drop_ratio)   # stall → 1, progress → 0
  backtrack_sig   = 1.0 if backtracks > 0 else 0.0
  raw_score = w_rate * rate_signal + w_backtrack * backtrack_sig + w_rotation * rotation_signal
  scale = clamp(raw_score, min_scale, max_scale)                # [0,1]
  noise_mag_t = base_magnitude * (decay ** iter_idx) * scale    # effective magnitude
  ```
- Enable by setting `enable_orthogonal_noise=True, auto_noise_controller=True` and choosing `noise_magnitude` (base) and `noise_schedule_decay`.
- First‑order property remains: noise is projected orthogonally to ∇F, so exploration is energy‑neutral to first order; controller only changes magnitude.

### 5.2 Metric‑aware variant (optional)
- Name (design): `metric_aware_noise_controller`. Use a problem‑specific metric \(M\) (e.g., Fisher for KL trust regions) to project/scale noise.
- Key distinctions:
  - IEON (Euclidean projection) is iso‑energy for \(F\): \( \nabla F^\top z_{\perp} = 0 \).
  - Metric‑aware variant is iso‑energy in \(M\): \( g^\top M\, z_{\perp,M} = 0 \). This is only approximately iso‑energy for \(F\) unless \(M\) matches the local geometry (e.g., Fisher for KL).
  - If you need strict iso‑energy for \(F\), keep Euclidean projection; use \(M\) only to scale magnitude.
- Availability rule of thumb:
  - Available if you can compute Fisher/metric info (logits/probs + reference + autograd → diag Fisher or Fv).
  - Otherwise fall back to Euclidean IEON with the auto controller.
- See `docs/METRIC_AWARE_NOISE_CONTROLLER.md` for formulas, availability checks, and caveats.

Code references:

```96:121:core/energy.py
def project_noise_orthogonal(
    noise: np.ndarray,
    grad: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """Project noise vector onto the subspace orthogonal to the gradient.
    
    z_orth = z - (z · g) * g / ||g||²
    
    This ensures exploration happens along the level sets of the energy function
    (iso-energy contours), avoiding ascent/descent directions.
    """
    grad_norm_sq = np.sum(grad * grad)
    
    if grad_norm_sq < eps:
        # Gradient is zero (at min/max/saddle) => all directions are valid
        return noise
        
    # Compute projection scalar: (z · g) / ||g||²
    projection_scalar = np.sum(noise * grad) / grad_norm_sq
    
    # Subtract component parallel to gradient
    noise_orth = noise - projection_scalar * grad
    
    return noise_orth
```

```305:313:core/coordinator.py
# optional normalization/clipping
if self.normalize_grads:
    norm = float(np.linalg.norm(np.asarray(grads, dtype=float)))
    if norm > 0.0:
        grads = [g / norm for g in grads]
```

```32:49:tests/test_orthogonal_noise.py
def test_coordinator_orthogonal_noise_integration() -> None:
    """Verify coordinator injects orthogonal noise in a non-degenerate (2D) case."""
    coord2 = EnergyCoordinator(
        modules=mods2,
        couplings=[],
        constraints={},
        enable_orthogonal_noise=True,
        noise_magnitude=0.5,
        noise_schedule_decay=1.0,
        step_size=1e-6,
        line_search=False
    )
```

## 6. Interaction with Gating and AGM
- Gating (FPP lens): IEON’s iso‑energy exploration helps surface beneficial non‑local moves without over‑activating gates. It reduces spurious increases that would otherwise tighten gate costs (or trigger hazard spikes).
- AGM (phase‑adaptive term weighting): IEON stabilizes early exploration phases; as AGM emphasizes different terms by phase, IEON keeps the search well‑behaved when energy curvature is uncertain.

## 7. Practice: Signals and Tuning
When to turn on or increase orthogonal noise:
- Large gradient rotation (curved valleys)
- ΔF much smaller than predicted by a first‑order model under normalized step
- Frequent backtracks or tight contraction margins
- Flat but anisotropic regions (small ‖g‖, high anisotropy proxies)

Tuning tips:
- Start with small `noise_magnitude` (e.g., 0.02–0.1 relative to η scale); enable `auto_noise_controller=True`.
- Keep `line_search=True` and stability guard on; IEON plays well with both.
- Log contraction margins and ΔF; ensure IEON raises exploration without harming acceptance rates.

## 8. Reproducibility (Windows PowerShell)
Run ΔF/conditioning benchmarks with normalized grads:
```powershell
python .\experiments\benchmark_delta_f90.py --configs vect gradnorm agm smallgain --steps 80 --log_budget
```
Run orthogonal noise unit tests:
```powershell
pytest -q tests\test_orthogonal_noise.py
```

## 9. Limitations and Failure Modes
- If curvature is highly non‑Euclidean, unit‑norm Euclidean gradient may be suboptimal vs metric‑aware updates (natural gradient). IEON is deliberately lightweight.
- Very high noise can still cause second‑order energy increases; use the controller or line search to keep acceptance healthy.
- At exact stationary points, any direction is tangent; IEON reduces to isotropic noise without the first‑order guarantee (still fine, but no special structure to exploit).

## 10. Related Work and References
- Prototype reference Normalized Dynamics OPT (unit‑gradient + tangent‑plane noise):
  - `https://github.com/Gman-Superfly/Normalized_Dynamic_OPT`
- Natural gradient methods:
  - Amari, S. (1998). “Natural gradient works efficiently in learning.” Neural Computation. `https://doi.org/10.1162/089976698300017746`
  - Martens, J. (2014). “New insights and perspectives on the natural gradient method.” arXiv:1412.1193. `https://arxiv.org/abs/1412.1193`
- Manifold‑aware stochastic dynamics:
  - Girolami, M., & Calderhead, B. (2011). “Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods.” Proc. Royal Society A. `https://doi.org/10.1098/rspa.2011.0411`
- Noise‑injection in optimization/inference:
  - Welling, M., & Teh, Y. W. (2011). “Bayesian Learning via Stochastic Gradient Langevin Dynamics.” arXiv:1112.5745. `https://arxiv.org/abs/1112.5745`

## 11. Relationship to This Repository’s Framework
- Objective: Free‑energy minimization with composable local modules and couplings.
- Coordination: Event‑driven, with gating for rare non‑local moves and AGM for phase‑adaptive term weighting.
- IEON’s role: A structure‑preserving exploration mechanism compatible with line search, stability caps, homotopy, and adaptive weighting.

## 12. Origin and Attribution
- Disambiguation: The Normalized_Dynamic_OPT prototype noise injection uses isotropic Gaussian noise added to embeddings; IEON projects noise onto the tangent plane (orthogonal to ∇F) in η‑space, so the noise component is first‑order energy‑neutral.
- The core geometric intuition—direction‑only flow paired with tangent‑plane noise—was explored in the prototype repo Normalized_Dynamic_OPT (`https://github.com/Gman-Superfly/Normalized_Dynamic_OPT`). Here we formalize, integrate, and validate the method within our energy‑based coordination stack, including tests (`tests/test_orthogonal_noise.py`) and benchmarks.

---

For questions or contributions, please open an issue or PR. If this work informs your research or engineering, consider citing per the notice at the top of this page.


