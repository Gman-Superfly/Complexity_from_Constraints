# Precision Layer: Diagonal Preconditioning and Precision‑Aware Noise

This document explains the Precision Layer features that make certainty (stiffness/curvature) and uncertainty (slack) first‑class in the coordinator.

- Diagonal preconditioning of gradients using per‑module curvature.
- Precision‑aware redistribution of orthogonal noise toward low‑curvature directions.

Both features are opt‑in and controlled by explicit flags. Status and tests are listed in `docs/CHECKLIST.md`.

---

## 1) Diagonal Preconditioning (Newton‑aware step)

Concept
- Scale each gradient component by the inverse of the local curvature:
  - \( g_i^{eff} = g_i / (\varepsilon + \mathrm{curv}_i) \)
- Stiff directions (high curvature) receive smaller steps; flat directions receive larger steps.
- This yields faster, more stable progress without inverting a full precision matrix.

How it works
- Modules can implement `SupportsPrecision.curvature(eta)` to expose local stiffness.
- The coordinator maintains a `_precision_cache` (diagonal curvatures) and refreshes it each iteration.
- When `use_precision_preconditioning=True`, the coordinator scales the step using the cached curvature.

Usage (Python)
```python
from core.coordinator import EnergyCoordinator

coord = EnergyCoordinator(
    modules=...,
    couplings=...,
    constraints={},
    use_analytic=True,
    line_search=False,                 # works with or without line search
    use_precision_preconditioning=True,
    precision_epsilon=1e-8,            # small ε for stability
)
```

Notes
- Compatible with line search (preconditioning applied to the search direction).
- Works with or without orthogonal noise.
- Defaults: OFF (opt‑in), see flags.

Validation
- Test: `tests/test_precision_preconditioning.py` verifies stiff dimensions update less than loose ones.

---

## 2) Precision‑Aware Noise (inverse‑precision redistribution)

Concept
- Iso‑energy orthogonal noise is stabilizing, but not all coordinates should be treated equally.
- Redistribute the noise magnitude toward low‑curvature directions to explore flat valleys more.
- Weights: \( w_i \propto 1 / (\varepsilon + \mathrm{curv}_i) \). After weighting, noise is re‑projected to remain orthogonal.

How it works
- The `PrecisionNoiseController` computes per‑dimension weights from cached curvature.
- The coordinator reweights an already orthogonalized noise vector and re‑projects to preserve orthogonality (Euclidean or metric‑aware).
- Magnitude is still governed by the base (orthogonal) noise controller schedule/signals.

Usage (Euclidean)
```python
from core.coordinator import EnergyCoordinator

coord = EnergyCoordinator(
    modules=...,
    couplings=...,
    constraints={},
    enable_orthogonal_noise=True,
    noise_magnitude=0.1,               # base magnitude
    auto_noise_controller=True,        # adapt magnitude by signals (rate/backtracks/rotation)
    precision_aware_noise_controller=True,  # inverse-precision redistribution
    # Optional preconditioning together with precision-aware noise:
    use_precision_preconditioning=True,
    precision_epsilon=1e-8,
)
```

Usage (Metric‑aware)
```python
import numpy as np
from core.coordinator import EnergyCoordinator

M = np.diag([2.0, 1.0, 0.5])  # example SPD metric

coord = EnergyCoordinator(
    modules=...,
    couplings=...,
    constraints={},
    enable_orthogonal_noise=True,
    noise_magnitude=0.1,
    auto_noise_controller=True,
    precision_aware_noise_controller=True,
    metric_aware_noise_controller=True,
    metric_matrix=M,                    # or provide metric_vector_product=lambda v: M @ v
)
```

Quick toggles (all‑in‑one; illustrative)
```python
from core.coordinator import EnergyCoordinator

coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints=constraints,
    # Precision-aware step preconditioning
    use_precision_preconditioning=True,
    precision_epsilon=1e-8,
    # Orthogonal noise with adaptive controller and inverse-precision redistribution
    enable_orthogonal_noise=True,
    auto_noise_controller=True,
    precision_aware_noise_controller=True,
    noise_magnitude=0.05,
    # Optional: metric-aware projection (provide either metric_matrix or metric_vector_product)
    metric_aware_noise_controller=True,
    # metric_matrix=M,  # or
    # metric_vector_product=Mv,
)
```

Notes
- `enable_orthogonal_noise=True` with `noise_magnitude=0.0` produces no noise; set magnitude > 0 to activate.
- Precision‑aware redistribution is a directional weighting; magnitude scheduling remains under the orthogonal controller.
- Defaults: OFF (opt‑in), see flags.

Validation
- Test: `tests/test_precision_noise_controller.py` checks inverse‑curvature weights and normalization.
- Existing tests cover orthogonal projection and metric‑aware orthogonality.

---

## Flags & Defaults

- `use_precision_preconditioning=False`
- `precision_epsilon=1e-8`
- `precision_aware_noise_controller=False`
- `enable_orthogonal_noise=True` (but `noise_magnitude=0.0` by default)
- `auto_noise_controller=False`
- `metric_aware_noise_controller=False`

These controls are independent and can be combined as needed.

---

## Observability & Best Practices

- Track per‑step energy and ΔF with `RelaxationTracker` (accepted steps only).
- Use `EnergyBudgetTracker` to log per‑term energies/grad norms and contraction margins.
- For noise experiments, log sharpness proxies and escape events (see roadmap), and compare at matched loss to show improved robustness with precision‑aware noise.

---

## FAQ

- Q: Do I need both preconditioning and precision‑aware noise?
  - A: No. They are complementary. Preconditioning improves step quality; precision‑aware noise improves exploration. Use both for best results on stiff/flat mixes.

- Q: Does metric‑aware projection conflict with precision weighting?
  - A: No. Weight, then re‑project. The coordinator preserves M‑orthogonality after weighting.

- Q: What if modules don’t implement `SupportsPrecision`?
  - A: Curvature defaults to 0.0; preconditioning becomes a no‑op and noise weights become uniform. You can progressively add curvature support to modules that benefit.

---

## References

- Roadmap (§ Phase 2: The Precision Layer)
- `core/noise_controller.py` (PrecisionNoiseController)
- `core/coordinator.py` (flags and step logic)
- Tests: `tests/test_precision_preconditioning.py`, `tests/test_precision_noise_controller.py`


