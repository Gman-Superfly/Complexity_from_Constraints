# Metric-Aware Noise Controller (IEON: M-orthogonal variant)

This note describes an optional extension of Iso‑Energy Orthogonal Noise (IEON) that uses a problem‑specific metric \(M\) (e.g., Fisher for KL trust regions) to scale and project the exploration noise. Use when a reliable metric is available; otherwise fall back to standard IEON (Euclidean projection).

## Summary
- IEON (Euclidean) is strictly iso‑energy for \(F\) to first order: \( \nabla F^\top z_{\perp} = 0 \).
- Metric‑aware IEON is iso‑energy in metric \(M\): \( g^\top M\, z_{\perp,M} = 0 \). This is only approximately iso‑energy for \(F\) unless \(M\) matches the local geometry (e.g., Fisher for KL‑based trust regions).
- If you need strict iso‑energy for \(F\), keep Euclidean projection for the noise, and (optionally) use the metric only to scale the magnitude.

## When is the metric available?
“Available” if you can compute Fisher/metric information from the current model and reference (examples for KL trust region):
- You have policy logits/probabilities and a reference policy distribution.
- Autograd is enabled to form score functions \(\nabla_\theta \log \pi_\theta\).
- You can compute one of: a Fisher‑vector product (Fv), a diagonal Fisher estimate, or a practical proxy (e.g., per‑step KL increments). If none are feasible (no reference, no autograd, or too costly), treat metric as unavailable and use Euclidean IEON.

## Formulas
- M‑orthogonal projection of noise:
  \[
  z_{\perp,M} \;=\; z \;-\; \frac{z^\top M g}{g^\top M g + \varepsilon}\, g
  \]
  where \(g = \nabla F\).
- Optional M‑aware sampling or scaling:
  - Precondition descent: \( g_{\text{nat}} = M^{-1} g \) (natural gradient).
  - Sample geometry‑aware noise: \( z \sim \mathcal{N}(0, M^{-1}) \) or whiten Euclidean noise through \(M^{-1/2}\).

## Controller and switches (design)
- Existing toggles:
  - `enable_orthogonal_noise`: on/off for IEON.
  - `auto_noise_controller`: on/off to adapt magnitude based on stall/rotation/backtracks.
- Optional toggle (design name): `metric_aware_noise_controller`
  - If enabled and metric is available: use \(M\)-orthogonal projection and (optionally) M‑aware magnitude scaling.
  - Else: revert to Euclidean IEON with the same auto controller signals.

### Precision-aware redistribution (optional)

Combine metric-aware projection with inverse‑precision weighting to emphasize low‑curvature directions:

```python
from core.coordinator import EnergyCoordinator

coord = EnergyCoordinator(
    modules=...,
    couplings=...,
    constraints={},
    enable_orthogonal_noise=True,
    noise_magnitude=0.1,
    auto_noise_controller=True,
    metric_aware_noise_controller=True,
    metric_vector_product=lambda v: M @ v,   # or metric_matrix=M
    precision_aware_noise_controller=True,   # inverse-precision weighting
    # optional: diagonal preconditioning for the step
    use_precision_preconditioning=True,
    precision_epsilon=1e-8,
)
```

Order of operations:
1) Generate raw noise
2) Project to (metric‑)orthogonal
3) Weight by inverse curvature
4) Re‑project to preserve orthogonality
5) Normalize to target magnitude

## Pseudocode
```python
# Inputs per step: grad g, raw noise z, metric M (optional), base_magnitude, decay, signals
if metric_available:
    # M-orthogonal projection
    alpha = (z.T @ (M @ g)) / (g.T @ (M @ g) + eps)
    z_perp = z - alpha * g
    # Optionally scale with metric-aware controller (not required)
else:
    # Euclidean IEON
    alpha = (z.T @ g) / (g.T @ g + eps)
    z_perp = z - alpha * g

noise_mag = base_magnitude * (decay ** iter_idx) * controller_scale(signals)
eta_next = project_box(eta - step_size * g_hat + noise_mag * z_perp)
```

## Relation to GSPO (KL trust region)
- For KL trust regions, using the Fisher metric aligns the tangent‑plane projection with the constraint geometry. See the KL‑orthogonal variant in `docs/paper_extensions/GSPO_SMALLGAIN_KL_ALLOCATOR.md` (Section 10.2).

## Caveats
- Iso‑energy in \(M\) is not the same as iso‑energy for \(F\) unless \(M\) closely matches the local geometry of \(F\) (e.g., Fisher for KL‑constrained objectives). For strict first‑order neutrality w.r.t. \(F\), keep Euclidean projection and use \(M\) only for magnitude scaling.

## References
- Amari (1998), Natural Gradient (doi: 10.1162/089976698300017746)
- Martens (2014), Natural Gradient perspective (arXiv:1412.1193)
- Girolami & Calderhead (2011), Riemann Manifold Langevin/HMC (doi: 10.1098/rspa.2011.0411)


