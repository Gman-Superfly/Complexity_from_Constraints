# When To Use Noise (IEON and Metric‑Aware Variants)

This guide explains when and how to use orthogonal (tangent‑plane) noise in our energy minimization framework, both in the default Euclidean form (IEON) and in the optional metric‑aware variant. It consolidates practical switches, recommended scenarios, caveats, and links to detailed docs and code.

## TL;DR
- IEON (Euclidean projection) is first‑order iso‑energy for F: ∇F^T z⊥ = 0.
- Metric‑aware variant is iso‑energy in metric M: g^T M z⊥,M = 0. That is only approximately iso‑energy for F unless M ≈ local geometry of F (e.g., Fisher for KL).
- Any noise (adaptive or fixed) breaks strict determinism → monotonic energy assertion is auto‑skipped.

## IEON (Euclidean) — Fixed vs Adaptive
- Fixed (non‑adaptive)
  - enable_orthogonal_noise=True
  - noise_magnitude > 0
  - auto_noise_controller=False
  - Optional: noise_schedule_decay < 1.0 for simple annealing

  When it makes sense:
  - Ablations/repro; hold exploration constant across runs
  - Short runs with known‑good magnitude
  - Sanity checks of first‑order iso‑energy behavior

- Adaptive
  - enable_orthogonal_noise=True
  - auto_noise_controller=True (uses stall/backtracks/rotation signals)
  - noise_magnitude acts as a base; noise_schedule_decay applies annealing

  When it makes sense:
  - Unknown or heterogeneous landscapes (varying curvature/rotation)
  - Keep acceptance/backtracks healthy over long runs
  - Turn exploration up when progress stalls, down when rate is good

Practical guidance:
- With fixed noise, keep noise_magnitude small, especially without line_search, to limit second‑order ΔF spikes and backtracks.
- Combine with line_search=True and stability guard when in doubt.
- Log contraction margins and ΔF to verify exploration is helpful, not harmful.

### Gradient normalization (recommended)
- For IEON, `normalize_grads=True` is recommended: it uses the unit gradient direction, which makes step_size interpretation stable and plays well with first‑order reasoning. IEON’s orthogonality does not require normalization, but normalization reduces sensitivity to ‖g‖ changes.
- Without normalization, IEON remains first‑order iso‑energy, but the relative scale of the descent vs noise changes with ‖g‖. Use smaller noise_magnitude and consider line_search.
- Metric‑aware note: if you use natural‑gradient style preconditioning (M⁻¹g), gradient normalization may be less critical; if you keep Euclidean descent with M‑orthogonal projection, you can still normalize the Euclidean gradient before projection for consistent steps.
- Switch: see `core/coordinator.py` (`normalize_grads=True`).

## Metric‑Aware Variant (optional, M‑orthogonal)
- Switches (design)
  - metric_aware_noise_controller=True to enable M‑orthogonal projection
  - Provide either metric_matrix (M) or metric_vector_product (Mv)
  - Works with fixed or adaptive magnitude (independent of adaptivity)

- Fixed (non‑adaptive, metric‑aware)
  - enable_orthogonal_noise=True
  - metric_aware_noise_controller=True
  - auto_noise_controller=False
  - Set noise_magnitude (and optional noise_schedule_decay)
  - Projection uses M (Fisher/KL or your Mv); magnitude is constant/annealed

- Adaptive (metric‑aware)
  - enable_orthogonal_noise=True
  - metric_aware_noise_controller=True
  - auto_noise_controller=True (same stall/backtracks/rotation signals)
  - Projection remains M‑orthogonal; controller modulates magnitude only

Caveat reminder:
- Metric‑aware is iso‑energy in M (g^T M z⊥,M = 0). It’s only approximately iso‑energy for F unless M ≈ F’s local geometry (e.g., Fisher for KL). For strict iso‑energy in F, use Euclidean projection and (optionally) use M only to scale magnitude.

Metric availability (rule of thumb):
- “Available” if you can compute Fisher/metric info from current model and reference:
  - logits/probabilities + reference distribution
  - autograd enabled for score functions (∇θ log πθ)
  - can compute Fisher‑vector products or a diagonal Fisher estimate
- Otherwise: treat as unavailable and use IEON (Euclidean projection) with the same controller.

## GSPO‑KL allocator: noise usage
- Purpose: optional, structure‑preserving exploration during GSPO‑token updates that respects the KL trust region to first order.
- Mechanism:
  - Reward‑orthogonal: project noise orthogonal to the reward gradient g (first‑order neutral for reward).
  - KL‑orthogonal (stricter TR): project noise orthogonal to g_KL (first‑order neutral for the KL constraint; pairs well with Fisher/metric‑aware setups).
- Magnitude control: fixed or adaptive (stall, backtracks/clip events, rotation). Prefer small base magnitudes; anneal if fixed.
- Budget interplay (SmallGain): noise is “first‑order free,” but second‑order curvature can still spend KL. Optionally reserve a small κ‑fraction of budget and skip noise when near the limit.
- Where: parameter or logit space (commonly logit space for RLHF/GSPO).
- When to use:
  - High gradient rotation, stalled reward improvement, variance spikes
  - You want exploration that doesn’t blow up the KL gate to first order
- Caveat: reward‑orthogonal is neutral for reward; KL‑orthogonal is neutral for KL. For metric‑aware iso‑energy, prefer Fisher; else use Euclidean projection and treat metric as a magnitude preconditioner.
- See: `docs/paper_extensions/GSPO_SMALLGAIN_KL_ALLOCATOR.md` (Section 10).

## Commands (Windows PowerShell)
- Run orthogonal‑noise unit tests:
```powershell
uv run -m pytest tests\test_orthogonal_noise.py -q
```

- Run metric‑aware tests (M‑orthogonal projection):
```powershell
uv run -m pytest tests\test_metric_aware_noise.py -q
```

- IEON repeats with adaptive controller:
```powershell
uv run python -m experiments.ieon_repeats --configs vect gradnorm agm `
  --repeats 5 --steps 80 --scenario dense --dense_size 32 `
  --noise_magnitude 0.05 --auto_noise_controller --log_budget --run_id_prefix ieon_noise
```

- IEON repeats with fixed noise (non‑adaptive):
```powershell
uv run python -m experiments.ieon_repeats --configs vect gradnorm agm `
  --repeats 5 --steps 80 --scenario dense --dense_size 32 `
  --noise_magnitude 0.05 --log_budget --run_id_prefix ieon_fixed
```

## References and Cross‑Links
- Concept and implementation details:
  - `docs/ISO-ENERGY_ORTHOGONAL_NOISE.md` (IEON, Euclidean projection)
  - `docs/METRIC_AWARE_NOISE_CONTROLLER.md` (M‑orthogonal variant, availability and caveats)
- Coordinator switches and projection code:
  - `core/coordinator.py` (enable_orthogonal_noise, auto_noise_controller, metric_aware_noise_controller, metric_vector_product/metric_matrix)
  - `core/energy.py` (project_noise_orthogonal, project_noise_metric_orthogonal)
- Tests:
  - `tests/test_orthogonal_noise.py` (IEON orthogonality and integration)
  - `tests/test_metric_aware_noise.py` (M‑orthogonal projection and coordinator path)


