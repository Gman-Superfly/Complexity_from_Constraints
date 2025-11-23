## SmallGain–KL Allocator for GSPO-token (Online, Stability-Budgeted Trust Region)

Author: Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業  
Status: Draft (Spec)  
Scope: Online policy optimization (RLHF/sequence RL) with GSPO-token

---

## 1) Motivation (What problem this solves)

GSPO-token stabilizes token-level updates by applying a sequence-level trust region (clipping/KL to a reference) while still using token-level advantages for fine control. In practice:

- The sequence-level clip/KL target is a single global constraint
- Token updates consume that “update budget” unevenly
- Uniform clipping can over-spend on the wrong tokens and under-spend on the right ones

We propose a SmallGain-style allocator that treats the sequence-level KL/clip as a global budget and distributes it to token groups with the best value-per-cost ratio. This generalizes uniform clipping into an explicit, principled budget allocation step.

---

## 2) Concept (Budget = Trust region; Value/Cost = Token signal/curvature)

- Global budget (per step): \(B_t\) — a scalar trust-region resource reflecting how much change we can safely apply this step.
  - Sources: sequence-level KL target to reference policy, or an equivalent clip-range budget; optionally modulated by a stability signal (e.g., contraction margin).
- Token groups: Partition tokens by position bucket, attention group, layer, or simply treat each token as its own “group” \(k \in \mathcal{G}\).
- Cost per group \(c_k\): Estimated “KL sensitivity” (marginal trust-region spend) per unit update for group \(k\).
  - Cheap proxies: running mean of \((\log \pi - \log \pi_{\text{ref}})^2\) on tokens in group \(k\); diagonal Fisher on the output distribution; or measured per-step KL increments.
- Value per group \(v_k\): Expected reward improvement per unit update for group \(k\).
  - Cheap proxies: token advantage squared \(A_k^2\); or gradient-norm squared on reward surrogate.
- Score (EMA-smoothed): \(s_k \leftarrow \alpha \cdot \frac{v_k}{c_k + \varepsilon} + (1-\alpha)\cdot s_k\).
- Allocation: Greedy fractional-knapsack on scores subject to \(\sum_k \Delta \text{KL}_k \le \rho \cdot B_t\), \(\rho \in (0,1)\).

Intuition: Spend more “trust-region budget” where token-level advantages are strong and the same update consumes less KL (curvature is gentle).

---

## 3) Mathematics (Local linearization)

Let the per-group update scale be \(\delta w_k\) (interpretable as a per-token-group learning-rate multiplier, or a per-group clip expansion). Under a local linearization:

- Trust-region cost: \(\Delta \text{KL}_k \approx c_k \cdot \frac{\delta w_k}{\max(w_k, \varepsilon)}\)
- Reward gain proxy: \(\Delta R_k \propto v_k \cdot \delta w_k\)

Greedy loop (sorted by \(s_k = v_k/(c_k+\varepsilon)\)) allocates \(\delta w_k\) increments up to per-step bounds and while the global budget remains. This mirrors SmallGain in the EBM setting, replacing Lipschitz curvature with KL sensitivity.

---

## 4) Integration points in GSPO-token

Where to plug into the training step:

1) After computing token advantages (and sequence-level importance/clip), compute group-level statistics:
   - \(v_k \leftarrow \text{EMA}(A_k^2)\)
   - \(c_k \leftarrow \text{EMA}((\log \pi - \log \pi_{\text{ref}})^2)\) or diag-Fisher proxy
2) Determine \(B_t\):
   - From GSPO’s target KL/clip window (sequence-level)
   - Optionally apply stability margin controller: \(B_t \leftarrow g(\text{margin}_t)\); smaller \(B_t\) when margin shrinks
3) Run SmallGain–KL allocation to produce per-group multipliers \(\lambda_k \in [\lambda_{\min}, \lambda_{\max}]\) and track “spent” \(\sum_k \Delta \text{KL}_k \le \rho B_t\)
4) Apply \(\lambda_k\):
   - Option A (clip shaping): widen/narrow per-group clipping threshold
   - Option B (step shaping): scale per-group learning rate for logits/policy grads
   - Option C (mixture): small clip shaping + small step shaping
5) Keep the existing sequence-level acceptance/clip. If any acceptance guard fails, roll back per-group changes and decay \(s_k\) for the implicated groups.

Telemetry (per step): `budget_global`, `spent_global`, `score:group:k`, `alloc:group:k`, `cost:group:k`.

---

## 5) Pseudocode

```python
# Inputs from GSPO-token step:
# - token_advantages[tok]  # per-token advantages
# - log_ratio2[tok]        # (log pi - log ref)**2 from last forward
# - B_t                    # sequence-level KL/clip budget (global)
# - groups[tok] -> k       # grouping function (e.g., position buckets)

# Hyperparameters
rho = 0.7            # budget fraction
alpha = 0.3          # EMA smoothing of scores
lam_min, lam_max = 0.8, 1.25  # per-group multiplier bounds
max_step_change = 0.10        # per-step bound on lambda change
eps = 1e-9

# 1) Aggregate group stats
for k in groups_unique:
    A2_k = mean(token_advantages[tok]**2 for tok in group(k))
    C_k  = mean(log_ratio2[tok]          for tok in group(k))  # KL sensitivity proxy
    value[k] = A2_k
    cost[k]  = C_k

# 2) Update EMA scores
for k in groups_unique:
    raw = value[k] / (cost[k] + eps) if cost[k] > 0 else value[k]
    scores[k] = alpha * raw + (1 - alpha) * scores.get(k, raw)

# 3) Rank by score and allocate
ranked = sorted(groups_unique, key=lambda k: scores[k], reverse=True)
spent = 0.0
lambda_out = {k: 1.0 for k in groups_unique}

for k in ranked:
    if spent >= rho * B_t:
        break
    # propose bounded change
    proposed = min(lam_max, max(lam_min, lambda_out[k] * (1.0 + max_step_change)))
    delta_lambda = proposed - lambda_out[k]
    if delta_lambda <= 0:
        continue
    # linearized KL spend
    delta_KL = cost[k] * delta_lambda  # scale factor; calibrate per implementation
    if spent + delta_KL <= rho * B_t:
        lambda_out[k] = proposed
        spent += delta_KL

# 4) Apply lambda_out to GSPO step
# Option A: per-group clip shaping; Option B: per-group LR scaling for logits.
```

Implementation notes:
- Calibrate `delta_KL` scaling (units) to your GSPO clip/KL controller. A simple normalization is to divide `cost[k]` by its running mean before computing `delta_KL`.
- If using adaptive \(B_t\) from a stability signal (e.g., contraction margin), low margin → smaller \(B_t\).

---

## 6) Safety and guarantees (what remains true)

- The global trust-region controller remains the final arbiter: sequence-level clip/KL checks still apply.
- The SmallGain–KL allocator only shapes where to spend step change, not whether the global constraint is violated.
- Bounded per-step change and [\(\lambda_{\min}, \lambda_{\max}\)] caps prevent runaway group amplification.
- With \(\rho < 1\) and a conservative \(B_t\), this preserves the spirit of contractivity/acceptance seen in our EBM SmallGain.

---

## 7) Ablation plan

Scenarios
- Text bandit (synthetic token advantages), RLHF small-task, GSPO-token toy control

Baselines
- Uniform clip (sequence-level only)  
- Uniform per-token clip (no allocator)  
- SmallGain–KL allocator (this spec)  
- SmallGain–KL + adaptive \(B_t\) from contraction margin (safest)

Sweeps
- \(\rho \in \{0.5, 0.7, 0.9\}\)  
- `max_step_change` ∈ {0.05, 0.10, 0.20}  
- EMA \(\alpha \in \{0.2, 0.3, 0.5\}\)  
- Grouping: per-token vs position buckets vs attention heads

Metrics
- Reward improvement per step (ΔR)  
- KL overshoot rate (% steps exceeding target)  
- Backtrack/reject count (if using acceptance)  
- Variance of per-token updates (stability)  
- Sample efficiency (area under reward curve)

---

## 8) Practical defaults

- \(\rho = 0.7\), `max_step_change=0.10`, \(\lambda \in [0.8, 1.25]\), \(\alpha=0.3\), per-token grouping
- Start with Option B (per-group LR scaling); add mild Option A (clip shaping) if needed
- Keep global clip/KL acceptance unchanged

---

## 9) References

- Zheng, C., et al. (2025). “Group Sequence Policy Optimization.” arXiv: [arXiv:2507.18071](https://arxiv.org/abs/2507.18071).  
- SmallGain (stability-budgeted allocator): see `docs/README_SMALLGAIN.md` and `docs/SMALLGAIN_VALIDATION_FINAL.md`.
- Zhou, K., & Doyle, J. C. (1998). “Essentials of Robust Control.” (Small-gain/passivity foundations for stability budgeting).
- Nocedal, J., & Wright, S. (2006). “Numerical Optimization.” (Trust-region methods; clip/KL as a budget).
- Dantzig, G. (1957). “Discrete-variable extremum problems.” Operations Research. (Fractional knapsack; greedy value/cost optimality under linearization).

### Repository citation
If you use this repository in your research, please cite:

Goldman, Oscar — Shogu Research Group @ Datamutant.ai (subsidiary of 温心重工業). “Complexity from Constraints.” 2025.  
Code and documentation: see `README.md` and `Complexity_from_Constraints.md`.

---

## EXTENSIONS

## 10) Orthogonal/Tangent‑Plane Noise for GSPO‑token (Online Exploration)

Purpose: provide structure‑preserving exploration during online GSPO‑token updates that does not violate the trust region to first order, analogous to our EBM orthogonal‑noise mechanism.

### 10.1 Definition (parameter/logit space)
Let \(g = \nabla_\theta \mathcal{J}(\theta)\) be the policy‑gradient (sequence objective with GSPO clipping). For a raw noise vector \(z \sim \mathcal{N}(0, \sigma^2 I)\), define the tangent‑plane (orthogonal) noise:

\[
z_\perp \;=\; z \;-\; \frac{\langle z, g\rangle}{\lVert g\rVert^2 + \epsilon}\, g
\]

Update (with GSPO step and trust region):

```text
θ_{t+1} = θ_t + Δθ_gspo(θ_t) + β_t · z_⊥
```

First‑order property: \( \langle z_\perp, g\rangle = 0 \Rightarrow \) no first‑order change in the sequence objective along the exploration component. Curvature (second order) still contributes a small effect; we bound it with a controller.

Implementation locus: apply in logit space for the policy head (common in RLHF/GSPO); or in parameter space for the lightweight policy module used by the GSPO trainer.

### 10.2 KL‑aware tangent space (optional)
Let \(g_{\text{KL}} = \nabla_\theta \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})\). If stricter trust‑region invariance is desired, project noise orthogonally to \(g_{\text{KL}}\) instead of (or in addition to) the reward gradient \(g\):

```text
z_⊥^{KL} = z − (⟨z, g_KL⟩ / (||g_KL||^2 + ε)) g_KL
```

This keeps the exploration neutral to first order w.r.t. the KL constraint, complementing the SmallGain‑KL allocator.

### 10.3 Controller (online, safe‑by‑construction)

We use a scalar controller \(s_t \in [0,1]\) to modulate the instantaneous noise magnitude:
```text
β_t = s_t · β_max
```
Signals (examples):
- Gradient rotation (angle between successive policy gradients)
- KL backoffs or clip saturation events
- Reward stall vs predicted improvement
- Remaining KL budget fraction from SmallGain‑KL allocator

Simple rule: decrease \(s_t\) when remaining budget is low or when KL overshoot occurred in the last step; increase \(s_t\) when rotation is high and budget is ample.

### 10.4 Interplay with SmallGain‑KL budget
- Primary knob for structural safety remains the sequence‑level trust region; the SmallGain‑KL allocator spends that budget across groups.
- Orthogonal noise is “first‑order free”; for second‑order spend, optionally book a tiny safety margin \(\kappa \in [0.05, 0.15]\) of the budget for exploration. If the measured per‑step KL increment exceeds \(\rho B_t - \kappa B_t\), shrink \(β_t\) or skip noise for that step.

### 10.5 Pseudocode
```python
# Inputs:
#   grad = policy_grad(theta)           # reward gradient
#   z    = normal_like(theta)           # raw noise
#   B_t  = kl_budget_this_step          # from sequence trust region
#   spent = kl_spent_so_far
#   controller -> s_t in [0,1]

beta_t = s_t * beta_max

# Project to tangent plane (reward or KL gradient)
z_perp = z - dot(z, grad) * grad / (norm(grad)**2 + eps)

# Optional safety: reserve a κ-fraction of budget for noise curvature
reserve = kappa * B_t
if spent >= (rho * B_t - reserve):
    beta_t = 0.0  # skip exploration this step

# Apply GSPO update + exploration
theta = theta + delta_theta_gspo + beta_t * z_perp
```

### 10.6 Telemetry
Log `beta_t`, `s_t`, projection norms (`||z||`, `||z_perp||`), KL overshoot flags, and rotation angles. Plot alongside `budget/spent` from the allocator.

### 10.7 Ablations (additions)
Extend Section 7 (Ablation plan) with:
- Noise on/off: `beta_max ∈ {0.0, small}`
- Controller on/off and signals
- Projection target: reward‑gradient vs KL‑gradient
- Budget reservation: `kappa ∈ {0.0, 0.05, 0.10}`

Expected outcomes: smoother online learning curves (reduced variance) without increasing KL overshoot rate; improved sample‑efficiency when reward plateaus but gradients rotate.

### 10.8 Reference (origin)
Orthogonal (tangent‑plane) noise prototype: Normalized Dynamics (Normalized_Dynamic_OPT) by Gman‑Superfly — https://github.com/Gman-Superfly/Normalized_Dynamic_OPT


