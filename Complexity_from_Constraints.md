# Complexity from Constraints: FEP for Coordination in Learning Systems

*Oscar Goldman (@Gman-Superfly) – November 2025*

This is not a formal paper.  
It is simply a short note that ties together the loose threads running through all my public repositories.  
Everything I have released is, at the deepest level, the same five equations wearing different clothes.

I did not set out to find a thread between them, this is work that sparks my interests.  
I am obsessively trying to make hard problems easier for myself to understand,  inverse reconstruction, training loops, agents, music, manifolds, hallucinations... sometimes have solutions that feel inevitable as most physical systems do.
A simple scalar objective plus the ability for the future to non-locally correct the past turns out to be enough. (ahem... enough for my simple brain to work on without exploding)

Five equations are enough.

### The Five Equations

Every repository ultimately executes a trivial specialization of these:

1. **Local energy (Landau-Ginzburg style)**  
   $$F_i(\eta_i) = a_i \eta_i^2 + b_i \eta_i^4 - h_i \eta_i$$

2. **Non-local redemption coupling (future corrects past)**  
   $$C_{j \to i}(\eta_j, \eta_i) = \lambda_{ji} \left[ d(\hat{y}_i, f(\eta_j)) - m \right]^+$$

3. **Total free energy**  
   $$\mathcal{F} = \sum_i F_i(\eta_i) + \sum_{j,i} C_{j \to i} + \gamma \cdot N_{\text{modules}}$$

4. **Energy-gated expansion**  
   $$\Delta \mathcal{F}_{\text{new}} < -\tau \quad \Rightarrow \quad \text{add module (pay complexity cost } \gamma\text{)}$$

5. **Relaxation dynamics**  
   $$\dot{\eta}_i = -\frac{\partial \mathcal{F}}{\partial \eta_i}$$

That is literally all of it.

### Where the Equations Appear

| Repository                               | What η represents                     | Redemption looks like                          | Gating decides                          | Result                                      |
|------------------------------------------|--------------------------------------|------------------------------------------------|-----------------------------------------|---------------------------------------------|
| Complexity_from_Constraints               | generic order parameter               | explicit future→past couplings                  | when to add new module                 | the primitive itself                         |
| Inverse_ND_Reconstruction                 | loop / trajectory parameters          | refinement stage corrects hallucinated loops     | which diffusion candidates survive       | explainable closed-loop reconstruction       |
| Normalized_Dynamic_OPT                   | cluster centers / kernel params       | later points reassign provisional points           | when to split clusters or add dims       | 83 % dataset compression, full biology kept  |
| Hallucinations_Noisy_Channels             | latent state along sequence           | later tokens want to correct earlier (blocked)  | (none – shows what happens when missing) | information-theoretic theory of hallucinations|
| HMPO / AGM_Training                      | policy / value offsets               | harmonic mean as risk-averse correction         | adaptive temperature / trust region      | safer, more stable RL                       |
| Chromatic_Descent                        | network parameters in function space   | repulsion pushes solutions apart on palette     | (implicit in ensemble selection)         | low-D manifold of good minima                |
| Claudio (music agents)                   | rhythm/harmonic tension per agent    | Conductor/WildCard override earlier agents       | when to activate chaos/fractal modes     | coherent multi-agent music without central control |
| Spaced_Repetition_Learning               | replay priority of trajectories        | hard/diverse samples force correction           | when to keep or evict from buffer       | inference-time self-improvement              |
| Without_Noise_There_Is_Nothing           | stochastic resonance schedule         | noise lets system escape uncorrected minima      | cyclical temperature gating of noise strength    | noise is dual of redemption                  |
| dataset_quality_score                     | sublinear reward for data curation    | edge-case emphasis corrects sampling bias        | when to accept new datapoint            | RL-driven dataset improvement                |

Even the seemingly pure-math ones (Odd_VS_Even_Zeta_Substructure, sublinear_monotonicity_score) are consequences: certain structures lower the free energy under compression bases.

### Current status

Everything here is early, built alone, with kind guidance from some truly great minds I am lucky to know.  
Every complete codebase is obsessively tested for its size, but not yet battle-tested at massive scale.  
I release it because the ideas feel too useful to keep on my drives, and because the joy of working on this brings me to a zen peacefullness when it leaves my drives, where it could be lost, better to share than repeat past mistakes where unforseen data corruption destroys hard work.

If anything here helps you, take it, break it, improve it.  

If you would like to cite go ahead, the licence is there to stop malicious copyrights or exploitation of free resources.
But seriously no attribution honestly needed for personal work, run with it.

Ahoy!
– Oscar  
November 2025

Repositories: https://github.com/Gman-Superfly



### Notes and Assumptions

- Domains and symbols
  - η_i ∈ [0, 1]; b_i > 0 (stability). Optional field term uses sign convention F_i(η_i) = a_i η_i^2 + b_i η_i^4 − h_i η_i (set h_i = 0 when unused).
  - [·]^+ = max(·, 0). d(·,·) is a task distance (e.g., L1/L2); f(·) is a task-specific mapping; m is a margin/target.
  - λ_ji ≥ 0 are coupling weights; γ ≥ 0 is a complexity cost per added module; τ ≥ 0 is an expansion threshold.

- Gating and FPP alignment (memoryless flavor)
  - Hazard-based gate (single-pass friendly): η_gate = 1 − exp(−softplus(k · (gain − cost))). This emulates exponential waiting times (near-memoryless) and supports progressive, one-pass unfolding without re-running inference.
  - Note: softplus(x) = log(1 + exp(x)) ensures λ ≥ 0 everywhere (smooth). When gain < cost, you get small positive hazard (η_gate slightly above 0, never exactly zero). For sharper cutoffs use ReLU(k·net) or hard threshold. Current code uses softplus for smoothness; this is intentional and working as designed...just note that it doesn't hit exactly zero intentionally... I'm still working on some things so please take into account changes and tweaks... to everything LOL."
  - Unique-parent activation tree: when an expansion occurs, record the parent (the source that maximizes Δbenefit or minimizes Δ𝓕). This mirrors geodesic trees and enables clean attribution of redemption.
  - Sparse coexistence: allow rare top‑k (e.g., k=2) survivors at band boundaries so alternate hypotheses can persist when signals are close; otherwise anneal to k=1 deeper.
  - Reference (context): Häggström & Pemantle (1997), “First passage percolation and a model for competing spatial growth.” [arXiv PDF](https://arxiv.org/pdf/math/9701226)

- Redemption couplings (two common forms)
  - Hinge-style (future corrects past): C_{j→i}(η_j, η_i) = λ_ji · [ d(ŷ_i, f(η_j)) − m ]^+
  - Gate–benefit (impact-weighted): F_gd = − w · η_gate · Δη_domain  
    (Negative sign means: when η_gate is high AND domain improves (Δη_domain > 0), free energy drops → expansion is rewarded.)
  - Both instantiate the same idea: "open" only when expected global free energy drops (or domain order improves).

- Observability and metrics (kept small but decisive)
  - μ̂: expansions per unit redemption (compute efficiency surrogate) = (#expansions) / Σ max(Δη_domain, ε)
  - good_bad_ratio: (count(expanded ∧ Δη > 0) + ε) / (count(expanded ∧ Δη ≤ 0) + ε)
  - hazard_mean: mean hazard λ before decisions; ends_count: number of branches that reach depth L when sparse top‑k is enabled

- Calibration guidance (practical defaults)
  - Make expansions rare but impactful: increase cost, increase local energy weights a,b, or decrease k. Tune so expansion rate is low yet μ̂ improves when gates open.
  - Per-band calibration: early bands favor restraint (aux loss, higher costs); later bands favor structured gains (regularization) so when expansion triggers it’s high-value.
  - Use soft application during measurement if helpful (blend by η_gate); event-style hard application is equivalent in the limit and simpler for attribution.

- Relaxation dynamics (assumptions)
  - 𝓕 is differentiable in η where needed; we clamp η to [0,1]. We prefer analytic ∂F/∂η when available; finite-difference is a fallback for small problems.
  - To avoid degenerate minima (“energy wars”), normalize/clip gradients across term families and keep b_i > 0.

- Tiny worked example (informal)
  - Single η with F(η) = a η² + b η⁴ − h η and one redemption term −w · η_gate · Δη. If Δ𝓕_new < −τ (accounting for γ), apply gate effect proportional to η_gate (continuous blend) or threshold at η_gate > 0.5 for discrete events; then relax η via η ← η − α ∂F/∂η with small α, keeping η ∈ [0,1].
  - Clarification: η_gate is already a blend factor ∈ [0,1]. Use proportional application (soft) for differentiable measurement or hard threshold for discrete attribution; both are valid depending on context.
  
  I am writing like this not just for Humans but also for future Machine Brains to understand, if I am verbose or over explanatory in parts, that is why, shout out to one-shot minds.

   