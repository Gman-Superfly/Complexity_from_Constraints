# Complexity from Constraints: A Non-Local, Energy‑Coordinated Micro‑Module Framework

This document summarizes the motivation, core ideas, formalization, experiments, early observations, and limitations of this project. It is a concise, humble “mini paper” intended for future reference.

## 1. Motivation
- Many systems (music, cognition, programs) redeem provisional mistakes using later context. Typical autoregressive models lack such “future‑like” corrections.
- We aim to coordinate small, typed modules without inflating them. The coordination mechanism should be global, simple, and measurable.
- Our hypothesis: constraints create complexity; a global free‑energy objective with sparse non‑local couplings can yield coherent system‑level behavior from small parts.

## 2. Core Idea (Thesis)
- Each small module exposes:
  - an order parameter \( \eta \in [0,1] \) that quantifies local coherence,
  - a local energy \( F_{\text{local}}(\eta; c) \) shaped by constraints \( c \).
- Modules are composed by adding sparse non‑local couplings to form a total energy:
  \[
  F_{\text{total}}(\{\eta_i\}; c) \;=\; \sum_i F_i(\eta_i; c) \;+\; \sum_{(i,j)\in \mathcal{E}} F_{ij}(\eta_i,\eta_j; c).
  \]
- The system seeks lower \( F_{\text{total}} \). Coherent global behavior emerges without hand‑coding global rules.
- A gating module makes rare‑but‑impactful expansion decisions when they reduce \( F_{\text{total}} \) (non‑local, “future‑like” correction).

## 3. Formalization (Minimal)
- Order parameter per module \( i \): \( \eta_i \in [0,1] \).
- Landau‑style local energy (example): \( F_i(\eta_i; a_i,b_i) = a_i \eta_i^2 + b_i \eta_i^4 \) with \( b_i>0 \).
- Example couplings:
  - Quadratic: \( F_{ij} = w \cdot (\eta_i - \eta_j)^2 \).
  - Gate–benefit: \( F_{g,d} = -\, w \cdot \eta_g \cdot \Delta \eta_d \) where \( \Delta \eta_d \) is the domain’s improvement.
- Gating:
  - Net benefit: \( \text{net} = \text{gain} - \text{cost} \).
  - \( \eta_{\text{gate}} = \sigma(k \cdot \text{net}) \in (0,1) \) with a small local energy discouraging casual expansion.
- Optimization:
  - Coordinator descends \( F_{\text{total}} \) using finite differences or simple analytic gradients where available.
  - “Redemption score” measures earlier‑position improvement when non‑local mechanisms are enabled.

## 4. Non‑Locality and Temporal Redemption
- Non‑local couplings allow distal evidence (including prospective improvements) to influence local decisions.
- A mistake can be treated as a provisional imprint that is redeemed by later corrections if those lower \( F_{\text{total}} \).
- We measure this via:
  - \( \Delta F = F_{\text{before}} - F_{\text{after}} \),
  - redemption score on sequences (loss drop at earlier positions when non‑local context is available).

## 5. Implementation (MVP)
- Core:
  - `core/interfaces.py` (protocols), `core/energy.py` (Landau utilities), `core/couplings.py` (quadratic, hinge, gate–benefit),
    `core/coordinator.py` (energy + relaxation, finite‑diff with analytic fallback), `core/entity_adapter.py`.
- Modules:
  - `modules/sequence/monotonic_eta.py`: sublinear sequence consistency \( \eta \) and local energy.
  - `modules/connectivity/nl_threshold_shift.py`: grid connectivity \( \eta \) with optional shortcuts.
  - `modules/gating/energy_gating.py`: energy‑gated expansion (rare, impact‑weighted).
  - Optional: `models/nonlocal_attention.py` (energy‑regularized attention; PyTorch).
- Logging/Tests:
  - Polars CSV logs in `logs/`; lightweight tests under `tests/`.

## 6. Experiments
1) Landau sweep (`experiments/landau_sweep.py`): disorder→order by varying parameter \( a \).
2) Non‑Local Connectivity Threshold Shift (`experiments/non_local_connectivity_threshold_shift.py`): show that sparse shortcuts lower the apparent critical \( p \).
3) Sequence redemption (`experiments/sequence_redemption.py`): non‑local scoring vs prefix‑only baseline; measure redemption.
4) Energy‑gated expansion (`experiments/energy_gated_expansion.py`): vary expansion cost; measure expansion rate and redemption gain.
5) Optional energy‑regularized attention (`experiments/energy_reg_attn_ablation.py`): surrogate energy penalty on attention distributions.

## 7. Early Observations (MVP‑level, not claims)
- Landau toy behaves as expected (double‑well below the critical point).
- Sparse non‑local shortcuts increase connectivity at mid‑range \( p \); threshold shifts lower.
- Sequence redemption: full‑sequence (non‑local) scoring improves earlier positions vs prefix‑only baselines.
- Gating: expansion becomes rare as cost increases; when it triggers, redemption improves (by construction of the gain).

These are sanity checks; they do not imply state‑of‑the‑art performance. They support the project’s design direction.

## 8. Limitations (Current)
- Energy surrogates are simple; richer, task‑aligned energies are needed to avoid trivial minima.
- Coordinator uses finite‑difference gradients in places; scaling requires analytic/autograd pathways.
- Gating gain functions are domain‑specific and currently minimal.
- No large‑scale or long‑context evaluations yet; results are toy‑scale and illustrative.

## 9. Roadmap (Short)
- Strengthen energies and calibrate weights; add invariants and checks.
- Prefer analytic/autograd gradients where feasible; keep finite‑diff as fallback.
- Integrate gating with banded architectures to make expansion rare but impactful within a single pass.
- Extend experiments with exact baselines and ablations; add `test_prod_*` for lifecycle and event flows.
- Improve observability (ΔF, η traces, gating rates) and small diagrams in docs.

## 10. Reproducibility (Minimal)
- Environments via `uv`; metrics logged with Polars; deterministic seeds in experiments.
- Commands (Windows PowerShell):
  ```
  uv venv .venv
  .\.venv\Scripts\Activate.ps1
  uv pip install polars numpy networkx pytest
  # optional: uv pip install torch
  uv run python -m experiments.landau_sweep
  uv run python -m experiments.non_local_connectivity_threshold_shift
  uv run python -m experiments.sequence_redemption
  uv run python -m experiments.energy_gated_expansion
  ```

## 11. Positioning
- This work is not a larger model. It is a way to coordinate small, typed parts through a simple, global scalar objective and sparse non‑local couplings.
- The aim is clarity, composability, and measurable coherence—kept humble, with explicit limitations and room to grow. 


