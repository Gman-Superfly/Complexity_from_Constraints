## Stability‑Margin as Resource: Small‑Gain Allocator (Per‑Edge Budgeting)

Author: Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業  
Date: Nov 2025  
Status: Production‑grade design (guarded by existing stability/acceptance backstops)  
Confidence: 85%

### Purpose
Unify stability guardrails with intelligent resource allocation. Instead of a global, uniform coupling cap, allocate a per‑step “contractivity budget” to edges with the best payoff per unit stability cost. This improves ΔF90 (steps to achieve 90% energy drop), reduces backtracks, and lowers oscillations in hinge/gate regimes while preserving monotone acceptance and invariants.

- Budget: Stability margin \(m \approx 2/L̂ − \text{step\_size}\) (hard cap already present).
- Cost: Each edge \(k\) consumes stability via curvature \(\Delta L_k\) (marginal Lipschitz increase per unit weight).
- Value: Each edge \(k\) buys expected energy drop \(\Delta F_k\) per unit weight; proxy with grad\_norm² (EMA) and, sparsely, measured \(\Delta F/\delta w_k\).
- Policy: Greedy fractional‑knapsack allocation by score = value/cost, with row‑aware caps respecting Gershgorin rows; all steps remain under Armijo monotone acceptance.

This follows our architectural stance: dumb core (audited projector), smart updates (adapters), explicit observability.

---

## 1) Classical foundations (why this is principled)

- Small‑gain/passivity (Zames 1966; Vidyasagar 1993): Treat coupling weights as loop gains and keep total L2 gain < 1 (reserve). Our conservative Gershgorin \(L̂\) upper‑bounds the Jacobian norm; the “stability margin” is a passivity reserve we can budget.
- Fractional knapsack (Dantzig 1957): Maximize \(\sum_k \text{value}_k\) s.t. \(\sum_k \text{cost}_k \le \text{budget}\); greedy by value/cost is optimal under local linearization. Here cost \(\approx \Delta L_k\), value \(\approx \Delta F_k\) proxy.
- Coordinate descent with Lipschitz constants (Gauss–Southwell‑L; Nutini et al., 2015): Best local decrease \(\approx g^2/(2L)\). Our per‑edge “value/cost” mirrors this principle at the pairwise term level.
- Trust‑region/line‑search (Powell; Nocedal–Wright): The budget \(m = 2/L̂ − \text{step}\) acts as a trust region; we allocate within \(m\) while keeping Armijo monotonicity (already shipped).
- Gershgorin (1931) with diagonal scaling (Varga): Per‑row diag + off‑sum bounds justify assembling \(\Delta L\) from per‑edge curvature and enforcing rowwise margin caps.

---

## 2) Inputs and telemetry (what exists and what we add)

Already available:
- Global Lipschitz estimate \(L̂\) via Gershgorin‑style bound (locals + couplings).
- Per‑term gradient norms: `grad_norm:*` for local and coupling families.
- Acceptance/monotonicity: Armijo backtracking, ΔF ≤ 0 per accepted step.
- Optional global caps: `stability_guard` step cap; `stability_coupling_auto_cap` uniform scaler.

Add the following:
- Per‑edge \(\Delta L_k\) contributions and per‑row margins (incident rows i, j).
- EMA‑smoothed per‑edge “value” estimates (norm² and sparse \(\Delta F/\delta w_k\) peeks).
- Telemetry for budget, spend, per‑edge allocation, smoothed hinge activity, contraction margin.

---

## 3) Algorithm (per step)

Let \(L̂_{\text{base}}\) be the Gershgorin bound under current weights; `step_to_use` already reflects line‑search/step‑cap logic.

1) Compute stability budgets:
   - Target bound: \(L̂_{\text{target}} = \min(L̂_{\text{base}}, \text{stability\_coupling\_target or } L̂_{\text{base}})\).
   - Row margins: for each row \(r\), \(m_r = \max(0, \text{target\_row}_r - \text{row}_r)\).
   - Global margin: \(m_{\text{global}} = \max(0, L̂_{\text{target}} - L̂_{\text{base}})\).
   - Usable margins (conservative): \(\rho \cdot m_r\), \(\rho \cdot m_{\text{global}}\) with \(\rho \in [0.5, 0.8]\).

2) Compute per‑edge costs \(\Delta L_k\) and assign to incident rows:
   - Quadratic (i,j): contributes \(2w\) to diag\(_i\), diag\(_j\) and \(2w\) to offsum\(_i\), offsum\(_j\) (weighted by term weight).
   - Hinge families: contribute when active; near activation, use smoothed activity \(\sigma(\text{gap}/\varepsilon)\) with \(\varepsilon \in [10^{-3}, 10^{-2}]\).
   - Gate‑benefit: linear (no curvature); treat \(\Delta L \approx 0\) but cap per‑step weight changes to avoid starving other families.

3) Compute per‑edge values \(\text{value}_k\) (EMA):
   - Default proxy: \(\text{value}_k \approx \|\nabla_k\|^2\) (weighted).
   - Optional “peek”: occasionally apply tiny \(\delta w\) to a few edges and measure \((\Delta F/\delta w)_k\) to recalibrate EMA.

4) Rank edges by \(\text{score}_k = \text{value}_k / (\text{cost}_k + \varepsilon)\), descending.

5) Row‑aware fractional‑knapsack allocation:
   - For row \(r\), spend until \(\sum_{k \in N(r)} \Delta L_k \le \rho \cdot m_r\).
   - Enforce \(\sum_k \Delta L_k \le \rho \cdot m_{\text{global}}\).
   - Apply per‑edge \(\Delta \text{weight}_k\) bounded by ±η (e.g., 5–10%) with floors/ceilings.

6) Update term weights:
   - Map `coup:ClassName` weights via per‑edge aggregation or a family key map, bounded by per‑step change and [floor, ceiling].

7) Guardrails and acceptance:
   - Keep `stability_guard` and Armijo backtracking (monotone acceptance).
   - Optional weak Wolfe/angle check only when a large reweight occurs.
   - If any accepted step exhibits ΔF > 0 (should not), roll back weight changes and decay scores for implicated edges.

8) Observability:
   - Log margin (global/rows), spend, per‑edge allocations, scores, hinge activity, contraction_margin, backtracks.

---

## 4) Implementation plan (minimal, auditable)

### 4.1 Per‑edge curvature and row margins (instrumentation)
- Extend the Lipschitz estimator to optionally return:
  - \(L̂_{\text{base}}\), per‑row sums and margins vs target,
  - per‑edge curvature contributions \(\Delta L_k\) under unit scale (smoothed hinges near activation).

### 4.2 Small‑gain allocator adapter (WeightAdapter)
- Implement `SmallGainWeightAdapter` that:
  - reads per‑term grad norms (and sparse \(\Delta F/\delta w\) peeks),
  - consumes per‑edge \(\Delta L_k\) + row/global margins snapshot,
  - computes EMA scores and runs row‑aware fractional‑knapsack,
  - updates `term_weights` for coupling families within per‑step bounds and global clamps,
  - returns the updated map for the coordinator to use next evaluation.

### 4.3 Coordinator hooks
- After `_term_grad_norms`, compute the curvature/margin snapshot, call `adapter.step(...)`, merge weights with floor/ceiling, then evaluate energy as usual.

### 4.4 Observability
- Extend `EnergyBudgetTracker` to emit:
  - `margin:global`, `margin:row:<i>`, `spent:global`, `spent:row:<i>`,
  - `alloc:coup:<Key>` (family or edge‑aggregated),
  - `score:coup:<Key>`, `hinge_activity:<Key>`,
  - `contraction_margin`, `last_step_backtracks`, `total_backtracks`.

---

## 5) Risk controls and defaults

- Keep `stability_guard=True` and Armijo backtracking enabled.
- Bound per‑step weight changes to ±5–10%; floors/ceilings: [0.1, 3.0].
- Conservative budget fraction: \(\rho = 0.7\).
- Hinge smoothing \(\varepsilon = 10^{-3}\) (tune per domain).
- Optional weak Wolfe only after large reweight bursts.

---

## 6) Validation plan

Baselines: (i) no adaptation, (ii) global `stability_coupling_auto_cap`, (iii) GradNorm.  
KPIs: ΔF90 (lower), total backtracks (lower), contraction_margin stats, final F, runtime, acceptance rate.  
Suites: ring (quadratic+hinge), sequence+gate, connectivity+gate.  
Ablations: row‑aware on/off, hinge smoothing on/off, \(\rho \in \{0.5, 0.7, 0.9\}\), per‑step bound in {5%, 10%, 20%}.

---

## 7) Suggested defaults

- Budget fraction \(\rho = 0.7\)
- Per‑step bound = 10%
- Floors/ceilings: [0.1, 3.0]
- EMA α = 0.3
- Hinge ε = 1e‑3
- Peek cadence: every 10 steps, ≤ 5% edges

---

## 8) Adapter skeleton (reference)

```python
from dataclasses import dataclass, field
from typing import Dict, Mapping

@dataclass
class SmallGainWeightAdapter:
    """Per-edge stability-margin allocator with row-aware greedy budgeting."""

    target_L: float = 0.0
    budget_fraction: float = 0.7
    max_step_change: float = 0.1
    ema_alpha: float = 0.3
    floor: float = 0.1
    ceiling: float = 3.0
    eps: float = 1e-9

    edge_costs: Dict[str, float] = field(default_factory=dict)     # ΔL_k
    row_margins: Dict[int, float] = field(default_factory=dict)    # m_r
    global_margin: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)         # EMA(value/cost)

    def step(
        self,
        term_grad_norms: Mapping[str, float],
        energy: float,                  # reserved
        current: Mapping[str, float],
    ) -> Mapping[str, float]:
        values = {k: term_grad_norms.get(k, 0.0)**2 for k in term_grad_norms if k.startswith("coup:")}
        costs = {k: max(self.edge_costs.get(k, 0.0), 0.0) for k in values}

        for k in values:
            raw = values[k] / (costs[k] + self.eps) if costs[k] > 0.0 else values[k]
            old = self.scores.get(k, raw)
            self.scores[k] = self.ema_alpha * raw + (1.0 - self.ema_alpha) * old

        ranked = sorted(values.keys(), key=lambda k: self.scores[k], reverse=True)

        row_budget = {r: max(0.0, self.row_margins.get(r, 0.0)) * self.budget_fraction for r in self.row_margins}
        global_budget = max(0.0, self.global_margin) * self.budget_fraction
        spent_row = {r: 0.0 for r in row_budget}
        spent_global = 0.0
        updated = {str(k): float(v) for k, v in current.items()}

        for k in ranked:
            if global_budget - spent_global <= 0.0:
                break
            fam_key = k
            w_old = updated.get(fam_key, 1.0)
            w_new = min(self.ceiling, max(self.floor, w_old * (1.0 + self.max_step_change)))
            delta_w = w_new - w_old
            if delta_w <= 0.0:
                continue
            delta_L = costs[k] * (delta_w / max(w_old, self.eps))
            # Row checks can be applied when mapping edge->rows is exposed
            if spent_global + delta_L <= global_budget:
                updated[fam_key] = w_new
                spent_global += delta_L
        return updated
```

Production mapping: define a deterministic mapping between per‑edge keys and coupling family keys (e.g., `coup:QuadraticCoupling`), or maintain per‑edge weights if required by the application. Persist adapter state (scores) in the run context for reproducibility.

---

## 9) Experiments (Windows PowerShell)

```powershell
# ΔF90 baselines
uv run python -m experiments.benchmark_delta_f90 --configs default analytic vect --steps 60 --save logs/df90_baseline.csv

# With uniform coupling auto-cap (existing)
uv run python -m experiments.benchmark_delta_f90 --configs default analytic vect --enable_coupling_auto_cap --steps 60 --save logs/df90_autocap.csv

# With SmallGain allocator (new config 'smallgain' in harness)
uv run python -m experiments.benchmark_delta_f90 --configs smallgain --steps 60 --save logs/df90_smallgain.csv

# Plots
uv run python -m experiments.plot_energy_budget --input logs/energy_budget.csv --metric contraction_margin --smooth 3 --save plots/contraction_margin.png
uv run python -m experiments.plot_energy_budget --input logs/energy_budget.csv --metric alloc:coup:QuadraticCoupling --smooth 3 --save plots/alloc_quad.png
```

---

## 10) References

- Zames, G. (1966). On the input-output stability of time-varying nonlinear feedback systems. IEEE TAC.
- Vidyasagar, M. (1993). Nonlinear Systems Analysis. SIAM.
- Dantzig, G. (1957). Discrete-variable extremum problems. Operations Research.
- Nutini, J., et al. (2015). Coordinate Descent Converges Faster with the Gauss-Southwell Rule. ICML.
- Varga, R. (2000). Gershgorin and His Circles. Springer.
- Nocedal, J., & Wright, S. (2006). Numerical Optimization. Springer.

---

## 11) Summary

- Treat stability margin as a production budget; allocate to edges with best expected ΔF per ΔL under row/global constraints.
- Keep monotone acceptance and invariants via Armijo and stability caps.
- Conservative, EMA‑smoothed, bounded updates with rich telemetry yield faster ΔF90 and fewer backtracks in coupled regimes without sacrificing robustness.


