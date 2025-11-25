# Proximal Methods & ADMM in Complexity from Constraints

## Overview

This document explains the proximal operator and ADMM (Alternating Direction Method of Multipliers) modes available in `EnergyCoordinator`, when to use them, and how they differ from standard gradient descent.

**Status**: PRODUCTION READY ✅ (All coupling families supported, 120 tests passing)

---

## Quick Start

### Standard Proximal Mode

```python
from core.coordinator import EnergyCoordinator

coord = EnergyCoordinator(
    modules=my_modules,
    couplings=my_couplings,
    constraints={},
    operator_splitting=True,
    prox_tau=0.05,          # proximal step size
    prox_steps=50,
    prox_block_mode="star",  # Jacobi star blocks (optional)
)

etas = coord.relax_etas(etas0, steps=50)
```

### ADMM Mode

```python
coord = EnergyCoordinator(
    modules=my_modules,
    couplings=my_couplings,
    constraints={},
    use_admm=True,
    admm_rho=1.0,           # penalty parameter
    admm_steps=50,
    admm_step_size=0.05,
    admm_gate_prox=True,    # prox-linear for gate-benefit
    admm_gate_damping=0.5,  # blend factor
)

etas = coord.relax_etas_admm(etas0, steps=50, rho=1.0, step_size=0.05)
```

---

## What Are Proximal Operators?

### Intuition

**Gradient Descent**: "Move in the direction that reduces energy most"  
**Proximal Update**: "Find the point that balances energy reduction with staying near the current position"

Mathematically, the proximal operator solves:

\[
x^{k+1} = \arg\min_x \left\{ F(x) + \frac{1}{2\tau} \|x - x^k\|^2 \right\}
\]

The second term is a "trust region" that penalizes large steps.

### Why Use Proximal Methods?

1. **Better for non-smooth energies**: Hinge constraints (ReLU-like) have kinks where gradients are undefined
2. **Implicit regularization**: The trust region automatically prevents overshooting
3. **Parallelization**: Block updates can be done in parallel (Jacobi scheme)
4. **Convergence guarantees**: Proven to converge even when gradient descent fails

---

## Implementation Details

### Coupling Families & Prox Operators

| Coupling Type | Prox Operator | Closed-Form? | Notes |
|---------------|---------------|--------------|-------|
| **QuadraticCoupling** | `prox_quadratic_pair` | ✅ Yes | 2×2 linear system, analytical inversion |
| **DirectedHingeCoupling** | `prox_asym_hinge_pair` | ✅ Yes | Checks gap sign, solves 2×2 when active |
| **AsymmetricHingeCoupling** | `prox_asym_hinge_pair` | ✅ Yes | Scales α, β asymmetrically |
| **GateBenefitCoupling** | `prox_linear_gate` | ✅ Yes | Linear term: η ← η₀ + τ·coeff |
| **DampedGateBenefitCoupling** | `prox_linear_gate` | ✅ Yes | With damping blend |

All prox operators are in `core/prox_utils.py`.

### Block Modes

#### Sequential (Default)

Updates locals first, then couplings sequentially:

```python
for each module i:
    η_i ← prox(local energy)
for each coupling (i,j):
    (η_i, η_j) ← prox(coupling energy)
```

#### Star Blocks (`prox_block_mode="star"`)

Updates each module together with its adjacent couplings (Jacobi-style):

```python
for each module i in parallel:
    η_i ← prox(local energy + all incident couplings)
```

**Benefit**: More parallelizable, smoother energy descent  
**Cost**: Slightly more compute per iteration

##### Visual: Star Block Around Node i

```
      j
      │
  k ──●── m     (update i together with all incident couplings)
      │
      ℓ

Block prox at i:
  η_i ← prox( F_local(i) + Σ_{(i,•)} F_coupling(i, •) )
(Run these star updates for all i in parallel, Jacobi-style)
```

---

## ADMM Mode (Advanced)

### What Is ADMM?

ADMM splits the optimization into:
1. **Primal update** (η variables)
2. **Dual update** (Lagrange multipliers u)
3. **Auxiliary update** (s variables for consensus)

For each quadratic coupling \( w(η_i - η_j)^2 \), ADMM introduces:
- Auxiliary variable: \( s_{ij} = η_i - η_j \)
- Dual variable: \( u_{ij} \) (enforces consensus)

Update steps:

\[
\begin{align}
s^{k+1} &= \arg\min_s \left\{ w s^2 + \frac{\rho}{2}(s - (η_i - η_j) - u)^2 \right\} \\
η^{k+1} &= \arg\min_η \left\{ F_{local}(η) + \frac{\rho}{2}\sum (η_i - η_j - s + u)^2 \right\} \\
u^{k+1} &= u + (s - (η_i - η_j))
\end{align}
\]

#### Visual: ADMM Update Cycle and Consensus

```
      +------------+        primal variables (η)
      |  η-update  |  ← minimize local energy + penalties
      +------------+
             │
             v
      +------------+        auxiliary differences (s)
      |  s-update  |  ← minimize per-edge quadratic with ρ
      +------------+
             │
             v
      +------------+        duals (u) enforce agreement
      |  u-update  |  ← u ← u + (s − (η_i − η_j))
      +------------+
             │
             └──────────── repeat until residuals small

Consensus on edge (i, j):
  s_ij  ≈  η_i − η_j     (auxiliary matches actual difference)
  u_ij enforces the constraint via augmented Lagrangian
```

### When to Use ADMM vs Proximal

| Use Case | Recommended Method | Why |
|----------|-------------------|-----|
| **Dense coupling graphs** | ADMM | Better handles many constraints simultaneously |
| **Hard constraints** (hinges) | ADMM | Auxiliary variables make constraints explicit |
| **Sparse graphs** | Proximal (star blocks) | Simpler, less overhead |
| **Gate-heavy systems** | Proximal | Gate-benefit uses linear prox (simple) |

### ADMM Parameters

- **`admm_rho`**: Penalty parameter (default: 1.0)
  - Higher ρ → faster consensus, but tighter coupling (may need smaller step_size)
  - Lower ρ → slower consensus, looser coupling (more stable)
  
- **`admm_step_size`**: Step size for η-update (default: 0.05)
  - Similar to gradient descent step size
  - Tune based on Lipschitz constant of local energies

- **`admm_gate_prox`**: Apply prox-linear update for gate-benefit couplings (default: True)
  - Uses `prox_linear_gate` with damping
  
- **`admm_gate_damping`**: Blend factor for gate prox (default: 0.5)
  - 0.0 = no prox correction (gradient only)
  - 1.0 = full prox correction
  - 0.5 = blend gradient + prox

---

## Performance Comparison

From `experiments/benchmark_delta_f90.py` results:

| Config | ΔF90 Steps | Final Energy | Wall Time | Notes |
|--------|-----------|--------------|-----------|-------|
| **Gradient (analytic)** | 22 | -0.000385 | 0.0052s | Baseline |
| **Proximal** | 15 | -0.015 | 0.008s | 30% faster, smoother |
| **Proximal (star)** | 12 | -0.018 | 0.010s | 45% faster |
| **ADMM** | 18 | -0.012 | 0.012s | Better for dense graphs |

**Key Takeaway**: Proximal star blocks achieve best ΔF90 on sparse-moderate graphs. ADMM shines on dense constraint graphs (10+ couplings per node).

---

## Choosing the Right Method

### Decision Tree

```
Is your graph dense (>5 couplings per node)?
  YES → Use ADMM
  NO → Continue

Do you have hinge constraints?
  YES → Use Proximal (better for non-smooth)
  NO → Continue

Do you need parallelization?
  YES → Use Proximal with star blocks
  NO → Use standard gradient descent
```

### Practical Guidelines

**Use Gradient Descent When**:
- Small graphs (<5 modules)
- Smooth energies (no hinges)
- Prototyping (simplest to debug)

**Use Proximal When**:
- Non-smooth constraints (hinges, gates)
- Moderate graphs (5-20 modules)
- Want implicit step-size regularization

**Use Proximal Star Blocks When**:
- Sparse-moderate graphs
- Can parallelize across modules
- Want fastest ΔF90

**Use ADMM When**:
- Dense coupling graphs (10+ couplings per node)
- Hard constraints that must be exact
- Need formal consensus convergence guarantees

---

## API Reference

### Proximal Mode Toggles

```python
EnergyCoordinator(
    operator_splitting=True,    # Enable proximal mode
    prox_tau=0.05,              # Proximal step size
    prox_steps=50,              # Number of proximal iterations
    prox_block_mode="star",     # Optional: "star" for Jacobi blocks
)
```

### ADMM Mode Toggles

```python
EnergyCoordinator(
    use_admm=True,             # Enable ADMM mode
    admm_rho=1.0,              # Penalty parameter
    admm_steps=50,             # Number of ADMM iterations
    admm_step_size=0.05,       # Step size for η-update
    admm_gate_prox=True,       # Use prox for gate-benefit
    admm_gate_damping=0.5,     # Damping for gate prox
)
```

### Calling ADMM Relaxation

```python
# Standard gradient relaxation
etas = coord.relax_etas(etas0, steps=50)

# ADMM relaxation (when use_admm=True)
etas = coord.relax_etas_admm(
    etas0,
    steps=coord.admm_steps,
    rho=coord.admm_rho,
    step_size=coord.admm_step_size
)
```

---

## Test Coverage

Comprehensive test suite validates all proximal/ADMM paths:

### Proximal Tests

- `tests/test_coordinator_prox.py`:
  - ✅ Non-increasing energy
  - ✅ Parity with gradient descent on small cases
  
- `tests/test_prox_blocks.py`:
  - ✅ Star blocks match sequential
  - ✅ Energy decreases with star mode

- `tests/test_prox_operators.py`:
  - ✅ Quadratic pair symmetry
  - ✅ Hinge inactive region identity
  - ✅ Linear gate moves correctly

### ADMM Tests

- `tests/test_coordinator_admm.py`:
  - ✅ Non-increasing energy with quadratic couplings
  
- `tests/test_coordinator_admm_hinge.py`:
  - ✅ Directed hinge non-increasing
  - ✅ Asymmetric hinge non-increasing
  
- `tests/test_coordinator_admm_gate_benefit.py`:
  - ✅ Gate-benefit non-increasing energy
  
- `tests/test_admm_damped_gate_benefit.py`:
  - ✅ Damped gate-benefit non-increasing
  - ✅ Parity with gradient descent

- `tests/test_admm_parity_small.py`:
  - ✅ Mixed problem parity (quadratic + hinge + gate-benefit)

**Run all proximal/ADMM tests**:

```powershell
uv run -m pytest tests/ -k "prox or admm" -v
```

---

## Troubleshooting

### ADMM Not Converging

**Symptoms**: Energy oscillates or increases  
**Fixes**:
1. Reduce `admm_step_size` (try 0.01-0.03)
2. Increase `admm_rho` (try 2.0-5.0)
3. Enable monotonicity guard: `enforce_invariants=True`
4. Check for numerical issues: set `admm_gate_damping=0.3` (more conservative)

### Proximal Mode Too Slow

**Symptoms**: Wall time much higher than gradient descent  
**Fixes**:
1. Use star blocks: `prox_block_mode="star"`
2. Reduce `prox_steps` (try 20-30 instead of 50)
3. Increase `prox_tau` slightly (try 0.08-0.10)
4. Consider gradient descent if graph is small/smooth

### Energy Still Increasing

**Symptoms**: Monotonicity assertions fail  
**Fixes**:
1. Check that all modules/couplings implement gradients correctly
2. Enable `line_search=True` in proximal mode
3. Reduce step sizes (`prox_tau` or `admm_step_size`)
4. Use `stability_guard=True` to auto-cap steps

---

## Mathematical Background

### Proximal Operator Definition

For a function \( F: \mathbb{R}^n \to \mathbb{R} \), the proximal operator is:

\[
\text{prox}_{\tau F}(x) = \arg\min_z \left\{ F(z) + \frac{1}{2\tau} \|z - x\|^2 \right\}
\]

### Properties

1. **Firmly non-expansive**: \( \|\text{prox}_F(x) - \text{prox}_F(y)\| \leq \|x - y\| \)
2. **Convergence**: Fixed-point iterations converge for convex F
3. **Monotonicity**: Energy is non-increasing when τ is small enough

### ADMM Consensus Form

For a separable problem \( \min F(η) + \sum G_k(Cη) \), ADMM alternates:

\[
\begin{align}
η^{k+1} &= \arg\min_η \left\{ F(η) + \frac{\rho}{2}\|Cη - z^k + u^k\|^2 \right\} \\
z^{k+1} &= \arg\min_z \left\{ \sum G_k(z) + \frac{\rho}{2}\|Cη^{k+1} - z + u^k\|^2 \right\} \\
u^{k+1} &= u^k + (Cη^{k+1} - z^{k+1})
\end{align}
\]

In our case:
- \( F(η) \) = sum of local energies
- \( G_k \) = coupling energies
- \( C \) = constraint matrix (e.g., \( η_i - η_j \) for quadratic)

---

## Implementation Notes

### Closed-Form Prox for Quadratic Coupling

For \( F(η_i, η_j) = w(η_i - η_j)^2 \), the prox step solves a 2×2 linear system:

\[
\begin{bmatrix}
2w + 1/\tau & -2w \\
-2w & 2w + 1/\tau
\end{bmatrix}
\begin{bmatrix}
η_i \\
η_j
\end{bmatrix}
=
\begin{bmatrix}
η_i^0 / \tau \\
η_j^0 / \tau
\end{bmatrix}
\]

**Implementation**: `prox_quadratic_pair` in `core/prox_utils.py`

### Prox for Hinge (Directional Constraint)

For \( F = w \max(0, β η_j - α η_i)^2 \):

- If gap ≤ 0: prox is identity (constraint inactive)
- If gap > 0: solve 2×2 system with penalty on gap

**Implementation**: `prox_asym_hinge_pair` in `core/prox_utils.py`

### Prox for Gate-Benefit (Linear Term)

For \( F = -w · η_{gate} · \Delta \), the prox step is:

\[
η^{k+1} = \text{clip}_{[0,1]}(η^k + \tau · w · \Delta)
\]

**Implementation**: `prox_linear_gate` in `core/prox_utils.py`

With damping (ADMM mode):

\[
η^{k+1} = (1 - d) · η^k + d · \text{prox}(η^k)
\]

where \( d \in [0, 1] \) is `admm_gate_damping`.

---

## When NOT to Use Proximal/ADMM

1. **Very small graphs** (<3 modules): Gradient descent is simpler and faster
2. **Smooth energies only**: No hinges/gates → gradient descent is sufficient
3. **Real-time systems**: Proximal/ADMM have ~2-3x overhead vs gradient descent
4. **When you need max speed**: Use analytic gradients with vectorization instead

---

## Validation Results

From `experiments/benchmark_delta_f90.py`:

### Baseline Scenario (2 modules, 2 couplings)

| Method | ΔF90 | Final Energy | Wall Time |
|--------|------|--------------|-----------|
| Analytic | 22 | -0.000385 | 0.0052s |
| Proximal | 15 | -0.015 | 0.008s |
| **Prox Star** | **12** ✅ | **-0.018** ✅ | 0.010s |
| ADMM | 18 | -0.012 | 0.012s |

### Dense Scenario (16 modules, dense graph)

| Method | ΔF90 | Final Energy | Wall Time |
|--------|------|--------------|-----------|
| Analytic | 40 | +0.019 ❌ | 0.020s |
| Proximal | 25 | -0.05 | 0.045s |
| Prox Star | 20 | -0.06 | 0.050s |
| **ADMM** | **18** ✅ | **-0.08** ✅ | 0.080s |

**Conclusion**: Proximal star blocks optimal for sparse-moderate; ADMM optimal for dense.

---

## References

### Code

- Implementation: `core/coordinator.py` (`relax_etas_proximal`, `relax_etas_admm`)
- Operators: `core/prox_utils.py`
- Tests: `tests/test_coordinator_prox.py`, `tests/test_admm_*.py`

### Theory

- Boyd, S., et al. (2011). "Distributed Optimization and Statistical Learning via ADMM." Foundations and Trends in Machine Learning.
- Parikh, N., & Boyd, S. (2014). "Proximal Algorithms." Foundations and Trends in Optimization.

### Related

- `docs/STABILITY_GUARANTEES.md` — How proximal methods help with stability
- `docs/META_LEARNING.md` — Using proximal with weight adapters
- `README.md` — Quick-start examples

---

## Future Work

- [ ] Proximal for higher-order couplings (3-way constraints)
- [ ] Accelerated ADMM (Nesterov momentum)
- [ ] Warm-start strategies (use amortizer proposals)
- [ ] Distributed ADMM (consensus across workers)

