# Energy Conservation and Monotonic Energy Assertions

## Overview

The `EnergyCoordinator` enforces a default‑on **monotonic energy assertion** during deterministic gradient descent relaxation. This makes the "energy conservation" framing explicit and testable while remaining production‑safe via automatic guards.

## Quick Start

```python
# Default ON in deterministic mode
coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={},
    monotonic_energy_tol=1e-10,        # Tolerance for numeric jitter
    noise_magnitude=0.0,               # Must be deterministic
    line_search=False                  # Line search has its own logic
)
```

Disable explicitly when desired:
```python
coord = EnergyCoordinator(..., assert_monotonic_energy=False)
```

## What It Does

When active (default unless auto‑skipped or disabled), the coordinator asserts after every gradient descent step that:

```
F(η_{t+1}) ≤ F(η_t) + tolerance
```

If energy increases beyond the tolerance, the assertion fails with a detailed error message showing the energy delta and suggesting potential causes.

## Core Principle: Energy Conservation in Gradient Descent

Gradient descent with a sufficiently small step size on a smooth energy landscape should **never increase energy**. The update rule is:

```
η_{t+1} = η_t - α ∇F(η_t)
```

By Taylor expansion:

```
F(η_{t+1}) ≈ F(η_t) - α ||∇F||² + O(α²)
```

For small enough α, the linear term dominates and energy decreases. Any violation indicates:
- Gradient computation bug (wrong sign, missing terms, incorrect derivatives)
- Numerical instability (step size too large, ill-conditioned Hessian)
- Misconfigured coupling (incorrect weights, conflicting constraints)
- Floating-point accumulation errors

## Auto‑skip conditions

The assertion is automatically skipped when it doesn't conceptually apply:
- Exploration noise is active: `noise_magnitude > 1e-12` (second‑order effects can raise energy)
- Line search is enabled: `line_search=True` (trial steps may increase energy before backtracking)
- Energy function is changing: homotopy schedules (`homotopy_steps > 0`) or dynamic term weights (`weight_adapter is not None`)

## When to Enable `assert_monotonic_energy=True`

### ✅ Use Cases (Strongly Recommended)

1. **Unit Tests and CI/CD**
   - Enable in all deterministic test cases to catch regressions early.
   - Validates that gradient implementations are correct.
   - Ensures coupling logic preserves energy conservation.
   
   ```python
   def test_energy_monotonic():
       coord = EnergyCoordinator(
           modules=[...],
           couplings=[...],
           constraints={},
           assert_monotonic_energy=True,
           noise_magnitude=0.0,
           step_size=0.01  # Conservative for testing
       )
       etas_final = coord.relax_etas(etas0, steps=100)
       # If assertion passes, gradients are correct
   ```

2. **Debugging Gradient Implementations**
   - When adding new `EnergyModule` types with custom `d_local_energy_d_eta`.
   - When implementing new `EnergyCoupling` with analytic derivatives.
   - Catches sign errors, missing normalization, incorrect chain rule application.
   
   ```python
   # Testing a new CustomModule
   coord = EnergyCoordinator(
       modules=[CustomModule()],
       couplings=[],
       constraints={},
       assert_monotonic_energy=True,
       use_analytic=True  # Test analytic gradients
   )
   ```

3. **Validating Coupling Configurations**
   - Ensures term weights, coupling strengths, and constraint parameters don't create pathological landscapes.
   - Detects when competing objectives cause instability.
   
   ```python
   # Validate multi-term balance
   coord = EnergyCoordinator(
       modules=[gate_mod, seq_mod],
       couplings=[(0, 1, QuadraticCoupling(weight=10.0))],
       constraints={"term_weights": {"local:EnergyGatingModule": 1.0, ...}},
       assert_monotonic_energy=True
   )
   ```

4. **Benchmarking Deterministic Baselines**
   - When establishing performance baselines for comparison.
   - Ensures reported energy trajectories are physically valid.
   - Provides confidence that improvements come from better algorithms, not artifacts.

5. **Numerical Stability Analysis**
   - Test step size limits: gradually increase `step_size` until assertion fails to find stability boundary.
   - Validate stability guards: confirm that `stability_guard=True` prevents violations.
   - Check Lipschitz estimation accuracy.
   
   ```python
   # Find maximum safe step size
   for step_size in [0.001, 0.01, 0.05, 0.1, 0.2]:
       try:
           coord = EnergyCoordinator(..., assert_monotonic_energy=True, step_size=step_size)
           coord.relax_etas(etas0, steps=100)
           print(f"✓ step_size={step_size} is safe")
       except AssertionError:
           print(f"✗ step_size={step_size} violates monotonicity")
           break
   ```

## When to Disable `assert_monotonic_energy=False` (Default)

### ❌ Scenarios Where Assertion Should Be Off

1. **Exploration with Orthogonal Noise (`noise_magnitude > 0`)**
   - **Why**: Orthogonal noise is projected onto the gradient's null space, so it doesn't increase energy **to first order**. However, second-order effects (curvature of the energy landscape) can cause small energy increases.
   - **Mathematical Detail**: 
     ```
     η_{t+1} = η_t - α∇F + βz_⊥
     F(η_{t+1}) ≈ F(η_t) - α||∇F||² + (β²/2)z_⊥ᵀ H z_⊥ + O(α², β³)
     ```
     If the Hessian H has negative eigenvalues in the tangent space (saddle point or negative curvature direction), z_⊥ᵀ H z_⊥ < 0 can cause F to increase.
   - **Action**: Keep assertion **disabled** when `noise_magnitude > 1e-12`.
   - Prototype noise injection reference: docs/ISO-ENERGY_ORTHOGONAL_NOISE.md


   ```python
   # Exploration mode: assertion auto-disabled by guard condition
   coord = EnergyCoordinator(
       modules=[...],
       couplings=[...],
       constraints={},
       assert_monotonic_energy=False,  # Or True; guard will skip check
       noise_magnitude=0.05,           # Exploration active
       noise_schedule_decay=0.99
   )
   ```

2. **Line Search Enabled (`line_search=True`)**
   - **Why**: Line search (Armijo backtracking) internally tries steps and **expects** some trial steps to increase energy. It backtracks until finding an acceptable step. The assertion would spuriously trigger on intermediate rejected steps.
   - **Detail**: The backtracking loop computes:
     ```
     F(η_t - α_trial ∇F) vs F(η_t) - c α_trial ||∇F||²
     ```
     and accepts only when Armijo condition holds. Intermediate trials may violate monotonicity.
   - **Action**: Assertion is **automatically skipped** when `line_search=True`.
   
   ```python
   # Line search mode: assertion disabled by guard
   coord = EnergyCoordinator(
       modules=[...],
       couplings=[...],
       constraints={},
       assert_monotonic_energy=True,   # Guard skips check anyway
       line_search=True,
       backtrack_factor=0.5,
       max_backtrack=5
   )
   ```

3. **Adaptive or Exotic Optimization Methods**
   - **Why**: Methods like adaptive coordinate descent, ADMM, or proximal splitting may have transient energy increases as part of their convergence strategy.
   - **Examples**:
     - **Adaptive coordinate descent**: Switching between global gradient descent and coordinate updates can cause small overshoots.
     - **ADMM**: The augmented Lagrangian includes penalty terms that don't strictly decrease original energy F every iteration.
     - **Operator splitting**: Proximal steps may increase F temporarily while enforcing constraints.
   - **Action**: Disable assertion for these advanced modes.
   
   ```python
   # ADMM mode: allow transient increases
   coord = EnergyCoordinator(
       modules=[...],
       couplings=[...],
       constraints={},
       assert_monotonic_energy=False,
       use_admm=True,
       admm_rho=1.0
   )
   ```

4. **Homotopy and Continuation Schedules**
   - **Why**: When sweeping coupling scales (`homotopy_coupling_scale_start`) or term weights over iterations, the energy function **itself changes**. Comparing F_t (under schedule₁) to F_{t+1} (under schedule₂) is meaningless—you're measuring different functions.
   - **Mathematical Detail**:
     ```
     F_t(η) = Σ w_i(t) E_i(η)
     F_{t+1}(η) = Σ w_i(t+1) E_i(η)
     ```
     w_i changes → not the same function → monotonicity doesn't apply.
   - **Action**: Disable when `homotopy_steps > 0` or using dynamic weight adapters that rescale during relaxation.
   
   ```python
   # Homotopy continuation: energy function morphs
   coord = EnergyCoordinator(
       modules=[...],
       couplings=[...],
       constraints={},
       assert_monotonic_energy=False,
       homotopy_coupling_scale_start=0.1,
       homotopy_steps=50
   )
   ```

5. **Floating-Point Accumulation in Long Runs**
   - **Why**: Over thousands of iterations with many coupling terms, floating-point rounding can accumulate. Energy computed as `sum([term1, term2, ..., term_N])` may have O(εN) jitter even with correct gradients.
   - **Symptom**: Assertion fails sporadically with tiny deltas like `Δ = 1.8e-14`.
   - **Action**: Either disable, or increase `monotonic_energy_tol` to something like `1e-8` for long runs.
   
   ```python
   # Long relaxation with many terms
   coord = EnergyCoordinator(
       modules=[...] * 50,  # Many modules
       couplings=[...] * 100,
       constraints={},
       assert_monotonic_energy=True,
       monotonic_energy_tol=1e-8  # Relax tolerance for accumulation
   )
   coord.relax_etas(etas0, steps=10000)  # Long run
   ```

6. **Production / User-Facing Applications**
   - **Why**: Assertions are for **development and testing**. In production, you want logging/metrics, not crashes. Use `EnergyBudgetTracker` to log monotonicity violations as metrics instead.
   - **Action**: Keep assertion disabled; monitor via telemetry.
   
   ```python
   # Production: log, don't assert
   from cf_logging.observability import EnergyBudgetTracker
   
   coord = EnergyCoordinator(
       modules=[...],
       couplings=[...],
       constraints={},
       assert_monotonic_energy=False  # No crashes in prod
   )
   
   tracker = EnergyBudgetTracker(run_id="prod-run-123")
   tracker.attach(coord)
   # Logs energy deltas; analyze offline for violations
   ```

## Guard Conditions (Automatic Disabling)

The assertion includes **guard conditions** that automatically skip the check even if `assert_monotonic_energy=True`:

```python
if (
    self.assert_monotonic_energy           # User enabled
    and self.noise_magnitude <= 1e-12      # Deterministic mode
    and not self.line_search               # Not using line search
    and prev_energy_value is not None      # Not first iteration
):
    assert energy_value <= prev_energy_value + self.monotonic_energy_tol
```

This means:
- You can safely enable in tests even if some code paths use noise/line search—guards prevent false positives.
- Assertion only fires in pure gradient descent with no exploration.

## Tuning `monotonic_energy_tol`

**Default**: `1e-10` (tight, catches real bugs while allowing minimal numeric jitter)

**When to adjust**:
- **Tighten to `1e-12` or `1e-14`**: When testing very clean, low-dimensional problems with exact arithmetic expectations.
- **Relax to `1e-8`**: When running long (>1000 steps) relaxations with many (>20) coupling terms where floating-point accumulation is expected.
- **Relax to `1e-6`**: When step size is large (`step_size > 0.1`) and you want to allow small overshoots from linearization error.

**Rule of thumb**:
```
tolerance ≈ max(machine_epsilon * num_terms * num_steps, step_size² * ||Hessian||)
```

For typical problems: `1e-10` to `1e-8` is the sweet spot.

## Assertion Failure: What It Means and How to Fix

### Error Message Example
```
AssertionError: Energy increased: 2.453871629403e+00 → 2.453871631821e+00 (Δ=2.418e-09). 
This indicates a gradient bug, numerical instability, or misconfigured coupling. 
Disable assert_monotonic_energy if using exploration noise, line search, or exotic schedules.
```

### Diagnostic Steps

1. **Check step size**: Try `step_size=0.001`. If assertion passes, you need smaller steps or stability guards.
   
2. **Verify gradients**: Switch to finite differences and compare:
   ```python
   coord_analytic = EnergyCoordinator(..., use_analytic=True)
   coord_numeric = EnergyCoordinator(..., use_analytic=False, grad_eps=1e-6)
   # Compare final etas and trajectories
   ```

3. **Inspect coupling weights**: Print `constraints["term_weights"]`. Look for:
   - Very large weights (>1e6) causing stiffness
   - Negative weights (should never happen unless intentional)
   - Weights that grow over time (homotopy active?)

4. **Check for NaNs/Infs**: Look at `_check_invariants` output. If etas or energy are non-finite, you have a deeper numerical issue.

5. **Enable stability guard**: If Lipschitz constant is large, add:
   ```python
   coord = EnergyCoordinator(..., stability_guard=True, stability_cap_fraction=0.9)
   ```

6. **Simplify the problem**: Remove couplings one by one. If assertion passes with fewer couplings, you've identified the culprit.

## Integration with Energy Budget Tracking

For production-grade monitoring, combine with `EnergyBudgetTracker`:

```python
from cf_logging.observability import EnergyBudgetTracker

coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={},
    assert_monotonic_energy=True  # Hard fail in dev/test
)

tracker = EnergyBudgetTracker(run_id="experiment-42")
tracker.attach(coord)

# After relaxation, analyze energy trajectory
df = pd.read_csv("logs/energy_budget.csv")
df_run = df[df["run_id"] == "experiment-42"]

# Check for violations
energy_deltas = df_run["energy"].diff()
violations = energy_deltas[energy_deltas > 1e-10]
print(f"Monotonicity violations: {len(violations)}")
```

## Comparison to Early-Stop Guard

The coordinator already has a **soft early-stop** guard:

```python
if prev_energy_value is not None and energy_value > prev_energy_value + 1e-12:
    break  # Stop iteration, don't crash
```

**Differences**:

| Feature | Early-Stop Guard | `assert_monotonic_energy` |
|---------|------------------|---------------------------|
| **Action** | Break loop, return current η | Crash with detailed error |
| **Purpose** | Prevent wasted compute on oscillations | Catch bugs during development |
| **Always active** | Yes | Opt-in (`assert_monotonic_energy=True`) |
| **Tolerance** | Hardcoded `1e-12` | Configurable `monotonic_energy_tol` |
| **Use case** | Production, experiments | Testing, validation, debugging |

**Recommendation**: Keep both. Early-stop prevents runaway loops; assertion catches bugs before they reach production.

## Summary: Decision Tree

```
Do you want strict energy conservation validation?
│
├─ YES → Enable assert_monotonic_energy=True
│   │
│   ├─ Using noise_magnitude > 0?
│   │   └─ YES → Guard auto-skips; no action needed
│   │
│   ├─ Using line_search=True?
│   │   └─ YES → Guard auto-skips; no action needed
│   │
│   ├─ Using homotopy or adaptive methods?
│   │   └─ YES → Disable assertion; energy function changes
│   │
│   ├─ Very long runs (>1000 steps) or many terms (>50)?
│   │   └─ YES → Increase monotonic_energy_tol to 1e-8
│   │
│   └─ Pure gradient descent, deterministic?
│       └─ YES → Assertion active; catches bugs ✓
│
└─ NO → Keep assert_monotonic_energy=False (default)
    └─ Use EnergyBudgetTracker for soft monitoring
```

## Theoretical Background: Why Monotonicity = Energy Conservation

In statistical mechanics and optimization, **energy conservation** and **monotonic energy decrease** are closely related but distinct concepts:

1. **Physical Energy Conservation**: Total energy (kinetic + potential) is conserved in isolated Hamiltonian systems. In dissipative systems (like gradient descent with friction), energy decreases.

2. **Free Energy Minimization**: Gradient flow on a Lyapunov function F(η) naturally decreases F. The rate is:
   ```
   dF/dt = -||∇F||² ≤ 0
   ```
   This is "energy dissipation," not "conservation."

3. **Why "Energy Conservation" in This Context**: We're checking that the **numerical scheme conserves the property** that F should decrease. If F increases, the discretized dynamics violate the continuous-time guarantee—a bug.

So "monotonic energy assertion" is really checking **conservation of the monotonicity property** under discretization. It's a discrete analog of energy conservation in the sense that we're validating the optimization respects the structure of the continuous system.

## References and Further Reading

- **Lyapunov Stability**: Any gradient descent on a smooth function is stable because energy is a Lyapunov function. Violations indicate the discrete scheme breaks stability.
- **Armijo-Goldstein Condition**: Line search ensures sufficient decrease; relaxing monotonicity requires proving convergence via other means.
- **Symplectic Integrators**: In physics, structure-preserving integrators (symplectic, variational) conserve energy to machine precision. This assertion is the optimization analog.

## Code Example: Full Workflow

```python
from core.coordinator import EnergyCoordinator
from core.couplings import QuadraticCoupling
from modules.gating.energy_gating import EnergyGatingModule
from modules.sequence.monotonic_eta import SequenceConsistencyModule
from cf_logging.observability import EnergyBudgetTracker

# Setup
gate_mod = EnergyGatingModule(cost=0.5)
seq_mod = SequenceConsistencyModule()

coord = EnergyCoordinator(
    modules=[gate_mod, seq_mod],
    couplings=[(0, 1, QuadraticCoupling(weight=1.0))],
    constraints={},
    # Strict validation in test mode
    assert_monotonic_energy=True,
    monotonic_energy_tol=1e-10,
    # Deterministic
    noise_magnitude=0.0,
    line_search=False,
    # Conservative step size for testing
    step_size=0.01,
    # Enable invariant checks
    enforce_invariants=True
)

# Attach tracker
tracker = EnergyBudgetTracker(run_id="test-monotonic")
tracker.attach(coord)

# Run
etas0 = [0.0, 0.5]
etas_final = coord.relax_etas(etas0, steps=100)

# If we get here without assertion, gradients are correct ✓
print("✓ Energy monotonicity validated")
print(f"Final etas: {etas_final}")
```

## Conclusion

The `assert_monotonic_energy` feature completes the energy conservation picture in deterministic mode. It's a powerful debugging tool that catches subtle gradient bugs, numerical instability, and misconfigured couplings early—before they cause silent failures in experiments.

**Key Takeaways**:
1. **Enable in tests and CI** to validate gradient implementations.
2. **Disable in production** and use soft monitoring instead.
3. **Guards automatically skip** when noise, line search, or other non-deterministic features are active.
4. **Tune tolerance** based on problem size and step size.
5. **Use with EnergyBudgetTracker** for comprehensive energy analysis.

When in doubt: enable during development, disable in production. The guards make it safe to leave on in test suites even when some paths use exploration or line search.

