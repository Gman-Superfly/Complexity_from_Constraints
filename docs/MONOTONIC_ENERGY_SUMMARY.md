# Summary: Monotonic Energy Assertion Feature

## What Was Added

Added optional **monotonic energy assertion** feature to `EnergyCoordinator` for debugging and validation.

### Code Changes

1. **`core/coordinator.py`**:
   - Added two new parameters (both **opt-in**, defaults preserve existing behavior):
     - `assert_monotonic_energy: bool = False` — Enable strict energy conservation check
     - `monotonic_energy_tol: float = 1e-10` — Tolerance for numeric jitter
   - Added assertion logic in `relax_etas()` loop with automatic guard conditions
   - Guards skip assertion when: `noise_magnitude > 1e-12`, `line_search=True`, or first iteration

2. **`docs/ENERGY_CONSERVATION_AND_MONOTONICITY.md`**:
   - Comprehensive 400+ line guide explaining:
     - When to enable (tests, debugging, validation, benchmarking)
     - When to disable (noise, line search, ADMM, homotopy, production)
     - Mathematical background (Taylor expansion, Lyapunov stability)
     - Troubleshooting guide with diagnostic steps
     - Integration with `EnergyBudgetTracker`

3. **`tests/test_monotonic_energy.py`**:
   - 6 new tests validating:
     - ✅ Assertion passes in deterministic mode
     - ✅ Guards skip assertion with noise
     - ✅ Guards skip assertion with line search
     - ✅ Assertion catches energy increase from large step size
     - ✅ Tolerance allows small numeric jitter
     - ✅ Disabled by default (backward compatibility)

4. **`README.md`**:
   - Added section on energy conservation with quick-start example
   - Clear guidance on when to enable/disable
   - Link to detailed documentation

## Backward Compatibility ✅

- **Default behavior unchanged**: `assert_monotonic_energy=False` by default
- **All existing tests pass**: 73 passed, 1 skipped
- **Existing soft guard remains**: The `break` at line 361-362 still prevents oscillations in all cases
- **Guards prevent false positives**: Assertion auto-skips when noise/line search are active

## Safety Guarantees

1. **Opt-in only**: Feature must be explicitly enabled
2. **Automatic guards**: Skips when non-deterministic features are active
3. **Soft fallback exists**: Existing early-stop guard at line 361-362 still protects all code paths
4. **Production-safe**: Default `False` means production code is unaffected

## Use Cases

### Enable (`assert_monotonic_energy=True`)
- Unit tests and CI/CD
- Debugging new gradient implementations
- Validating coupling configurations
- Benchmarking deterministic baselines
- Finding maximum safe step size

### Keep Disabled (default)
- Production deployments
- Exploration runs (`noise_magnitude > 0`)
- Line search modes
- Adaptive methods (ADMM, operator-splitting)
- Homotopy schedules

## How It Works

```python
# In relax_etas() loop, after computing energy_value:
if (
    self.assert_monotonic_energy           # User enabled
    and self.noise_magnitude <= 1e-12      # Deterministic mode
    and not self.line_search               # Not using line search
    and prev_energy_value is not None      # Not first iteration
):
    assert energy_value <= prev_energy_value + self.monotonic_energy_tol, (
        f"Energy increased: {prev_energy_value:.12e} → {energy_value:.12e} ..."
    )
```

## Key Insight: Why This Matters

Gradient descent should **never increase energy** in deterministic mode. Any violation indicates:
- Gradient bug (wrong sign, missing terms)
- Numerical instability (step size too large)
- Misconfigured coupling (conflicting constraints)
- Floating-point issues

By making this explicit and testable, we catch bugs **before** they reach experiments or production.

## Testing Results

```
tests/test_monotonic_energy.py::test_monotonic_energy_assertion_passes_in_deterministic_mode PASSED
tests/test_monotonic_energy.py::test_monotonic_energy_assertion_skipped_with_noise PASSED
tests/test_monotonic_energy.py::test_monotonic_energy_assertion_skipped_with_line_search PASSED
tests/test_monotonic_energy.py::test_monotonic_energy_assertion_triggers_on_large_step PASSED
tests/test_monotonic_energy.py::test_monotonic_energy_tolerance_allows_small_jitter PASSED
tests/test_monotonic_energy.py::test_monotonic_energy_disabled_by_default PASSED

All 73 existing tests still pass ✅
```

## Relation to "Slop" Concept

In our earlier discussion, we explored "slop" as compressibility vs value. In energy terms:

- **Slop in text** = redundancy unjustified by utility U
- **Slop in energy** = excess energy (ΔF) over the minimum needed for the same utility
- **Energy conservation** = ensuring we don't add slop (unnecessary energy) during optimization

The monotonic energy assertion ensures that gradient descent **removes slop** (reduces F) rather than adding it. It's the operational check that we're moving toward compression/minimization, not away from it.

## Documentation Structure

```
README.md
  ↓ Quick start + when to enable/disable
  ↓
docs/ENERGY_CONSERVATION_AND_MONOTONICITY.md
  ↓ Deep dive:
  ├─ Mathematical background (Taylor expansion, Lyapunov)
  ├─ When to enable (detailed use cases)
  ├─ When to disable (with mathematical reasons)
  ├─ Guard conditions explained
  ├─ Tolerance tuning guide
  ├─ Troubleshooting assertion failures
  ├─ Integration with EnergyBudgetTracker
  └─ Decision tree + code examples
```

## Next Steps (Optional)

If you want to use this in practice:

1. **Enable in CI**: Add `assert_monotonic_energy=True` to test fixtures
2. **Validate new modules**: Enable when adding custom `EnergyModule` or `EnergyCoupling`
3. **Find step size limits**: Use to determine maximum safe `step_size` for your problem
4. **Soft monitoring**: In production, use `EnergyBudgetTracker` to log (not assert) violations

## Conclusion

- ✅ Feature added with full backward compatibility
- ✅ Default disabled to preserve existing behavior
- ✅ Comprehensive documentation and tests
- ✅ Guards prevent false positives
- ✅ All existing tests pass

The system now has explicit, testable energy conservation validation available **on-demand** for development and testing, while production code remains untouched.

