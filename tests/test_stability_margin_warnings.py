from __future__ import annotations

import warnings
from typing import Any, List, Tuple, Dict

from core.coordinator import EnergyCoordinator
from modules.gating.energy_gating import EnergyGatingModule
from core.couplings import QuadraticCoupling


def _make_tight_coupling_setup() -> Tuple[List[Any], List[Tuple[int, int, Any]], Dict[str, Any], List[Any]]:
    """Setup with strong coupling to stress stability margin."""
    m0 = EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.5, b=0.5)
    m1 = EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.5, b=0.5)
    modules = [m0, m1]
    # Strong coupling to trigger margin warnings
    couplings = [(0, 1, QuadraticCoupling(weight=10.0))]
    constraints: Dict[str, Any] = {}
    inputs: List[Any] = [None, None]
    return modules, couplings, constraints, inputs


def test_stability_margin_warning_emitted_below_threshold() -> None:
    """Coordinator should emit warning when contraction margin drops below threshold."""
    modules, couplings, constraints, inputs = _make_tight_coupling_setup()
    
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints=constraints,
        stability_guard=True,
        log_contraction_margin=True,
        warn_on_margin_shrink=True,
        margin_warn_threshold=0.01,  # Set high threshold to trigger warning
        step_size=0.15,  # Large step to stress margin
        use_analytic=True,
    )
    
    etas = coord.compute_etas(inputs)
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        etas_final = coord.relax_etas(etas, steps=5)
        
        # Should have emitted at least one warning
        margin_warnings = [warning for warning in w if "Contraction margin" in str(warning.message)]
        assert len(margin_warnings) > 0, "Expected contraction margin warning to be emitted"
        
        # Warning message should mention Lipschitz bound and safe step
        msg = str(margin_warnings[0].message)
        assert "Lipschitz bound" in msg, f"Expected Lipschitz mention in: {msg}"
        assert "safe step" in msg, f"Expected safe step mention in: {msg}"
        assert "Consider reducing step_size" in msg, f"Expected advice in: {msg}"


def test_stability_margin_warning_not_emitted_when_disabled() -> None:
    """Coordinator should NOT emit warning when warn_on_margin_shrink=False."""
    modules, couplings, constraints, inputs = _make_tight_coupling_setup()
    
    coord = EnergyCoordinator(
        modules=modules,
        couplings=couplings,
        constraints=constraints,
        stability_guard=True,
        log_contraction_margin=True,
        warn_on_margin_shrink=False,  # Disabled
        margin_warn_threshold=0.01,
        step_size=0.15,
        use_analytic=True,
    )
    
    etas = coord.compute_etas(inputs)
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        etas_final = coord.relax_etas(etas, steps=5)
        
        # Should NOT emit any margin warnings
        margin_warnings = [warning for warning in w if "Contraction margin" in str(warning.message)]
        assert len(margin_warnings) == 0, "Should not emit warning when disabled"


def test_stability_margin_warning_not_emitted_above_threshold() -> None:
    """Coordinator should NOT emit warning when margin is healthy."""
    modules, couplings, constraints, inputs = _make_tight_coupling_setup()
    
    # Use weak coupling to keep margin healthy
    modules_weak = [
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.2),
        EnergyGatingModule(gain_fn=lambda _: 0.0, a=0.2, b=0.2),
    ]
    couplings_weak = [(0, 1, QuadraticCoupling(weight=0.3))]  # Weak coupling
    
    coord = EnergyCoordinator(
        modules=modules_weak,
        couplings=couplings_weak,
        constraints={},
        stability_guard=True,
        log_contraction_margin=True,
        warn_on_margin_shrink=True,
        margin_warn_threshold=1e-6,  # Low threshold
        step_size=0.05,  # Small safe step
        use_analytic=True,
    )
    
    etas = coord.compute_etas([None, None])
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        etas_final = coord.relax_etas(etas, steps=10)
        
        # Should NOT emit warnings with healthy margin
        margin_warnings = [warning for warning in w if "Contraction margin" in str(warning.message)]
        assert len(margin_warnings) == 0, f"Should not emit warning with healthy margin, got: {[str(w.message) for w in margin_warnings]}"

