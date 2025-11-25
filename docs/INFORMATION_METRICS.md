# Information-Structure Metrics in the Homeostat

This document defines the information-structure metrics used by the Homeostat for validation, observability, and (in later phases) meta-learning, using strictly functional, domain-agnostic terminology.

## Overview
We instrument several metrics that help diagnose whether the current relaxation trajectory is moving toward well-structured, supported solutions:

- **Alignment (`info:alignment`)**: Cosine similarity between the current state and a reference state (or concept vector). Higher is better.
- **Drift (`info:drift`)**: Euclidean distance to a reference state or trajectory. Lower is better.
- **Constraint Violation Rate (`info:constraint_violation_rate`)**: Fraction of violated constraints among all checked constraints.

## Definitions

### Alignment (a)
Measures how aligned the current vector of order parameters \(\eta\) is with a reference vector \(\eta^*\) (e.g., a target configuration or concept embedding):

\[ a(\eta, \eta^*) = \frac{\langle \eta, \eta^* \rangle}{\|\eta\| \cdot \|\eta^*\|} \in [-1, 1] . \]

- **High** alignment (near 1) suggests the system is tracking the intended configuration or concept.
- **Low/negative** alignment suggests the system is orthogonal or opposed to the intended direction.

### Drift (Δ)
Measures the geometric deviation from a reference state:

\[ \Delta(\eta, \eta^*) = \| \eta - \eta^* \|_2 . \]

- **Low** drift indicates stability around a reference.
- **High** drift indicates the system is moving away; this can be benign (exploration) or problematic (instability), depending on context.

### Constraint Violation Rate (h)
Defines the fraction of constraints violated in a given step or evaluation window:

\[ h = \frac{\text{violated constraints}}{\text{total constraints checked}} \in [0, 1]. \]

- A **constraint** can be a hard rule (e.g., bounds, logical consistency), a domain-specific assertion (e.g., schema conformance), or a factual check (e.g., retrieval consistency).
- In many domains you may keep task-specific error metrics (e.g., semantic consistency checks). The framework intentionally surfaces the neutral **constraint violation rate**, leaving domain-specific interpretations to downstream consumers.
- **Drift vs. Constraint Violations**: Drift is a geometric measure of distance to a reference and does not, by itself, imply a constraint violation. A system can drift (e.g., explore a new basin) without violating constraints, and conversely, it can have low drift while still violate certain task-specific rules.

## Implementation Details

- **Library**: See `core/info_metrics.py` for reference implementations
  - `compute_alignment(current, reference)`
  - `compute_drift(current, reference)`
  - `compute_constraint_violation_rate(violations, total)` (also aliased as `compute_hallucination_rate` for backward compatibility)
- **Logging**: `EnergyBudgetTracker` logs these fields when provided in `coord.constraints`:
  - `info:alignment` and `info:drift` are logged if `constraints["reference_etas"]` is present and matches the length of `etas`.
  - `info:constraint_violation_rate` is logged if both `constraints["constraint_violation_count"]` and `constraints["total_constraints_checked"]` are present (use task-specific naming if desired; these keys are the canonical defaults).

### Span-level definition (Information Structures origin)
- h was introduced as a span-level error rate for LLM tasks:
  - “Unsupported or contradicted factual spans, measured with provenance (document IDs/offsets or KG triples).”
- Practical calculation:
  - Count every answer span marked unsupported/contradicted by a verifier (with provenance),
  - Divide by the total checked spans in the window/episode.
- Provenance components:
  - **Document IDs**: unique identifiers for each evidence source (e.g., `trial_2021_phase3.pdf`).
  - **Offsets**: byte/character/token ranges pointing to the exact supporting span (e.g., `2315–2420`).
  - **KG Triples**: subject–predicate–object facts `(DrugX, reduces, HbA1c)` with optional provenance metadata.

## ρ–h Complementarity

Top ρ (redundancy) and h (constraint-violation rate) are complementary in the coding-theory sense: ρ is “extra structure/information” you add; h is the residual error rate the structure fails to prevent.

### How they relate
- ρ→h curve: As ρ increases, h typically falls sharply after a domain-specific threshold ρ*. Below ρ*, you’re under the “channel capacity” and h stays high; above ρ*, h drops and then shows diminishing returns.
- Quality of ρ matters: Non-duplicative, relevant redundancy lowers h. Blind duplication can inflate ρ without moving h. Use diversity controls (e.g., MMR) to keep ρ “useful.”
- ρ vs a (alignment): ρ supplies evidence; a measures how well the internal state tracks the target concept. High ρ raises the ceiling for a; stable a correlates with lower h.
- ρ vs Δ (drift): Good redundancy stabilizes trajectories (Δ↓), reducing the chance of unsupported detours (h↓).
- ρ vs E (entropy): With structured evidence, uncertainty falls (E↓) for the same a; combine with a verification pass to ensure E↓ is justified (h stays low).

### Design patterns
- Target minimal ρ*: Calibrate the least redundancy that drives h below your threshold. Anything beyond ρ* costs latency/compute; only keep if it measurably reduces h further.
- Structured redundancy: Add labeled, non-overlapping evidence (Facts/Claims/Constraints) rather than longer prompts. Pair with a light verifier that flags unsupported spans.
- Typed h: Track multiple h variants (e.g., policy/schema/factual) as separate counters; improvements in ρ may lower only some types of h.
- Dynamic control: Use an accept/verify/abstain policy:
  - If (a high AND E low) AND h_estimated low → accept
  - Else if evidence available → verify once
  - Else → abstain/escalate

### What to measure (practical)
- Plot h vs ρ with iso-a slices (hold a roughly constant) to see true redundancy effects.
- Plot Δ vs ρ and E vs ρ to confirm stability and justified confidence.
- Track “useful ρ”: redundancy that actually reduces h (exclude duplicates/contradictions).
- Report ρ*, the smallest redundancy at which h ≤ target.

### How to implement here
- Emit `constraint_violation_count` and `total_constraints_checked` (and typed variants) to log `info:constraint_violation_rate` (h).
- Compute ρ using a simple, task-fit proxy (e.g., TF-IDF/embedding coverage of unique, relevant evidence). Keep duplicates out.
- Log both ρ and h together per step/run; add a small dashboard panel to trend them and their ratio over time.

### Mental model
- ρ is the “parity/check bits” you add to your system; h is the empirical error rate after decoding/verification.
- Good systems tune ρ to the minimum needed to keep h acceptably low, then spend the saved budget on speed or more checks where it matters most.

### Example (supplying references)
```python
from cf_logging.observability import EnergyBudgetTracker
from core.coordinator import EnergyCoordinator

coord = EnergyCoordinator(
    modules=mods,
    couplings=coups,
    constraints={
        "reference_etas": [0.5] * len(mods),  # alignment/drift reference
        "constraint_violation_count": 3,
        "total_constraints_checked": 50,
    },
    step_size=0.05,
)
tracker = EnergyBudgetTracker(name="energy_budget", run_id="demo")
tracker.attach(coord)
coord.relax_etas(etas0, steps=20)
# Logs contain info:alignment, info:drift, info:constraint_violation_rate
```

## Context & Relation to Literature
- **“Hallucination” vs. Constraint Violations**: In many domains (e.g., LLMs), hallucination denotes **factually incorrect or unsupported outputs**, often measured via task-specific evaluators. In our general-purpose energy-based setting, the closest universal notion is the rate of **violations of well-defined constraints**. We therefore **name the logged metric `info:constraint_violation_rate`** while acknowledging its historical use in exploratory research by O.G.such as [Hallucinations_Noisy_Channels](https://github.com/Gman-Superfly/Hallucinations_Noisy_Channels) (work in progress).
- **“Constraint violation rate (h)”**: This metric originates from the Information Structures research by O.G., where h measures unsupported or contradicted factual spans, using provenance (document IDs, offsets, or KG triples) to audit each claim. We adopt the same idea here but use the neutral term “constraint violation rate” in code and logs so downstream tasks can map it to whatever error notion applies (LLM hallucination, schema breach, policy violation, etc.). Provenance components:
  - **Document IDs**: unique identifiers for each evidence source (e.g., `trial_2021_phase3.pdf`).
  - **Offsets**: byte/character/token ranges pointing to the exact supporting span (e.g., `2315–2420`).
  - **KG Triples**: subject–predicate–object facts `(DrugX, reduces, HbA1c)` with optional provenance metadata.
  
- **Drift is not Hallucination**: Drift is a geometric measure of distance to a reference and **does not, by itself, imply hallucination**. A system can drift (e.g., explore a new basin) without violating constraints, and conversely, it can have low drift while still violating constraints.
- **Use in Meta-Learning**: These metrics can serve as **validation signals** or **reward components** in the meta-learning outer loop (e.g., `core/meta_env.py`), enabling reward functions that penalize high violation rates and reward high alignment/low drift.

## Best Practices
- Provide `reference_etas` only when a meaningful target trajectory/state exists (e.g., supervised or replay settings).
- Treat `constraint_violation_rate` as a high-level **task-dependent** signal. Define clear, auditable constraint checks in your task wrapper.
- Use these metrics primarily for **diagnostics and validation**; integrate them into training or meta-learning loops as appropriate for your domain.

---
**Status**: Implemented and logged (`EnergyBudgetTracker`), tested in `tests/test_info_metrics.py` and `tests/test_info_metrics_logging.py`.

---

## Constraint Violation: Functional Taxonomy and Synonyms

In this framework, a “constraint violation” is any event in which an observed state, action, or output fails to satisfy a specified rule or invariant. The term is intentionally **domain-agnostic** and can encompass a wide spectrum of application-specific checks. Depending on context, the same concept may be referred to by different names; below is a non-exhaustive mapping to common terminology:

- **Mathematical/Optimization:** constraint breach, infeasible assignment, inequality/equality violation (e.g., \(A\,x \le b\), \(g(x) \le 0\), \(h(x)=0\))
- **Software Engineering & APIs:** precondition/postcondition failure, contract violation, assertion failure, guard failure, input validation error
- **Data/Schema Integrity:** schema mismatch, type error, referential-integrity violation (FK), uniqueness/primary-key conflict, NOT NULL breach, range/out-of-bounds value
- **Safety & Compliance:** safety-rule violation, policy breach, PII exposure, regulatory/compliance failure (e.g., HIPAA/GDPR constraints), access-control/authorization failure
- **Control & Robotics:** actuator/sensor limits exceeded, collision/keep-out zone breach, stability/feasibility violation, saturation exceeded
- **Scientific/Physical Models:** physical-law violation (energy/mass conservation breach), monotonicity/positivity constraint broken, units/dimensional consistency error
- **Quality/Performance Gates:** SLA/SLO breach, latency/throughput limits exceeded, resource/budget overrun, service health check failure
- **Fairness/Risk Constraints:** fairness threshold violation, risk/ exposure cap exceeded, safety margin under-run
- **Concurrency/State Machines:** illegal state transition, invariant break, liveness/safety property violation

Functionally, a **constraint violation rate** is the proportion of failed checks among all checks performed over a window or episode. It can be:

- **Binary** (pass/fail per check) or **graded** (severity-weighted, partial credit for near-miss), aggregated via counts or weighted sums.
- **Pointwise** (per step) or **windowed** (e.g., last N steps, per episode), with optional time-weighting.
- **Typed/Tagged** to enable breakdowns by constraint class (e.g., `type=range`, `type=policy`, `type=schema`), and **severity** (e.g., `critical`, `warning`).

### Practical Usage Patterns
- **Instrumentation:** expose `constraint_violation_count` and `constraint_total` (or typed variants) from your task wrapper; the logger computes `info:constraint_violation_rate`.
- **Reward Shaping / Meta-Learning:** incorporate negative weight for violation rate to bias parameter search or training toward compliant solutions.
- **Gating & Early Exit:** terminate or back off when violation rate exceeds a threshold; trigger mitigation (e.g., reduce step size, increase regularization, tighten constraints).
- **Auditing:** persist per-constraint IDs, timestamps, and context for traceability; compute per-class histograms for root-cause analysis.

### Relationship to Other Signals
- **Drift (\(\Delta\))** is a geometric deviation from a reference and does not by itself indicate a violation; it is useful for diagnosing stability and trajectory quality but must be interpreted alongside constraint outcomes.
- **Alignment (\(a\))** complements violation metrics by indicating semantic proximity to a desired target or concept; again, alignment alone does not guarantee compliance.
