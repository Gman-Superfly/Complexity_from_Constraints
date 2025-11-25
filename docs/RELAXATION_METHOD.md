### Visual Motif: Tiny parts, intelligent flow

```
                          [Adapters / Weight Policies]
                    (GradNorm | AGM | SmallGain | GSPO-token)
                                    ↑
                                    │
   ┌─────────┐     springs/hinges   │    springs/hinges      ┌─────────┐
   │ Module A│───(Quadratic/Hinge)──┼──(Quadratic/Hinge)─────│ Module B│
   └─────────┘                      │                        └─────────┘
        │                           │                             │
   F_A(η)      gate/latch           │                           F_B(η)
        │        (open when ΔF↓)    │
        └─── wormhole / redemption ─┘  (future-like benefit pulls closed gates)
                                    │
                     ┌──────────────────────────────┐
                     │        Coordinator           │
                     │  F(η) = ΣF_local + ΣF_couple │
                     │  η ← η − α∇F / prox / ADMM   │
                     └──────────────────────────────┘
                                    │
                  logs/telemetry ←──┼──→ events (gate_opened, backtrack, budget_spent)
                                    │
                        "Flow is the intelligence"
```

Legend:
- springs/hinges: sparse couplings (Quadratic, Hinge, Gate‑Benefit)
- gate/latch: hazard‑based gating; opens only when total energy drops
- coordinator: minimizes total energy with guards (line search, stability, budgets)
- adapters: meta‑policies that reweight terms based on observed stress
- events/telemetry: explicit observability of the run (ΔF, margins, gates)


The "springs" are not just metaphors; they are the literal mathematical terms that the relaxation loop tries to minimize.
Here is the direct mapping from that diagram to the code mechanics:
The Springs (Couplings):
Physics: A stretched spring stores potential energy ($E = \frac{1}{2}k x^2$). It "wants" to be length 0.
Code: QuadraticCoupling adds $E = \lambda (\eta_i - \eta_j)^2$ to the total energy.
Relaxation: When the coordinator runs eta -= grad, it is literally "relaxing" the tension in that spring, pulling the two modules ($\eta_i, \eta_j$) closer together to lower the energy.
The Latches (Hinges/Gates):
Physics: A latch holds potential energy until a threshold is met, then releases.
Code: HingeCoupling or GateBenefitCoupling.
Relaxation: These add non-linear energy terms. The relaxation method pushes against them until the "force" (gradient) is strong enough to flip the gate (open the latch), releasing the pent-up energy (the "redemption").
The Coordinator:
It is the physics engine. It sums up all the spring tensions ($F_{total}$) and moves the parts ($\eta$) to find the resting state (equilibrium).
So, the "springs" define what the system cares about (the constraints), and the "relaxation method" is how the system satisfies those cares (by minimizing the tension).