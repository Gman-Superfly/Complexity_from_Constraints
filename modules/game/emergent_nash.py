"""Emergent Nash module with simple regret-based local energy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple
import numpy as np

from core.interfaces import EnergyModule, OrderParameter

__all__ = ["symmetric_2x2_payoff", "strategy_regret", "NashModule", "replicator_step"]


def symmetric_2x2_payoff(T: float = 5.0, R: float = 3.0, P: float = 1.0, S: float = 0.0) -> np.ndarray:
    """Prisoner's Dilemma-style payoff matrix for the row player.
    
    Actions: 0=Cooperate, 1=Defect
    Returns:
        2x2 matrix A where A[a_row, a_col] is row player's payoff.
    """
    A = np.array([[R, S],
                  [T, P]], dtype=float)
    return A


def strategy_regret(A: np.ndarray, p_row: float, p_col: float) -> float:
    """Compute row player's regret under mixed strategies (p_row, p_col).
    
    p in [0,1] is probability of action 1 (Defect).
    """
    assert 0.0 <= p_row <= 1.0 and 0.0 <= p_col <= 1.0, "probabilities must be in [0,1]"
    # Expected payoff for row given p_col
    # u_row(a) for a in {0,1}
    u0 = (1 - p_col) * A[0, 0] + p_col * A[0, 1]
    u1 = (1 - p_col) * A[1, 0] + p_col * A[1, 1]
    u_mixed = (1 - p_row) * u0 + p_row * u1
    u_best = max(u0, u1)
    regret = max(0.0, u_best - u_mixed)
    return float(regret)


def replicator_step(p: float, payoff_against: Tuple[float, float], lr: float = 0.05) -> float:
    """Single-player replicator update for 2 actions given opponent mix summarized as (u0, u1)."""
    u0, u1 = payoff_against
    u_bar = (1 - p) * u0 + p * u1
    # dp/dt = p(1-p)(u1 - u0)
    dp = p * (1 - p) * (u1 - u0)
    return float(np.clip(p + lr * dp, 0.0, 1.0))


@dataclass
class NashModule(EnergyModule):
    """Order parameter is alignment with equilibrium (η = 1 - normalized regret)."""
    T: float = 5.0
    R: float = 3.0
    P: float = 1.0
    S: float = 0.0

    def compute_eta(self, x: Any) -> OrderParameter:
        """x = (p_row, p_col)."""
        assert isinstance(x, tuple) and len(x) == 2, "x must be (p_row, p_col)"
        p_row, p_col = float(x[0]), float(x[1])
        A = symmetric_2x2_payoff(self.T, self.R, self.P, self.S)
        # Normalize regret by max payoff spread
        reg = strategy_regret(A, p_row, p_col)
        max_spread = float(np.max(A) - np.min(A))
        norm_reg = 0.0 if max_spread <= 0 else min(1.0, reg / max_spread)
        eta = 1.0 - norm_reg
        assert 0.0 <= eta <= 1.0, "η must be within [0,1]"
        return float(eta)

    def local_energy(self, eta: OrderParameter, constraints: Mapping[str, Any]) -> float:
        # Minimize regret ⇒ maximize η; Landau-like around 1.0
        assert 0.0 <= eta <= 1.0, "η must be within [0,1]"
        a = float(constraints.get("nash_alpha", 1.0))
        b = float(constraints.get("nash_beta", 1.0))
        assert a >= 0.0 and b >= 0.0, "alpha/beta must be non-negative"
        delta = (1.0 - float(eta))
        return float(a * (delta ** 2) + b * (delta ** 4))


