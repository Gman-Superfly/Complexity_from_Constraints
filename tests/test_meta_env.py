from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

from core.meta_env import EnergyLandscapeSearchEnv
from core.coordinator import EnergyCoordinator
from core.interfaces import EnergyModule, OrderParameter
from core.couplings import QuadraticCoupling


class QuadModule(EnergyModule):
    def __init__(self, a: float = 1.0):
        self.a = float(a)

    def local_energy(self, eta: OrderParameter, constraints: dict) -> float:
        e = float(eta)
        return self.a * e * e


@dataclass
class ToyTask:
    initial_etas: List[float]


def build_coord(params: Dict[str, Any]) -> EnergyCoordinator:
    # Use params to adjust coupling weight or step size
    weight = float(params.get("coupling_weight", 0.1))
    step_size = float(params.get("step_size", 0.05))
    mods = [
        QuadModule(1.0),
        QuadModule(1.2),
    ]
    coups = [(0, 1, QuadraticCoupling(weight=weight))]
    return EnergyCoordinator(mods, coups, {}, step_size=step_size)


def reward_fn(coord: EnergyCoordinator, task: ToyTask, final_etas: List[float]) -> float:
    # Negative total energy as reward
    return -float(coord.energy(final_etas))


def obs_fn(coord: EnergyCoordinator, task: ToyTask, final_etas: List[float]):
    return {"energy": float(coord.energy(final_etas))}


def test_meta_env_step():
    tasks = [ToyTask(initial_etas=[0.5, 0.5])]
    env = EnergyLandscapeSearchEnv(
        build_coordinator=build_coord,
        tasks=tasks,
        reward_fn=reward_fn,
        obs_fn=obs_fn,
        steps_per_episode=5,
        step_size=0.05,
    )

    obs = env.reset()
    assert "t" in obs
    action = {"coupling_weight": 0.2, "step_size": 0.05}
    obs2, reward, done, info = env.step(action)
    assert "energy" in obs2
    assert isinstance(reward, float)
    assert isinstance(done, bool)
