from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .coordinator import EnergyCoordinator


BuildCoordinatorFn = Callable[[Dict[str, Any]], EnergyCoordinator]
Task = Any  # user-defined task object
RewardFn = Callable[[EnergyCoordinator, Task, List[float]], float]
ObsFn = Callable[[EnergyCoordinator, Task, List[float]], Dict[str, float]]


@dataclass
class EnergyLandscapeSearchEnv:
    """Minimal RL-style environment for parameter search over energy configurations.

    This is a light-weight scaffold that does not require gym. It allows an external
    agent to propose a parameter dictionary (the 'action'), constructs an
    EnergyCoordinator via a provided builder function, runs a short relaxation on a
    selected task, and returns a scalar reward plus an observation dict.
    """

    build_coordinator: BuildCoordinatorFn
    tasks: Sequence[Task]
    reward_fn: RewardFn
    obs_fn: Optional[ObsFn] = None
    steps_per_episode: int = 20
    step_size: float = 0.05

    _current_task: Optional[Task] = field(default=None, init=False, repr=False)
    _t: int = field(default=0, init=False, repr=False)

    def reset(self, task_index: Optional[int] = None) -> Dict[str, float]:
        if not self.tasks:
            raise ValueError("No tasks provided")
        if task_index is None:
            # default to first task for determinism; users can randomize externally
            self._current_task = self.tasks[0]
        else:
            self._current_task = self.tasks[int(task_index) % len(self.tasks)]
        self._t = 0
        return {"t": float(self._t)}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        if self._current_task is None:
            self.reset()
        # Build coordinator with proposed parameters
        coord = self.build_coordinator(action)
        # User must provide a way to get initial etas from the task
        if not hasattr(self._current_task, "initial_etas"):
            raise AttributeError("Task must provide 'initial_etas' list")
        etas0: List[float] = list(getattr(self._current_task, "initial_etas"))
        # Run a short relaxation
        final_etas = coord.relax_etas(etas0, steps=self.steps_per_episode)
        # Compute reward
        reward = float(self.reward_fn(coord, self._current_task, final_etas))
        # Build observation
        obs: Dict[str, float] = {"t": float(self._t)}
        if self.obs_fn is not None:
            try:
                obs.update(self.obs_fn(coord, self._current_task, final_etas))
            except Exception:
                pass
        self._t += 1
        done = self._t >= 1  # one-step episodes by default; caller controls looping
        info: Dict[str, Any] = {}
        return obs, reward, done, info
