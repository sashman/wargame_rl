"""Base classes for scripted baseline policies that drive the player's models.

Baselines exist to give every measurement a floor and a reference. A
``success_rate`` quoted without them says nothing: the trained policy at 945
epochs scored 17% where a squad-marching heuristic scores 80%, and neither
number is interpretable alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel
    from wargame_rl.wargame.envs.wargame import WargameEnv


class BaselinePolicy(ABC):
    """Selects actions for the player's models as a scripted reference."""

    def select_action(
        self,
        models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        """Return one action per player model.

        Baselines act only in the movement phase; every other phase yields
        ``STAY``. They deliberately do not shoot, so they measure positional
        play alone and stay a conservative bar for the learned policy.
        """
        if env.game_clock_state.phase is not BattlePhase.movement:
            return WargameEnvAction(actions=[STAY_ACTION] * len(models))
        return self.select_movement(models, env)

    @abstractmethod
    def select_movement(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """Return one movement-phase action per player model."""
        ...


def step_toward_objective(
    model: WargameModel, objective_location: np.ndarray, radius: float, env: WargameEnv
) -> int:
    """Return the action moving `model` toward an objective, or STAY if inside it.

    The step is capped at the distance to the objective's boundary so a model
    settles on the disc instead of overshooting and oscillating across it.
    """
    delta = np.asarray(objective_location, dtype=float) - np.asarray(
        model.location, dtype=float
    )
    distance = float(np.linalg.norm(delta))
    if distance <= radius:
        return STAY_ACTION
    return env.player_action_handler.best_action_toward(
        float(delta[0]), float(delta[1]), max_step_length=distance - radius
    )


class ScriptedObjectiveAssignmentPolicy(BaselinePolicy):
    """Walks every model to an assigned objective and holds it.

    Subclasses differ only in `assign_objective`; the movement is shared.
    """

    @abstractmethod
    def assign_objective(
        self, model_index: int, model: WargameModel, env: WargameEnv
    ) -> int:
        """Return the index of the objective this model should occupy."""
        ...

    def select_movement(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """Send each alive model toward its assigned objective."""
        objectives = env.objectives
        actions: list[int] = []
        for index, model in enumerate(models):
            if not model.is_alive or not objectives:
                actions.append(STAY_ACTION)
                continue
            objective = objectives[self.assign_objective(index, model, env)]
            actions.append(
                step_toward_objective(
                    model, objective.location, float(objective.radius_size), env
                )
            )
        return WargameEnvAction(actions=actions)
