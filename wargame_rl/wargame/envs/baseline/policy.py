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
    from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
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

        Command, movement, shooting and charge are dispatched separately; every
        other phase yields ``STAY``. Most baselines do not shoot, which makes
        them a conservative positional bar — see `ScriptedSquadMarchShootPolicy`
        for the one that does.

        ⚠ **The charge branch is the one that was missing, and its absence was
        a measurement blocker rather than a gap in coverage.** This method
        returned STAY for every phase outside command, movement and shooting,
        so **no scripted baseline and no scripted opponent could charge at
        all**. An agent trained with melee on would have been scored against a
        bar that cannot use a core rule — verbatim the failure this project
        already paid for on Advance. See `docs/melee.md`.
        """
        phase = env.game_clock_state.phase
        if phase is BattlePhase.command:
            return self.select_command(models, env)
        if phase is BattlePhase.movement:
            return self.select_movement(models, env)
        if phase is BattlePhase.shooting:
            return self.select_shooting(models, env, action_mask)
        if phase is BattlePhase.charge:
            return self.select_charge(models, env)
        return WargameEnvAction(actions=[STAY_ACTION] * len(models))

    @abstractmethod
    def select_movement(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """Return one movement-phase action per player model."""
        ...

    def select_command(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """Return one command-phase action per player model — the move type.

        Defaults to STAY, which the handler reads as declaring a **normal**
        move. That is what keeps every baseline written before the declaration
        existed working unchanged: a policy that does not choose a move type
        gets the one it always had.
        """
        return WargameEnvAction(actions=[STAY_ACTION] * len(models))

    def select_shooting(
        self,
        models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        """Return one shooting-phase action per player model.

        Defaults to holding fire, so existing baselines are unchanged and stay
        a pure measure of positional play.
        """
        return WargameEnvAction(actions=[STAY_ACTION] * len(models))

    def select_charge(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """Return one charge-phase action per player model.

        Defaults to declining every charge, which is what every baseline did
        before this hook existed — so adding it changes no measured number.
        Overriding it is what makes a bar able to use the rule.

        The charge phase runs **after** shooting, so a declined or failed
        charge costs a scripted policy nothing at all: the referee reverts an
        illegal one to where the unit started, and this turn's fire is already
        spent. What a *successful* charge costs is next turn — an engaged unit
        cannot shoot, and cannot be shot at.
        """
        return WargameEnvAction(actions=[STAY_ACTION] * len(models))


def objective_extent(objective: WargameObjective) -> float:
    """How far from its centre an objective reaches, whichever kind it is.

    A disc reports its radius; an area reports the radius of a disc with the
    same area, the same convention the observation builder uses so "how big is
    this objective" means one thing everywhere.

    Needed because `radius_size` is 0.0 for an area objective by design, so any
    "have we arrived yet" test written against it waits for the squad's centroid
    to reach the objective's centroid exactly -- which on a real ruin means
    marching the whole squad onto one point.
    """
    if objective.area is not None:
        return float(np.sqrt(objective.area.area / np.pi))
    return float(objective.radius_size)


def step_toward_objective(
    model: WargameModel,
    objective: WargameObjective,
    env: WargameEnv,
    model_idx: int | None = None,
) -> int:
    """Return the action moving `model` toward `objective`, or STAY once on it.

    Arrival is a test against the objective's own *extent*, and the step is
    capped at the distance to its boundary, so a model settles on the near edge
    instead of overshooting and oscillating across it. Settling at the edge is
    also what spreads a squad out: models arriving from different bearings stop
    at different points on the perimeter rather than converging on one.

    **This used to take a location and a radius, and that was the whole bug.**
    An area objective reports `radius_size` of 0.0 by design -- its extent is
    the outline, and distance is reported to that edge through the
    `norms_offset` seam -- so every model steered at the *centroid* and stopped
    only once within zero of it. On a marker objective that is merely the centre
    of a small disc; on a terrain objective the size of a real ruin it means the
    whole squad walking to a single point. With bases on they then collide, and
    the ones behind stop dead in the open: measured at final occupancy 0.375 for
    `greedy_nearest` and 0.542 for `split_evenly`, against 1.000 before bases
    existed.
    """
    location = np.asarray(model.location, dtype=float)
    if objective.area is not None:
        if objective.area.contains(float(location[0]), float(location[1])):
            return STAY_ACTION
        gap = objective.area.distance_to_point(float(location[0]), float(location[1]))
    else:
        gap = float(
            np.linalg.norm(np.asarray(objective.location, dtype=float) - location)
        ) - float(objective.radius_size)
        if gap <= 0.0:
            return STAY_ACTION

    delta = np.asarray(objective.location, dtype=float) - location
    distance = float(np.linalg.norm(delta))
    if distance <= 0.0:
        return STAY_ACTION
    return env.player_action_handler.best_action_toward(
        float(delta[0]), float(delta[1]), max_step_length=gap, model_idx=model_idx
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
            actions.append(step_toward_objective(model, objective, env, index))
        return WargameEnvAction(actions=actions)
