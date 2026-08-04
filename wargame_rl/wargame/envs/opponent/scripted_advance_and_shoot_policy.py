"""Opponent policy: advance onto objectives, and shoot whenever a target is up."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.opponent.registry import register_policy
from wargame_rl.wargame.envs.opponent.scripted_advance_to_objective_policy import (
    ScriptedAdvanceToObjectivePolicy,
)
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ScriptedAdvanceAndShootPolicy(ScriptedAdvanceToObjectivePolicy):
    """`scripted_advance_to_objective`, plus firing every shooting phase.

    Every other opponent policy is movement-only, so opponent models carry
    weapons they never fire and the player faces an enemy that cannot answer.
    That is what makes the `squad_march_shoot` baseline's 1.00 win rate a
    weaker bar than it looks.

    Target choice is a uniform draw over the valid targets rather than
    nearest-first. Nearest-first would concentrate fire and make the opponent a
    sharper threat than "returns fire" warrants; uniform keeps it a plain
    two-sided-game fixture whose damage output is easy to reason about.
    """

    # The env only refines this policy's mask with range/LOS validity because
    # of this flag -- without it the shots below would be unchecked.
    shoots = True

    def select_action(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        """Shoot during the shooting phase, otherwise advance as the base does."""
        if env.game_clock_state.phase is BattlePhase.shooting:
            return self._select_shooting(opponent_models, env, action_mask)
        return super().select_action(opponent_models, env, action_mask)

    def _select_shooting(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None,
    ) -> WargameEnvAction:
        """Fire each model at a random valid target, or hold if it has none."""
        actions = [STAY_ACTION] * len(opponent_models)
        shooting_slice = env.opponent_action_handler.shooting_slice
        if shooting_slice is None or action_mask is None or not env.wargame_models:
            return WargameEnvAction(actions=actions)

        for index, model in enumerate(opponent_models):
            if not model.is_alive:
                continue
            # The env mask already encodes target-alive, range, line of sight
            # and engagement-range validity, so honouring it is what keeps this
            # opponent playing by the same rules as the player.
            valid = np.flatnonzero(
                action_mask[index, shooting_slice.start : shooting_slice.end]
            )
            if valid.size == 0:
                continue
            # env.np_random rather than the global RNG so a seeded episode
            # replays identically -- the same generator the combat rolls and
            # placement already derive from.
            target = int(env.np_random.choice(valid))
            actions[index] = shooting_slice.start + target

        return WargameEnvAction(actions=actions)


register_policy("scripted_advance_and_shoot", ScriptedAdvanceAndShootPolicy)
