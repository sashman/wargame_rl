"""Opponent policy that samples random actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import register_policy
from wargame_rl.wargame.envs.types import WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class RandomPolicy(OpponentPolicy):
    """Each opponent model picks a uniformly random action."""

    def __init__(self, env: WargameEnv, **kwargs: object) -> None:
        self._env = env

    def select_action(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
    ) -> WargameEnvAction:
        return WargameEnvAction(list(env.opponent_action_space.sample()))


register_policy("random", RandomPolicy)
