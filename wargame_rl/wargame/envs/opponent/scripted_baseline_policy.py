"""Opponent policy: play any registered scripted *baseline* on the opponent side.

The baselines are the strongest scripted play in the repo — `squad_march_take`
scores 115.0 vp_margin on the real tables against `scripted_advance_and_shoot`'s
opponent — but they were written to drive the *player*, and the two policy
hierarchies are separate. That asymmetry is a scenario limit, not a design one:
the agent has only ever faced `scripted_advance_and_shoot`, which picks its
target uniformly and never revises where a squad is going.

This adapter closes the gap without a second copy of any policy. A baseline
reads the env from the player's side; it is handed a `_MirroredEnv` instead, in
which the two sides are swapped, so the same code plays the opponent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import register_policy

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.types import WargameEnvAction
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class _MirroredEnv:
    """The env as the opponent sees it: the two sides swapped, nothing else.

    Every attribute a baseline reads is either **shared** (objectives, the game
    clock, the board) or **side-specific**, and only the second kind is
    overridden here. `tests/test_scripted_baseline_opponent.py` enumerates the
    side-specific reads across the whole baseline package and fails if a new one
    appears, because a baseline that reaches for an un-mirrored side attribute
    would silently play for the wrong army — reading its own models as targets,
    or steering with the player's action handler.

    Attribute lookup falls through to the real env, so this stays a mirror
    rather than a second implementation of `WargameEnv` that has to be kept in
    step with it.
    """

    def __init__(self, env: WargameEnv) -> None:
        self._env = env
        # Swapped once at construction rather than per call: the config is
        # immutable for the episode, and `model_copy` re-points fields without
        # re-running validation, which would reject a config whose model list
        # no longer matches its own count field.
        self.config = env.config.model_copy(
            update={
                "number_of_wargame_models": env.config.number_of_opponent_models,
                "number_of_opponent_models": env.config.number_of_wargame_models,
                "models": env.config.opponent_models,
                "opponent_models": env.config.models,
                "deployment_zone": env.config.opponent_deployment_zone,
                "opponent_deployment_zone": env.config.deployment_zone,
            }
        )

    @property
    def wargame_models(self) -> list[WargameModel]:
        """Our models — the opponent's, from the real env's point of view."""
        return self._env.opponent_models

    @property
    def player_models(self) -> list[WargameModel]:
        """`BattleView`'s name for the same list `wargame_models` returns."""
        return self._env.opponent_models

    @property
    def opponent_models(self) -> list[WargameModel]:
        """The enemy — the player's models."""
        return self._env.wargame_models

    @property
    def player_action_handler(self) -> Any:
        """The handler that moves *our* models and indexes *their* units."""
        return self._env.opponent_action_handler

    def __getattr__(self, name: str) -> Any:
        """Fall through to the real env for everything not side-specific.

        The `__dict__` lookup rather than `self._env` is load-bearing, and it is
        not defensive programming. `__getattr__` runs for any name normal lookup
        misses, and `copy.deepcopy` reconstructs an instance *without* calling
        `__init__` — so during the copy `_env` is missing, `self._env` re-enters
        here, and the two recurse until the stack ends. Lightning deep-copies the
        env in `save_hyperparameters`, so with the plain version every training
        run on a config using this policy died at startup with `RecursionError`.
        """
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)


class ScriptedBaselineOpponentPolicy(OpponentPolicy):
    """Run a registered baseline (`squad_march_take`, `squad_march_shoot`, …).

    Configured by name, so every baseline present and future is available as an
    opponent without a class per pairing:

    ```yaml
    opponent_policy:
      type: scripted_baseline
      params:
        baseline: squad_march_take
    ```

    **Switching a config's opponent invalidates every number measured on it** —
    both the baselines and the agent — so a config that adopts this must
    re-measure the floor and the bar before quoting anything.
    """

    def __init__(self, env: WargameEnv, **kwargs: object) -> None:
        baseline = kwargs.pop("baseline", None)
        if not isinstance(baseline, str):
            raise ValueError(
                "scripted_baseline requires a `baseline` param naming a registered "
                "baseline policy, e.g. params: {baseline: squad_march_take}"
            )
        self._name = baseline
        self._baseline = build_baseline_policy(baseline, **kwargs)
        self._mirror = _MirroredEnv(env)
        # `shoots` gates whether the env refines this policy's mask with range,
        # line of sight and engagement validity. It is not a performance switch:
        # left False for a baseline that fires, the mask would be phase-and-alive
        # only and every shot would be taken unchecked. Derived from whether the
        # baseline overrides the hold-fire default, so a new shooting baseline
        # cannot forget to declare it.
        self.shoots = (
            type(self._baseline).select_shooting is not BaselinePolicy.select_shooting
        )

    @property
    def baseline_name(self) -> str:
        """The registered baseline this opponent is playing."""
        return self._name

    def select_action(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        """Ask the baseline what to do, with the board handed to it side-swapped."""
        return self._baseline.select_action(
            opponent_models,
            cast("WargameEnv", self._mirror),
            action_mask=action_mask,
        )


register_policy("scripted_baseline", ScriptedBaselineOpponentPolicy)
