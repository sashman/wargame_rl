"""Opponent policy: play any registered scripted *baseline* on the opponent side.

The baselines are the strongest scripted play in the repo — `squad_march_take`
scores 115.0 vp_margin on the real tables against `scripted_advance_and_shoot`'s
opponent — but they were written to drive the *player*, and the two policy
hierarchies are separate. That asymmetry is a scenario limit, not a design one:
the agent has only ever faced `scripted_advance_and_shoot`, which picks its
target uniformly and never revises where a squad is going.

This adapter closes the gap without a second copy of any policy. A baseline
reads the env from the player's side; it is handed a `MirroredEnv` instead, in
which the two sides are swapped, so the same code plays the opponent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.opponent.mirror import MirroredEnv
from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import register_policy

# `MirroredEnv` lived here as `_MirroredEnv` until it was extracted to
# `opponent/mirror.py`. A checkpoint pickles the whole `WargameEnv`, and this
# policy holds a mirror, so every checkpoint saved before that extraction names
# a class this module no longer has and fails to load at all -- and
# `checkpoints/` is the only copy of those weights. The instance state is
# `{_env, config}` in both versions and everything else is a property, so
# unpickling the old name into the new class restores exactly the old object.
_MirroredEnv = MirroredEnv

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.types import WargameEnvAction
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


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
        self._mirror = MirroredEnv(env)
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
