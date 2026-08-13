"""Letting a human take individual opponent models off their policy.

The player's actions are chosen by the session and can simply be overwritten
before `step`. The opponent's are not: `run_after_player_action` runs the
opponent's whole turn *inside* the player's `step()`, so by the time the session
gets control back the opponent has already moved. The only seam is the policy
itself, which is why this wraps it rather than editing an action vector.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.envs.wargame_model import WargameModel


class OverridableOpponentPolicy(OpponentPolicy):
    """Delegates to `inner`, then replaces the actions a human authored.

    The overrides are read at `select_action` time rather than baked in, so one
    wrapper serves a whole session. `shoots` is copied from the inner policy
    because the env uses it to decide whether to pay for the opponent's
    range/LOS mask — a wrapper that answered for itself would quietly change how
    the opponent is masked, which is not something a debug tool may do.
    """

    def __init__(self, inner: OpponentPolicy, overrides: dict[int, int]) -> None:
        self.inner = inner
        self.overrides = overrides
        self.shoots = inner.shoots

    def select_action(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        """The inner policy's turn, with the authored models overwritten."""
        action = self.inner.select_action(opponent_models, env, action_mask=action_mask)
        for index, chosen in self.overrides.items():
            if 0 <= index < len(action.actions):
                action.actions[index] = chosen
        return action
