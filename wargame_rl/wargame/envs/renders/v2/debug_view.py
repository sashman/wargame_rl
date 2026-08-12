"""What the inspector needs to read, beyond what `BattleView` exposes.

`BattleView` is deliberately narrow — reward calculators, the replay adapter and
every renderer depend on it, and a snapshot has to be able to implement it. The
inspector needs two things that are not on it and should not be: the per-model
reward vector, and the live observation.

Widening `BattleView` to suit one debug panel would push both onto the replay
adapter too, which cannot supply either. So this is a separate protocol that
*extends* `BattleView`; `WargameEnv` satisfies it structurally, and nothing else
has to.

It is `runtime_checkable` because the presenter is handed a `BattleView` by an
ABC it must not change, so the extra reads are guarded by an `isinstance` rather
than by the signature. Note that only the *names* are checked at runtime, which
is enough here: anything reaching this presenter is the live env.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.types import WargameEnvObservation


@runtime_checkable
class DebugView(BattleView, Protocol):
    """A live env, read-only, for the debug inspector."""

    @property
    def last_per_model_reward(self) -> np.ndarray: ...

    @property
    def observation(self) -> WargameEnvObservation: ...
