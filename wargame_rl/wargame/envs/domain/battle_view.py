"""Read-only view of battle state for renderers and reward calculators."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.types.config import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import GameState


@runtime_checkable
class BattleView(Protocol):
    """Read-only interface to battle state.

    Implemented by WargameEnv (and later by Battle when used standalone).
    Allows renderers and reward calculators to depend on this protocol
    instead of the full Gymnasium environment.
    """

    @property
    def board_width(self) -> int: ...
    @property
    def board_height(self) -> int: ...
    @property
    def config(self) -> WargameEnvConfig: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
    @property
    def player_models(self) -> list[WargameModel]: ...
    @property
    def opponent_models(self) -> list[WargameModel]: ...
    @property
    def objectives(self) -> list[WargameObjective]: ...
    @property
    def deployment_zone(self) -> np.ndarray: ...
    @property
    def opponent_deployment_zone(self) -> np.ndarray: ...
    @property
    def current_turn(self) -> int: ...
    @property
    def last_reward(self) -> float | None: ...
    @property
    def game_clock_state(self) -> GameState: ...
    @property
    def n_rounds(self) -> int: ...
    @property
    def player_vp(self) -> int: ...
    @property
    def opponent_vp(self) -> int: ...
    @property
    def player_vp_delta(self) -> int: ...
    @property
    def opponent_vp_delta(self) -> int: ...

    def has_line_of_sight_between_cells(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> bool: ...
