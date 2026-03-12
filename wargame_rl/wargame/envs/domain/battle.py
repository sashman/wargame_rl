"""Battle aggregate: current state of one battle (models, objectives, board, zones)."""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.value_objects import BoardDimensions, DeploymentZone


class Battle:
    """Aggregate root for the current battle state.

    Holds player models, opponent models, objectives, board dimensions,
    and deployment zones. All mutations to battle state go through this
    aggregate (e.g. placement, action application).
    """

    def __init__(
        self,
        *,
        board_dimensions: BoardDimensions,
        player_models: list[WargameModel],
        opponent_models: list[WargameModel],
        objectives: list[WargameObjective],
        deployment_zone: DeploymentZone,
        opponent_deployment_zone: DeploymentZone,
    ) -> None:
        self._board_dimensions = board_dimensions
        self._player_models = player_models
        self._opponent_models = opponent_models
        self._objectives = objectives
        self._deployment_zone = deployment_zone
        self._opponent_deployment_zone = opponent_deployment_zone
        self._player_vp = 0
        self._opponent_vp = 0
        self._player_vp_delta = 0
        self._opponent_vp_delta = 0

    @property
    def board_width(self) -> int:
        return self._board_dimensions.width

    @property
    def board_height(self) -> int:
        return self._board_dimensions.height

    @property
    def player_models(self) -> list[WargameModel]:
        return self._player_models

    @property
    def opponent_models(self) -> list[WargameModel]:
        return self._opponent_models

    @property
    def objectives(self) -> list[WargameObjective]:
        return self._objectives

    @property
    def deployment_zone(self) -> np.ndarray:
        return self._deployment_zone.as_array()

    @property
    def opponent_deployment_zone(self) -> np.ndarray:
        return self._opponent_deployment_zone.as_array()

    @property
    def player_vp(self) -> int:
        return self._player_vp

    @property
    def opponent_vp(self) -> int:
        return self._opponent_vp

    @property
    def player_vp_delta(self) -> int:
        return self._player_vp_delta

    @property
    def opponent_vp_delta(self) -> int:
        return self._opponent_vp_delta

    def add_player_vp(self, amount: int) -> None:
        """Add victory points for the player and accumulate delta for this step."""
        self._player_vp += amount
        self._player_vp_delta += amount

    def add_opponent_vp(self, amount: int) -> None:
        """Add victory points for the opponent and accumulate delta for this step."""
        self._opponent_vp += amount
        self._opponent_vp_delta += amount

    def reset_vp_deltas(self) -> None:
        """Reset per-step VP deltas (call at start of each env step)."""
        self._player_vp_delta = 0
        self._opponent_vp_delta = 0

    def reset_for_episode(self) -> None:
        """Clear episode state on all models before new placement."""
        self._player_vp = 0
        self._opponent_vp = 0
        self._player_vp_delta = 0
        self._opponent_vp_delta = 0
        for model in self._player_models:
            model.reset_for_episode()
        for model in self._opponent_models:
            model.reset_for_episode()
