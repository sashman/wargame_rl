"""Battle aggregate: current state of one battle (models, objectives, board, zones)."""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.terrain import Terrain
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
        terrain: Terrain,
    ) -> None:
        self._board_dimensions = board_dimensions
        self._player_models = player_models
        self._opponent_models = opponent_models
        self._objectives = objectives
        self._deployment_zone = deployment_zone
        self._opponent_deployment_zone = opponent_deployment_zone
        self._terrain = terrain
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
    def terrain(self) -> Terrain:
        return self._terrain

    def set_terrain(self, terrain: Terrain) -> None:
        """Replace the terrain layout — used to regenerate it between episodes.

        Every LOS query resolves terrain through this aggregate at call time
        (there is no precomputed blocking cache), so a replacement takes effect
        immediately.
        """
        self._terrain = terrain

    def set_objectives(self, objectives: list[WargameObjective]) -> None:
        """Replace the objectives — used when a drawn layout brings its own.

        Mutates the list in place rather than rebinding it, because the env and
        the renderers hold the same list object from construction. Rebinding
        would leave every one of those aliases pointing at the previous
        episode's objectives, silently and with no exception.

        Objective *count* may change between episodes when a `map_pool` mixes
        layouts of different sizes: the distance cache is rebuilt from this list
        every step, and `objective_budget` is what keeps the observation a fixed
        width across the change.
        """
        self._objectives[:] = objectives

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

    def restore_victory_points(
        self,
        *,
        player_vp: int,
        opponent_vp: int,
        player_vp_delta: int,
        opponent_vp_delta: int,
    ) -> None:
        """Set victory points outright, for restoring a snapshot.

        Distinct from `add_player_vp` because loading a state is not scoring:
        the totals are known and must land exactly, not accumulate onto
        whatever the aggregate happened to hold.

        The deltas are restored rather than zeroed. `player_vp_delta` is a
        feature of the observation the policy acts on, so zeroing it would make
        a loaded state disagree with the live state it was captured from and
        round-trip to a different snapshot. Both are always in the snapshot, so
        there is nothing to reconstruct.
        """
        self._player_vp = player_vp
        self._opponent_vp = opponent_vp
        self._player_vp_delta = player_vp_delta
        self._opponent_vp_delta = opponent_vp_delta

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
