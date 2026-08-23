from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache


@dataclass(slots=True)
class StepContext:
    """Extensible data carrier assembled by the environment each step.

    Passed to all reward calculators and success criteria so their
    signatures stay stable as new mechanics (combat, terrain, VP) are
    added -- just add fields here.
    """

    distance_cache: DistanceCache
    current_turn: int
    max_turns: int
    board_width: int
    board_height: int
    is_terminated: bool = False
    current_round: int = 1
    battle_phase: BattlePhase = BattlePhase.command
    player_damage_dealt: int = 0
    opponent_damage_dealt: int = 0
    player_models_killed: int = 0
    opponent_models_killed: int = 0
    # Kills made by each player model this step, shape (n_player_models,).
    # `player_models_killed` is its sum; the vector exists so shooting can be
    # credited to the model that actually fired rather than shared flat across
    # the army. None when no shooting has been resolved this step.
    player_kills_by_model: np.ndarray | None = None
    # Lazily built by `opponent_distances`; never set by the env.
    _opponent_distance_cache: DistanceCache | None = None

    def opponent_distances(self, view: BattleView) -> DistanceCache:
        """The opponent's model-to-objective distances, built at most once a step.

        Five calculators were each building this from scratch with byte-identical
        arguments and their own private memo -- `objective_coverage`,
        `closest_objective_v2`, `objective_flip_bonus` and `objective_hold`
        twice. `compute_distances` is ~37% of `env.step()`, so that is the same
        shape as the bug that once made two calculators ~80% of a 25v25 step,
        one abstraction level up.

        Safe to cache on the context because the context is built AFTER the
        opponent's turn has been executed (`wargame.py`: `run_after_player_action`
        at :1083, the context at :1152), so the board is final for the step and
        every reader sees the same one. It has its own field rather than sharing
        a key with anything else -- `objective_hold._player_occupancy` documents
        what sharing one memo key across two quantities does.
        """
        if self._opponent_distance_cache is None:
            from wargame_rl.wargame.envs.domain.entities import alive_mask_for
            from wargame_rl.wargame.envs.env_components.distance_cache import (
                compute_distances,
            )

            self._opponent_distance_cache = compute_distances(
                view.opponent_models,
                view.objectives,
                alive_mask=alive_mask_for(view.opponent_models),
            )
        return self._opponent_distance_cache
