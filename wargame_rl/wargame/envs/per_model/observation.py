"""The per-model observation: the next decision, plus what this turn revealed.

A presentation adapter over the env. Its one rule of its own is the *reveal*:
a unit's advance D6 and charge 2D6 are shown only once that unit has declared,
so the policy declares blind and acts seeing the roll -- the rules' order of
events, kept on the observation rather than in the domain because the value
exists on the model either way and only who may see it changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.per_model.types import DecisionPoint, PerModelObservation

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.per_model.env import PerModelEnv


def build_per_model_observation(
    env: PerModelEnv, point: DecisionPoint
) -> PerModelObservation:
    """Assemble the observation for the player's seat at `point`."""
    seat = env.player_seat
    models = seat.models
    revealed_advance = np.array(
        [m.advance_roll if m.declared_advance else 0.0 for m in models], dtype=float
    )
    revealed_charge = np.array(
        [m.charge_roll if m.declared_charge else 0.0 for m in models], dtype=float
    )
    modes = env.player_consolidation_modes()
    state = env.game_clock_state
    return PerModelObservation(
        decision=point,
        battle_round=int(state.battle_round or 0),
        current_turn=env.current_turn,
        sub_step=env.sub_step,
        episode_step=env.episode_step,
        active_seat_is_player=state.active_player == env.player_side,
        revealed_advance_roll=revealed_advance,
        revealed_charge_roll=revealed_charge,
        consolidate_mode=modes,
        advanced=np.array([m.advanced_this_turn for m in models], dtype=bool),
        fell_back=np.array([m.fell_back_this_turn for m in models], dtype=bool),
        declared_charge=np.array([m.declared_charge for m in models], dtype=bool),
        player_vp=env.player_vp,
        opponent_vp=env.opponent_vp,
        player_vp_delta=env.player_vp_delta,
        opponent_vp_delta=env.opponent_vp_delta,
    )
