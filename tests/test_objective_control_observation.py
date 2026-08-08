"""Per-objective control state on the objective token.

VP is scored on `player_count > opponent_count` per objective, but an objective
otherwise reaches the network as *nothing but a location*. Two separate reward
levers keyed on those counts -- `objective_hold`'s surplus discount and
`closest_objective_v2`'s overstack penalty -- both collapsed objective
occupancy when tested (rounds 1 and 2, both seeds each). The diagnosis is that
neither was attributable: with no input encoding the counts, the policy can only
experience either as "standing on objectives pays less", and does less of it.

`observe_objective_control` is off by default and must stay byte-identical when
off, because every existing checkpoint's objective embedding is 2-wide.
"""

from __future__ import annotations

import numpy as np
import torch

from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import (
    observation_to_tensor,
    observations_to_tensor_batch,
)
from wargame_rl.wargame.model.net import TransformerNetwork

CONTESTED = (20, 20)
EMPTY = (36, 36)


def _make_env(observe: bool, n_opponents_on_objective: int = 4) -> WargameEnv:
    """Four player models on objective 0; N opponents there too, objective 1 empty."""
    opponents = [
        ModelConfig(x=CONTESTED[0], y=CONTESTED[1], group_id=0)
        for _ in range(n_opponents_on_objective)
    ]
    opponents += [
        ModelConfig(x=2, y=2, group_id=0) for _ in range(4 - n_opponents_on_objective)
    ]
    config = WargameEnvConfig(
        render_mode=None,
        board_width=40,
        board_height=40,
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=2,
        objective_radius_size=3,
        number_of_battle_rounds=6,
        max_groups=1,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        models=[
            ModelConfig(x=CONTESTED[0], y=CONTESTED[1] + i, group_id=0)
            for i in range(4)
        ],
        opponent_models=opponents,
        objectives=[
            ObjectiveConfig(x=CONTESTED[0], y=CONTESTED[1]),
            ObjectiveConfig(x=EMPTY[0], y=EMPTY[1]),
        ],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
        observe_objective_control=observe,
    )
    return WargameEnv(config=config)


def test_flag_off_leaves_the_objective_token_two_wide() -> None:
    """Backward compatibility: every existing checkpoint expects width 2."""
    env = _make_env(observe=False)
    observation, _ = env.reset(seed=0)

    tensors = observation_to_tensor(observation)

    assert tuple(tensors[1].shape) == (2, 2)


def test_flag_on_adds_exactly_three_columns() -> None:
    """Location (2) plus player count, opponent count and radius."""
    env = _make_env(observe=True)
    observation, _ = env.reset(seed=0)

    tensors = observation_to_tensor(observation)

    assert tuple(tensors[1].shape) == (2, 5)


def test_counts_reflect_who_is_actually_on_each_objective() -> None:
    """The whole point: a contested disc and an empty one must read differently."""
    env = _make_env(observe=True, n_opponents_on_objective=4)
    observation, _ = env.reset(seed=0)

    contested, empty = observation.objectives

    assert contested.player_count == 1.0, "all 4 player models are on objective 0"
    assert contested.opponent_count == 1.0, "all 4 opponents are on objective 0"
    assert empty.player_count == 0.0
    assert empty.opponent_count == 0.0


def test_counts_are_normalised_by_army_size_not_left_raw() -> None:
    """Scaled to O(1). A raw count would sit beside features of order 0.01-1.0."""
    env = _make_env(observe=True, n_opponents_on_objective=2)
    observation, _ = env.reset(seed=0)

    contested = observation.objectives[0]

    assert contested.opponent_count == 0.5, "2 of 4 opponents"
    assert contested.player_count is not None and contested.radius is not None
    assert 0.0 <= contested.player_count <= 1.0
    assert 0.0 <= contested.radius <= 1.0


def test_counts_exclude_dead_models() -> None:
    """A corpse holds nothing; control is a count of *alive* models."""
    env = _make_env(observe=True, n_opponents_on_objective=4)
    env.reset(seed=0)
    for model in env.opponent_models[:2]:
        model.take_damage(model.stats["max_wounds"])

    observation = env._get_obs()

    assert observation.objectives[0].opponent_count == 0.5, "2 of 4 still alive"


def test_two_observations_batch_cleanly_with_the_flag_on() -> None:
    """`observations_to_tensor_batch` stacks the objective array; width must agree."""
    env = _make_env(observe=True)
    observation, _ = env.reset(seed=0)

    batched = observations_to_tensor_batch([observation, observation])

    assert tuple(batched[1].shape) == (2, 2, 5)


def test_the_transformer_resizes_its_objective_embedding_on_its_own() -> None:
    """`from_env` reads objective_size off the tensor, so there is no index trap.

    Unlike a per-model column -- where `_alive_feature_index` counts backwards
    from the end and a new column silently breaks the key-padding mask -- the
    objective token has no trailing structure to disturb.
    """
    env = _make_env(observe=True)
    observation, _ = env.reset(seed=0)
    network = TransformerNetwork.from_env(env=env, is_policy=True)

    with torch.no_grad():
        logits = network(observation_to_tensor(observation, network.device))

    assert network.objective_size == 5
    assert logits.shape[1] == 4, "one row per player model"
    assert torch.isfinite(logits).any()


def test_control_state_tracks_movement_within_an_episode() -> None:
    """It must be recomputed each observation, not frozen at reset.

    A cached value would tell the agent an objective is still held after the
    army has walked off it.
    """
    env = _make_env(observe=True, n_opponents_on_objective=0)
    env.reset(seed=0)
    before = env._get_obs().objectives[0].player_count

    for model in env.wargame_models:
        model.location = np.array(EMPTY, dtype=model.location.dtype)

    after = env._get_obs().objectives[0].player_count

    assert before == 1.0
    assert after == 0.0, "counts did not follow the models"
