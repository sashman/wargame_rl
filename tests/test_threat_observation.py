"""Tests for the `observe_threat_count` observation feature.

The feature exists because the policy has no line-of-sight information at
movement time — the shooting mask is built only during the shooting phase and
only masks logits, so it never reaches the encoder. Cover cannot be learned from
an observation that does not say whether you are visible.

Two things have to be exactly right and neither fails loudly:

1. **Timing.** The value must describe the position the model is in *now*, not
   the one it left. A cached-during-the-opponent's-turn implementation is stale
   within the round and would tell the agent it is safe where it used to be.
2. **Column position.** `TransformerNetwork._alive_feature_index` finds `alive`
   by counting backwards from the last column. A threat column appended after
   the combat stats shifts that index and makes the key-padding mask read
   `wound_ratio` as `alive` — silently.

Geometry is fixed by hand throughout so every expected value is checkable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    TerrainPieceConfig,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import WeaponProfile
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import (
    observation_to_tensor,
    observations_to_tensor_batch,
)
from wargame_rl.wargame.model.net import TransformerNetwork

PLAYER_X = 10
OPPONENT_X = 18
N_MODELS = 2
STAY = 0

# Spans the full height between the two lines, so every player-opponent
# sightline crosses it.
BLOCKING_RUIN = TerrainPieceConfig(footprint=(14, 0, 15, 39))


def _make_env(
    *,
    weapon_range: int = 12,
    terrain: list[TerrainPieceConfig] | None = None,
    observe_threat_count: bool = True,
) -> WargameEnv:
    """Two player models facing two armed opponents across open ground.

    The objective sits on the opponents so `scripted_advance_and_shoot` keeps
    them still, and high wounds keep everyone alive, so the geometry is fixed
    for the whole episode.
    """
    config = WargameEnvConfig(
        board_width=40,
        board_height=40,
        number_of_wargame_models=N_MODELS,
        number_of_opponent_models=N_MODELS,
        number_of_objectives=1,
        objective_radius_size=3,
        number_of_battle_rounds=3,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        models=[
            ModelConfig(
                x=PLAYER_X,
                y=18 + i,
                max_wounds=50,
                weapons=[WeaponProfile(range=weapon_range)],
            )
            for i in range(N_MODELS)
        ],
        opponent_models=[
            ModelConfig(
                x=OPPONENT_X,
                y=18 + i,
                max_wounds=50,
                weapons=[WeaponProfile(range=weapon_range)],
            )
            for i in range(N_MODELS)
        ],
        objectives=[ObjectiveConfig(x=OPPONENT_X, y=19)],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
        terrain=terrain,
        observe_threat_count=observe_threat_count,
    )
    return WargameEnv(config, renderer=None)


def _player_threats(env: WargameEnv) -> list[float | None]:
    obs = env.observation
    return [m.threat_count for m in obs.wargame_models]


def test_flag_off_leaves_the_feature_width_unchanged() -> None:
    """Every existing checkpoint depends on this: off must be byte-compatible."""
    off = _make_env(observe_threat_count=False)
    on = _make_env(observe_threat_count=True)
    try:
        off.reset(seed=0)
        on.reset(seed=0)
        width_off = observation_to_tensor(off.observation)[2].shape[-1]
        width_on = observation_to_tensor(on.observation)[2].shape[-1]
    finally:
        off.close()
        on.close()

    assert width_on == width_off + 1


def test_flag_off_leaves_threat_count_unset() -> None:
    """None, not 0.0 — 0.0 would read as 'perfectly covered'."""
    env = _make_env(observe_threat_count=False)
    try:
        env.reset(seed=0)
        assert _player_threats(env) == [None] * N_MODELS
    finally:
        env.close()


def test_open_ground_counts_every_enemy() -> None:
    """Clear sight, in range: the whole opposing force is a threat."""
    env = _make_env()
    try:
        env.reset(seed=0)
        assert _player_threats(env) == [1.0] * N_MODELS
    finally:
        env.close()


def test_blocking_ruin_removes_the_threat() -> None:
    """The point of the feature: it must fall when line of sight breaks."""
    env = _make_env(terrain=[BLOCKING_RUIN])
    try:
        env.reset(seed=0)
        assert _player_threats(env) == [0.0] * N_MODELS
    finally:
        env.close()


@pytest.mark.parametrize("weapon_range, expected", [(4, 0.0), (12, 1.0)])
def test_range_gates_the_threat(weapon_range: int, expected: float) -> None:
    """A threat is 'could shoot me', so range gates it as well as sight."""
    env = _make_env(weapon_range=weapon_range)
    try:
        env.reset(seed=0)
        assert _player_threats(env) == [expected] * N_MODELS
    finally:
        env.close()


def test_opponent_tokens_carry_the_mirror_reading() -> None:
    """Opponents get 'player models that can shoot me' from the same scan."""
    env = _make_env()
    try:
        env.reset(seed=0)
        obs = env.observation
        assert [m.threat_count for m in obs.opponent_models] == [1.0] * N_MODELS
    finally:
        env.close()


def test_dead_models_neither_threaten_nor_are_threatened() -> None:
    """A corpse is not a gun and does not need cover."""
    env = _make_env()
    try:
        env.reset(seed=0)
        env.opponent_models[0].take_damage(
            int(env.opponent_models[0].stats["current_wounds"])
        )
        env.wargame_models[0].take_damage(
            int(env.wargame_models[0].stats["current_wounds"])
        )
        obs = env.observation
    finally:
        env.close()

    assert obs.wargame_models[0].threat_count == 0.0
    # One of two opponents left, so the survivor's threat halves.
    assert obs.wargame_models[1].threat_count == pytest.approx(0.5)
    assert obs.opponent_models[1].threat_count == pytest.approx(0.5)


def _make_move_env() -> WargameEnv:
    """One player model that can step from open ground into cover.

    Player starts at (13, 20), opponent stands at (10, 30) with the objective on
    it so it never moves. The ruin is a two-cell bar at x in [9, 10], y in
    [24, 25]:

    - from (13, 20) the sightline runs down and to the left, crossing x=11-12 at
      those rows, so it misses the bar entirely — exposed;
    - from (10, 20) it is the straight column x=10, which runs into the bar —
      covered.

    Both endpoints stay outside the footprint, so the see-out/see-into rule does
    not drop the ruin from the query.
    """
    config = WargameEnvConfig(
        board_width=40,
        board_height=40,
        number_of_wargame_models=1,
        number_of_opponent_models=1,
        number_of_objectives=1,
        objective_radius_size=1,
        number_of_battle_rounds=3,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        models=[
            ModelConfig(x=13, y=20, max_wounds=50, weapons=[WeaponProfile(range=12)])
        ],
        opponent_models=[
            ModelConfig(x=10, y=30, max_wounds=50, weapons=[WeaponProfile(range=12)])
        ],
        objectives=[ObjectiveConfig(x=10, y=30)],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
        terrain=[TerrainPieceConfig(footprint=(9, 24, 10, 25))],
        observe_threat_count=True,
    )
    return WargameEnv(config, renderer=None)


def test_threat_is_measured_after_the_move_not_before() -> None:
    """The observation must describe where the model is, not where it was.

    A value cached during the opponent's turn is stale within the round: with
    the usual `skip_phases` the opponent hook does not run on a movement step at
    all, so the agent would be told it is safe in the cell it just left.
    """
    env = _make_move_env()
    try:
        env.reset(seed=0)
        assert env.observation.wargame_models[0].threat_count == 1.0

        move_west = env.player_action_handler.best_action_toward(
            -3.0, 0.0, max_step_length=3.0
        )
        obs, _reward, _terminated, _truncated, _info = env.step(
            WargameEnvAction(actions=[move_west])
        )
        moved_to = env.wargame_models[0].location.copy()
    finally:
        env.close()

    assert tuple(moved_to) == (10, 20), "test geometry broke: model did not arrive"
    assert obs.wargame_models[0].threat_count == 0.0


def test_alive_column_is_still_locatable_with_the_flag_on() -> None:
    """Regression guard for the column-position trap.

    If the threat column lands after the combat stats, `_alive_feature_index`
    points one column early and the transformer's key-padding mask starts
    reading `wound_ratio` as `alive`. Nothing raises; it just degrades.
    """
    env = _make_env()
    try:
        net = TransformerNetwork.policy_from_env(env)
        env.reset(seed=0)
        env.wargame_models[0].take_damage(
            int(env.wargame_models[0].stats["current_wounds"])
        )
        tensors = observation_to_tensor(env.observation, net.device)
    finally:
        env.close()

    players = tensors[2]
    alive_idx = net._alive_feature_index(
        int(players.shape[-1]), int(tensors[3].shape[0])
    )
    assert torch.equal(
        players[:, alive_idx], torch.tensor([0.0, 1.0], device=players.device)
    )


def test_observations_batch_cleanly_with_the_flag_on() -> None:
    """Widths must agree across episodes or the rollout buffer cannot collate."""
    env = _make_env()
    try:
        env.reset(seed=0)
        first = env.observation
        env.reset(seed=1)
        second = env.observation
        batched = observations_to_tensor_batch([first, second])
    finally:
        env.close()

    assert batched[2].shape[0] == 2
    assert batched[3].shape[0] == 2


def test_threat_counts_survive_an_episode_without_nan() -> None:
    """Smoke test through the real step loop, opponent shooting and all."""
    env = _make_env()
    try:
        env.reset(seed=0)
        action = WargameEnvAction(actions=[STAY] * N_MODELS)
        terminated = truncated = False
        seen: list[float] = []
        while not (terminated or truncated):
            obs, _r, terminated, truncated, _info = env.step(action)
            seen.extend(
                m.threat_count for m in obs.wargame_models if m.threat_count is not None
            )
    finally:
        env.close()

    assert seen
    assert all(0.0 <= value <= 1.0 for value in seen)
    assert not any(np.isnan(value) for value in seen)
