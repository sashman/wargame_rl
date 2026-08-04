"""Tests for the shooting baseline.

Every other baseline is movement-only, so their win rate is the ceiling of a
policy class the learned agent is not in — it gets a shooting decision every
other step. Measured on the 25v25 config over 40 held-out episodes,
`squad_march` wins 0.78 while `squad_march_shoot` wins 1.00, so calibrating a
final gate against the movement-only bar would aim at 80% of what is reachable.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import WeaponProfile
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv


def _make_env(*, armed: bool = True) -> WargameEnv:
    """Player models in weapon range of the opponent but not engaged with it."""
    weapons = [WeaponProfile(range=12, attacks=2)] if armed else []
    config = WargameEnvConfig(
        render_mode=None,
        board_width=40,
        board_height=40,
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=2,
        objective_radius_size=2,
        number_of_battle_rounds=6,
        max_groups=2,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        models=[
            ModelConfig(x=10, y=18 + i, group_id=i // 2, weapons=weapons)
            for i in range(4)
        ],
        opponent_models=[ModelConfig(x=18, y=18 + i, group_id=0) for i in range(4)],
        objectives=[ObjectiveConfig(x=20, y=20), ObjectiveConfig(x=20, y=30)],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
    )
    return WargameEnv(config=config)


def _run(policy_name: str, env: WargameEnv, seed: int = 0) -> list[int]:
    """Run one episode and return the shooting actions actually issued."""
    policy = build_baseline_policy(policy_name)
    observation, _ = env.reset(seed=seed)
    shooting_slice = env.player_action_handler.shooting_slice
    assert shooting_slice is not None
    fired: list[int] = []

    terminated = truncated = False
    while not (terminated or truncated):
        is_shooting = env.game_clock_state.phase is BattlePhase.shooting
        action = policy.select_action(
            env.wargame_models, env, action_mask=observation.action_mask
        )
        if is_shooting:
            fired.extend(
                a
                for a in action.actions
                if shooting_slice.start <= a < shooting_slice.end
            )
        observation, _r, terminated, truncated, _info = env.step(action)
    return fired


def test_shooting_baseline_actually_fires() -> None:
    """The shooting baseline issues shooting-slice actions; squad_march does not."""
    assert _run("squad_march_shoot", _make_env()) != []
    assert _run("squad_march", _make_env()) == []


def test_only_masked_in_targets_are_chosen() -> None:
    """Every shot respects the env mask: range, line of sight, alive, engagement.

    Honouring the mask is what keeps the baseline playing by the same rules as
    the learned policy, so its score is a fair bar rather than a cheat.
    """
    env = _make_env()
    policy = build_baseline_policy("squad_march_shoot")
    observation, _ = env.reset(seed=0)
    shooting_slice = env.player_action_handler.shooting_slice
    assert shooting_slice is not None

    terminated = truncated = False
    while not (terminated or truncated):
        mask = observation.action_mask
        assert mask is not None
        action = policy.select_action(env.wargame_models, env, action_mask=mask)
        if env.game_clock_state.phase is BattlePhase.shooting:
            for index, chosen in enumerate(action.actions):
                if shooting_slice.start <= chosen < shooting_slice.end:
                    assert bool(mask[index, chosen])
        observation, _r, terminated, truncated, _info = env.step(action)


def test_unarmed_models_hold_fire() -> None:
    """With no weapons the mask offers no targets, so nothing is fired."""
    assert _run("squad_march_shoot", _make_env(armed=False)) == []


def test_shooting_baseline_kills_more_than_the_movement_only_one() -> None:
    """The point of the bar: firing changes the outcome.

    Uses opponent survivors rather than win rate, which needs many episodes to
    separate; casualties separate in one.
    """

    def survivors(policy_name: str) -> int:
        env = _make_env()
        _run(policy_name, env)
        return sum(1 for m in env.opponent_models if m.is_alive)

    assert survivors("squad_march_shoot") < survivors("squad_march")


def test_target_choice_is_the_nearest_valid_one() -> None:
    """Ties aside, each model shoots the closest target the mask allows."""
    env = _make_env()
    policy = build_baseline_policy("squad_march_shoot")
    observation, _ = env.reset(seed=0)
    shooting_slice = env.player_action_handler.shooting_slice
    assert shooting_slice is not None

    while env.game_clock_state.phase is not BattlePhase.shooting:
        action = policy.select_action(
            env.wargame_models, env, action_mask=observation.action_mask
        )
        observation, _r, _t, _tr, _i = env.step(action)

    mask = observation.action_mask
    assert mask is not None
    action = policy.select_action(env.wargame_models, env, action_mask=mask)
    locations = np.array([m.location for m in env.opponent_models], dtype=float)
    for index, chosen in enumerate(action.actions):
        if not (shooting_slice.start <= chosen < shooting_slice.end):
            continue
        valid = np.flatnonzero(mask[index, shooting_slice.start : shooting_slice.end])
        distances = np.linalg.norm(
            locations[valid]
            - np.asarray(env.wargame_models[index].location, dtype=float),
            axis=1,
        )
        assert chosen - shooting_slice.start == pytest.approx(
            valid[int(np.argmin(distances))]
        )
