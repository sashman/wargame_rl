"""Tests for the shooting opponent.

Every other opponent policy is movement-only, so opponent models carry weapons
they never fire and the player faces an enemy that cannot answer. These tests
pin down the two halves of `scripted_advance_and_shoot`: that it fires at all,
and that its shots obey the same range / line-of-sight / engagement rules the
player's do — the env only started enforcing those for the opponent alongside
this policy.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.shooting import PairedShootingResult
from wargame_rl.wargame.envs.opponent.registry import (
    build_opponent_policy,
    get_registry,
)
from wargame_rl.wargame.envs.opponent.scripted_advance_and_shoot_policy import (
    ScriptedAdvanceAndShootPolicy,
)
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import WeaponProfile
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

SHOOTING_POLICY = "scripted_advance_and_shoot"
MOVEMENT_POLICY = "scripted_advance_to_objective"

PLAYER_X = 10
OPPONENT_X = 18
WEAPON_RANGE = 12
WALL_X = 14


def _wall_at(column: int, width: int, height: int) -> list[list[bool]]:
    """A full-height blocking column, as a `blocking_mask` grid (rows are y)."""
    return [[x == column for x in range(width)] for _ in range(height)]


def _make_env(
    *,
    policy: str = SHOOTING_POLICY,
    armed: bool = True,
    walled: bool = False,
    objective: tuple[int, int] = (OPPONENT_X, 19),
) -> WargameEnv:
    """Opponents holding an objective, in weapon range of a stationary player.

    By default the objective sits on top of the opponents so
    `scripted_advance_to_objective` keeps them still: the geometry then stays
    fixed for the whole episode, which is what lets the line-of-sight
    assertions below read positions after the fact.
    """
    width, height = 40, 40
    weapons = [WeaponProfile(range=WEAPON_RANGE, attacks=2)] if armed else []
    config = WargameEnvConfig(
        render_mode=None,
        board_width=width,
        board_height=height,
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=1,
        objective_radius_size=3,
        number_of_battle_rounds=4,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        models=[ModelConfig(x=PLAYER_X, y=18 + i, max_wounds=20) for i in range(4)],
        opponent_models=[
            ModelConfig(x=OPPONENT_X, y=18 + i, weapons=weapons) for i in range(4)
        ],
        objectives=[ObjectiveConfig(x=objective[0], y=objective[1])],
        opponent_policy=OpponentPolicyConfig(type=policy),
        blocking_mask=_wall_at(WALL_X, width, height) if walled else None,
    )
    return WargameEnv(config=config)


def _run_episode(env: WargameEnv, seed: int = 42) -> list[PairedShootingResult]:
    """Play a full episode with the player holding still; collect opponent shots."""
    env.reset(seed=seed)
    shots: list[PairedShootingResult] = []
    terminated = truncated = False
    while not (terminated or truncated):
        stay = WargameEnvAction(actions=[0] * len(env.wargame_models))
        _obs, _r, terminated, truncated, _info = env.step(stay)
        shots.extend(env.last_opponent_shooting_results)
    return shots


def test_the_shooting_opponent_fires_and_the_movement_one_does_not() -> None:
    """The gap this policy closes: armed opponents that never pulled a trigger."""
    assert _run_episode(_make_env(policy=SHOOTING_POLICY)) != []
    assert _run_episode(_make_env(policy=MOVEMENT_POLICY)) == []


def test_the_shooting_opponent_inflicts_casualties() -> None:
    """Firing has to change the game, not just appear in the results list."""
    env = _make_env()
    _run_episode(env)
    total_wounds = sum(m.stats["current_wounds"] for m in env.wargame_models)
    assert total_wounds < 4 * 20


def test_every_shot_is_in_range_and_in_line_of_sight() -> None:
    """Shots obey the rules the player's do — the mask is not decorative."""
    env = _make_env()
    shots = _run_episode(env)
    assert shots != []

    for shot in shots:
        attacker = env.opponent_models[shot.attacker_idx]
        target = env.wargame_models[shot.target_idx]
        distance = float(
            np.linalg.norm(
                np.asarray(attacker.location, dtype=float)
                - np.asarray(target.location, dtype=float)
            )
        )
        assert distance <= WEAPON_RANGE
        assert env.has_line_of_sight_between_cells(
            int(attacker.location[0]),
            int(attacker.location[1]),
            int(target.location[0]),
            int(target.location[1]),
        )


def test_a_wall_between_the_lines_stops_every_shot() -> None:
    """Blocked line of sight means hold fire, not shoot through terrain.

    This is the regression test for the opponent mask: before it was refined,
    the same policy would have fired straight through the wall.
    """
    assert _run_episode(_make_env(walled=True)) == []
    assert _run_episode(_make_env(walled=False)) != []


def test_unarmed_opponents_hold_fire() -> None:
    """No weapons means no valid targets, and no crash."""
    assert _run_episode(_make_env(armed=False)) == []


def test_movement_matches_the_policy_it_extends() -> None:
    """Shooting is added to `scripted_advance_to_objective`, not swapped in.

    The objective sits away from both lines so the opponents actually march;
    with the default static setup this assertion would hold vacuously. The
    opponents are unarmed so neither arm can end the episode early by wiping
    the player out, which would make the two tracks differ in length rather
    than in movement.
    """

    def opponent_track(policy: str) -> list[list[tuple[int, int]]]:
        env = _make_env(policy=policy, armed=False, objective=(30, 30))
        env.reset(seed=7)
        track: list[list[tuple[int, int]]] = []
        terminated = truncated = False
        while not (terminated or truncated):
            stay = WargameEnvAction(actions=[0] * len(env.wargame_models))
            _obs, _r, terminated, truncated, _info = env.step(stay)
            track.append(
                [(int(m.location[0]), int(m.location[1])) for m in env.opponent_models]
            )
        return track

    assert opponent_track(SHOOTING_POLICY) == opponent_track(MOVEMENT_POLICY)


def test_target_choice_is_reproducible_for_a_seed() -> None:
    """Targets come from `env.np_random`, so a seeded episode replays exactly."""

    def targets(seed: int) -> list[tuple[int, int]]:
        return [
            (s.attacker_idx, s.target_idx) for s in _run_episode(_make_env(), seed=seed)
        ]

    assert targets(42) == targets(42)


def test_registered_under_its_config_name() -> None:
    """The YAML identifier resolves to the policy class."""
    assert get_registry()[SHOOTING_POLICY] is ScriptedAdvanceAndShootPolicy
    env = _make_env()
    policy = build_opponent_policy(OpponentPolicyConfig(type=SHOOTING_POLICY), env)
    assert isinstance(policy, ScriptedAdvanceAndShootPolicy)
    assert policy.shoots is True


@pytest.mark.parametrize("policy", [SHOOTING_POLICY, MOVEMENT_POLICY])
def test_episode_completes_without_error(policy: str) -> None:
    """Backward compatibility: both opponents drive a full episode."""
    env = _make_env(policy=policy)
    _run_episode(env)
    assert env.current_turn > 0
