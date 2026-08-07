"""Regression tests: the opponent must not act more often than the player.

The scripted opponent's policy emits movement actions regardless of phase, and
`run_until_player_phase` used to invoke it once per battle phase. With
`skip_phases: [command, charge, fight]` that gave the opponent four moves per
round against the player's one — enough to seize every objective before VP
scoring opens, which silently invalidated every calibrated phase threshold.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

STAY = 0


def _make_env(skip_phases: list[BattlePhase] | None = None) -> WargameEnv:
    """One player model and one opponent model, both far from a central objective."""
    config = WargameEnvConfig(
        board_width=60,
        board_height=40,
        number_of_wargame_models=1,
        number_of_objectives=1,
        number_of_opponent_models=1,
        objective_radius_size=2,
        number_of_battle_rounds=10,
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
        models=[ModelConfig(x=2, y=20, group_id=0)],
        opponent_models=[ModelConfig(x=57, y=20, group_id=0)],
        objectives=[ObjectiveConfig(x=30, y=20)],
        skip_phases=(
            skip_phases
            if skip_phases is not None
            else [BattlePhase.command, BattlePhase.charge, BattlePhase.fight]
        ),
    )
    return WargameEnv(config=config)


def _opponent_distance(env: WargameEnv) -> float:
    return float(
        np.linalg.norm(env.opponent_models[0].location - env.objectives[0].location)
    )


def test_opponent_moves_once_per_round() -> None:
    """The opponent closes no more ground per round than one move allows.

    Both models start 27 cells from the objective and share `max_move_speed`,
    so after one round of the player standing still the opponent may not have
    closed more than a single move's worth of distance.
    """
    env = _make_env()
    env.reset(seed=0)
    before = _opponent_distance(env)

    # One full round: the player acts in the movement and shooting phases.
    env.step(WargameEnvAction(actions=[STAY]))
    env.step(WargameEnvAction(actions=[STAY]))

    closed = before - _opponent_distance(env)
    assert closed <= env.rules_quantities.max_move_speed + 1e-6


def test_opponent_cannot_reach_the_objective_faster_than_one_move_a_round() -> None:
    """The opponent needs as many rounds to arrive as its move allows.

    Arrival *time* is the quantity that matters, not distance closed: both
    sides stop on the objective, so total ground covered is equal however
    often each side moved. Seizing objectives before VP scoring opens at
    round 2 is exactly what the extra moves bought.
    """
    env = _make_env()
    env.reset(seed=0)
    distance = _opponent_distance(env)
    radius = float(env.objectives[0].radius_size)
    fewest_moves = int(
        np.ceil((distance - radius) / env.rules_quantities.max_move_speed)
    )

    rounds = 0
    terminated = truncated = False
    while not (terminated or truncated) and _opponent_distance(env) > radius:
        # One round: the player stands still through movement and shooting.
        for _ in range(2):
            _obs, _r, terminated, truncated, _info = env.step(
                WargameEnvAction(actions=[STAY])
            )
            if terminated or truncated:
                break
        rounds += 1

    assert rounds >= fewest_moves


def test_models_only_displace_in_the_movement_phase() -> None:
    """A movement action applied outside the movement phase is a no-op.

    The action mask already prevents a learned policy from sending one, but
    scripted policies bypass the mask entirely.
    """
    env = _make_env(skip_phases=[])
    env.reset(seed=0)
    handler = env.player_action_handler
    move = handler.best_action_toward(1.0, 0.0)

    for _ in range(env.max_turns):
        phase = env.game_clock_state.phase
        before = env.wargame_models[0].location.copy()
        _obs, _r, terminated, truncated, _info = env.step(
            WargameEnvAction(actions=[move])
        )
        moved = not np.array_equal(before, env.wargame_models[0].location)
        if phase is not BattlePhase.movement:
            assert not moved, f"model displaced during {phase}"
        if terminated or truncated:
            break
