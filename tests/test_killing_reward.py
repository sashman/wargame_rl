"""Regression tests for KillingReward.

Two defects, both silent while the scripted opponent could not shoot:
it read `opponent_models_killed` (the player's *own* losses, by the naming
convention where `player_` means "by the player"), and it subtracted a running
total from an already-per-step count, so the reward telescoped to the final
step and went negative on the step after any kill.
"""

from __future__ import annotations

import pytest

from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.calculators.killing import KillingReward
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


@pytest.fixture
def env() -> WargameEnv:
    """Small env used only as a BattleView for the calculator."""
    wargame_env = WargameEnv(
        config=WargameEnvConfig(render_mode=None, number_of_battle_rounds=5)
    )
    wargame_env.reset(seed=0)
    return wargame_env


def _context(
    env: WargameEnv, player_models_killed: int = 0, opponent_models_killed: int = 0
) -> StepContext:
    return StepContext(
        distance_cache=compute_distances(env.wargame_models, env.objectives),
        current_turn=1,
        max_turns=env.max_turns,
        board_width=env.board_width,
        board_height=env.board_height,
        player_models_killed=player_models_killed,
        opponent_models_killed=opponent_models_killed,
    )


def test_rewards_kills_the_player_made(env: WargameEnv) -> None:
    """Two opponent models killed pays twice the bonus."""
    calculator = KillingReward(bonus_killing_opponent=5.0)
    reward = calculator.calculate(env, _context(env, player_models_killed=2))
    assert reward == pytest.approx(10.0)


def test_does_not_reward_the_players_own_losses(env: WargameEnv) -> None:
    """Losing models earns nothing.

    This is the defect: `opponent_models_killed` counts player models the
    opponent eliminated, so reading it paid +5 per friendly casualty.
    """
    calculator = KillingReward(bonus_killing_opponent=5.0)
    reward = calculator.calculate(env, _context(env, opponent_models_killed=3))
    assert reward == pytest.approx(0.0)


def test_reward_does_not_telescope_across_steps(env: WargameEnv) -> None:
    """Each step is paid on its own kill count, with no negative rebound.

    The context field is already a per-step delta. Subtracting the previous
    step's value made a kill-then-no-kill sequence pay +5 then -5, summing to
    zero over the episode.
    """
    calculator = KillingReward(bonus_killing_opponent=5.0)
    rewards = [
        calculator.calculate(env, _context(env, player_models_killed=killed))
        for killed in (1, 0, 2, 0)
    ]
    assert rewards == pytest.approx([5.0, 0.0, 10.0, 0.0])
    assert sum(rewards) == pytest.approx(15.0)
