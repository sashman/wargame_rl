"""Tests for ModelsLostPenalty — the cost side of the shooting trade.

Two things are easy to get wrong here and neither raises:

- **The field.** `opponent_models_killed` means "killed by the opponent", i.e.
  player losses. `KillingReward` once read it the other way round and paid the
  agent for its own casualties; this calculator has the mirror-image hazard.
- **The scope.** It has to be global. `RewardPhaseManager` runs per-model
  calculators over alive models only, and every model in these scenarios has one
  wound, so a per-model damage penalty never fires — the model that took the
  damage is dead by the time the loop runs. That is asserted end-to-end below
  rather than argued in a comment.
"""

from __future__ import annotations

import pytest

from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.calculators.models_lost import ModelsLostPenalty
from wargame_rl.wargame.envs.reward.calculators.registry import (
    CALCULATOR_REGISTRY,
    build_calculator,
)
from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager
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


def test_charges_for_player_models_lost(env: WargameEnv) -> None:
    """Three friendly casualties cost three times the penalty."""
    calculator = ModelsLostPenalty(penalty_per_loss=1.0)
    reward = calculator.calculate(env, _context(env, opponent_models_killed=3))
    assert reward == pytest.approx(-3.0)


def test_does_not_charge_for_kills_the_player_made(env: WargameEnv) -> None:
    """The mirror of KillingReward's defect: killing must not cost anything."""
    calculator = ModelsLostPenalty(penalty_per_loss=1.0)
    reward = calculator.calculate(env, _context(env, player_models_killed=4))
    assert reward == pytest.approx(0.0)


def test_a_clean_step_is_free(env: WargameEnv) -> None:
    """No losses, no penalty — the term must not leak a per-step drag."""
    calculator = ModelsLostPenalty(penalty_per_loss=1.0)
    assert calculator.calculate(env, _context(env)) == pytest.approx(0.0)


def test_registered_under_models_lost() -> None:
    """YAML configs address it by name, so the registry entry is the contract."""
    assert CALCULATOR_REGISTRY["models_lost"] is ModelsLostPenalty
    built = build_calculator(
        "models_lost", weight=0.1, params={"penalty_per_loss": 2.0}
    )
    assert isinstance(built, ModelsLostPenalty)
    assert built.weight == pytest.approx(0.1)
    assert built.penalty_per_loss == pytest.approx(2.0)


def test_penalty_reaches_the_reward_through_the_phase_manager(env: WargameEnv) -> None:
    """End-to-end: a loss must actually move the number the agent is trained on.

    This is the check that a per-model formulation would fail. Global terms are
    broadcast whole, so the loss lands regardless of who died.
    """
    manager = RewardPhaseManager.from_configs(
        [
            RewardPhaseConfig(
                name="test",
                reward_calculators=[
                    RewardCalculatorConfig(
                        type="models_lost",
                        weight=0.5,
                        params={"penalty_per_loss": 2.0},
                    )
                ],
                success_criteria=SuccessCriteriaConfig(
                    type="player_ahead_on_vp", params={}
                ),
            )
        ]
    )

    clean = manager.calculate_reward(env, _context(env))
    after_losses = manager.calculate_reward(
        env, _context(env, opponent_models_killed=2)
    )

    assert clean == pytest.approx(0.0)
    assert after_losses == pytest.approx(-2.0)  # 0.5 weight x 2.0 penalty x 2 losses
