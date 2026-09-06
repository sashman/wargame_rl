"""The re-timed reward (issue #286): action terms pay on the acting model's
step, state terms / globals / VP pay once per turn cycle on the closing step,
and the terminal bonuses fire on the last round's close."""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.per_model import PerModelAction, PerModelEnv, StepKind
from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig


def _config(terminal_vp_bonus: float = 0.0) -> WargameEnvConfig:
    # The army starts ON an objective, so the mission actually scores VP and
    # the closing step's `vp_gain` assertion is not a 0 == 0 tautology.
    return WargameEnvConfig(
        render_mode=None,
        board_width=20,
        board_height=20,
        number_of_wargame_models=4,
        max_groups=2,
        number_of_objectives=2,
        objective_radius_size=2,
        models=[
            {"x": 5, "y": 5, "group_id": 0},  # type: ignore[list-item]
            {"x": 5, "y": 6, "group_id": 0},  # type: ignore[list-item]
            {"x": 6, "y": 5, "group_id": 1},  # type: ignore[list-item]
            {"x": 6, "y": 6, "group_id": 1},  # type: ignore[list-item]
        ],
        objectives=[{"x": 5, "y": 5}, {"x": 15, "y": 15}],  # type: ignore[list-item]
        number_of_battle_rounds=3,
        reward_phases=[
            RewardPhaseConfig(
                name="mixed",
                reward_calculators=[
                    RewardCalculatorConfig(type="closest_objective_v2", weight=1.0),
                    RewardCalculatorConfig(type="objective_hold", weight=1.0),
                    RewardCalculatorConfig(type="vp_gain", weight=1.0),
                ],
                success_criteria=SuccessCriteriaConfig(type="all_at_objectives"),
                terminate_on_success=False,
                terminal_vp_bonus=terminal_vp_bonus,
            ),
        ],
    )


def _drive(
    env: PerModelEnv, seed: int
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Every step's breakdown, split into model steps and closing steps."""
    observation, _ = env.reset(seed=seed)
    model_breakdowns: list[dict[str, float]] = []
    close_breakdowns: list[dict[str, float]] = []
    terminated = False
    while not terminated:
        if observation.kind is StepKind.turn_close:
            action = PerModelAction()
        else:
            index = int(np.flatnonzero(observation.selection_mask)[0])
            # Unit 0 holds its objective (so the mission scores and the state
            # term earns); unit 1 walks (so the progress term has something to
            # pay on its members' own steps).
            moves = int(env.wargame_models[index].group_id) == 1
            action = PerModelAction(model_index=index, action=5 if moves else 0)
        observation, _reward, terminated, _t, _ = env.step(action)
        if action.model_index is None:
            close_breakdowns.append(dict(env.last_reward_breakdown))
        else:
            model_breakdowns.append(dict(env.last_reward_breakdown))
    return model_breakdowns, close_breakdowns


def test_action_terms_pay_on_model_steps_and_state_terms_on_closes() -> None:
    env = PerModelEnv(_config(), build_info=False)
    model_breakdowns, close_breakdowns = _drive(env, seed=3)
    assert len(close_breakdowns) == 3  # one per round

    model_keys = {key for breakdown in model_breakdowns for key in breakdown}
    close_keys = {key for breakdown in close_breakdowns for key in breakdown}
    assert model_keys <= {"closest_objective_v2"}, (
        "a model step may pay only the acting model's action terms"
    )
    assert "closest_objective_v2" in model_keys, "the progress term never paid"
    assert "closest_objective_v2" not in close_keys
    assert "objective_hold" in close_keys or "vp_gain" in close_keys
    for breakdown in model_breakdowns:
        assert "objective_hold" not in breakdown
        assert "vp_gain" not in breakdown


def test_the_close_pays_the_turns_net_vp() -> None:
    """Summed over the episode, the closing steps' `vp_gain` equals the final
    net VP over the cap — the same telescoped total the whole-phase facade
    pays in per-phase instalments."""
    env = PerModelEnv(_config(), build_info=False)
    _model, closes = _drive(env, seed=11)
    assert env.player_vp > 0, "the scenario scored nothing; the test is vacuous"
    paid = sum(b.get("vp_gain", 0.0) for b in closes)
    cap = env.config.mission.per_round_cap
    expected = (env.player_vp - env.opponent_vp) / float(cap)
    assert paid == pytest.approx(expected)


def test_the_terminal_bonus_fires_on_the_last_close() -> None:
    env = PerModelEnv(_config(terminal_vp_bonus=7.0), build_info=False)
    _model, closes = _drive(env, seed=3)
    # With no opponent every objective the player takes scores, so the VP
    # threshold is reachable; the bonus may pay only on the LAST close.
    for breakdown in closes[:-1]:
        assert "terminal_vp_bonus" not in breakdown
    if env.player_vp >= (env.config.mission.per_round_cap or 0):
        assert closes[-1].get("terminal_vp_bonus", 0.0) == 7.0


def test_an_unclassified_per_model_calculator_is_refused() -> None:
    from wargame_rl.wargame.envs.per_model.reward_timing import PerModelRewardTimer
    from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator
    from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager

    class UnclassifiedCalculator(PerModelRewardCalculator):
        def calculate(self, i, model, view, ctx):  # type: ignore[no-untyped-def]
            return 0.0

    manager = RewardPhaseManager.from_configs(_config().reward_phases)
    manager.phases[0].per_model_calculators.append(
        ("unclassified", UnclassifiedCalculator(weight=1.0))
    )
    with pytest.raises(ValueError, match="timing classification"):
        PerModelRewardTimer(manager)
