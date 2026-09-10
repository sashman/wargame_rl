"""The evaluation kernel: one result value object and one set of end-state
readouts for both facades."""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.baseline.evaluate import BaselineResult
from wargame_rl.wargame.envs.domain.kernel.entities import (
    WargameModel,
    WargameObjective,
)
from wargame_rl.wargame.envs.domain.kernel.value_objects import position
from wargame_rl.wargame.envs.evaluation import EvalResult, read_end_of_episode


def _model(x: float, y: float, group: int, *, alive: bool = True) -> WargameModel:
    model = WargameModel(
        location=position(x, y),
        stats={"max_wounds": 1, "current_wounds": 1 if alive else 0},
        distances_to_objectives=np.zeros(2),
        group_id=group,
    )
    return model


def test_the_old_name_is_the_new_type() -> None:
    assert BaselineResult is EvalResult


def test_read_end_of_episode_on_a_hand_placed_board() -> None:
    objectives = [
        WargameObjective(location=position(10.0, 10.0), radius_size=3.0),
        WargameObjective(location=position(30.0, 10.0), radius_size=3.0),
    ]
    models = [
        _model(10.0, 10.0, 0),
        _model(11.0, 10.0, 0),
        _model(30.0, 10.0, 1),
        _model(50.0, 10.0, 1, alive=False),
    ]
    opponents = [_model(30.5, 10.0, 0), _model(31.0, 10.0, 0)]
    end = read_end_of_episode(models, opponents, objectives)
    # Three alive, all on an objective; the first is held 2 v 0, the second
    # contested 1 v 2 -- a tie or a deficit scores for nobody.
    assert end.fraction_alive == 0.75
    assert end.at_objectives == 1.0
    assert end.objectives_held == 1.0
    assert end.worst_cohesion_gap > 0.0


def test_the_per_model_readouts_are_none_safe() -> None:
    result = EvalResult(
        name="x",
        n_episodes=0,
        final_fraction_at_objectives=0.0,
        win_rate=0.0,
        player_vp=0.0,
        opponent_vp=0.0,
        worst_cohesion_gap=0.0,
        final_fraction_alive=0.0,
        exposure_rate=None,
        terrain_proximity=None,
        firepower_ratio=None,
        objectives_held=0.0,
    )
    assert result.mean_reward is None
    assert result.max_reward is None
    assert result.mean_decisions is None
    assert result.success_rate is None
    measured = EvalResult(
        **{
            **result.__dict__,
            "episode_rewards": (1.0, 3.0),
            "success_per_episode": (True, False),
        }
    )
    assert measured.mean_reward == 2.0
    assert measured.max_reward == 3.0 and measured.min_reward == 1.0
    assert measured.success_rate == 0.5
