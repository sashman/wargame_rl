"""The two gates that decide where the travel reward points.

`closest_objective_v2` is the only calculator that pays a model to move
*between* objectives, and `scripts/measure_shaping_gates.py` exists to report
what its two gates decide. These tests pin the behaviour that diagnosis rests
on, so a later change to the calculator cannot quietly invalidate a published
measurement.

The load-bearing one is `test_an_objective_held_by_two_is_not_a_candidate`: the
whole finding is that taking a properly-held objective pays **nothing** until
one model can flip it single-handed, and if that ever stops being true the
report that says so is wrong.
"""

from __future__ import annotations

import pytest

from scripts.measure_shaping_gates import GateCounts, find_calculator, instrument
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.reward.calculators.closest_objective_v2 import (
    ClosestObjectiveV2Calculator,
)
from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.wargame import WargameEnv


class TestTheCandidateGate:
    @pytest.mark.parametrize(
        ("player_count", "opponent_count", "is_candidate"),
        [
            (0, 0, True),  # neutral -> ours, one model is enough
            (0, 1, True),  # theirs by one -> contested, worth arriving
            (1, 1, True),  # contested -> ours
            (0, 2, False),  # theirs by two: one arrival changes nothing
            (0, 4, False),  # theirs by four: the coordinated-assault case
            (2, 0, False),  # already ours
        ],
    )
    def test_an_objective_held_by_two_is_not_a_candidate(
        self, player_count: int, opponent_count: int, is_candidate: bool
    ) -> None:
        """The gate asks whether ONE more model flips the control label."""
        calculator = ClosestObjectiveV2Calculator()
        assert (
            calculator._is_positive_transition(player_count, opponent_count)
            is is_candidate
        )

    @pytest.mark.parametrize("opponent", [0, 1, 2, 3, 4])
    def test_only_the_last_two_arrivals_into_a_held_point_are_paid(
        self, opponent: int
    ) -> None:
        """The gate pays a two-model window, wherever the opponent's count puts it.

        With the opponent holding `o`, the transitions that count are
        `opponent -> contested` at `p = o - 1` and `contested -> player` at
        `p = o`. So arrivals 1..o-1 earn **nothing**, arrivals o and o+1 are
        paid, and everything after is refused as already-ours.

        That is the whole finding: an assault on a properly-held objective is
        unpaid for its opening, and the deeper the enemy is dug in, the longer
        the unpaid stretch. At `o = 4` the first three models to walk over earn
        zero for doing it.
        """
        calculator = ClosestObjectiveV2Calculator()
        paid = [
            calculator._is_positive_transition(already_there, opponent)
            for already_there in range(6)
        ]
        expected_window = {max(opponent - 1, 0), opponent}
        assert {i for i, v in enumerate(paid) if v} == expected_window
        # The unpaid opening, stated as the number of models who get nothing.
        assert sum(1 for v in paid[: max(opponent - 1, 0)] if not v) == max(
            opponent - 1, 0
        )


class TestTheAssignment:
    def test_one_group_can_own_several_objectives(self) -> None:
        """Which starves the others onto `fallback_to_nearest`.

        Driven through a real env rather than a stub calculator: the assignment
        is memoised per step and reads positions, group ids and the candidate
        mask together, so a hand-built fake is a second implementation of the
        thing under test.

        Group 0 is parked beside both objectives and group 1 is in the far
        corner, so group 0 is the nearest eligible group for each of them.
        """
        # Group 0 sits equidistant from both objectives and OUTSIDE each of
        # them -- standing on one would make it already-ours and drop it out of
        # the candidate mask, which is a different reason for a missing
        # assignment than the one this test is about.
        env = _env_with_shaping(
            player_positions=[(5, 15), (6, 15), (28, 28), (28, 27)],
        )
        calculator = find_calculator(env)
        assert calculator is not None

        env.reset(seed=5)
        env.step(WargameEnvAction(actions=[STAY_ACTION] * 4))

        assignment = calculator._cached_group_assignment
        assert assignment, "no objective was assigned at all"
        owners = set(assignment.values())
        assert owners == {0}, (
            f"group 0 should own every assigned objective, got {assignment}"
        )
        assert len(assignment) >= 2, "both objectives should be owned by one group"


def _env_with_shaping(
    player_positions: list[tuple[int, int]] | None = None,
) -> WargameEnv:
    """A small board whose reward phase carries the calculator under test.

    `player_positions` pins the four models where a test needs them; without it
    they are placed randomly in the deployment zone.
    """
    models = [
        ModelConfig(group_id=i // 2, weapons=[WeaponProfile(range=12)])
        if player_positions is None
        else ModelConfig(
            group_id=i // 2,
            x=player_positions[i][0],
            y=player_positions[i][1],
            weapons=[WeaponProfile(range=12)],
        )
        for i in range(4)
    ]
    return WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_battle_rounds=3,
            board_width=30,
            board_height=30,
            number_of_wargame_models=4,
            number_of_opponent_models=4,
            number_of_objectives=2,
            max_groups=2,
            objectives=[ObjectiveConfig(x=15, y=10), ObjectiveConfig(x=15, y=20)],
            models=models,
            opponent_models=[
                ModelConfig(group_id=i // 2, weapons=[WeaponProfile(range=12)])
                for i in range(4)
            ],
            opponent_policy=OpponentPolicyConfig(
                type="scripted_advance_to_objective", params={}
            ),
            reward_phases=[
                RewardPhaseConfig(
                    name="travel",
                    reward_calculators=[
                        RewardCalculatorConfig(
                            type="closest_objective_v2",
                            weight=1.0,
                            params={
                                "progress_scale": 6.0,
                                "fallback_to_nearest": True,
                            },
                        ),
                        RewardCalculatorConfig(type="vp_gain", weight=1.0),
                    ],
                    success_criteria=SuccessCriteriaConfig(type="player_ahead_on_vp"),
                )
            ],
        )
    )


class TestTheInstrumentation:
    def test_it_finds_the_live_calculator(self) -> None:
        env = _env_with_shaping()
        assert isinstance(find_calculator(env), ClosestObjectiveV2Calculator)

    def test_every_model_step_lands_in_exactly_one_bucket(self) -> None:
        """The three paths are exhaustive; a leak would silently deflate a rate."""
        env = _env_with_shaping()
        calculator = find_calculator(env)
        assert calculator is not None
        counts = GateCounts()
        instrument(calculator, counts)

        env.reset(seed=3)
        for _ in range(3):
            env.step(WargameEnvAction(actions=[STAY_ACTION] * 4))

        assert counts.model_steps > 0
        assert (
            counts.model_steps_assigned
            + counts.model_steps_fallback
            + counts.model_steps_no_target
            == counts.model_steps
        )
        assert counts.objectives_candidate <= counts.objectives_total
        assert counts.units_assigned <= counts.units_total

    def test_wrapping_does_not_change_the_reward(self) -> None:
        """A diagnostic that perturbs what it measures is worthless."""
        rewards = []
        for instrumented in (False, True):
            env = _env_with_shaping()
            calculator = find_calculator(env)
            assert calculator is not None
            if instrumented:
                instrument(calculator, GateCounts())
            env.reset(seed=7)
            total = 0.0
            for _ in range(3):
                _obs, reward, _done, _trunc, _info = env.step(
                    WargameEnvAction(actions=[STAY_ACTION] * 4)
                )
                total += float(reward)
            rewards.append(total)
        assert rewards[0] == rewards[1]
