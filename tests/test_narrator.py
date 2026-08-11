"""Tests for StepNarrator and public describe_action API."""

from __future__ import annotations

import pytest

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.state.narrator import StepNarrator
from wargame_rl.wargame.envs.state.snapshot import describe_action
from wargame_rl.wargame.envs.types import TurnOrder, WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv


@pytest.fixture
def shooting_env() -> WargameEnv:
    cfg = WargameEnvConfig(
        board_width=30,
        board_height=30,
        number_of_wargame_models=2,
        number_of_objectives=1,
        number_of_opponent_models=2,
        models=[
            ModelConfig(
                x=5,
                y=5,
                max_wounds=3,
                weapons=[
                    WeaponProfile(
                        range=50,
                        attacks=4,
                        ballistic_skill=2,
                        strength=8,
                        ap=2,
                        damage=2,
                    )
                ],
            ),
            ModelConfig(
                x=6,
                y=5,
                max_wounds=3,
                weapons=[
                    WeaponProfile(
                        range=50,
                        attacks=4,
                        ballistic_skill=2,
                        strength=8,
                        ap=2,
                        damage=2,
                    )
                ],
            ),
        ],
        # Two opponent UNITS: shooting names a unit, and `group_id` defaults
        # to 0, so without these the pair collapses into one target.
        opponent_models=[
            ModelConfig(
                x=20, y=5, group_id=0, max_wounds=3, weapons=[WeaponProfile(range=50)]
            ),
            ModelConfig(
                x=21, y=5, group_id=1, max_wounds=3, weapons=[WeaponProfile(range=50)]
            ),
        ],
        opponent_policy=OpponentPolicyConfig(type="random"),
        turn_order=TurnOrder.player,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        n_movement_angles=8,
        n_speed_bins=3,
    )
    return WargameEnv(config=cfg)


class TestNarrateAfterReset:
    def test_narrate_produces_sections(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()
        narrator = StepNarrator()
        text = narrator.narrate(snap)

        assert "===" in text
        assert "BOARD:" in text
        assert "FORCE STATUS:" in text
        assert "PLAYER MODELS:" in text
        assert "OBJECTIVES:" in text
        assert "REWARD" in text
        assert "STATUS:" in text


class TestNarrateAfterShooting:
    def test_combat_section_present(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        n = len(shooting_env.wargame_models)
        shooting_env.step(WargameEnvAction(actions=[STAY_ACTION] * n))

        ss = shooting_env._action_handler.shooting_slice
        assert ss is not None
        shooting_env.step(WargameEnvAction(actions=[ss.start, ss.start + 1]))

        snap = shooting_env.to_snapshot()
        narrator = StepNarrator()
        text = narrator.narrate(snap)

        assert "COMBAT:" in text
        assert "expected" in text
        assert "% hit" in text
        assert "% wound" in text


class TestNarrateRewardPhase:
    def test_includes_reward_phase_name(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        n = len(shooting_env.wargame_models)
        shooting_env.step(WargameEnvAction(actions=[STAY_ACTION] * n))

        snap = shooting_env.to_snapshot()
        narrator = StepNarrator()
        text = narrator.narrate(snap)

        assert snap.reward.phase_name in text


class TestNarrateForceBalance:
    def test_includes_alive_counts(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()
        narrator = StepNarrator()
        text = narrator.narrate(snap)

        assert f"{snap.player_alive_count} alive" in text
        assert f"{snap.opponent_alive_count} alive" in text
        assert f"{snap.player_total_wounds} wounds" in text


class TestDescribeActionPublic:
    @pytest.mark.parametrize(
        "action,expected",
        [
            (0, "Stay"),
        ],
    )
    def test_stay(self, action: int, expected: str) -> None:
        result = describe_action(action, 8, 3, None, None)
        assert result == expected

    def test_move(self) -> None:
        result = describe_action(1, 8, 3, None, None)
        assert "Move" in result
        assert "speed" in result

    def test_shoot(self) -> None:
        result = describe_action(25, 8, 3, 25, 27)
        assert result == "Shoot at enemy unit 0"

    def test_compass_directions_8_angles(self) -> None:
        directions = []
        for angle_idx in range(8):
            action = 1 + angle_idx * 3
            desc = describe_action(action, 8, 3, None, None)
            direction = desc.split("Move ")[1].split(" at")[0]
            directions.append(direction)
        assert directions == ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
