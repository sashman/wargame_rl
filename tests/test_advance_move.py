"""The advance move: reach bought with the turn's shooting.

`docs/rules/09-movement-phase.md` § Advance move -- maximum distance is the
unit's Move plus one D6 rolled before moving, and afterwards only `[RUN AND
GUN]` weapons may fire. No weapon here has that ability, so an advance forfeits
the turn's shooting outright.

⚠ The encoding is a slice appended AFTER shooting, not extra `n_speed_bins`.
Widening the speed bins renumbers every existing action, because `decode_action`
is angle-major and speed-minor -- action 7 would stop meaning (angle 1, speed 0)
and start meaning (angle 0, speed 6). Warm starts load with `strict=False`, so
every checkpoint would load and be silently wrong. That property is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.domain.game_clock import BattlePhase
from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

SHOOT_TARGETS = 5


def _handler(bins: int) -> ActionHandler:
    return ActionHandler(
        WargameEnvConfig(n_advance_speed_bins=bins, number_of_wargame_models=4),
        n_shoot_targets=SHOOT_TARGETS,
    )


def test_no_advance_bins_is_a_byte_for_byte_no_op() -> None:
    """The default must not move a single action index."""
    # Arrange / Act
    handler = _handler(0)

    # Assert
    assert handler.advance_slice is None
    assert handler.n_actions == 102


def test_advance_actions_are_APPENDED_so_old_indices_keep_their_meaning() -> None:
    """The property that makes every existing checkpoint still valid."""
    # Arrange
    plain, advancing = _handler(0), _handler(3)

    # Act / Assert: every index that decodes as a MOVE is unchanged. The
    # shooting indices are excluded because `decode_action` does not interpret
    # them at all -- it would read past the displacement table for either
    # handler, which is pre-existing and not what this test is about.
    movement = advancing.movement_slice
    for action in [0, *range(movement.start, movement.end)]:
        np.testing.assert_array_equal(
            plain.decode_action(action, model_idx=0),
            advancing.decode_action(action, model_idx=0),
        )
    assert advancing.advance_slice is not None
    assert advancing.advance_slice.start == plain.n_actions


def test_the_longest_advance_beats_the_fastest_normal_move() -> None:
    """Reach is what an advance buys, so the top bin must exceed a normal move."""
    # Arrange
    config = WargameEnvConfig(n_advance_speed_bins=3, number_of_wargame_models=4)
    handler = ActionHandler(config, n_shoot_targets=SHOOT_TARGETS)
    slice_ = handler.advance_slice
    assert slice_ is not None

    # Act
    fastest_normal = np.linalg.norm(
        handler.decode_action(
            handler.encode_action(0, config.n_speed_bins - 1), model_idx=0
        )
    )
    longest_advance = np.linalg.norm(
        handler.decode_action(slice_.end - 1, model_idx=0, advance_roll=1.0)
    )

    # Assert: even the WORST roll must beat the best normal move.
    assert longest_advance > fastest_normal


def test_an_advance_can_stop_SHORT_of_a_normal_move() -> None:
    """`M + roll` is a maximum, not a fixed distance -- there must be a brake.

    ⚠ This admits dominated actions on purpose: a 4" advance costs the turn's
    shooting and buys nothing a 4" normal move would not. The first version of
    this encoding pruned them, which optimised the action space against the
    rules and removed the ability to advance toward an objective and halt to
    keep unit coherency -- the binding constraint in this project.
    """
    # Arrange
    config = WargameEnvConfig(n_advance_speed_bins=3, number_of_wargame_models=4)
    handler = ActionHandler(config, n_shoot_targets=SHOOT_TARGETS)
    slice_ = handler.advance_slice
    assert slice_ is not None

    # Act
    shortest_advance = np.linalg.norm(
        handler.decode_action(slice_.start, model_idx=0, advance_roll=6.0)
    )

    # Assert
    assert shortest_advance < config.max_move_speed


@pytest.mark.parametrize("roll", [1.0, 3.0, 6.0])
def test_the_bins_divide_the_whole_allowance(roll: float) -> None:
    """The top bin is exactly `M + roll`; the rest are even fractions of it."""
    # Arrange
    config = WargameEnvConfig(n_advance_speed_bins=3, number_of_wargame_models=4)
    handler = ActionHandler(config, n_shoot_targets=SHOOT_TARGETS)
    slice_ = handler.advance_slice
    assert slice_ is not None

    # Act
    top_bin = handler.decode_action(slice_.start + 2, model_idx=0, advance_roll=roll)
    mid_bin = handler.decode_action(slice_.start + 1, model_idx=0, advance_roll=roll)

    # Assert: the top bin is the full allowance, and the bins divide it evenly.
    allowance = config.max_move_speed + roll
    assert np.linalg.norm(top_bin) == pytest.approx(allowance)
    assert np.linalg.norm(mid_bin) == pytest.approx(allowance * 2 / 3)


def test_a_roll_of_zero_gives_exactly_the_normal_move_allowance() -> None:
    """The die is the whole gain. (A D6 never rolls 0; this pins the boundary.)"""
    # Arrange
    config = WargameEnvConfig(n_advance_speed_bins=3, number_of_wargame_models=4)
    handler = ActionHandler(config, n_shoot_targets=SHOOT_TARGETS)
    slice_ = handler.advance_slice
    assert slice_ is not None

    # Act
    displacement = handler.decode_action(
        slice_.start + 2, model_idx=0, advance_roll=0.0
    )

    # Assert
    assert np.linalg.norm(displacement) == pytest.approx(config.max_move_speed)


def test_every_angle_is_reachable_by_an_advance() -> None:
    """The slice is angle-major over the same directions as a normal move."""
    # Arrange
    config = WargameEnvConfig(n_advance_speed_bins=2, number_of_wargame_models=4)
    handler = ActionHandler(config, n_shoot_targets=SHOOT_TARGETS)
    slice_ = handler.advance_slice
    assert slice_ is not None

    # Act
    headings = {
        round(
            float(
                np.arctan2(
                    *reversed(handler.decode_action(a, model_idx=0, advance_roll=6.0))
                )
            ),
            6,
        )
        for a in range(slice_.start, slice_.end)
    }

    # Assert
    assert len(headings) == config.n_movement_angles


class TestAdvanceInAnEpisode:
    """The roll, the per-turn reset, and the cost, through a real env."""

    @staticmethod
    def _env(bins: int) -> WargameEnv:
        config = load_env_config("configs/dev/4v4_two_phases.yaml")
        varied = config.model_copy(deep=True)
        varied.n_advance_speed_bins = bins
        return create_environment(env_config=varied)

    def test_every_unit_gets_a_d6_and_its_models_share_it(self) -> None:
        """One roll per UNIT, not per model -- the rules roll for the unit."""
        # Arrange
        env = self._env(3)
        try:
            # Act
            env.reset(seed=1234)

            # Assert
            by_group: dict[int, set[float]] = {}
            for model in env.wargame_models:
                by_group.setdefault(int(model.group_id), set()).add(model.advance_roll)
            assert by_group, "no models"
            for group, rolls in by_group.items():
                assert len(rolls) == 1, f"unit {group} has mixed rolls {rolls}"
                roll = rolls.pop()
                assert 1.0 <= roll <= 6.0
        finally:
            env.close()

    def test_no_advance_bins_means_no_roll_at_all(self) -> None:
        """The dice are not drawn, so no existing config's RNG stream moves."""
        # Arrange
        env = self._env(0)
        try:
            # Act
            env.reset(seed=1234)

            # Assert
            assert all(m.advance_roll == 0.0 for m in env.wargame_models)
        finally:
            env.close()

    def test_advancing_forfeits_the_turns_shooting(self) -> None:
        """The cost the rules attach, and the flag both shooting masks read.

        Asserted at the handler seam rather than after `step()`, because a whole
        step runs the clock on to the NEXT turn's command -> movement boundary,
        where `begin_turn()` correctly clears the flag again. Observing it after
        a step would be observing the reset, not the advance.
        """
        # Arrange
        env = self._env(3)
        try:
            env.reset(seed=1234)
            handler = env.player_action_handler
            slice_ = handler.advance_slice
            assert slice_ is not None
            assert isinstance(env.action_space, spaces.Tuple)
            assert not any(m.advanced_this_turn for m in env.wargame_models)

            # Act
            handler.apply(
                WargameEnvAction(actions=[slice_.start for _ in env.wargame_models]),
                env.wargame_models,
                env.config.board_width,
                env.config.board_height,
                env.action_space,
                phase=BattlePhase.movement,
                enemy_models=env.opponent_models,
            )

            # Assert
            assert all(m.advanced_this_turn for m in env.wargame_models)
        finally:
            env.close()

    def test_a_normal_move_does_NOT_forfeit_shooting(self) -> None:
        """The control: without it the test above would pass on a stuck flag."""
        # Arrange
        env = self._env(3)
        try:
            env.reset(seed=1234)
            handler = env.player_action_handler
            move = handler.encode_action(0, 0)
            assert isinstance(env.action_space, spaces.Tuple)

            # Act
            handler.apply(
                WargameEnvAction(actions=[move for _ in env.wargame_models]),
                env.wargame_models,
                env.config.board_width,
                env.config.board_height,
                env.action_space,
                phase=BattlePhase.movement,
                enemy_models=env.opponent_models,
            )

            # Assert
            assert not any(m.advanced_this_turn for m in env.wargame_models)
        finally:
            env.close()

    def test_the_flag_and_the_roll_are_cleared_each_turn(self) -> None:
        """⚠ `advanced_this_turn` was previously cleared only at EPISODE reset.

        Had anything ever set it, the model could never have shot again for the
        rest of the episode. `begin_turn` is the per-turn boundary that fixes it.
        """
        # Arrange
        env = self._env(3)
        try:
            env.reset(seed=1234)
            model = env.wargame_models[0]
            model.advanced_this_turn = True
            model.advance_roll = 99.0

            # Act: run a full step, which crosses a command -> movement boundary.
            env.step(WargameEnvAction(actions=env.action_space.sample()))

            # Assert
            assert model.advanced_this_turn is False
            assert model.advance_roll != 99.0
        finally:
            env.close()
