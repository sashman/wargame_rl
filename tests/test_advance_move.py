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
from wargame_rl.wargame.envs.env_components.actions import (
    MOVE_TYPE_ADVANCE,
    STAY_ACTION,
    ActionHandler,
)
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import NON_MOVEMENT_PHASES
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.envs.wargame_model import WargameModel
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.observation import observation_to_tensor

SHOOT_TARGETS = 5


def _declare(env: WargameEnv, kind: str) -> int:
    """The action index that declares `kind`, asserted to exist.

    ⚠ The move-type slice is sized to the move types the scenario HAS, so an
    offset is no longer a constant: `advance` sits at 1 on an advance-only
    config and would move if that scenario also fought in melee.
    """
    action = env.player_action_handler.move_type_action(kind)
    assert action is not None
    return action


def _phases_with_command() -> list[BattlePhase]:
    """Skip everything non-movement EXCEPT command, where the move type is declared.

    A config with advance rungs and no command phase is rejected at
    construction: the declaration would be unreachable and the advance silently
    unavailable for the whole run.
    """
    return [p for p in NON_MOVEMENT_PHASES if p is not BattlePhase.command]


def _handler(bins: int) -> ActionHandler:
    return ActionHandler(
        WargameEnvConfig(
            n_advance_speed_bins=bins,
            number_of_wargame_models=4,
            skip_phases=_phases_with_command(),
        ),
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
    config = WargameEnvConfig(
        n_advance_speed_bins=3,
        number_of_wargame_models=4,
        skip_phases=_phases_with_command(),
    )
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


def test_no_advance_rung_is_within_a_normal_move() -> None:
    """No action may spend the unit's shooting for a distance a walk reaches.

    ⚠ This REPLACES `test_an_advance_can_stop_SHORT_of_a_normal_move`, which
    pinned the opposite. That test's justification -- that a unit which cannot
    stop short cannot advance toward an objective and halt to keep coherency --
    was checked against `env.step` on 2026-08-23 and does not hold: only ONE
    model need choose an advance for the whole unit to advance, and its
    squadmates keep the entire normal slice. The brake is the normal slice, not
    a short advance. See `test_a_unit_advances_while_its_squadmates_stop_short`.
    """
    # Arrange
    config = WargameEnvConfig(
        n_advance_speed_bins=3,
        number_of_wargame_models=4,
        skip_phases=_phases_with_command(),
    )
    handler = ActionHandler(config, n_shoot_targets=SHOOT_TARGETS)
    slice_ = handler.advance_slice
    assert slice_ is not None

    # Act
    distances = [
        float(
            np.linalg.norm(
                handler.decode_action(
                    slice_.start + bin_idx, model_idx=0, advance_roll=6.0
                )
            )
        )
        for bin_idx in range(config.n_advance_speed_bins)
    ]

    # Assert
    assert all(d > config.max_move_speed for d in distances), distances


def test_an_advance_rung_means_the_same_distance_whatever_the_roll() -> None:
    """Stationary semantics: the roll gates legality, never meaning.

    Under the fractional encoding one index meant 2.33" on a roll of 1 and 4.00"
    on a roll of 6, so a policy had to read `advance_roll` to know what its own
    action did. It is the only slice in the game that ever behaved that way.
    """
    # Arrange
    config = WargameEnvConfig(
        n_advance_speed_bins=3,
        number_of_wargame_models=4,
        skip_phases=_phases_with_command(),
    )
    handler = ActionHandler(config, n_shoot_targets=SHOOT_TARGETS)
    slice_ = handler.advance_slice
    assert slice_ is not None

    # Act — the top rung, which every roll of 6 makes legal
    distances = {
        roll: float(
            np.linalg.norm(
                handler.decode_action(slice_.start + 2, model_idx=0, advance_roll=roll)
            )
        )
        for roll in (6.0,)
    }
    ladder = [handler.advance_distance(b, config.max_move_speed) for b in range(3)]

    # Assert
    assert ladder == [8.0, 10.0, 12.0]
    assert distances[6.0] == pytest.approx(12.0)


def test_the_roll_decides_which_rungs_are_legal() -> None:
    """A rung beyond `M + roll` must be masked, not silently shortened."""
    # Arrange
    config = WargameEnvConfig(
        n_advance_speed_bins=3,
        number_of_wargame_models=4,
        skip_phases=_phases_with_command(),
    )
    handler = ActionHandler(config, n_shoot_targets=SHOOT_TARGETS)
    slice_ = handler.advance_slice
    assert slice_ is not None

    class _Model:
        def __init__(self, roll: float) -> None:
            self.advance_roll = roll
            # A rung is legal only for a unit that DECLARED an advance; without
            # this every row would be False and the test would assert nothing.
            self.declared_advance = True

    # Act — M = 6, so the ladder is 8/10/12 and needs rolls of 2/4/6
    # `None` enemies: this test is about the ROLL gate, and an engaged unit
    # may not advance at all — passing no opposing force says so explicitly
    # rather than leaving the engagement check silently off.
    legality = handler.advance_legality(
        [_Model(1.0), _Model(2.0), _Model(4.0), _Model(6.0)], None
    )

    # Assert — one row per model, the first `n_bins` columns are the first angle
    assert legality.shape == (4, slice_.size)
    assert list(legality[0, :3]) == [False, False, False]
    assert list(legality[1, :3]) == [True, False, False]
    assert list(legality[2, :3]) == [True, True, False]
    assert list(legality[3, :3]) == [True, True, True]


@pytest.mark.parametrize("roll", [2.0, 4.0, 6.0])
def test_a_legal_rung_is_never_shortened(roll: float) -> None:
    """A rung the roll allows must be delivered in full.

    ⚠ Replaces `test_the_bins_divide_the_whole_allowance`, which pinned the
    fractional ladder: the top bin was `M + roll` and the rest even fractions of
    it, so the same index meant a different distance every turn. The rungs are
    absolute now, and the roll decides only which of them are legal.
    """
    # Arrange
    config = WargameEnvConfig(
        n_advance_speed_bins=3,
        number_of_wargame_models=4,
        skip_phases=_phases_with_command(),
    )
    handler = ActionHandler(config, n_shoot_targets=SHOOT_TARGETS)
    slice_ = handler.advance_slice
    assert slice_ is not None
    move = config.max_move_speed

    # Act — the longest rung this roll makes legal
    legal = [b for b in range(3) if handler.advance_distance(b, move) <= move + roll]
    top = legal[-1]
    delivered = np.linalg.norm(
        handler.decode_action(slice_.start + top, model_idx=0, advance_roll=roll)
    )

    # Assert
    assert delivered == pytest.approx(handler.advance_distance(top, move))
    assert delivered <= move + roll


def test_a_roll_of_zero_gives_exactly_the_normal_move_allowance() -> None:
    """The die is the whole gain. (A D6 never rolls 0; this pins the boundary.)"""
    # Arrange
    config = WargameEnvConfig(
        n_advance_speed_bins=3,
        number_of_wargame_models=4,
        skip_phases=_phases_with_command(),
    )
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
    config = WargameEnvConfig(
        n_advance_speed_bins=2,
        number_of_wargame_models=4,
        skip_phases=_phases_with_command(),
    )
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
        varied.skip_phases = _phases_with_command()
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

            # Act -- the DECLARATION is what costs the shooting now, not the
            # distance travelled. It is made in the command phase.
            move_type = handler.move_type_slice
            assert move_type is not None
            handler.apply(
                WargameEnvAction(
                    actions=[
                        _declare(env, MOVE_TYPE_ADVANCE) for _ in env.wargame_models
                    ]
                ),
                env.wargame_models,
                env.config.board_width,
                env.config.board_height,
                env.action_space,
                phase=BattlePhase.command,
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

            # Act: step, declaring NORMAL, until this side's NEXT turn begins.
            # ⚠ Not a random action: a sampled command-phase action can declare
            # an advance, which legitimately sets the flag again and made this
            # test flaky the moment the declaration existed.
            # STAY is legal in every phase and declares `normal` in the command
            # phase, so one action is correct for the whole turn.
            declare_normal = WargameEnvAction(
                actions=[STAY_ACTION] * len(env.wargame_models)
            )
            start_round = env.game_clock_state.battle_round
            for _ in range(40):
                env.step(declare_normal)
                if env.game_clock_state.battle_round != start_round:
                    break

            # Assert
            assert env.game_clock_state.battle_round != start_round, (
                "never reached the next turn, so the clearing was never exercised"
            )
            assert model.advanced_this_turn is False
            assert model.advance_roll != 99.0
        finally:
            env.close()


class TestAdvanceObservation:
    """Both halves of the trade must be visible to the policy."""

    @staticmethod
    def _obs(bins: int):  # type: ignore[no-untyped-def]
        config = load_env_config("configs/dev/4v4_two_phases.yaml").model_copy(
            deep=True
        )
        config.n_advance_speed_bins = bins
        config.skip_phases = _phases_with_command()
        env = create_environment(env_config=config)
        try:
            observation, _info = env.reset(seed=7)
            return observation, env.wargame_models[0].advance_roll
        finally:
            env.close()

    def test_no_advance_bins_adds_no_observation_columns(self) -> None:
        """The no-op guarantee: an unchanged tensor width keeps checkpoints."""
        # Arrange / Act
        observation, _roll = self._obs(0)

        # Assert
        model = observation.wargame_models[0]
        assert model.advance_roll is None
        assert model.advanced_this_turn is None

    def test_the_roll_is_observable_and_normalised_by_the_die(self) -> None:
        """A D6 raw would be 6x the scale of every neighbouring feature."""
        # Arrange / Act
        observation, raw_roll = self._obs(3)

        # Assert
        model = observation.wargame_models[0]
        assert model.advance_roll is not None
        assert model.advance_roll == pytest.approx(raw_roll / 6.0)
        assert 0.0 < model.advance_roll <= 1.0

    def test_the_spent_shooting_flag_is_observable(self) -> None:
        """For the VALUE head: the action mask already forbids the shot."""
        # Arrange / Act
        observation, _roll = self._obs(3)

        # Assert
        assert observation.wargame_models[0].advanced_this_turn == 0.0

    def test_the_two_columns_widen_the_per_model_tensor_by_exactly_two(self) -> None:
        """⚠ A tensor-shape change: it orphans every checkpoint deliberately."""
        # Arrange
        plain, _ = self._obs(0)
        advancing, _ = self._obs(3)

        # Act
        narrow = observation_to_tensor(plain)[2].shape[1]
        wide = observation_to_tensor(advancing)[2].shape[1]

        # Assert
        assert wide - narrow == 2


class TestAdvanceIsResolvedPerUnit:
    """The unit's LEADER declares, and the whole unit is bound by it."""

    @staticmethod
    def _env(bins: int) -> WargameEnv:
        config = load_env_config("configs/dev/4v4_two_phases.yaml").model_copy(
            deep=True
        )
        config.n_advance_speed_bins = bins
        config.skip_phases = _phases_with_command()
        return create_environment(env_config=config)

    def _apply(
        self,
        env: WargameEnv,
        actions: list[int],
        phase: BattlePhase = BattlePhase.movement,
    ) -> None:
        assert isinstance(env.action_space, spaces.Tuple)
        env.player_action_handler.apply(
            WargameEnvAction(actions=actions),
            env.wargame_models,
            env.config.board_width,
            env.config.board_height,
            env.action_space,
            phase=phase,
            enemy_models=env.opponent_models,
        )

    def test_the_leader_declares_and_the_whole_unit_is_bound(self) -> None:
        """⚠ Replaces `test_one_advancing_model_commits_its_whole_unit`.

        That test pinned an OR over five per-model movement actions: any one
        model choosing an advance rung spent all five models' shooting. The move
        type is a UNIT choice in the rules, so it is now declared once, in the
        command phase, by the unit's lowest-indexed alive model.
        """
        # Arrange
        env = self._env(3)
        try:
            env.reset(seed=7)
            handler = env.player_action_handler
            move_type = handler.move_type_slice
            assert move_type is not None
            # Two units of two. Set explicitly: the 4v4 fixture puts every model
            # in its own unit, where "the whole unit advances" is vacuously true.
            for index, model in enumerate(env.wargame_models):
                model.group_id = index // 2
            groups = [int(m.group_id) for m in env.wargame_models]
            first = groups[0]
            assert groups.count(first) > 1, "need a unit of 2+ models"

            actions = [STAY_ACTION] * len(env.wargame_models)
            actions[0] = _declare(env, MOVE_TYPE_ADVANCE)  # the leader of unit 0

            # Act
            self._apply(env, actions, phase=BattlePhase.command)

            # Assert: every model of the leader's unit is bound and pays; no other.
            for model, group in zip(env.wargame_models, groups, strict=True):
                assert model.declared_advance is (group == first)
                assert model.advanced_this_turn is (group == first), (
                    f"unit {group} wrongly {'' if model.advanced_this_turn else 'not '}"
                    "charged for the advance"
                )
        finally:
            env.close()

    def test_a_follower_cannot_commit_its_unit(self) -> None:
        """The whole point of the change: only the leader's declaration counts.

        Under the previous OR-resolution this exact input advanced the unit, so
        one model's exploration spent four squadmates' shooting.
        """
        # Arrange
        env = self._env(3)
        try:
            env.reset(seed=7)
            handler = env.player_action_handler
            move_type = handler.move_type_slice
            assert move_type is not None
            for index, model in enumerate(env.wargame_models):
                model.group_id = index // 2

            actions = [STAY_ACTION] * len(env.wargame_models)
            actions[1] = _declare(env, MOVE_TYPE_ADVANCE)  # a FOLLOWER, not leader

            # Act
            self._apply(env, actions, phase=BattlePhase.command)

            # Assert
            assert not any(m.declared_advance for m in env.wargame_models)
            assert not any(m.advanced_this_turn for m in env.wargame_models)
        finally:
            env.close()

    def test_a_declared_unit_may_still_move_its_members_short(self) -> None:
        """Formation survives an advance, which a leader-BINDS rule would break.

        Separating the declaration from the displacement is what makes the unit
        decision safe: every member of a declared unit keeps the whole movement
        slice, so a squad can advance and still hold its 2" chain.
        """
        # Arrange
        env = self._env(3)
        try:
            env.reset(seed=7)
            handler = env.player_action_handler
            move_type = handler.move_type_slice
            advance = handler.advance_slice
            assert move_type is not None and advance is not None
            for index, model in enumerate(env.wargame_models):
                model.group_id = index // 2
                model.advance_roll = 6.0
            declare = [STAY_ACTION] * len(env.wargame_models)
            declare[0] = _declare(env, MOVE_TYPE_ADVANCE)
            self._apply(env, declare, phase=BattlePhase.command)
            before = [m.location.copy() for m in env.wargame_models]

            # Act — the leader takes the top rung, its squadmate a slow normal move
            actions = [handler.encode_action(0, 0)] * len(env.wargame_models)
            actions[0] = advance.start + (env.config.n_advance_speed_bins - 1)
            self._apply(env, actions)

            # Assert
            travelled = [
                float(np.linalg.norm(m.location - b))
                for m, b in zip(env.wargame_models, before, strict=True)
            ]
            assert travelled[0] > env.config.max_move_speed
            assert 0.0 < travelled[1] <= env.config.max_move_speed

        finally:
            env.close()

    def test_a_unit_nobody_advanced_keeps_its_shooting(self) -> None:
        """The control: without it the test above passes on an always-true flag."""
        # Arrange
        env = self._env(3)
        try:
            env.reset(seed=7)
            handler = env.player_action_handler

            # Act
            self._apply(
                env,
                [handler.encode_action(0, 0)] * len(env.wargame_models),
            )

            # Assert
            assert not any(m.advanced_this_turn for m in env.wargame_models)
        finally:
            env.close()


def test_a_declaration_action_in_the_movement_phase_does_not_decode_as_a_MOVE() -> None:
    """Regression: `apply` guarded the shooting slice and nothing else.

    ⚠ A real crash on `main`, found by `tests/test_scripted_advance.py` sampling
    the raw action space: a `move_type` index reaching the movement phase fell
    through to `decode_action`, which read it as `action - 1` into the angle
    table and raised `IndexError: index 24 is out of bounds for axis 0 with size
    16`. It fired on roughly 1 run in 8 because it needs an unmasked sample to
    land in a 2-action slice.

    Every phase-valid mask forbids this, which is why no policy hit it -- but
    `apply` deliberately does not trust the mask for the shooting slice, and had
    no reason to trust it here either. A declaration carries no displacement, so
    outside the phase it is declared in it is simply nothing.
    """
    # Arrange
    handler = _handler(3)
    move_type = handler.move_type_slice
    assert move_type is not None
    models = [
        WargameModel(
            location=np.array([10.0 + index, 10.0]),
            stats={"toughness": 3, "save": 4, "max_wounds": 1, "current_wounds": 1},
            distances_to_objectives=np.zeros(1),
            group_id=0,
            base_radius=0.0,
        )
        for index in range(4)
    ]
    before = [model.location.copy() for model in models]

    # Act
    handler.apply(
        WargameEnvAction(actions=[move_type.start] * len(models)),
        models,
        60,
        44,
        handler.action_space,
        phase=BattlePhase.movement,
        enemy_models=[],
    )

    # Assert
    for start, model in zip(before, models):
        np.testing.assert_array_equal(start, model.location)
