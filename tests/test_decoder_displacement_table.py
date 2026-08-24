"""The joint decoder must model every displacing action at its true displacement.

⚠ Regression. `_displacement_table` built only stay + the movement slice and
zero-padded the rest, but the ADVANCE slice is registered after shooting and
does displace — 8" to 12" of it. Modelling that as zero did not merely mis-score
advances, it inverted the decoder's opinion of them: with `ends == positions`,
`_coherent_mask` returns True whenever the unit is *already* coherent, so
advance combinations were certified legal far above their true rate, and among
legal candidates the highest log-prob wins. `verify_moves` shared the same
array, so the check built to catch exactly this class of error could not.

These tests fail on the old table and pass on the new one.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import NON_MOVEMENT_PHASES, BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.decoding import _displacement_table
from wargame_rl.wargame.model.common.factory import create_environment

ADVANCE_BINS = 3


def _env() -> WargameEnv:
    """A four-model env with advance rungs, so the slice exists and is legal."""
    config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        opponent_policy=OpponentPolicyConfig(type="random"),
        n_advance_speed_bins=ADVANCE_BINS,
        skip_phases=[p for p in NON_MOVEMENT_PHASES if p is not BattlePhase.command],
    )
    env = create_environment(config)
    env.reset(seed=7)
    return env


def _advance_to_movement(env: WargameEnv) -> None:
    """Step out of the command phase, declaring a normal move, into movement."""
    while env.game_clock_state.phase is not BattlePhase.movement:
        n = len(env.player_models)
        env.step(WargameEnvAction(actions=[STAY_ACTION] * n))


def test_the_table_agrees_with_decode_action_on_every_displacing_action() -> None:
    """One source of truth. A drift here moves models where the decoder never looked."""
    env = _env()
    _advance_to_movement(env)
    handler = env.player_action_handler
    table = _displacement_table(env)

    slices = [handler.movement_slice, handler.advance_slice]
    for model_idx, model in enumerate(env.player_models):
        roll = float(model.advance_roll)
        for action_slice in slices:
            assert action_slice is not None
            for action in range(action_slice.start, action_slice.end):
                expected = handler.decode_action(
                    action, model_idx=model_idx, advance_roll=roll
                )
                np.testing.assert_allclose(table[model_idx, action], expected)


def test_an_advance_action_is_not_modelled_as_standing_still() -> None:
    """The defect itself: an 8-12" move that the decoder scored as a 0" move."""
    env = _env()
    _advance_to_movement(env)
    handler = env.player_action_handler
    advance = handler.advance_slice
    assert advance is not None

    table = _displacement_table(env)
    reach = np.linalg.norm(table[:, advance.start : advance.end, :], axis=2)

    assert reach.max() > 0.0, "every advance action still reads as zero displacement"
    # Every rung is beyond a normal move by construction, so nothing in the
    # slice may be shorter than the shortest normal move.
    movement = handler.movement_slice
    assert movement is not None
    normal = np.linalg.norm(table[:, movement.start : movement.end, :], axis=2)
    assert reach.max() > normal.max() * 0.99


def test_a_shooting_action_still_displaces_nobody() -> None:
    """The zeros were right for shooting, and must stay."""
    env = _env()
    shooting = env.player_action_handler.shooting_slice
    assert shooting is not None
    table = _displacement_table(env)
    np.testing.assert_array_equal(
        table[:, shooting.start : shooting.end, :],
        np.zeros_like(table[:, shooting.start : shooting.end, :]),
    )


@pytest.mark.parametrize("bin_idx", range(ADVANCE_BINS))
def test_the_table_predicts_where_env_step_actually_puts_the_model(
    bin_idx: int,
) -> None:
    """⚠ The rule this file exists to honour: assert against `env.step`.

    Seven tests covered this module and none called it, which is how a decoder
    came to judge candidates against its own relaxation. The table is a forward
    model; if it disagrees with the env, the decoder is certifying moves nobody
    can make.
    """
    env = _env()
    # Declare an advance for every unit, so the rungs are legal this turn.
    handler = env.player_action_handler
    move_type = handler.move_type_slice
    assert move_type is not None
    n = len(env.player_models)
    declare = [move_type.start + 1] * n
    env.step(WargameEnvAction(actions=declare))
    assert env.game_clock_state.phase is BattlePhase.movement

    advance = handler.advance_slice
    assert advance is not None
    table = _displacement_table(env)

    # Angle bin 0 (east), the requested rung. Legality is gated on the roll.
    action = advance.start + bin_idx
    legality = handler.advance_legality(env.player_models)
    if not legality[0, action - advance.start]:
        pytest.skip("this rung is not legal at this turn's roll")

    predicted = table[0, action].copy()
    start = env.player_models[0].location.copy()
    actions = [STAY_ACTION] * n
    actions[0] = int(action)
    env.step(WargameEnvAction(actions=actions))
    travelled = np.linalg.norm(env.player_models[0].location - start)

    # The env clamps to the board and resolves collisions, so it may deliver
    # less — but never more, and never zero when the table promised a long move.
    assert travelled <= np.linalg.norm(predicted) + 1e-6
    assert travelled > 0.0
