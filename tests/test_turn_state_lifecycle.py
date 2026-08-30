"""Per-turn model state is cleared at the start of every side's turn.

⚠ Regression. The `begin_turn()` loop lived inside `_roll_advance_dice`, behind
its `advance_slice is None` early return, so on any config without advance rungs
-- which is most of them -- it never ran at all. That was harmless only by
coincidence: the one writer of `advanced_this_turn` is gated on the same
condition, so the flags were provably already clear.

It stops being harmless for any mechanic that keeps its own per-turn state, so
these tests assert the hook FIRES, not merely that today's flags end up right.
A test that only checked the flags would have passed throughout the defect.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config import MeleeConfig
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import NON_MOVEMENT_PHASES, BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment


def _env(advance_bins: int) -> WargameEnv:
    skip = list(NON_MOVEMENT_PHASES)
    if advance_bins:
        skip = [p for p in skip if p is not BattlePhase.command]
    config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        opponent_policy=OpponentPolicyConfig(type="random"),
        n_advance_speed_bins=advance_bins,
        skip_phases=skip,
    )
    env = create_environment(config)
    env.reset(seed=11)
    return env


def _count_begin_turns(env: WargameEnv, steps: int) -> int:
    """Run the env, counting how many times any model's turn state is cleared."""
    calls = 0

    def wrap(model: object) -> Callable[[], None]:
        original = model.begin_turn  # type: ignore[attr-defined]

        def counted() -> None:
            nonlocal calls
            calls += 1
            original()

        return counted

    for model in env.wargame_models + env.opponent_models:
        model.begin_turn = wrap(model)  # type: ignore[method-assign]
    n = len(env.wargame_models)
    for _ in range(steps):
        env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
    return calls


def test_turn_state_is_cleared_even_when_the_scenario_has_no_advance() -> None:
    """The defect: on an advance-off config the hook never fired at all."""
    env = _env(advance_bins=0)
    assert _count_begin_turns(env, steps=6) > 0


def test_turn_state_is_cleared_when_the_scenario_has_advance() -> None:
    """And the path that always worked still works."""
    env = _env(advance_bins=3)
    assert _count_begin_turns(env, steps=6) > 0


def test_hoisting_the_hook_changed_no_flag_on_an_advance_off_config() -> None:
    """The no-op proof, asserted rather than argued.

    With no advance slice there is no writer for any of the three fields
    `begin_turn` clears, so clearing them more often must change nothing.
    """
    env = _env(advance_bins=0)
    n = len(env.wargame_models)
    for _ in range(8):
        env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        for model in env.wargame_models + env.opponent_models:
            assert model.advanced_this_turn is False
            assert model.declared_advance is False
            assert model.advance_roll == 0.0


def test_melee_requires_the_phases_it_is_played_in() -> None:
    """A feature whose actions are never legal measures nothing for hours."""
    with pytest.raises(ValueError, match="remove 'charge' from skip_phases"):
        WargameEnvConfig(
            number_of_wargame_models=4,
            melee=MeleeConfig(enabled=True),
            skip_phases=list(NON_MOVEMENT_PHASES),
        )


def test_melee_does_NOT_require_the_fight_phase_to_be_stepped() -> None:
    """⚠ The fight carries no agent action and resolves at the phase boundary.

    An earlier validator demanded both phases, which rejected the config the
    design actually wants -- fight skipped, charge stepped -- and would have
    cost an agent step per round for a mask with one legal option.
    """
    config = WargameEnvConfig(
        number_of_wargame_models=4,
        melee=MeleeConfig(enabled=True),
        skip_phases=[BattlePhase.fight],
    )
    assert BattlePhase.fight in config.skip_phases


def test_melee_off_does_not_care_what_is_skipped() -> None:
    """⚠ `skip_phases: []` is a documented setting and five test modules use it.

    The reverse check -- rejecting a stepped charge/fight while melee is off --
    was proposed and is NOT implemented, because it would reject legitimate
    configs to guard against a mistake nothing has made.
    """
    for skip in ([], list(NON_MOVEMENT_PHASES)):
        WargameEnvConfig(number_of_wargame_models=4, skip_phases=skip)


def test_melee_on_with_both_phases_stepped_is_accepted() -> None:
    config = WargameEnvConfig(
        number_of_wargame_models=4,
        melee=MeleeConfig(enabled=True),
        skip_phases=[BattlePhase.shooting],
    )
    assert config.melee.enabled
