"""A mission's NAME must not silently change what other layers do.

Three readers branched on the literal string `"default"` or duck-typed their way
into `mission.params`:

  * `player_vp_min._theoretical_max_vp` returned **0** for any other name, so the
    phase-advance threshold collapsed to `min_vp` and `success_rate` pinned at
    1.0 -- a curriculum advancing on epoch count alone;
  * `vp_threshold_for_terminal_bonus` returned None on the same branch, disabling
    `terminal_vp_bonus`;
  * `vp_gain` fell back to a cap of 15, rescaling every reward in the run.

None of the 115 configs in the repo sets a mission, so nothing exercised any of
them. The first config to would have hit all three at once, and none fails
loudly. These tests fail on the pre-fix code.
"""

from __future__ import annotations

import pytest

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.criteria.player_vp_min import PlayerVPMinCriteria
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

CONFIG = "configs/golden/25v25_maps_two_mode.yaml"


def _context(env: object) -> StepContext:
    """A real context -- `vp_gain` reads none of it, but no mocks."""
    return StepContext(
        distance_cache=compute_distances(
            env.player_models,  # type: ignore[attr-defined]
            env.objectives,  # type: ignore[attr-defined]
        ),
        current_turn=1,
        max_turns=20,
        board_width=int(env.board_width),  # type: ignore[attr-defined]
        board_height=int(env.board_height),  # type: ignore[attr-defined]
    )


def _view_with_mission_type(name: str) -> WargameEnv:
    """A live env whose mission carries `name` but the default numbers."""
    config = load_env_config(CONFIG)
    config.mission.type = name
    env = create_environment(env_config=config)
    env.reset(seed=700000)
    return env


# `none` is the only other REGISTERED mission type -- an unknown name is
# rejected at construction, which is correct and is not what this tests.
@pytest.mark.parametrize("mission_type", ["default", "none"])
def test_the_phase_gate_is_the_same_whatever_the_mission_is_called(
    mission_type: str,
) -> None:
    """Same numbers, different name -- the threshold must not move."""
    env = _view_with_mission_type(mission_type)
    criteria = PlayerVPMinCriteria(fraction_of_max=0.5)

    threshold = criteria.vp_threshold_for_terminal_bonus(env)
    env.close()

    assert threshold is not None, "a renamed mission must not disable the bonus"
    assert threshold > 0, "a renamed mission must not collapse the gate to min_vp"


def test_a_mission_that_pays_nothing_does_disable_the_bonus() -> None:
    """The one case where None is right: nothing to score against."""
    config = load_env_config(CONFIG, cap_per_turn="0")
    env = create_environment(env_config=config)
    env.reset(seed=700000)

    threshold = PlayerVPMinCriteria(
        fraction_of_max=0.5
    ).vp_threshold_for_terminal_bonus(env)
    env.close()

    assert threshold is None


def test_the_reward_normaliser_follows_the_mission_not_a_fallback() -> None:
    """`vp_gain` divides by the mission's own cap, not a hardcoded 15.

    Asserts the divisor the calculator actually uses, by giving the view a known
    VP delta and reading the reward back -- not merely that the property exists.
    """
    from wargame_rl.wargame.envs.reward.calculators.vp_gain import VPGainCalculator

    rewards = {}
    for cap in ("15", "30"):
        config = load_env_config(CONFIG, cap_per_turn=cap)
        # `vp_gain` never branched on the NAME -- it duck-typed into `params`
        # and fell back to 15. The cap is what this asserts.
        env = create_environment(env_config=config)
        env.reset(seed=700000)
        env._battle.restore_victory_points(  # noqa: SLF001
            player_vp=0, opponent_vp=0, player_vp_delta=30, opponent_vp_delta=0
        )
        rewards[cap] = VPGainCalculator().calculate(env, _context(env))
        env.close()

    # Same delta of 30, twice the cap, half the reward.
    assert rewards["15"] == pytest.approx(2.0)
    assert rewards["30"] == pytest.approx(1.0)
