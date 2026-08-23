"""The scripted policies use the Advance move, because it is a core rule.

⚠ **A bar that cannot use a rule the agent can is not a bar.** Every scripted
baseline and every opponent policy predates the Advance move, so before this
they marched at Move while the agent could run at `M + D6`. Scoring an
advancing agent against a walking script measures a handicap, not a policy --
this project's most expensive documented error, made twice before.

`advance_when_out_of_reach = False` reproduces the pre-Advance behaviour
exactly, so the two are comparable on one config.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment


def _config(bins: int) -> WargameEnvConfig:
    return WargameEnvConfig(
        render_mode=None, number_of_battle_rounds=6, n_advance_speed_bins=bins
    )


def _advance_actions(policy_name: str, bins: int, rounds: int = 6) -> tuple[int, int]:
    """Play an episode; return (advance actions chosen, movement actions chosen)."""
    env = create_environment(env_config=_config(bins))
    policy = build_baseline_policy(policy_name)
    observation, _ = env.reset(seed=3)
    handler = env.player_action_handler
    advance_slice = handler.advance_slice
    advances = moves = 0
    done = False
    while not done:
        action = policy.select_action(
            env.wargame_models, env, action_mask=observation.action_mask
        )
        for value in action.actions:
            if (
                advance_slice is not None
                and advance_slice.start <= value < advance_slice.end
            ):
                advances += 1
            elif value != 0:
                moves += 1
        observation, _r, done, _t, _i = env.step(action)
    env.close()
    return advances, moves


def test_a_squad_that_cannot_reach_its_objective_advances() -> None:
    """The whole point: run while far, walk once close."""
    advances, moves = _advance_actions("squad_march", bins=3)

    assert advances > 0, "the scripted bar never used the Advance move"
    assert moves > 0, "it advanced every step, so it never settles onto a point"


def test_the_toggle_reproduces_the_pre_advance_bar_exactly() -> None:
    """`advance_when_out_of_reach = False` must be the old policy, action for action."""
    env_old = create_environment(env_config=_config(3))
    env_new = create_environment(env_config=_config(3))
    # A subclass, not a mutated instance: mutating one would be a hazard for any
    # test that ran after it on the same worker.
    from wargame_rl.wargame.envs.baseline.scripted_squad_march import (
        ScriptedSquadMarchPolicy,
    )

    class NoAdvance(ScriptedSquadMarchPolicy):
        advance_when_out_of_reach = False

    old = NoAdvance()
    new = build_baseline_policy("squad_march")

    observation_old, _ = env_old.reset(seed=5)
    observation_new, _ = env_new.reset(seed=5)
    slice_ = env_new.player_action_handler.advance_slice
    assert slice_ is not None

    saw_difference = False
    done = False
    while not done:
        action_old = old.select_action(
            env_old.wargame_models, env_old, action_mask=observation_old.action_mask
        )
        action_new = new.select_action(
            env_new.wargame_models, env_new, action_mask=observation_new.action_mask
        )
        if list(action_old.actions) != list(action_new.actions):
            saw_difference = True
        assert not any(slice_.start <= a < slice_.end for a in action_old.actions), (
            "the toggle is off and the policy still advanced"
        )
        observation_old, _r, done, _t, _i = env_old.step(action_old)
        observation_new, _r2, _d2, _t2, _i2 = env_new.step(action_new)

    assert saw_difference, "the toggle changed nothing, so it is not a control"


def test_a_scenario_without_advance_bins_is_untouched() -> None:
    """Every existing config must play exactly as before -- no slice, no change."""
    advances, moves = _advance_actions("squad_march", bins=0)

    assert advances == 0
    assert moves > 0


@pytest.mark.parametrize("policy", ["squad_march", "squad_march_shoot"])
def test_best_advance_toward_respects_the_units_own_roll(policy: str) -> None:
    """The reachable distance is `M + roll`, so the roll has to be read, not assumed."""
    env = create_environment(env_config=_config(3))
    env.reset(seed=11)
    handler = env.player_action_handler

    far = handler.best_advance_toward(1.0, 0.0, advance_roll=6.0, model_idx=0)
    near = handler.best_advance_toward(
        1.0, 0.0, advance_roll=0.0, max_step_length=1.0, model_idx=0
    )
    assert far is not None and near is not None

    reach_far = np.linalg.norm(
        handler.decode_action(far, model_idx=0, advance_roll=6.0)
    )
    reach_near = np.linalg.norm(
        handler.decode_action(near, model_idx=0, advance_roll=0.0)
    )
    assert reach_far > reach_near, "the roll did not change the distance offered"


def test_no_advance_slice_returns_none_rather_than_raising() -> None:
    """So a caller can stay config-agnostic and fall back to a normal move."""
    env = create_environment(env_config=_config(0))
    env.reset(seed=1)

    assert env.player_action_handler.best_advance_toward(1.0, 0.0, 3.0) is None


def test_a_darkened_slice_offers_no_advance_action() -> None:
    """A darkened slice is masked, so offering one would be an illegal action.

    This is also what makes a darkened config a genuine "advance off" control
    for a scripted policy: the same policy code plays the pre-Advance game.
    """
    config = _config(3)
    config.dark_action_slices = ["advance"]
    env = create_environment(env_config=config)
    env.reset(seed=1)

    assert env.player_action_handler.best_advance_toward(1.0, 0.0, 3.0) is None


def test_a_scripted_policy_never_emits_an_illegal_action_when_darkened() -> None:
    """The behavioural version: play it out and check every action against the mask."""
    config = _config(3)
    config.dark_action_slices = ["advance"]
    env = create_environment(env_config=config)
    policy = build_baseline_policy("squad_march")
    observation, _ = env.reset(seed=4)

    done = False
    while not done:
        action = policy.select_action(
            env.wargame_models, env, action_mask=observation.action_mask
        )
        mask = np.asarray(observation.action_mask)
        for model_index, value in enumerate(action.actions):
            assert mask[model_index][value], (
                f"model {model_index} chose action {value}, which the mask forbids"
            )
        observation, _r, done, _t, _i = env.step(action)
    env.close()


def test_the_opponent_advances_too() -> None:
    """A rules feature the OPPONENT cannot use is the same defect as a bar that cannot.

    Until this existed, an advancing agent trained against an opponent walking at
    Move while it ran at `M + D6` — which flatters the agent at the very matchup
    it is scored on.

    ⚠ Exercises `scripted_advance_to_objective` directly rather than through a
    config, because the golden advance config's opponent is `scripted_baseline`
    — a *baseline* played on the opponent side, which takes its Advance from
    `ScriptedSquadMarchPolicy` and is covered by the tests above.
    """
    from scripts.scenario_overrides import load_env_config
    from wargame_rl.wargame.envs.opponent.registry import build_opponent_policy
    from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig

    config = load_env_config("configs/experiments/25v25_maps_advance.yaml")
    env = create_environment(env_config=config)
    env.reset(seed=8)
    policy = build_opponent_policy(
        OpponentPolicyConfig(type="scripted_advance_to_objective"), env
    )
    advance_slice = env.opponent_action_handler.advance_slice
    assert advance_slice is not None

    action = policy.select_action(env.opponent_models, env)
    advances = sum(
        1 for a in action.actions if advance_slice.start <= a < advance_slice.end
    )
    env.close()

    assert advances > 0, "the opponent never used the Advance move"


def test_the_opponent_toggle_reproduces_the_walking_opponent() -> None:
    """`advance_when_out_of_reach = False` must give the pre-Advance opponent."""
    from scripts.scenario_overrides import load_env_config
    from wargame_rl.wargame.envs.opponent.scripted_advance_to_objective_policy import (
        ScriptedAdvanceToObjectivePolicy,
    )

    class Walking(ScriptedAdvanceToObjectivePolicy):
        advance_when_out_of_reach = False

    config = load_env_config("configs/experiments/25v25_maps_advance.yaml")
    env = create_environment(env_config=config)
    env.reset(seed=8)
    advance_slice = env.opponent_action_handler.advance_slice
    assert advance_slice is not None

    action = Walking(env).select_action(env.opponent_models, env)
    env.close()

    assert not any(
        advance_slice.start <= a < advance_slice.end for a in action.actions
    ), "the toggle is off and the opponent still advanced"


def test_the_opponents_advance_columns_are_zeroed_not_stale() -> None:
    """Each side rolls on its own turn, so the opponent's values are a turn old.

    Zeroed rather than dropped: the two token types share a feature width, and a
    constant-zero column is informationally identical to no column while costing
    no shape change.
    """
    env = create_environment(env_config=_config(3))
    observation, _ = env.reset(seed=2)

    for _ in range(6):
        observation, _r, done, _t, _i = env.step(
            WargameEnvAction(actions=[int(a) for a in env.action_space.sample()])
        )
        if done:
            break
        for model in observation.opponent_models:
            assert model.advance_roll == 0.0
            assert model.advanced_this_turn == 0.0
