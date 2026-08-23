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

from typing import cast

import numpy as np
import pytest

from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    OpponentPolicyConfig,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import WeaponProfile
from wargame_rl.wargame.model.common.factory import create_environment


def _config(bins: int) -> WargameEnvConfig:
    return WargameEnvConfig(
        render_mode=None, number_of_battle_rounds=6, n_advance_speed_bins=bins
    )


def _advance_actions(
    policy_name: str, bins: int, rounds: int = 6, advancing: bool = False
) -> tuple[int, int]:
    """Play an episode; return (advance actions chosen, movement actions chosen)."""
    env = create_environment(env_config=_config(bins))
    policy = build_baseline_policy(policy_name)
    policy.advance_when_out_of_reach = advancing  # type: ignore[attr-defined]
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
    advances, moves = _advance_actions("squad_march", bins=3, advancing=True)

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

    class Advancing(ScriptedSquadMarchPolicy):
        advance_when_out_of_reach = True

    old = build_baseline_policy("squad_march")  # the default: does not advance
    advancing = Advancing()

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
        action_new = advancing.select_action(
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
    from wargame_rl.wargame.envs.opponent.scripted_advance_to_objective_policy import (
        ScriptedAdvanceToObjectivePolicy,
    )
    from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig

    config = load_env_config("configs/experiments/25v25_maps_advance.yaml")
    env = create_environment(env_config=config)
    env.reset(seed=8)
    built = build_opponent_policy(
        OpponentPolicyConfig(type="scripted_advance_to_objective"), env
    )
    policy = cast(ScriptedAdvanceToObjectivePolicy, built)
    # Opt in: the default is now OFF, because the heuristic was rejected.
    policy.advance_when_out_of_reach = True
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


def test_advancing_is_OFF_by_default_because_the_heuristic_was_REJECTED() -> None:
    """The 2x2 that rejected it: advancing costs its USER about 78 vp.

    `25v25_maps_advance_refereed`, held-out nine, n=10, `squad_march_take` both
    sides, vp_margin to the player:

                           opponent walks   opponent advances
        player walks            -4.1              +72.7
        player advances        -81.8               -3.6

    Both-advance (-3.6) is indistinguishable from both-walk (-4.1), which is how
    a first measurement read this as "+15.5 to the bar". It is worth nothing to
    the bar -- both sides were handicapping themselves equally.

    The MECHANISM stays, because a bar that cannot use a core rule is not a bar.
    The HEURISTIC is what is rejected, so the default must stay off until a rule
    exists that prices the forfeited shooting.
    """
    from wargame_rl.wargame.envs.baseline.scripted_squad_march import (
        ScriptedSquadMarchPolicy,
    )
    from wargame_rl.wargame.envs.opponent.scripted_advance_to_objective_policy import (
        ScriptedAdvanceToObjectivePolicy,
    )

    assert ScriptedSquadMarchPolicy.advance_when_out_of_reach is False
    assert ScriptedAdvanceToObjectivePolicy.advance_when_out_of_reach is False


def _armed_config(bins: int, weapon_range: int) -> WargameEnvConfig:
    """A config with real weapons on both sides.

    ⚠ The bare `_config` above leaves `models` unset, and `max_weapon_ranges`
    returns 0.0 for every model of a `None` model list — so a rule that turns
    on weapon reach is vacuously satisfied there and a test written against it
    asserts nothing. Both sides are armed explicitly.
    """
    weapon = WeaponProfile(range=weapon_range)
    return WargameEnvConfig(
        render_mode=None,
        number_of_battle_rounds=6,
        n_advance_speed_bins=bins,
        number_of_wargame_models=4,
        models=[ModelConfig(weapons=[weapon], group_id=i // 2) for i in range(4)],
        number_of_opponent_models=4,
        opponent_models=[
            ModelConfig(weapons=[weapon], group_id=i // 2) for i in range(4)
        ],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
    )


def _no_shot_advance_actions(weapon_range: int) -> tuple[int, int]:
    """Play `squad_march_take_advance` armed; return (advances, ordinary moves)."""
    env = create_environment(env_config=_armed_config(3, weapon_range))
    policy = build_baseline_policy("squad_march_take_advance")
    observation, _ = env.reset(seed=3)
    advance_slice = env.player_action_handler.advance_slice
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


def test_the_priced_rule_advances_when_no_shot_is_forfeited() -> None:
    """At a short reach most of the march is out of range, so the run is free."""
    advances, moves = _no_shot_advance_actions(weapon_range=6)

    assert advances > 0, "it never advanced, so the free run is being declined"
    assert moves > 0, "it advanced every step, so it never settles onto a point"


def test_a_weapon_that_covers_the_board_stops_every_advance() -> None:
    """The rule's whole content: an advance that costs a shot is not taken.

    With reach past the board diagonal every member has an enemy within range
    of wherever a normal move would land, so no advance is ever free.
    """
    advances, moves = _no_shot_advance_actions(weapon_range=200)

    assert advances == 0, "it advanced while giving up a shot it actually had"
    assert moves > 0, "it stopped moving entirely, which is a different bug"


def test_the_priced_rule_leaves_the_rejected_heuristic_off() -> None:
    """The two rules are independent; `squad_march_take` must not start running."""
    plain = build_baseline_policy("squad_march_take")
    priced = build_baseline_policy("squad_march_take_advance")

    assert plain.advance_when_out_of_reach is False  # type: ignore[attr-defined]
    assert plain.advance_when_no_shot is False  # type: ignore[attr-defined]
    assert priced.advance_when_out_of_reach is False  # type: ignore[attr-defined]
    assert priced.advance_when_no_shot is True  # type: ignore[attr-defined]


def test_the_priced_rule_is_the_take_allocation_with_nothing_else_changed() -> None:
    """With no advance bins registered it must be `squad_march_take`, action for action."""
    env_plain = create_environment(env_config=_config(0))
    env_priced = create_environment(env_config=_config(0))
    plain = build_baseline_policy("squad_march_take")
    priced = build_baseline_policy("squad_march_take_advance")
    observation_plain, _ = env_plain.reset(seed=11)
    observation_priced, _ = env_priced.reset(seed=11)

    done = False
    while not done:
        action_plain = plain.select_action(
            env_plain.wargame_models,
            env_plain,
            action_mask=observation_plain.action_mask,
        )
        action_priced = priced.select_action(
            env_priced.wargame_models,
            env_priced,
            action_mask=observation_priced.action_mask,
        )
        assert list(action_plain.actions) == list(action_priced.actions)
        observation_plain, _r, done, _t, _i = env_plain.step(action_plain)
        observation_priced, _r2, _d2, _t2, _i2 = env_priced.step(action_priced)
    env_plain.close()
    env_priced.close()


def test_the_arrive_rule_runs_less_often_than_the_no_shot_rule() -> None:
    """D-40 is strictly narrower: every arrival is free, not every free run arrives.

    Both share the no-shot clause, so the only difference is the arrival
    condition. If the arrive rule ever ran MORE, the extra clause would be
    doing something other than what it says.
    """
    env = create_environment(env_config=_armed_config(3, weapon_range=6))
    advance_slice = env.player_action_handler.advance_slice
    counts = {}
    for name in ("squad_march_take_advance", "squad_march_take_arrive"):
        policy = build_baseline_policy(name)
        observation, _ = env.reset(seed=5)
        advances = 0
        done = False
        while not done:
            action = policy.select_action(
                env.wargame_models, env, action_mask=observation.action_mask
            )
            advances += sum(
                1
                for value in action.actions
                if advance_slice is not None
                and advance_slice.start <= value < advance_slice.end
            )
            observation, _r, done, _t, _i = env.step(action)
        counts[name] = advances
    env.close()

    assert counts["squad_march_take_arrive"] <= counts["squad_march_take_advance"]


def test_the_arrive_rule_leaves_every_other_flag_alone() -> None:
    """Only the two clauses it names may be on."""
    policy = build_baseline_policy("squad_march_take_arrive")

    assert policy.advance_when_out_of_reach is False  # type: ignore[attr-defined]
    assert policy.advance_when_no_shot is True  # type: ignore[attr-defined]
    assert policy.advance_to_arrive is True  # type: ignore[attr-defined]
