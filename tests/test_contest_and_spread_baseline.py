"""The `contest_and_spread` baseline: read the opponent, and do not stack fire.

It was built to test whether `squad_march_shoot` -- the bar every result in
`reports/` is quoted against -- is a weak ceiling, since it allocates squads by
a fixed `k % n_objectives` and fires nearest-first, exploiting neither an
opponent that stops moving by round 9 nor the fact that a second shot on a
one-wound model is usually discarded.

**It is not.** Measured on seeds 700000-700029, this policy scores 0.60 win /
+18.8 vp_margin against the bar's 0.77 / +39.4. Massing 10/10/5 and winning two
objectives outright beats spreading to grab the abandoned one and losing the
contested ones -- control is a strict count comparison, so concentration is the
reason the bar wins rather than a defect in it. Kept as a reference point, and
as the record of a refuted hypothesis.

These tests pin the two behaviours that distinguish it, on hand-built geometry
rather than on aggregate win rates, so a regression names its own cause.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.baseline.scripted_contest_and_spread import (
    ScriptedContestAndSpreadPolicy,
)
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import WeaponProfile
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

WEAPONS = [WeaponProfile(range=12, attacks=1)]


def _make_env() -> WargameEnv:
    """Four player models in range of four opponents, two objectives.

    The opponents all sit on objective 0, leaving objective 1 empty -- the
    situation `scripted_advance_to_objective` creates in every episode once it
    parks, and the one the allocation is supposed to exploit.
    """
    config = WargameEnvConfig(
        render_mode=None,
        board_width=40,
        board_height=40,
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=2,
        objective_radius_size=2,
        number_of_battle_rounds=6,
        max_groups=2,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        models=[
            ModelConfig(x=10, y=18 + i, group_id=i // 2, weapons=WEAPONS)
            for i in range(4)
        ],
        # All four opponents inside objective 0's radius at (20, 20).
        opponent_models=[
            ModelConfig(x=20, y=19 + (i % 2), group_id=0, weapons=WEAPONS)
            for i in range(4)
        ],
        objectives=[ObjectiveConfig(x=20, y=20), ObjectiveConfig(x=20, y=32)],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
    )
    return WargameEnv(config=config)


def test_allocation_sends_a_squad_to_the_undefended_objective() -> None:
    """The whole point: an empty objective is cheaper than a defended one.

    `squad_march` would send squad 0 to objective 0 and squad 1 to objective 1
    by `k % n`, which only coincidentally does the right thing. This asserts the
    allocation is driven by the opponent count, by checking the *cheapest*
    objective is the one taken first.
    """
    env = _make_env()
    env.reset(seed=0)
    policy = ScriptedContestAndSpreadPolicy()

    allocation = policy._objective_allocation(env, n_squads=2)

    # Objective 1 is empty, objective 0 holds four opponents, so the empty one
    # must be claimed before any squad is committed to the contested one.
    assert allocation[0] == 1, f"cheapest objective not taken first: {allocation}"


def test_allocation_covers_every_squad() -> None:
    """No squad may be left without an objective, whatever the counts are."""
    env = _make_env()
    env.reset(seed=0)
    policy = ScriptedContestAndSpreadPolicy()

    for n_squads in (1, 2, 3, 5):
        allocation = policy._objective_allocation(env, n_squads=n_squads)
        assert len(allocation) == n_squads
        assert all(0 <= index < len(env.objectives) for index in allocation)


def _shooting_targets(policy_name: str) -> list[int]:
    """Targets chosen in the first shooting phase where anyone can fire."""
    env = _make_env()
    env.reset(seed=0)
    policy = build_baseline_policy(policy_name)
    shooting_slice = env.player_action_handler.shooting_slice
    assert shooting_slice is not None

    for _ in range(20):
        observation = env._get_obs()
        if env.game_clock_state.phase == BattlePhase.shooting:
            action = policy.select_action(
                env.wargame_models,
                env,
                action_mask=np.asarray(observation.action_mask),
            )
            fired = [
                a - shooting_slice.start
                for a in action.actions
                if shooting_slice.start <= a < shooting_slice.end
            ]
            if fired:
                return fired
        env.step(policy.select_action(env.wargame_models, env))
    return []


def test_fire_is_spread_across_distinct_targets() -> None:
    """Several shooters with the same nearest enemy must not all take it.

    With `max_wounds: 1`, every shot after the first successful one on a target
    is discarded, so this is the difference between 0.83 and 1.48 expected kills
    from five shots.
    """
    targets = _shooting_targets("contest_and_spread")

    assert targets, "precondition: someone must be able to fire"
    assert len(targets) == len(set(targets)), (
        f"targets were stacked rather than spread: {targets}"
    )


def test_it_is_registered_under_its_name() -> None:
    """`measure-baselines` builds it by string, so the name is the contract."""
    assert build_baseline_policy("contest_and_spread") is not None
