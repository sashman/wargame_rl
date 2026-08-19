"""Playing a scripted *baseline* on the opponent side.

The baselines are the strongest scripted play here — `squad_march_take` scores
115.0 vp_margin on the real tables — but they were written to drive the player,
and the opponent hierarchy is separate. `scripted_baseline` adapts one to the
other by handing the baseline a side-swapped view of the env instead of a second
copy of every policy.

The whole correctness question is that mirror. A baseline that reads an
un-mirrored side attribute plays for the wrong army: it steers with the player's
action handler, or reads its own models as targets. So the tests below pin the
swap directly, prove a real episode moves and shoots the *opponent's* models,
and keep a static guard on the set of side-specific attributes the baseline
package reads.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.opponent.mirror import MirroredEnv
from wargame_rl.wargame.envs.opponent.registry import (
    build_opponent_policy,
    get_registry,
)
from wargame_rl.wargame.envs.opponent.scripted_baseline_policy import (
    ScriptedBaselineOpponentPolicy,
)
from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import WeaponProfile
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

WEAPONS = [WeaponProfile(range=12, attacks=2)]

# Attributes of the env that mean different things to the two sides. Every one
# of these must be overridden by `MirroredEnv`; anything else is shared and is
# allowed to fall through to the real env.
SIDE_SPECIFIC = {
    "wargame_models",
    "player_models",
    "opponent_models",
    "player_action_handler",
    "opponent_action_handler",
    "config",
    "player_vp",
    "opponent_vp",
    "player_vp_delta",
    "opponent_vp_delta",
    "player_max_ranges",
    "opponent_max_ranges",
    "deployment_zone",
    "opponent_deployment_zone",
    "last_player_shooting_results",
    "last_opponent_shooting_results",
}

# Modules that read the env through a mirror. `observation_builder` joined the
# list when the mirror was extended to the observation path -- without it the
# guarantee stopped covering the code that had just come to depend on it.
MIRROR_CONSUMERS = (
    "wargame_rl/wargame/envs/baseline",
    "wargame_rl/wargame/envs/env_components/observation_builder.py",
    "wargame_rl/wargame/model/common/decoding.py",
)


def _make_env(baseline: str, *, player_x: int = 14) -> WargameEnv:
    """Two squads a side, four objectives, the player parked on the left pair.

    The geometry is chosen so a *mirrored* allocation is distinguishable from an
    un-mirrored one. Objectives 0 and 1 (x=14) are occupied by the player and 2
    and 3 (x=46) are empty, so an allocation policy playing the opponent — which
    ranks by how many *enemies* stand on each objective — must send both squads
    right. A mirror that failed to swap sides would read the opponent's own
    models as the occupants, find all four objectives equally empty, and take 0
    and 1 in index order instead.

    `player_x` moves the player's models without moving the objectives, so the
    shooting test can start inside 12" while the allocation test stays clean.
    """
    config = WargameEnvConfig(
        render_mode=None,
        board_width=60,
        board_height=40,
        number_of_wargame_models=6,
        number_of_opponent_models=6,
        number_of_objectives=4,
        objective_radius_size=3,
        number_of_battle_rounds=6,
        max_groups=2,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        models=[
            ModelConfig(
                x=player_x, y=10 + 20 * (i // 3), group_id=i // 3, weapons=WEAPONS
            )
            for i in range(6)
        ],
        opponent_models=[
            ModelConfig(x=56, y=10 + 20 * (i // 3), group_id=i // 3, weapons=WEAPONS)
            for i in range(6)
        ],
        objectives=[
            ObjectiveConfig(x=14, y=10),
            ObjectiveConfig(x=14, y=30),
            ObjectiveConfig(x=46, y=10),
            ObjectiveConfig(x=46, y=30),
        ],
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": baseline}
        ),
        # The default phase is `all_at_objectives` with `terminate_on_success`,
        # and the player is deployed standing on two objectives -- which ends
        # the episode on step one and makes every "run it out" assertion below
        # vacuous. Spelled out so the episode lasts its full six rounds.
        reward_phases=[
            RewardPhaseConfig(
                name="play_it_out",
                reward_calculators=[RewardCalculatorConfig(type="vp_gain")],
                success_criteria=SuccessCriteriaConfig(type="player_ahead_on_vp"),
                terminate_on_success=False,
            )
        ],
    )
    return WargameEnv(config=config)


def _hold(env: WargameEnv) -> WargameEnvAction:
    """Every player model stands still, so anything that moves is the opponent."""
    return WargameEnvAction(actions=[STAY_ACTION] * len(env.player_models))


def test_the_mirror_swaps_the_sides_and_shares_everything_else() -> None:
    """Own/enemy/handler/config flip; objectives and the clock do not."""
    env = _make_env("squad_march_take")
    env.reset(seed=0)

    mirror = MirroredEnv(env)

    assert mirror.wargame_models is env.opponent_models
    assert mirror.player_models is env.opponent_models
    assert mirror.opponent_models is env.wargame_models
    assert mirror.player_action_handler is env.opponent_action_handler
    assert (
        mirror.config.number_of_wargame_models == env.config.number_of_opponent_models
    )
    assert (
        mirror.config.number_of_opponent_models == env.config.number_of_wargame_models
    )
    assert mirror.config.deployment_zone == env.config.opponent_deployment_zone

    # Shared state falls through rather than being re-declared, which is what
    # keeps the mirror from drifting as the env grows.
    assert mirror.objectives is env.objectives
    assert mirror.game_clock_state == env.game_clock_state
    assert mirror.board_width == env.board_width


def test_every_side_specific_attribute_the_baselines_read_is_mirrored() -> None:
    """A new side-specific read must be classified, not silently fall through.

    Scans every module that reads the env through a mirror for `env.<attr>` and
    `view.<attr>`, and requires each name to be either overridden by
    `MirroredEnv` or explicitly shared. `evaluate.py` is excluded: it drives the
    player deliberately and never sees the mirror.

    ⚠ **This catches new *reads of known* side-specific names, not newly
    invented ones.** The assertion intersects with `SIDE_SPECIFIC`, so an
    attribute nobody adds to that set falls through in silence. The guard that
    cannot be fooled that way is `tests/test_swap_invariance.py`, which compares
    the mirrored observation against the other seat's tensor for tensor. Treat
    this scan as a cheap early warning, not as the guarantee.
    """
    root = Path(__file__).resolve().parents[1]
    paths: list[Path] = []
    for entry in MIRROR_CONSUMERS:
        target = root / entry
        paths.extend(sorted(target.glob("*.py")) if target.is_dir() else [target])

    reads: set[str] = set()
    for path in paths:
        if path.name in {"__init__.py", "evaluate.py"}:
            continue
        # `view.` as well as `env.`: the observation builder takes a BattleView
        # and names its parameter `view`, so an `env.`-only pattern read none of
        # the code this guarantee had just been extended to cover.
        reads |= set(
            re.findall(r"\b(?:env|view)\.([a-z_][a-z_0-9]*)", path.read_text())
        )

    env = _make_env("squad_march")
    # `dir` on the instance, not the class: `config` is swapped in `__init__`
    # rather than declared as a property.
    mirrored = {
        name
        for name in dir(MirroredEnv(env))
        if not name.startswith("__") and name != "_env"
    }
    unclassified = (reads & SIDE_SPECIFIC) - mirrored

    assert not unclassified, (
        f"baselines read side-specific env attributes the mirror does not swap: "
        f"{sorted(unclassified)}. Add a property to `MirroredEnv`, or add the "
        f"name to SIDE_SPECIFIC here if it is genuinely shared."
    )
    # And the mirror must not have gone stale in the other direction: every
    # attribute it swaps has to still exist on the real env.
    for name in mirrored:
        assert hasattr(env, name), f"MirroredEnv overrides a dead attribute: {name}"


def test_it_moves_the_opponents_army_and_leaves_the_player_alone() -> None:
    """The adapted baseline drives the opponent's models, not the player's."""
    env = _make_env("squad_march_take")
    env.reset(seed=0)

    before = [tuple(model.location) for model in env.opponent_models]
    player_before = [tuple(model.location) for model in env.player_models]

    # Two steps, because the opponent acts after the phase advances: the first
    # step's opponent turn falls in the shooting phase, the second in movement.
    env.step(_hold(env))
    env.step(_hold(env))

    after = [tuple(model.location) for model in env.opponent_models]
    assert after != before, "the opponent did not move"
    # Opponents deploy at x=56 and every objective is to their left.
    assert all(new[0] < old[0] for old, new in zip(before, after))
    assert [tuple(m.location) for m in env.player_models] == player_before


def test_the_allocation_counts_the_player_as_the_enemy() -> None:
    """The mirror is what makes `squad_march_take` rank the right occupancy.

    The player holds objectives 0 and 1, so the two cheapest objectives are 2
    and 3 and both opponent squads must head there. Un-mirrored, the policy
    would count its own models, see four empty objectives and march left.
    """
    env = _make_env("squad_march_take")
    env.reset(seed=0)

    for _ in range(env.max_turns):
        _, _, terminated, truncated, _ = env.step(_hold(env))
        if terminated or truncated:
            break

    # Objectives 2 and 3 sit at x=46; 0 and 1 at x=14. Run the episode out
    # rather than a couple of steps, because both allocations start by moving
    # left and only separate once the squads arrive.
    for model in env.opponent_models:
        x, y = float(model.location[0]), float(model.location[1])
        assert x > 40.0, (
            f"opponent marched at the objectives the player holds: {(x, y)}"
        )


def test_a_shooting_baseline_fires_for_the_opponent_and_hits_the_player() -> None:
    """`squad_march_take` shoots, and its shots resolve against player models."""
    # Deployed inside the 12" weapon range, so a target exists from round one.
    env = _make_env("squad_march_take", player_x=50)
    env.reset(seed=0)

    wounds_before = sum(model.stats["current_wounds"] for model in env.player_models)
    fired = False
    for _ in range(env.max_turns):
        _, _, terminated, truncated, _ = env.step(_hold(env))
        results = env.last_opponent_shooting_results
        if results:
            fired = True
            # Indices address the *player's* model list. A mirror that failed to
            # swap sides would have the opponent shooting into its own army.
            assert all(
                0 <= result.target_idx < len(env.player_models) for result in results
            )
            if (
                sum(model.stats["current_wounds"] for model in env.player_models)
                < wounds_before
            ):
                break
        if terminated or truncated:
            break

    assert fired, "the shooting baseline never fired as the opponent"
    assert (
        sum(model.stats["current_wounds"] for model in env.player_models)
        < wounds_before
    ), "the opponent's fire never landed on the player"


@pytest.mark.parametrize(
    ("baseline", "shoots"),
    [
        ("squad_march_take", True),
        ("squad_march_shoot", True),
        ("squad_march", False),
        ("random", False),
    ],
)
def test_shoots_is_derived_from_the_baseline(baseline: str, shoots: bool) -> None:
    """`shoots` gates mask refinement, so it must track the baseline it wraps.

    Left False for a baseline that fires, the env would build a phase-and-alive
    mask only and every shot would be taken unchecked by range and line of sight.
    """
    env = _make_env("squad_march")
    policy = build_opponent_policy(
        OpponentPolicyConfig(type="scripted_baseline", params={"baseline": baseline}),
        env,
    )
    assert isinstance(policy, ScriptedBaselineOpponentPolicy)
    assert policy.shoots is shoots
    assert policy.baseline_name == baseline


@pytest.mark.parametrize("params", [{}, {"baseline": "no_such_baseline"}])
def test_a_bad_baseline_name_fails_at_construction(params: dict[str, str]) -> None:
    """Validation at construction, not on the first step of a training run."""
    env = _make_env("squad_march")
    with pytest.raises(ValueError):
        build_opponent_policy(
            OpponentPolicyConfig(type="scripted_baseline", params=params), env
        )


def test_an_env_carrying_this_policy_survives_a_deep_copy() -> None:
    """The regression that killed every training run on this policy.

    Lightning deep-copies the env in `save_hyperparameters`, and `deepcopy`
    reconstructs an object *without* calling `__init__` — so a `__getattr__`
    that reaches through `self._env` re-enters itself while `_env` is still
    missing and recurses until the stack ends. Every other test here builds the
    mirror normally and never sees it.
    """
    env = _make_env("squad_march_take")
    env.reset(seed=0)

    clone = copy.deepcopy(env)

    policy = clone.opponent_policy
    assert isinstance(policy, ScriptedBaselineOpponentPolicy)
    assert policy.baseline_name == "squad_march_take"
    # And the copy is still playable, not merely constructible.
    clone.step(_hold(clone))


def test_the_existing_opponent_policies_are_untouched() -> None:
    """Adding the adapter must not disturb the registry every config names."""
    registry = get_registry()
    assert "scripted_baseline" in registry
    assert {
        "random",
        "scripted_advance_to_objective",
        "scripted_advance_and_shoot",
    } <= (set(registry))
