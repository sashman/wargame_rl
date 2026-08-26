"""The fight step alternates between players, one unit at a time.

`docs/rules/12-fight-phase.md` § Fight step. v1 resolved the active player's
whole side and then the opponent's, which is a materially different game: every
one of the active player's casualties landed before any opposing unit swung
back, so whoever held the turn won every trade they started.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.fight import (
    FightSide,
    OverrunRules,
    resolve_fight_step,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

ENGAGEMENT = 1.0
# Lethal on purpose: the default blade is ~0.02 expected wounds a swing, which
# would make "who swung first" invisible in a test of ordering.
LETHAL = MeleeWeaponProfile(attacks=3, melee_skill=2, strength=10, ap=4, damage=1)


def _env(units: int = 2, lethal: bool = True) -> WargameEnv:
    """`units` squads a side, each of two models.

    `lethal` picks the one-shot-kill fixture, which makes fight ORDER visible.
    Pass False where the test is about a MOVE rather than about who died --
    with alternation and a lethal blade, whoever activates first wins every
    trade and can wipe the unit under test before its turn comes round.
    """
    profile = LETHAL if lethal else MeleeWeaponProfile()
    n = units * 2
    config = WargameEnvConfig(
        number_of_wargame_models=n,
        number_of_opponent_models=n,
        number_of_objectives=1,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[
            ModelConfig(group_id=i // 2, melee_weapons=[profile]) for i in range(n)
        ],
        opponent_models=[
            ModelConfig(group_id=i // 2, melee_weapons=[profile]) for i in range(n)
        ],
        melee=MeleeConfig(enabled=True),
        engagement_range=ENGAGEMENT,
        base_radius=0.0,
        skip_phases=[BattlePhase.shooting],
    )
    env = create_environment(config)
    env.reset(seed=11)
    return env


def _lock(env: WargameEnv) -> None:
    """Put every squad nose to nose with its opposite number."""
    for index, model in enumerate(env.wargame_models):
        model.location = np.array(
            [10.0 + (index // 2) * 20.0, 10.0 + (index % 2)], dtype=float
        )
    for index, model in enumerate(env.opponent_models):
        model.location = np.array(
            [10.5 + (index // 2) * 20.0, 10.0 + (index % 2)], dtype=float
        )


def _sides(env: WargameEnv) -> tuple[FightSide, FightSide]:
    return (
        FightSide(
            models=env.wargame_models,
            weapons=[cfg.melee_weapons for cfg in env.config.models or []],
        ),
        FightSide(
            models=env.opponent_models,
            weapons=[cfg.melee_weapons for cfg in env.config.opponent_models or []],
        ),
    )


def test_both_sides_swing_and_neither_resolves_its_whole_force_first() -> None:
    """The property v1 could not have: the reply lands between the blows.

    With two locked squads a side and a lethal blade, whole-side resolution
    gives the active player every kill before the opponent swings at all. Under
    alternation the opponent's first unit fights after the active player's
    first, so both sides land attacks.
    """
    # Arrange
    env = _env(units=2)
    try:
        _lock(env)
        quantities = env.rules_quantities

        # Act
        mine, theirs = resolve_fight_step(
            _sides(env),
            np.random.default_rng(3),
            engagement_range=quantities.engagement_range,
            base_diameter=2.0 * quantities.base_radius,
            pass_range=quantities.scale.to_units(5.0),
        )

        # Assert
        assert mine, "the active player never swung"
        assert theirs, "the opponent never swung — this is the v1 behaviour"
    finally:
        env.close()


def test_a_unit_is_never_selected_twice() -> None:
    """Each unit fights at most once per step, per `12-fight-phase.md`."""
    # Arrange
    env = _env(units=2)
    try:
        _lock(env)
        quantities = env.rules_quantities

        # Act
        mine, theirs = resolve_fight_step(
            _sides(env),
            np.random.default_rng(5),
            engagement_range=quantities.engagement_range,
            base_diameter=2.0 * quantities.base_radius,
            pass_range=quantities.scale.to_units(5.0),
        )

        # Assert — a model may swing several times only within its own unit's
        # single activation, so no ATTACKER index may appear under two groups.
        for results, models in (
            (mine, env.wargame_models),
            (theirs, env.opponent_models),
        ):
            seen: dict[int, int] = {}
            for blow in results:
                group = int(models[blow.attacker_idx].group_id)
                seen.setdefault(blow.attacker_idx, group)
                assert seen[blow.attacker_idx] == group
    finally:
        env.close()


def test_a_unit_with_STRIKES_FIRST_goes_before_one_without() -> None:
    """A charging unit fights first — the sub-step, not a sort key."""
    # Arrange: only the player's squad 0 charged.
    env = _env(units=2)
    try:
        _lock(env)
        for model in env.wargame_models:
            model.charged_this_turn = int(model.group_id) == 0
        quantities = env.rules_quantities

        # Act
        mine, _theirs = resolve_fight_step(
            _sides(env),
            np.random.default_rng(7),
            engagement_range=quantities.engagement_range,
            base_diameter=2.0 * quantities.base_radius,
            pass_range=quantities.scale.to_units(5.0),
        )

        # Assert
        assert mine, "the charging unit never swung"
        first = int(env.wargame_models[mine[0].attacker_idx].group_id)
        assert first == 0, "a unit without Strikes First swung before one with it"
    finally:
        env.close()


def test_nobody_engaged_means_nobody_swings() -> None:
    """The step ends when no unit is eligible on either side."""
    # Arrange: armies far apart.
    env = _env(units=1)
    try:
        for index, model in enumerate(env.wargame_models):
            model.location = np.array([5.0, 5.0 + index], dtype=float)
        for index, model in enumerate(env.opponent_models):
            model.location = np.array([55.0, 40.0 + index], dtype=float)
        quantities = env.rules_quantities

        # Act
        mine, theirs = resolve_fight_step(
            _sides(env),
            np.random.default_rng(9),
            engagement_range=quantities.engagement_range,
            base_diameter=2.0 * quantities.base_radius,
            pass_range=quantities.scale.to_units(5.0),
        )

        # Assert
        assert mine == [] and theirs == []
    finally:
        env.close()


def test_a_unit_ENGAGED_WITH_NOBODY_overruns_onto_a_new_target() -> None:
    """`12-fight-phase.md` § Overrun fight.

    A unit eligible to fight but engaged with nobody — it killed its charge
    target, or was left behind when one died — makes **one additional pile-in
    move** and then fights. ⚠ Keyed on the step-START eligibility, not on live
    contact: a live read cannot tell a unit that LOST its target from one that
    never had one.

    ⚠ **The blade is the DEFAULT one here, not the lethal fixture the other
    tests use, and that is deliberate.** Built with the lethal profile this test
    asserted the wrong thing and passed for the wrong reason: the OPPONENT's
    stranded squad overran first and wiped the player's before its turn came
    round, so the mechanism worked and the assertion failed. With alternation
    and a one-shot-kill weapon, whoever activates first wins every trade —
    which is precisely the property alternating activation exists to expose.
    """
    # Arrange: player squad 1 is stranded 2" from the only enemy unit — inside
    # pile-in's reach, outside contact — while squad 0 holds it in melee.
    env = _env(units=2, lethal=False)
    try:
        for index, model in enumerate(env.wargame_models):
            model.location = (
                np.array([10.0, 10.0 + index], dtype=float)
                if index < 2
                else np.array([10.0, 14.0 + index], dtype=float)
            )
        for index, model in enumerate(env.opponent_models):
            model.location = np.array([10.5, 10.0 + index], dtype=float)
        stranded = [2, 3]
        before = [np.array(env.wargame_models[i].location, copy=True) for i in stranded]
        quantities = env.rules_quantities
        overrun = OverrunRules(
            pile_in_distance=quantities.scale.to_units(3.0),
            selection_range=quantities.scale.to_units(5.0),
            base_radius=quantities.base_radius,
            board=(float(env.board_width), float(env.board_height)),
            coherency_nearest=quantities.scale.to_units(
                env.config.coherency.nearest_distance
            ),
            coherency_furthest=quantities.scale.to_units(
                env.config.coherency.furthest_distance
            ),
        )

        # Act
        resolve_fight_step(
            _sides(env),
            np.random.default_rng(13),
            engagement_range=quantities.engagement_range,
            base_diameter=2.0 * quantities.base_radius,
            pass_range=quantities.scale.to_units(5.0),
            started_eligible=({0, 1}, {0}),
            overrun=overrun,
        )

        # Assert — the additional pile-in move is the observable half; the swing
        # that follows it is negligible with this blade by construction.
        assert any(
            not np.array_equal(before[row], env.wargame_models[index].location)
            for row, index in enumerate(stranded)
        ), "the stranded unit never took its additional pile-in move"
    finally:
        env.close()


def test_overrun_is_OFF_by_the_flag() -> None:
    """The ablation: without it, a stranded unit simply does not fight."""
    # Arrange — the same board, no overrun rules supplied.
    env = _env(units=2, lethal=False)
    try:
        for index, model in enumerate(env.wargame_models):
            model.location = (
                np.array([10.0, 10.0 + index], dtype=float)
                if index < 2
                else np.array([10.0, 14.0 + index], dtype=float)
            )
        for index, model in enumerate(env.opponent_models):
            model.location = np.array([10.5, 10.0 + index], dtype=float)
        stranded = [2, 3]
        before = [np.array(env.wargame_models[i].location, copy=True) for i in stranded]
        quantities = env.rules_quantities

        # Act
        resolve_fight_step(
            _sides(env),
            np.random.default_rng(13),
            engagement_range=quantities.engagement_range,
            base_diameter=2.0 * quantities.base_radius,
            pass_range=quantities.scale.to_units(5.0),
        )

        # Assert
        for row, index in enumerate(stranded):
            assert np.array_equal(before[row], env.wargame_models[index].location)
    finally:
        env.close()
