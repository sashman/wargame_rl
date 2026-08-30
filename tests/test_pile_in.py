"""Pile-in: engaged units close up 3" before blows are traded.

`docs/rules/12-fight-phase.md` § Pile-in step. Tested through `env.step` where
the rule crosses the phase clock, and at the domain level for the geometry —
this project has twice shipped a defect that a full suite of unit tests could
not see because none of them called `env.step`.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.domain.pile_in import (
    SELECTION_RANGE_INCHES,
    agent_move_is_legal,
    pile_in,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

ENGAGEMENT = 1.0


def _env(pile: bool = True, n: int = 2) -> WargameEnv:
    config = WargameEnvConfig(
        number_of_wargame_models=n,
        number_of_opponent_models=n,
        number_of_objectives=1,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[
            ModelConfig(group_id=0, melee_weapons=[MeleeWeaponProfile()])
            for _ in range(n)
        ],
        opponent_models=[
            ModelConfig(group_id=0, melee_weapons=[MeleeWeaponProfile()])
            for _ in range(n)
        ],
        melee=MeleeConfig(enabled=True, pile_in=pile),
        engagement_range=ENGAGEMENT,
        base_radius=0.0,
        skip_phases=[BattlePhase.shooting],
    )
    env = create_environment(config)
    env.reset(seed=5)
    return env


def _kwargs(env: WargameEnv) -> dict[str, object]:
    quantities = env.rules_quantities
    return {
        "max_distance": quantities.scale.to_units(env.config.melee.pile_in_distance),
        "selection_range": quantities.scale.to_units(SELECTION_RANGE_INCHES),
        "engagement_range": quantities.engagement_range,
        "base_radius": quantities.base_radius,
        "board": (float(env.board_width), float(env.board_height)),
        "coherency_nearest": quantities.scale.to_units(
            env.config.coherency.nearest_distance
        ),
        "coherency_furthest": quantities.scale.to_units(
            env.config.coherency.furthest_distance
        ),
    }


def test_a_model_in_base_contact_is_PINNED() -> None:
    """*"Models in base contact with an enemy model cannot be moved."*

    This is what stops a pile-in dragging a locked model off its opponent to
    chase a nearer one.
    """
    # Arrange: m0 touching, m1 a stride away, both of one unit.
    env = _env()
    try:
        m0, m1 = env.wargame_models
        m0.location = np.array([10.0, 10.0], dtype=m0.location.dtype)
        m1.location = np.array([10.0, 11.0], dtype=m1.location.dtype)
        for index, enemy in enumerate(env.opponent_models):
            enemy.location = np.array([10.0, 10.0 + index * 0.4], dtype=float)
        pinned = np.array(m0.location, copy=True)

        # Act
        pile_in(
            env.wargame_models,
            env.opponent_models,
            eligible_units={0},
            **_kwargs(env),  # type: ignore[arg-type]
        )

        # Assert
        assert np.array_equal(m0.location, pinned), "a model in base contact moved"
    finally:
        env.close()


def test_a_disengaged_model_CLOSES_on_its_target() -> None:
    """*"Every model that is moved must end closer to the closest target."*"""
    # Arrange: the unit is engaged through m0, so m1 must close.
    env = _env()
    try:
        m0, m1 = env.wargame_models
        m0.location = np.array([10.0, 10.0], dtype=m0.location.dtype)
        m1.location = np.array([8.5, 10.0], dtype=m1.location.dtype)
        for index, enemy in enumerate(env.opponent_models):
            enemy.location = np.array([10.5, 10.0 + index * 0.4], dtype=float)
        before = float(np.linalg.norm(m1.location - env.opponent_models[0].location))

        # Act
        moved = pile_in(
            env.wargame_models,
            env.opponent_models,
            eligible_units={0},
            **_kwargs(env),  # type: ignore[arg-type]
        )

        # Assert
        after = float(np.linalg.norm(m1.location - env.opponent_models[0].location))
        assert moved == [0]
        assert after < before, "a piling-in model did not end closer"
    finally:
        env.close()


def test_it_does_nothing_when_the_unit_is_engaged_with_NOBODY_in_range() -> None:
    """No targets, no move — the rules select from within 5" or not at all."""
    # Arrange
    env = _env()
    try:
        for index, model in enumerate(env.wargame_models):
            model.location = np.array([10.0, 10.0 + index], dtype=float)
        for index, enemy in enumerate(env.opponent_models):
            enemy.location = np.array([50.0, 10.0 + index], dtype=float)
        before = [np.array(m.location, copy=True) for m in env.wargame_models]

        # Act
        moved = pile_in(
            env.wargame_models,
            env.opponent_models,
            eligible_units={0},
            **_kwargs(env),  # type: ignore[arg-type]
        )

        # Assert
        assert moved == []
        for model, origin in zip(env.wargame_models, before, strict=True):
            assert np.array_equal(model.location, origin)
    finally:
        env.close()


def test_melee_off_pile_in_never_runs() -> None:
    """The no-op guarantee: `melee.pile_in` is read only when melee is on."""
    # Arrange
    config = WargameEnvConfig(
        number_of_wargame_models=1,
        number_of_opponent_models=1,
        number_of_objectives=1,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        melee=MeleeConfig(enabled=False, pile_in=True),
        skip_phases=[
            BattlePhase.command,
            BattlePhase.shooting,
            BattlePhase.charge,
            BattlePhase.fight,
        ],
    )
    env = create_environment(config)
    try:
        env.reset(seed=1)
        before = [np.array(m.location, copy=True) for m in env.wargame_models]

        # Act
        for _ in range(4):
            env.step(WargameEnvAction(actions=[0] * len(env.wargame_models)))

        # Assert — nothing here moves a model that was told to stay.
        for model, origin in zip(env.wargame_models, before, strict=True):
            assert np.array_equal(model.location, origin)
    finally:
        env.close()


def test_consolidate_ONGOING_is_pile_in_restricted_to_engaged_units() -> None:
    """`12-fight-phase.md` § Consolidate — Ongoing mode.

    Its conditions are pile-in's exactly, so it reuses the primitive with the
    5" selection branch suppressed. What must NOT happen is a disengaged unit
    consolidating in Ongoing mode: the rules put it in Engaging or Objective
    mode instead, and the modes are ordered and compulsory.
    """
    # Arrange: the unit is 3" away — inside pile-in's 5" branch, outside contact.
    env = _env()
    try:
        for index, model in enumerate(env.wargame_models):
            model.location = np.array([10.0, 10.0 + index], dtype=float)
        for index, enemy in enumerate(env.opponent_models):
            enemy.location = np.array([13.0, 10.0 + index], dtype=float)
        before = [np.array(m.location, copy=True) for m in env.wargame_models]
        kwargs = _kwargs(env)

        # Act: selection_range 0.0 is what Ongoing mode passes.
        ongoing = pile_in(
            env.wargame_models,
            env.opponent_models,
            eligible_units={0},
            **{**kwargs, "selection_range": 0.0},  # type: ignore[arg-type]
        )

        # Assert
        assert ongoing == [], "a disengaged unit consolidated in Ongoing mode"
        for model, origin in zip(env.wargame_models, before, strict=True):
            assert np.array_equal(model.location, origin)

        # And the same unit DOES pile in, which is the control that shows the
        # suppression is the selection range and not the geometry.
        assert pile_in(
            env.wargame_models,
            env.opponent_models,
            eligible_units={0},
            **kwargs,  # type: ignore[arg-type]
        ) == [0]
    finally:
        env.close()


def test_an_ENGAGED_unit_may_not_ADVANCE_out_of_combat() -> None:
    """`09-movement-phase.md`: an engaged unit's only move is a FALL BACK.

    A fall back is capped at M. Until 2026-08-26 an engaged model kept all 48
    advance rungs, so it could withdraw `M + roll` — which
    `implementation-status.md` row 63 names as the observable difference
    between this env's engaged-movement rule and the rules'.

    ⚠ Behaviourally a no-op wherever melee is off: `back_off_to_unengaged` runs
    on every mover, so engagement is 0.0000% of model-pairs without the charge's
    exemption, and the seeded digest is unchanged across this fix.
    """
    # Arrange
    from scripts.scenario_overrides import load_env_config

    config = load_env_config("configs/experiments/25v25_maps_melee.yaml")
    config.n_advance_speed_bins = 3
    env = create_environment(env_config=config)
    try:
        env.reset(seed=1)
        handler = env.player_action_handler
        model = env.wargame_models[0]
        model.advance_roll = 6.0
        model.declared_advance = True
        free = np.array(env.opponent_models[0].location, copy=True)

        # Act: unengaged first — the control that shows the rungs exist at all.
        env.opponent_models[0].location = np.array([55.0, 40.0], dtype=float)
        unengaged = handler.advance_legality(env.wargame_models, env.opponent_models)
        # Then locked in melee.
        model.location = np.array([20.0, 20.0], dtype=model.location.dtype)
        env.opponent_models[0].location = np.array([20.5, 20.0], dtype=free.dtype)
        engaged = handler.advance_legality(env.wargame_models, env.opponent_models)

        # Assert
        assert unengaged[0].any(), "the control failed — no advance rungs at all"
        assert not engaged[0].any(), "an engaged model kept its advance rungs"
    finally:
        env.close()


def test_a_model_ENGAGED_BUT_NOT_TOUCHING_is_NOT_pinned() -> None:
    """The pin is BASE CONTACT, not engagement range — regression, 2026-08-26.

    `12-fight-phase.md`: *"Models in base contact with an enemy model cannot be
    moved."* `agent_move_is_legal` pinned on `engagement_matrix` at
    `engagement_range`, which is the SAME predicate at the SAME threshold that
    `ActionHandler.short_move_legality` uses to decide a model is ELIGIBLE to
    pile in. One predicate used twice in opposition made **eligible identical to
    pinned**, so an agent-chosen pile-in could never be legal unless it stood
    still. Measured on the shipped melee config, 86.5% of ORDERED pile-in moves
    delivered exactly 0.000" and the mean was 0.103" against a 3" ladder;
    narrowing the pin took that to 28.3% and 0.525".

    ⚠ This is the AGENT path (`1cfebfa`). The engine path (`pile_in`,
    `9e11bf0`) constructs its own legal move and was never affected — a test
    written against `pile_in` passes either way and is vacuous. Verified to fail
    without the fix.
    """
    # Arrange — m0 engaged at 0.5 (< 1.0 engagement) but NOT touching at radius
    # 0, then moved 0.2 closer to its target. Legal under the rules; refused by
    # the old pin.
    env = _env()
    try:
        m0, m1 = env.wargame_models
        for index, enemy in enumerate(env.opponent_models):
            enemy.location = np.array([10.5, 10.0 + index * 0.4], dtype=float)
        before = np.array([[10.0, 10.0], [10.0, 10.6]], dtype=float)
        m0.location = np.array([10.2, 10.0], dtype=m0.location.dtype)
        m1.location = np.array([10.0, 10.6], dtype=m1.location.dtype)
        quantities = env.rules_quantities

        # Act
        legal = agent_move_is_legal(
            env.wargame_models,
            [0, 1],
            before,
            [m for m in env.opponent_models if m.is_alive],
            selection_range=quantities.scale.to_units(SELECTION_RANGE_INCHES),
            engagement_range=quantities.engagement_range,
            base_radius=quantities.base_radius,
            coherency_nearest=quantities.scale.to_units(
                env.config.coherency.nearest_distance
            ),
            coherency_furthest=quantities.scale.to_units(
                env.config.coherency.furthest_distance
            ),
        )

        # Assert
        assert legal, "a model engaged but not in base contact was pinned"
    finally:
        env.close()


def test_a_model_in_base_contact_is_STILL_pinned_for_an_agent_move() -> None:
    """The other half: narrowing the pin must not delete the rule."""
    # Arrange — m0 exactly touching (radius 0), then moved.
    env = _env()
    try:
        m0, m1 = env.wargame_models
        for index, enemy in enumerate(env.opponent_models):
            enemy.location = np.array([10.0, 10.0 + index * 0.4], dtype=float)
        before = np.array([[10.0, 10.0], [10.0, 10.6]], dtype=float)
        m0.location = np.array([10.2, 10.0], dtype=m0.location.dtype)
        m1.location = np.array([10.0, 10.6], dtype=m1.location.dtype)
        quantities = env.rules_quantities

        # Act
        legal = agent_move_is_legal(
            env.wargame_models,
            [0, 1],
            before,
            [m for m in env.opponent_models if m.is_alive],
            selection_range=quantities.scale.to_units(SELECTION_RANGE_INCHES),
            engagement_range=quantities.engagement_range,
            base_radius=quantities.base_radius,
            coherency_nearest=quantities.scale.to_units(
                env.config.coherency.nearest_distance
            ),
            coherency_furthest=quantities.scale.to_units(
                env.config.coherency.furthest_distance
            ),
        )

        # Assert
        assert not legal, "a model in base contact was allowed to move"
    finally:
        env.close()


def test_a_pile_in_ENDPOINT_may_lie_inside_an_enemy_engagement_range() -> None:
    """The other half of the pair, and it lives in `ActionHandler.apply`.

    Regression, 2026-08-26. `back_off_to_unengaged` was handed the enemy rings
    for every phase but `charge`, and it returns `start` for a model that BEGINS
    inside a ring -- which is every model a pile-in is offered to, since being
    engaged is what makes it eligible. `12-fight-phase.md` requires these moves
    to END engaged, exactly as the charge does.

    ⚠ **Neither half of this fix works alone, and that is measured**: with the
    pin narrowed but no exemption, delivery is bit-identical to the defect
    (0.103", 86.5% zero); with the exemption but the old pin it gets WORSE
    (0.051", 94.9%), because models can now move and the referee then reverts
    their whole unit. Together: 0.525" and 28.3%.
    """
    # Arrange -- one model, engaged, ordered a pile-in that closes on the enemy.
    env = _env(n=1)
    try:
        model = env.wargame_models[0]
        enemy = env.opponent_models[0]
        enemy.location = np.array([10.5, 10.0], dtype=float)
        model.location = np.array([10.0, 10.0], dtype=model.location.dtype)
        start = np.array(model.location, copy=True)
        handler = env.player_action_handler
        legality = handler.short_move_legality(
            env.wargame_models, env.opponent_models, BattlePhase.pile_in
        )
        legal_actions = np.nonzero(legality[0])[0]
        assert legal_actions.size > 0, "an engaged model was offered no pile-in"

        # Act -- take a legal rung that is not STAY.
        moved = False
        for candidate in legal_actions:
            model.location = np.array(start, dtype=model.location.dtype)
            handler.apply(
                WargameEnvAction(actions=[int(candidate)]),
                env.wargame_models,
                env.board_width,
                env.board_height,
                cast(spaces.Tuple, env.action_space),
                phase=BattlePhase.pile_in,
                enemy_models=env.opponent_models,
            )
            if not np.array_equal(model.location, start):
                moved = True
                break

        # Assert
        assert moved, (
            "every legal pile-in rung was backed off to the model's start "
            "position -- the endpoint exemption is missing"
        )
    finally:
        env.close()
