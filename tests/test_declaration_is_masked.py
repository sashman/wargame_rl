"""A unit the rules make ineligible may not DECLARE a move type.

⚠ **The declaration was unmasked on both seats until 2026-08-26.**
`ActionRegistry.get_action_mask` is purely phase-based, and nothing refined the
`move_type` slice, so any alive model could declare anything in any command
phase. Eligibility was enforced only later, on the MOVE, where an ineligible
unit simply found no legal rung.

Both halves of that cost something:

- a charge declared by an ineligible unit is a **bit-exact no-op** -- no state
  change, no reward difference -- so the policy got a free action with nothing
  to learn from. Measured on the first unshaped arm: **71.4% of declared
  model-steps held zero legal rungs**, and declarations landed on eligible units
  *below chance* (31.4%, z = −2.54) against the teacher's 40/40;
- an advance declaration sets `advanced_this_turn` immediately, which both
  shooting masks read, so an ENGAGED unit could forfeit its whole shooting phase
  for a move `advance_legality` then refused it.

⚠ The gap map rated charge eligibility **implemented**. It was implemented for
the *move*, not the *declaration*.

These go through `build_observation`, which is what `step` calls. A handler-level
assertion would have passed throughout the entire life of the bug -- the handler
always knew the answer, and nothing asked it.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.env_components.actions import MOVE_TYPE_CHARGE
from wargame_rl.wargame.envs.env_components.observation_builder import build_observation
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

CHARGE_RANGE = 12.0


def _env() -> WargameEnv:
    """Two squads of two a side with melee on and an opponent that holds still."""
    config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=2,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[
            ModelConfig(group_id=i // 2, melee_weapons=[MeleeWeaponProfile()])
            for i in range(4)
        ],
        opponent_models=[
            ModelConfig(group_id=i // 2, melee_weapons=[MeleeWeaponProfile()])
            for i in range(4)
        ],
        melee=MeleeConfig(enabled=True),
        engagement_range=1.0,
        base_radius=0.0,
        skip_phases=[BattlePhase.fight],
    )
    env = create_environment(config)
    env.reset(seed=11)
    return env


def _to_command_phase(env: WargameEnv) -> None:
    """Step STAY until the command phase.

    ⚠ Asserts it arrived. An earlier version called a method that does not
    exist and still passed every test, because `reset` leaves the clock IN the
    command phase and the loop body never ran -- a helper that cannot fail is
    indistinguishable from one that works.
    """
    stay = WargameEnvAction(actions=[0] * len(env.wargame_models))
    for _ in range(len(BattlePhase) * 2):
        if env.game_clock_state.phase is BattlePhase.command:
            return
        env.step(stay)
    raise AssertionError("never reached the command phase")


def _charge_column(env: WargameEnv) -> np.ndarray:
    """The declaration mask's charge column, as `step` would build it."""
    observation = build_observation(
        env, action_registry=env.player_action_handler.registry
    )
    mask = np.asarray(observation.action_mask, dtype=bool)
    handler = env.player_action_handler
    action = handler.move_type_action(MOVE_TYPE_CHARGE)
    assert action is not None
    return mask[:, action]


def _place(models: list, x: float, y: float) -> None:
    for offset, model in enumerate(models):
        model.location = np.array([x + 0.1 * offset, y], dtype=float)


def test_a_unit_with_no_enemy_within_charge_range_cannot_declare() -> None:
    """`11-charge-phase.md`: ineligible if *not within 12" of any enemy unit*."""
    # Arrange
    env = _env()
    try:
        _to_command_phase(env)
        _place(env.wargame_models, 5.0, 5.0)
        _place(env.opponent_models, 5.0 + CHARGE_RANGE + 5.0, 5.0)

        # Act
        charge = _charge_column(env)

        # Assert
        assert not charge.any(), "an out-of-range unit was offered a charge declaration"

        # ...and this is exactly what the bug looked like, so the test cannot go
        # vacuous: the phase-only mask -- all the old code ever applied -- still
        # says every alive model may declare a charge here.
        handler = env.player_action_handler
        unrefined = handler.registry.get_model_action_masks(
            BattlePhase.command,
            len(env.wargame_models),
            alive_mask=np.array([m.is_alive for m in env.wargame_models]),
        )
        action = handler.move_type_action(MOVE_TYPE_CHARGE)
        assert action is not None
        assert unrefined[:, action].all(), (
            "the unrefined mask no longer differs from the refined one, so this "
            "test would pass with the fix reverted"
        )
    finally:
        env.close()


def test_the_declaration_is_offered_once_an_enemy_is_within_range() -> None:
    """The positive control: without it, a mask that refuses everything passes.

    ⚠ Updated 2026-08-26 with the roll gate: eligibility now also requires the
    unit's 2D6 to cover the gap, so the fixture pins a covering roll explicitly
    rather than inheriting whatever the reset rolled. The old fixture sat at
    10" and passed or failed on the dice.
    """
    # Arrange
    env = _env()
    try:
        _to_command_phase(env)
        _place(env.wargame_models, 5.0, 5.0)
        _place(env.opponent_models, 5.0 + CHARGE_RANGE - 2.0, 5.0)
        for model in env.wargame_models:
            model.charge_roll = 12.0

        # Act
        charge = _charge_column(env)

        # Assert
        assert charge.any(), "an in-range unit was denied a charge declaration"
    finally:
        env.close()


def test_the_row_is_never_emptied_because_normal_is_always_legal() -> None:
    """STAY declares `normal`, so masking it would make the phase unsteppable."""
    # Arrange
    env = _env()
    try:
        _to_command_phase(env)
        _place(env.wargame_models, 5.0, 5.0)
        _place(env.opponent_models, 5.0 + CHARGE_RANGE + 5.0, 5.0)

        # Act
        observation = build_observation(
            env, action_registry=env.player_action_handler.registry
        )
        mask = np.asarray(observation.action_mask, dtype=bool)

        # Assert
        alive = [i for i, m in enumerate(env.wargame_models) if m.is_alive]
        assert alive
        for index in alive:
            assert mask[index].any(), "the mask must never empty a row"
    finally:
        env.close()


def test_both_seats_are_masked() -> None:
    """A rules difference between the seats is worth 24.6 vp on shooting alone."""
    # Arrange
    env = _env()
    try:
        _to_command_phase(env)
        _place(env.wargame_models, 5.0, 5.0)
        _place(env.opponent_models, 5.0 + CHARGE_RANGE + 5.0, 5.0)
        handler = env.opponent_action_handler

        # Act
        legality = handler.declaration_legality(env.opponent_models, env.wargame_models)

        # Assert
        slice_ = handler.move_type_slice
        action = handler.move_type_action(MOVE_TYPE_CHARGE)
        assert slice_ is not None and action is not None
        column = action - slice_.start
        assert not legality[:, column].any(), "the opponent seat was left unmasked"
    finally:
        env.close()


def test_a_unit_whose_roll_cannot_cover_the_gap_may_not_declare() -> None:
    """A doomed declaration is masked, exactly as a doomed advance is.

    Regression, 2026-08-26. Declaration eligibility checked the 12" range and
    the blocked flags but never the ROLL -- so a unit at gap 10" that rolled 3
    could declare a charge no rung could land, and the referee reverted the
    move to a bit-exact no-op. Measured on the arm of record: 13.8% of the
    trained agent's charge attempts were doomed at declaration, while the
    scripted comparator has always declined them (`_reachable_charge_units`).
    """
    # Arrange -- one unit inside the 12" declaration range whose roll is short.
    env = _env()
    try:
        handler = env.player_action_handler
        models = env.wargame_models
        enemies = env.opponent_models
        for index, model in enumerate(models):
            model.group_id = 0
            model.location = np.array([10.0, 10.0 + index * 0.2], dtype=float)
            model.charge_roll = 3.0
        gap_units = env.rules_quantities.scale.to_units(8.0)
        for index, enemy in enumerate(enemies):
            enemy.location = np.array(
                [10.0 + gap_units, 10.0 + index * 0.2], dtype=float
            )
        column = handler._move_types.index(MOVE_TYPE_CHARGE)

        # Act
        doomed = handler.declaration_legality(models, enemies)

        # A roll that covers the gap flips it back on -- same geometry, bigger
        # dice, so the test cannot pass by the unit being ineligible outright.
        for model in models:
            model.charge_roll = 12.0
        winnable = handler.declaration_legality(models, enemies)

        # Assert
        assert not doomed[:, column].any(), (
            "a unit whose roll cannot reach any enemy was allowed to declare"
        )
        assert winnable[:, column].any(), (
            "the same unit with a covering roll could not declare -- the "
            "fixture is testing ineligibility, not the roll gate"
        )
    finally:
        env.close()
