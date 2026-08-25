"""A charge is the one move allowed to end inside an enemy's engagement range.

⚠ This is the mechanic. Everything else in melee is bookkeeping around it.

`back_off_to_unengaged` runs on every mover on both seats, so engagement is
0.0000% of model-pairs across 60,520 observations — and the minimum edge-to-edge
gap is **1.000008740"** against an engagement range of 1.0. The army is not far
from contact; it is parked 8.7 micro-inches outside it and always has been. So
the charge does not need to cross a gap. It needs the exemption.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.engagement import engaged_with_any
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION, ActionHandler
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.envs.wargame_model import WargameModel
from wargame_rl.wargame.model.common.factory import create_environment

ENGAGEMENT = 1.0
# The overlap test needs models with real extent; the rest of this file uses 0.
BASE_RADIUS = 0.63


def _handler(melee: bool, n_models: int = 1, base_radius: float = 0.0) -> ActionHandler:
    skip = [BattlePhase.command, BattlePhase.shooting, BattlePhase.fight]
    if not melee:
        skip.append(BattlePhase.charge)
    return ActionHandler(
        WargameEnvConfig(
            number_of_wargame_models=n_models,
            models=[ModelConfig() for _ in range(n_models)],
            melee=MeleeConfig(enabled=melee),
            engagement_range=ENGAGEMENT,
            base_radius=base_radius,
            skip_phases=skip,
        )
    )


def _model(x: float, group: int = 0) -> WargameModel:
    return WargameModel(
        location=np.array([x, 10.0]),
        stats={"toughness": 3, "save": 4, "max_wounds": 1, "current_wounds": 1},
        distances_to_objectives=np.zeros(1),
        group_id=group,
        base_radius=0.0,
    )


def _walk_at(handler: ActionHandler, phase: BattlePhase) -> tuple[float, bool]:
    """Step one model 1" at an enemy 1.5" away; return its gap and whether engaged.

    Short deliberately: a full 6" move would carry it straight past the enemy
    and out the far side, which ends unengaged for a reason that has nothing to
    do with the rule under test.
    """
    mover = _model(5.0)
    enemy = _model(6.5, group=1)
    handler.apply(
        WargameEnvAction(
            actions=[handler.best_action_toward(1.0, 0.0, max_step_length=1.0)]
        ),
        [mover],
        60,
        44,
        handler.action_space,
        phase=phase,
        enemy_models=[enemy],
    )
    gap = float(np.linalg.norm(mover.location - enemy.location))
    engaged = bool(
        engaged_with_any(
            np.array([mover.location]),
            np.array([enemy.location]),
            np.array([True]),
            np.array([True]),
            engagement_range=ENGAGEMENT,
        )[0]
    )
    return gap, engaged


def test_a_normal_move_parks_just_outside_contact() -> None:
    """The behaviour that made engagement unreachable — and it is microscopic."""
    gap, engaged = _walk_at(_handler(melee=True), BattlePhase.movement)
    assert not engaged
    assert gap > ENGAGEMENT
    assert gap - ENGAGEMENT < 1e-4, (
        f"parked {gap - ENGAGEMENT:.3e} outside contact — the gap a charge must cross"
    )


def test_a_charge_move_may_end_engaged() -> None:
    """The exemption: the same walk, in the charge phase, reaches contact."""
    _, engaged = _walk_at(_handler(melee=True), BattlePhase.charge)
    assert engaged, "a charge could not reach contact"


def test_with_melee_off_the_charge_phase_displaces_nobody() -> None:
    """The switch. Off, the movement slice is not legal in the charge phase."""
    handler = _handler(melee=False)
    mover = _model(5.0)
    start = mover.location.copy()
    handler.apply(
        WargameEnvAction(
            actions=[handler.best_action_toward(1.0, 0.0, max_step_length=1.0)]
        ),
        [mover],
        60,
        44,
        handler.action_space,
        phase=BattlePhase.charge,
        enemy_models=[_model(6.5, group=1)],
    )
    assert np.array_equal(mover.location, start)


def test_a_charge_still_may_not_end_inside_another_model() -> None:
    """The exemption drops the engagement rings, never the occupied bases."""
    handler = _handler(melee=True)
    mover = WargameModel(
        location=np.array([5.0, 10.0]),
        stats={"toughness": 3, "save": 4, "max_wounds": 1, "current_wounds": 1},
        distances_to_objectives=np.zeros(1),
        group_id=0,
        base_radius=0.5,
    )
    enemy = WargameModel(
        location=np.array([6.0, 10.0]),
        stats={"toughness": 3, "save": 4, "max_wounds": 1, "current_wounds": 1},
        distances_to_objectives=np.zeros(1),
        group_id=1,
        base_radius=0.5,
    )
    handler.apply(
        WargameEnvAction(actions=[handler.best_action_toward(1.0, 0.0)]),
        [mover],
        60,
        44,
        handler.action_space,
        phase=BattlePhase.charge,
        enemy_models=[enemy],
    )
    separation = float(np.linalg.norm(mover.location - enemy.location))
    assert separation >= 1.0 - 1e-6, f"models overlap: centres {separation:.4f} apart"


def _legality(
    *,
    gap: float,
    roll: float,
    engaged: bool = False,
    advanced: bool = False,
    fell_back: bool = False,
) -> np.ndarray:
    """One model, one enemy `gap` away; return its charge-move legality row."""
    handler = _handler(melee=True)
    mover = _model(5.0)
    mover.charge_roll = roll
    mover.advanced_this_turn = advanced
    mover.fell_back_this_turn = fell_back
    enemy = _model(5.0 + (ENGAGEMENT / 2 if engaged else gap), group=1)
    row: np.ndarray = handler.charge_legality([mover], [enemy])[0]
    return row


def test_a_unit_out_of_the_declaration_range_may_not_charge() -> None:
    """12" is the band a charge may be declared within."""
    assert (
        _legality(gap=30.0, roll=12.0).any() is np.False_
        or not _legality(gap=30.0, roll=12.0).any()
    )
    assert _legality(gap=8.0, roll=12.0).any()


def test_an_already_engaged_unit_may_not_charge() -> None:
    """You cannot charge out of a fight."""
    assert not _legality(gap=8.0, roll=12.0, engaged=True).any()


def test_a_unit_that_advanced_or_fell_back_may_not_charge() -> None:
    """Both flags block the declaration, per 11-charge-phase.md § Eligibility."""
    assert not _legality(gap=8.0, roll=12.0, advanced=True).any()
    assert not _legality(gap=8.0, roll=12.0, fell_back=True).any()


def test_the_roll_caps_the_charge_distance() -> None:
    """The 2D6 is the move's maximum, so longer speed bins are masked out."""
    generous = _legality(gap=8.0, roll=12.0)
    stingy = _legality(gap=8.0, roll=1.0)
    assert generous.sum() > stingy.sum()


def test_a_roll_of_zero_leaves_no_legal_charge() -> None:
    """Nothing is rolled outside a charge phase, and nothing is then legal."""
    assert not _legality(gap=8.0, roll=0.0).any()


def _unit_charge(
    enemy_positions: list[tuple[float, int]], *, spread: float = 0.5
) -> tuple[list[WargameModel], list[np.ndarray]]:
    """Charge a two-model unit east; return the models and where they started."""
    handler = _handler(melee=True, n_models=2)
    movers = [_model(5.0), _model(5.0)]
    movers[1].location = np.array([5.0, 10.0 + spread])
    for m in movers:
        m.charge_roll = 12.0
    enemies = [_model(x, group=g) for x, g in enemy_positions]
    start = [np.array(m.location, copy=True) for m in movers]
    handler.apply(
        WargameEnvAction(
            actions=[handler.best_action_toward(1.0, 0.0, max_step_length=1.0)] * 2
        ),
        movers,
        60,
        44,
        handler.action_space,
        phase=BattlePhase.charge,
        enemy_models=enemies,
    )
    return movers, start


def test_a_charge_that_reaches_one_enemy_unit_stands() -> None:
    movers, start = _unit_charge([(6.5, 1)])
    assert not np.array_equal(movers[0].location, start[0]), "the charge was reverted"


def test_a_charge_that_clips_a_SECOND_enemy_unit_is_reverted_entirely() -> None:
    """`11-charge-phase.md`: engaged with no unit that was not a target.

    This is what makes a charge fail even on a long roll, and the revert is
    all-or-nothing — every model returns to where it started.
    """
    movers, start = _unit_charge([(6.5, 1), (6.5, 2)], spread=0.5)
    for model, origin in zip(movers, start, strict=True):
        assert np.array_equal(model.location, origin), "an illegal charge stood"


def test_a_charge_that_reaches_nobody_is_reverted() -> None:
    """A charge that ends unengaged did not happen."""
    movers, start = _unit_charge([(40.0, 1)])
    for model, origin in zip(movers, start, strict=True):
        assert np.array_equal(model.location, origin)


def test_the_revert_is_unconditional_and_not_the_coherency_referee() -> None:
    """⚠ `coherency.enforce_move` defaults to `off` on every shipped config.

    Routing the charge's own conditions through it would let an illegal charge
    simply stand wherever enforcement is off — which is everywhere.
    """
    handler = _handler(melee=True, n_models=2)
    assert handler._coherency_mode.value == "off"
    movers, start = _unit_charge([(6.5, 1), (6.5, 2)])
    assert np.array_equal(movers[0].location, start[0])


def _melee_env() -> WargameEnv:
    """One model a side, melee on, an opponent that holds still."""
    config = WargameEnvConfig(
        number_of_wargame_models=1,
        number_of_opponent_models=1,
        number_of_objectives=1,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[ModelConfig(melee_weapons=[MeleeWeaponProfile()])],
        opponent_models=[ModelConfig(melee_weapons=[MeleeWeaponProfile()])],
        melee=MeleeConfig(enabled=True),
        engagement_range=ENGAGEMENT,
        base_radius=0.0,
        skip_phases=[BattlePhase.command, BattlePhase.shooting],
    )
    env = create_environment(config)
    env.reset(seed=5)
    return env


def test_a_charge_reaches_contact_through_env_step() -> None:
    """The mechanic end to end, driven by `env.step` rather than by `apply`.

    ⚠ **Every other test in this file calls `ActionHandler.apply` directly.**
    This project has twice shipped a defect that a full suite of unit tests could
    not see because none of them called `env.step` — the joint decoder judged
    candidates against its own relaxation (worth +11.4 vp), and the endpoint
    back-off walked models into friendly bases. A charge crosses the action
    mask, the phase clock, the opponent seat and the referee, and only `step`
    exercises all four.
    """
    # Arrange: 1.5" apart, so any legal charge rung reaches contact and no rung
    # carries the model clean past the enemy and out the far side.
    env = _melee_env()
    player = env.wargame_models[0]
    opponent = env.opponent_models[0]
    player.location = np.array([10.0, 10.0], dtype=player.location.dtype)
    opponent.location = np.array([11.5, 10.0], dtype=opponent.location.dtype)
    # ⚠ Two inches, not `best_action_toward`, which returns the FASTEST rung.
    # A 6" charge from 1.5" away lands at 16.0 — clean past the enemy and out
    # the far side, which ends unengaged and is reverted whole. That is the rule
    # working, and it would read here as the exemption failing.
    east = env.player_action_handler.encode_action(0, 1)

    # Act: STAY through movement, then charge east.
    engaged = False
    for _ in range(4):
        phase = env.game_clock_state.phase
        charging = phase is BattlePhase.charge
        env.step(WargameEnvAction(actions=[east if charging else STAY_ACTION]))
        if charging:
            engaged = bool(
                engaged_with_any(
                    np.array([player.location], dtype=float),
                    np.array([opponent.location], dtype=float),
                    np.ones(1, dtype=bool),
                    np.ones(1, dtype=bool),
                    engagement_range=ENGAGEMENT,
                )[0]
            )
            break

    # Assert
    assert engaged, "the charge did not reach contact through env.step"
    assert player.charged_this_turn, "a charge that stood did not earn its flag"


def test_the_charge_phase_grants_no_free_MOVE_to_a_unit_that_cannot_charge() -> None:
    """The control that matters most for every existing scenario.

    Making the movement slice valid in the charge phase is what avoids a new
    action slice — and it would be worth nothing if a unit with no charge
    available could simply take a second movement phase every turn.

    ⚠ **This proves the REFEREE, not the mask, and the distinction was published
    the wrong way round.** `ActionHandler.apply` never consults the action mask —
    it is documented as deliberately not trusting it — so the move below IS
    applied and then undone by `_enforce_charge`, which reverts any unit that
    does not end engaged with exactly one enemy unit. Net displacement of zero
    therefore cannot distinguish "the mask forbade it" from "the referee undid
    it". The protection is real and is arguably the stronger of the two, since
    the revert is unconditional while a mask only binds an actor that reads it.

    ⚠ **A related claim is RETRACTED.** A smoke run reporting zero inches moved
    in the charge phase by `squad_march_take` was offered as evidence the mask
    works. It is vacuous: `BaselinePolicy.select_action` returns STAY for every
    phase that is not command, movement or shooting, so that policy could not
    have moved under no masking whatsoever.
    """
    # Arrange: the only enemy is 30" away, far outside the 12" declaration range.
    env = _melee_env()
    player = env.wargame_models[0]
    opponent = env.opponent_models[0]
    player.location = np.array([10.0, 10.0], dtype=player.location.dtype)
    opponent.location = np.array([40.0, 10.0], dtype=opponent.location.dtype)
    east = env.player_action_handler.best_action_toward(1.0, 0.0)

    # Act
    moved = 0.0
    for _ in range(4):
        phase = env.game_clock_state.phase
        if phase is not BattlePhase.charge:
            env.step(WargameEnvAction(actions=[STAY_ACTION]))
            continue
        before = player.location.copy()
        env.step(WargameEnvAction(actions=[east]))
        moved = float(np.linalg.norm(player.location - before))
        break

    # Assert
    assert moved == 0.0, "an ineligible unit took a free move in the charge phase"


def _placed(spots: list[tuple[float, float, int]]) -> list[WargameModel]:
    return [
        WargameModel(
            location=np.array([x, y]),
            stats={"toughness": 3, "save": 4, "max_wounds": 1, "current_wounds": 1},
            distances_to_objectives=np.zeros(1),
            group_id=group,
            base_radius=BASE_RADIUS,
        )
        for x, y, group in spots
    ]


def test_a_reverted_charge_does_not_land_on_top_of_another_unit() -> None:
    """All-or-nothing must not put a unit back onto ground somebody else took.

    ⚠ **Found in a demo recording, measured at 0.868in of base overlap** — 69% of
    a 32mm base, player models only, in the charge phase only. Every model moves
    before any unit is judged, so a unit whose charge fails is restored to a
    start position that a LATER unit may have advanced into. The board then holds
    two models inside each other, which no rule permits.

    The geometry here is the one from that recording, reduced: unit 1 is behind
    unit 0, unit 0 charges and cannot reach, unit 1 charges into the space unit 0
    vacated and reaches an enemy. Unit 0 is judged first, fails, and is put back
    on top of unit 1.
    """
    # Arrange
    handler = _handler(melee=True, n_models=4, base_radius=BASE_RADIUS)
    movers = _placed(
        [(10.0, 10.0, 0), (10.0, 11.5, 0), (8.0, 10.4, 1), (8.0, 11.9, 1)],
    )
    # One enemy unit far away, and one placed to engage unit 1 where it lands
    # WITHOUT engaging unit 0 where it starts — that is what lets unit 1 stand
    # while unit 0 fails.
    enemy = _placed([(15.0, 10.0, 0), (10.0, 14.0, 1), (10.0, 15.5, 1)])
    east = handler.encode_action(0, 1)

    # Act
    handler.apply(
        WargameEnvAction(actions=[east] * 4),
        movers,
        60,
        44,
        handler.action_space,
        phase=BattlePhase.charge,
        enemy_models=enemy,
    )

    # Assert
    positions = np.array([m.location for m in movers], dtype=float)
    gaps = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    gaps += np.eye(len(movers)) * 999.0
    worst = float((2 * BASE_RADIUS - gaps).max())
    assert worst <= 1e-9, f"models overlap by {worst:.4f}in after a reverted charge"
