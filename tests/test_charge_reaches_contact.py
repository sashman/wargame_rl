"""A charge is the one move allowed to end inside an enemy's engagement range.

⚠ This is the mechanic. Everything else in melee is bookkeeping around it.

`back_off_to_unengaged` runs on every mover on both seats, so engagement is
0.0000% of model-pairs across 60,520 observations — and the minimum edge-to-edge
gap is **1.000008740"** against an engagement range of 1.0. Contact is not
unreachable; it is fenced off by one branch, and the charge is the exemption.

⚠ **RETRACTED: "so the charge does not need to cross a gap".** That 8.7
micro-inch figure is the MINIMUM pair on the board and was read as a typical
one. The median charge-eligible unit is **5.99"** from its nearest enemy and
**0.0%** of declarations are within one speed bin, so a charge needs the
exemption *and* the distance. The no-new-actions design survives on the pairing
argument alone — see `docs/melee.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.domain.engagement import engaged_with_any
from wargame_rl.wargame.envs.env_components.actions import (
    MOVE_TYPE_CHARGE,
    STAY_ACTION,
    ActionHandler,
)
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.envs.wargame_model import WargameModel
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.observation import observation_to_tensor

ENGAGEMENT = 1.0
# The overlap test needs models with real extent; the rest of this file uses 0.
BASE_RADIUS = 0.63


def _handler(melee: bool, n_models: int = 1, base_radius: float = 0.0) -> ActionHandler:
    skip = [BattlePhase.shooting, BattlePhase.fight]
    if not melee:
        skip.extend([BattlePhase.command, BattlePhase.charge])
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
    # ⚠ The declaration and the roll are BOTH preconditions the referee
    # re-checks at resolution since 2026-08-25 (`apply` takes no mask, so the
    # mask alone left a hole). A handler-level test has to supply what the
    # command phase would.
    mover.declared_charge = True
    mover.charge_roll = 12.0
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
    """One model, one enemy `gap` away; return its charge-move legality row.

    ⚠ The mover is marked as having DECLARED. A rung is legal only for a unit
    whose leader declared a charge in the command phase, so without this every
    row is empty and these tests would all pass for the wrong reason.
    """
    handler = _handler(melee=True)
    mover = _model(5.0)
    mover.declared_charge = True
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
        # ⚠ The declaration is a real precondition, re-checked by the referee
        # since 2026-08-25 -- `apply` takes no mask, so a policy bypassing one
        # could otherwise charge without ever declaring. The env sets this in
        # the command phase; a handler-level test has to set it itself.
        m.declared_charge = True
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
        # ⚠ command is STEPPED: the charge is declared there by the unit's
        # leader, so skipping it would leave no legal declaration and every
        # charge would revert.
        skip_phases=[BattlePhase.shooting],
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
    # A long charge from 1.5" away lands clean past the enemy and out the far
    # side, which ends unengaged and is reverted whole. That is the rule
    # working, and it would read here as the exemption failing.
    #
    # ⚠ Rung 0, not rung 1, since 2026-08-26: in the charge phase the movement
    # slice decodes to the CHARGE ladder, whose rungs are `(s + 1) x 2"` because
    # the rules cap a charge at the 2D6 and not at Move. Rung 1 used to be 2"
    # and is now 4".
    east = env.player_action_handler.encode_action(0, 0)
    # ⚠ The unit must DECLARE in the command phase or the referee reverts the
    # charge -- the declaration is a precondition re-checked at resolution since
    # 2026-08-25, not just a mask. Driving this end to end is the point of the
    # test, and the declaration is part of "end to end".
    declare = env.player_action_handler.move_type_action(MOVE_TYPE_CHARGE)
    assert declare is not None, "the melee config must carry a charge declaration"

    # Act: declare in command, STAY through movement, then charge east.
    engaged = False
    for _ in range(4):
        phase = env.game_clock_state.phase
        charging = phase is BattlePhase.charge
        if phase is BattlePhase.command:
            action = declare
        elif charging:
            action = east
        else:
            action = STAY_ACTION
        env.step(WargameEnvAction(actions=[action]))
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


class TestChargeObservation:
    """The 2D6 gates which rungs are legal, so the policy has to be able to see it.

    ⚠ Before this, `charge_roll` reached the network **only** as a `masked_fill`
    on the final logits (`net.py`). A mask is not an observation: it removes
    illegal actions from the head, but it never enters the trunk, so no layer
    could condition on the roll and the CRITIC could not see it at all. The value
    of a state where the unit rolled 11 differs from one where it rolled 3, and
    the critic was being asked to price both the same. `advance_roll` had this
    right; the charge copied the mask and not the column.
    """

    @staticmethod
    def _obs(melee: bool, charge_phase: bool = True):  # type: ignore[no-untyped-def]
        config = WargameEnvConfig(
            number_of_wargame_models=1,
            number_of_opponent_models=1,
            number_of_objectives=1,
            opponent_policy=OpponentPolicyConfig(
                type="scripted_baseline", params={"baseline": "hold_deployment"}
            ),
            models=[ModelConfig(melee_weapons=[MeleeWeaponProfile()])],
            opponent_models=[ModelConfig(melee_weapons=[MeleeWeaponProfile()])],
            melee=MeleeConfig(enabled=melee),
            engagement_range=ENGAGEMENT,
            base_radius=0.0,
            skip_phases=(
                [BattlePhase.shooting]
                if charge_phase
                else [BattlePhase.command, BattlePhase.shooting, BattlePhase.charge]
            ),
        )
        env = create_environment(config)
        try:
            observation, _info = env.reset(seed=11)
            return observation, env.wargame_models[0].charge_roll
        finally:
            env.close()

    def test_a_config_that_skips_the_charge_phase_adds_no_columns(self) -> None:
        """The no-op guarantee: every golden config skips `charge`."""
        # Arrange / Act
        observation, _roll = self._obs(melee=False, charge_phase=False)

        # Assert
        model = observation.wargame_models[0]
        assert model.charge_roll is None
        assert model.fell_back_this_turn is None

    def test_the_DARK_CONTROL_keeps_the_columns_so_the_pair_stays_paired(self) -> None:
        """⚠ The gate is the PHASE, not `melee.enabled`, and this is why.

        `25v25_maps_melee.yaml` and `..._melee_dark.yaml` differ in exactly one
        scalar so that the arm and its control share an init and the per-seed
        difference is a paired estimator. Gating these columns on that same
        scalar would give them different tensor widths — different weights at
        step 0 — and destroy the only thing the pair exists to provide.
        """
        # Arrange
        arm, _ = self._obs(melee=True, charge_phase=True)
        dark, _ = self._obs(melee=False, charge_phase=True)

        # Act
        arm_width = observation_to_tensor(arm)[2].shape[1]
        dark_width = observation_to_tensor(dark)[2].shape[1]

        # Assert
        assert arm_width == dark_width
        assert dark.wargame_models[0].charge_roll == 0.0, "no roll is ever taken"
        assert dark.wargame_models[0].fell_back_this_turn == 0.0

    def test_the_roll_is_observable_and_normalised_by_the_two_dice(self) -> None:
        """2D6 raw would be 12x the scale of every neighbouring feature."""
        # Arrange / Act
        observation, raw_roll = self._obs(melee=True)

        # Assert
        model = observation.wargame_models[0]
        assert raw_roll > 0.0, "the roll must have happened before the observation"
        assert model.charge_roll is not None
        assert model.charge_roll == pytest.approx(raw_roll / 12.0)
        assert 0.0 < model.charge_roll <= 1.0

    def test_the_fell_back_flag_is_observable(self) -> None:
        """For the VALUE head: falling back spends the shooting AND the charge."""
        # Arrange / Act
        observation, _roll = self._obs(melee=True)

        # Assert
        assert observation.wargame_models[0].fell_back_this_turn == 0.0

    def test_the_opponents_columns_are_zeroed_because_they_are_stale(self) -> None:
        """Each side rolls at the start of its OWN turn, so theirs says nothing."""
        # Arrange / Act
        observation, _roll = self._obs(melee=True)

        # Assert
        assert observation.opponent_models[0].charge_roll == 0.0
        assert observation.opponent_models[0].fell_back_this_turn == 0.0

    def test_the_melee_columns_widen_the_per_model_tensor_by_exactly_three(
        self,
    ) -> None:
        """⚠ A tensor-shape change: it orphans a melee checkpoint deliberately.

        ⚠ **Was two, and is three since 2026-08-25.** `declared_charge` joined
        `charge_roll` and `fell_back_this_turn` because a charge-gated reward
        term has to key on the declaration, and CLAUDE.md's cheapest standing
        check — *"check the agent can OBSERVE what the lever keys on"* — has
        burned ~10 GPU-hours twice on terms that keyed on invisible state. It
        also fixes a real perceptual gap: the declaration binds a unit a whole
        phase before the charge is aimed, so a model aiming one had no input
        saying it was under one.
        """
        # Arrange
        plain, _ = self._obs(melee=False, charge_phase=False)
        fighting, _ = self._obs(melee=True)

        # Act
        narrow = observation_to_tensor(plain)[2].shape[1]
        wide = observation_to_tensor(fighting)[2].shape[1]

        # Assert
        assert wide - narrow == 3


def test_a_61_wide_CHECKPOINT_can_still_be_scored_on_a_melee_config() -> None:
    """⚠ The columns forked the melee family off the whole checkpoint corpus.

    They widen the per-model tensor 61 -> 64, so after they landed no existing
    agent could be scored on a melee config at all — "what does melee do to our
    agent" became answerable only by training a new one, and a melee arm's
    numbers would have been an island with no allocation statistic, no
    offence/defence split and no five-opponent table beside them. Found by an
    audit panel.

    Making the columns unconditional would have made every config 64 wide and
    orphaned every 61-wide checkpoint instead, which is worse. `observe_charge`
    is the escape hatch: the melee configs keep their columns and stay paired
    with each other, and a scoring run can turn them off to match the corpus.
    """
    # Arrange
    config = load_env_config("configs/experiments/25v25_maps_melee.yaml")
    golden = load_env_config("configs/golden/25v25_maps_two_mode.yaml")

    # Act
    widths = {}
    for label, cfg in (("golden", golden), ("melee", config)):
        env = create_environment(cfg)
        try:
            observation, _ = env.reset(seed=3)
            widths[label] = observation_to_tensor(observation)[2].shape[1]
        finally:
            env.close()
    config.melee.observe_charge = False
    env = create_environment(config)
    try:
        observation, _ = env.reset(seed=3)
        widths["melee_narrowed"] = observation_to_tensor(observation)[2].shape[1]
    finally:
        env.close()

    # Assert
    assert widths["melee"] == widths["golden"] + 3
    assert widths["melee_narrowed"] == widths["golden"]


def test_a_model_that_moves_AWAY_reverts_its_units_charge() -> None:
    """`11-charge-phase.md` § Charge move, *While moving*.

    *"Each model must end its move CLOSER to one or more charge targets."*

    ⚠ **This condition did not exist until 2026-08-25 and no gap-map row
    recorded its absence.** Without it a charge was satisfied by ONE model
    reaching contact while its squadmates moved anywhere coherency allowed —
    a materially easier charge than the rules', and the half of the mechanic a
    learned policy is most likely to exploit. A rules-lawyer audit measured
    2.4% of a rigid script's charging models (5.3% of its standing charges)
    already violating it by accident.
    """
    # Arrange: m0 closes on the enemy, m1 walks the opposite way.
    handler = _handler(melee=True, n_models=2)
    movers = [_model(5.0), _model(5.0)]
    movers[1].location = np.array([5.0, 10.5])
    for model in movers:
        model.charge_roll = 12.0
        model.declared_charge = True
    enemies = [_model(6.5, group=1)]
    start = [np.array(m.location, copy=True) for m in movers]
    east = handler.best_action_toward(1.0, 0.0, max_step_length=1.0)
    west = handler.best_action_toward(-1.0, 0.0, max_step_length=1.0)

    # Act
    handler.apply(
        WargameEnvAction(actions=[east, west]),
        movers,
        60,
        44,
        handler.action_space,
        phase=BattlePhase.charge,
        enemy_models=enemies,
    )

    # Assert — all or nothing, so BOTH models go back.
    for model, origin in zip(movers, start, strict=True):
        assert np.array_equal(model.location, origin), (
            "a charge stood while one of its models moved away from the target"
        )


def test_an_UNDECLARED_unit_cannot_charge_even_if_the_mask_is_bypassed() -> None:
    """The referee, not the mask, is the authority.

    ⚠ `ActionHandler.apply` takes NO mask, so until 2026-08-25 the declaration
    and the 2D6 cap lived only in `charge_legality` — a policy that bypassed the
    mask could charge without declaring. That is the defect class this project
    paid +11.4 vp for on the joint decoder and paid again on
    `back_off_to_unengaged`: the constraint lived in one layer and the layer
    that actually moved models did not know about it.
    """
    # Arrange
    movers, start = _unit_charge([(6.5, 1)])
    assert not np.array_equal(movers[0].location, start[0]), "the control failed"

    # Act: the same charge from a unit that never declared.
    handler = _handler(melee=True, n_models=2)
    undeclared = [_model(5.0), _model(5.0)]
    undeclared[1].location = np.array([5.0, 10.5])
    for model in undeclared:
        model.charge_roll = 12.0
        model.declared_charge = False
    origins = [np.array(m.location, copy=True) for m in undeclared]
    handler.apply(
        WargameEnvAction(
            actions=[handler.best_action_toward(1.0, 0.0, max_step_length=1.0)] * 2
        ),
        undeclared,
        60,
        44,
        handler.action_space,
        phase=BattlePhase.charge,
        enemy_models=[_model(6.5, group=1)],
    )

    # Assert
    for model, origin in zip(undeclared, origins, strict=True):
        assert np.array_equal(model.location, origin), "an undeclared unit charged"


def test_a_charge_may_not_travel_further_than_its_2D6() -> None:
    """`11-charge-phase.md`: *"Maximum distance | The charge roll."*

    ⚠ Masked on both seats and, until 2026-08-25, enforced NOWHERE else — a
    declared model with a roll of 2 handed the 6" rung travelled the full 6.0"
    and was granted `charged_this_turn`. The advance has had belt-and-braces
    since it shipped (`_advance_displacement` clamps at resolution as well as
    masking); this gives the charge the same.
    """
    # Arrange: contact is reachable, but only by out-running the dice.
    handler = _handler(melee=True, n_models=1)
    mover = _model(5.0)
    mover.declared_charge = True
    mover.charge_roll = 2.0
    origin = np.array(mover.location, copy=True)

    # Act: a 6" rung, which the mask would have forbidden.
    handler.apply(
        WargameEnvAction(actions=[handler.best_action_toward(1.0, 0.0)]),
        [mover],
        60,
        44,
        handler.action_space,
        phase=BattlePhase.charge,
        enemy_models=[_model(10.5, group=1)],
    )

    # Assert
    assert np.array_equal(mover.location, origin), (
        "a charge outran its 2D6 because only the mask was stopping it"
    )
    assert not mover.charged_this_turn


def test_the_comparator_DECLARES_on_the_full_roll_not_on_min_move_roll() -> None:
    """The declaration and the aim must ask the same question.

    Regression, 2026-08-26. `ScriptedSquadMarch._reachable_charge_units` gated
    the DECLARATION on `min(Move, charge_roll)` -- the cap retired at `5da54ed`,
    when `DEFERRED: charge.beyond_move_ladder` closed and the charge ladder began
    reaching `CHARGE_DICE_MAX` = 12" -- while `select_charge` aimed with the full
    roll, through the scale. It also compared a roll in INCHES against speeds in
    board units. The unit declined charges it could actually make: the
    comparator under-declared by ~22%, and every target in
    `docs/melee-teaching-goal.md`, including `stood/ep > 3.0`, descends from it.

    ⚠ Fixing this MOVES THE SCRIPTED BAR on every melee config. That is the
    point -- a bar that declines legal charges is not the bar.
    """
    # Arrange -- the enemy sits beyond Move but inside a maximum roll, which is
    # exactly the band the retired cap discarded.
    from wargame_rl.wargame.envs.baseline.scripted_squad_march import (
        ScriptedSquadMarchPolicy,
    )

    env = _melee_env()
    try:
        env.reset(seed=3)
        scripted = ScriptedSquadMarchPolicy()
        models = env.wargame_models
        move = float(env.player_action_handler.move_speeds.max())
        scale = env.rules_quantities.scale
        contact = float(env.rules_quantities.engagement_range) + 2.0 * float(
            env.rules_quantities.base_radius
        )
        gap = move + 0.5 * move

        for index, model in enumerate(models):
            model.group_id = 0
            model.location = np.array([10.0, 10.0 + index * 0.2], dtype=float)
        for index, enemy in enumerate(env.opponent_models):
            enemy.group_id = 0
            enemy.location = np.array(
                [10.0 + gap + contact, 10.0 + index * 0.2], dtype=float
            )

        # A roll that covers the gap, expressed in inches as the env stores it.
        for model in models:
            model.charge_roll = float(scale.to_inches(gap + contact)) + 1.0

        # Act
        reachable = scripted._reachable_charge_units(models, env)

        # Assert -- the gap is beyond Move, so the retired cap declared nothing.
        assert gap > move, "the fixture does not exercise the band the cap cut"
        assert reachable, (
            "a unit whose ROLL covers the gap was not declared reachable -- the "
            "declaration is still capped at Move"
        )
    finally:
        env.close()


def test_the_approach_mask_keeps_only_target_closing_moves() -> None:
    """`charge_approach_mask`: every legal charge move ends closer to the target.

    The referee's *while moving* clause reverts a whole charge when any mover
    ends not-closer; the mask applies the same test at action time so the
    policy cannot pick a self-voiding move. Measured motivation
    (`docs/melee-teaching-goal.md` §16-18): 76.7% of a trained arm's failed
    charges reached NOBODY, median heading error 92°, and both shaping forms
    failed their pre-registrations.
    """
    # Arrange -- a declared unit due east of its target, roll covering the gap.
    env = _melee_env()
    try:
        handler = env.player_action_handler
        models = env.wargame_models
        for index, model in enumerate(models):
            model.group_id = 0
            model.location = np.array([10.0, 10.0 + index * 0.2], dtype=float)
            model.charge_roll = 12.0
            model.declared_charge = True
        for index, enemy in enumerate(env.opponent_models):
            enemy.location = np.array([16.0, 10.0 + index * 0.2], dtype=float)

        handler._charge_approach_mask = True
        masked = handler.charge_legality(models, env.opponent_models)
        handler._charge_approach_mask = False
        unmasked = handler.charge_legality(models, env.opponent_models)

        # Assert -- strictly fewer actions survive, and every survivor closes.
        assert masked.sum() < unmasked.sum(), "the mask removed nothing"
        assert masked.any(), "the mask emptied the action set entirely"
        flat = handler._charge_displacements.reshape(-1, 2)
        target = np.array([16.0, 10.0], dtype=float)
        start = np.array(models[0].location, dtype=float)
        before = float(np.linalg.norm(target - start))
        for action_index in np.nonzero(masked[0])[0]:
            endpoint = start + flat[action_index]
            assert float(np.linalg.norm(target - endpoint)) < before + 1e-9, (
                f"masked-legal action {action_index} does not approach the target"
            )
    finally:
        env.close()


def test_the_approach_mask_never_empties_a_declared_units_set() -> None:
    """A unit with no approaching rung keeps the distance gate's actions.

    ⚠ Driven at the guard's own contract, not through `charge_legality`: on
    this ladder the case is geometrically unreachable from a legal position —
    no rung approaches only when the gap is at most half the smallest rung
    (r < 2·gap is the approach condition for the direct angle), and a unit
    that close is ENGAGED and therefore ineligible to declare at all. The
    guard is defensive, for geometries bases and misalignment could open.
    """
    # Arrange — a legality row of all-True and a target INSIDE the smallest
    # rung's overshoot band, so nothing ends closer.
    env = _melee_env()
    try:
        handler = env.player_action_handler
        models = env.wargame_models[:1]
        model = models[0]
        model.group_id = 0
        model.location = np.array([10.0, 10.0], dtype=model.location.dtype)
        enemy = env.opponent_models[0]
        enemy.group_id = 0
        enemy.location = np.array([10.4, 10.0], dtype=enemy.location.dtype)
        legality = np.ones((1, handler.movement_slice.size), dtype=bool)
        before = legality.copy()

        # Act
        handler._apply_charge_approach_mask(legality, models, [enemy])

        # Assert — no action approaches (gap 0.4 < rung 2 / 2), so the guard
        # must leave the row exactly as the distance gate had it.
        assert (legality == before).all(), (
            "the guard emptied or altered a unit with no approaching rung"
        )
    finally:
        env.close()
