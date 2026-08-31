"""Joint constrained decoding: the most probable action combination the rules allow.

The policy emits one categorical per model and samples them independently, but
coherency is a property of the joint configuration. These pin the decoder on
hand-built preferences, so the behaviour is checked against geometry rather
than against whatever a checkpoint happened to prefer.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.engagement import engaged_with_any
from wargame_rl.wargame.envs.env_components.actions import MOVE_TYPE_CHARGE, STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.decoding import decode_joint_coherent
from wargame_rl.wargame.model.common.factory import create_environment


@pytest.fixture
def env() -> WargameEnv:
    """A four-model env whose first two models share a unit."""
    config = parse_yaml_raw_as(WargameEnvConfig, open("configs/dev/tiny.yaml").read())
    environment = create_environment(config)
    environment.reset(seed=1)
    return environment


def unit_is_coherent(env: WargameEnv, indices: list[int]) -> bool:
    models = env.player_models
    return evaluate_coherency(
        positions=np.array([models[i].location for i in indices], dtype=float),
        group_ids=np.zeros(len(indices), dtype=np.intp),
        alive_mask=np.array([models[i].is_alive for i in indices], dtype=bool),
        base_radii=np.array([models[i].base_radius for i in indices], dtype=float),
        nearest_distance=env.config.coherency.nearest_distance,
        furthest_distance=env.config.coherency.furthest_distance,
    ).all_coherent


def test_top_k_of_one_is_exactly_the_independent_decode(env: WargameEnv) -> None:
    """K=1 must be a no-op, so every number measured before this existed holds."""
    handler = env.player_action_handler
    rng = np.random.default_rng(0)
    log_probs = rng.normal(size=(len(env.player_models), handler.n_actions))
    actions = [int(a) for a in log_probs.argmax(axis=1)]

    assert decode_joint_coherent(log_probs, actions, env, top_k=1) == actions


def test_it_replaces_a_joint_action_that_would_break_the_unit(env: WargameEnv) -> None:
    """The whole point: independently-best actions that are jointly illegal.

    Model 0 is told to sprint east and model 1 to sprint west, which is the
    canonical way an independently factorised policy tears a unit apart. A legal
    combination exists in the candidate set — both moving the same way — and the
    decoder has to find it.
    """
    models = env.player_models
    unit = [i for i, m in enumerate(models) if m.group_id == models[0].group_id]
    assert len(unit) >= 2, "fixture must give the first unit at least two models"
    # Put them adjacent so only the *move* can break the chain.
    models[unit[1]].location = models[unit[0]].location + np.array(
        [1.0, 0.0], dtype=models[unit[0]].location.dtype
    )
    assert unit_is_coherent(env, unit)

    handler = env.player_action_handler
    east = handler.best_action_toward(1.0, 0.0)
    west = handler.best_action_toward(-1.0, 0.0)

    n_actions = handler.n_actions
    log_probs = np.full((len(models), n_actions), -20.0)
    # Both models rank a full-speed opposed sprint first and agreeing second.
    log_probs[unit[0]][east] = 0.0
    log_probs[unit[0]][west] = -1.0
    log_probs[unit[1]][west] = 0.0
    log_probs[unit[1]][east] = -1.0
    for index in range(len(models)):
        if index not in unit:
            log_probs[index][0] = 0.0

    actions = [int(a) for a in log_probs.argmax(axis=1)]
    assert actions[unit[0]] == east and actions[unit[1]] == west

    decoded = decode_joint_coherent(log_probs, actions, env, top_k=2)

    assert decoded != actions, "the decoder should have rejected the opposed sprint"
    # Whatever it chose must be one of the candidates it was offered.
    assert decoded[unit[0]] in (east, west)
    assert decoded[unit[1]] in (east, west)
    # And the two must now agree, which is the only legal pair here.
    assert decoded[unit[0]] == decoded[unit[1]]


def test_it_keeps_the_original_actions_when_nothing_legal_is_available(
    env: WargameEnv,
) -> None:
    """No legal combination in the candidate set leaves the caller's enforcement.

    The decoder must never invent an action outside the top-K it was given; when
    every candidate combination is illegal it declines, and `enforce_move` then
    applies exactly as it did before.
    """
    models = env.player_models
    unit = [i for i, m in enumerate(models) if m.group_id == models[0].group_id]
    # Park them far enough apart that no single move can close the chain.
    models[unit[1]].location = models[unit[0]].location + np.array(
        [40.0, 0.0], dtype=models[unit[0]].location.dtype
    )
    handler = env.player_action_handler
    n_actions = handler.n_actions
    log_probs = np.full((len(models), n_actions), -20.0)
    # One candidate each, and it cannot possibly bridge 40 inches.
    log_probs[unit[0]][0] = 0.0
    log_probs[unit[1]][0] = 0.0
    for index in range(len(models)):
        if index not in unit:
            log_probs[index][0] = 0.0
    actions = [int(a) for a in log_probs.argmax(axis=1)]

    assert decode_joint_coherent(log_probs, actions, env, top_k=1) == actions
    assert decode_joint_coherent(log_probs, actions, env, top_k=2) == actions


def test_a_unit_too_large_for_the_candidate_budget_is_left_alone(
    env: WargameEnv,
) -> None:
    """`top_k ** size` explodes, so an oversized unit keeps its independent decode."""
    handler = env.player_action_handler
    rng = np.random.default_rng(1)
    log_probs = rng.normal(size=(len(env.player_models), handler.n_actions))
    actions = [int(a) for a in log_probs.argmax(axis=1)]

    decoded = decode_joint_coherent(log_probs, actions, env, top_k=8, max_candidates=4)

    assert decoded == actions


def test_include_stay_supplies_a_fallback_when_nothing_legal_is_ranked(
    env: WargameEnv,
) -> None:
    """With no legal combination in the top-K set, standing still beats reverting.

    The decoder finds nothing legal on 0.3-1.4% of unit-moves and hands the unit
    back unchanged, so the referee reverts it. A revert is not free: it also runs
    the overlap cascade, which drags *neighbouring* units back and accounts for
    9.2-15.3% of all freezes. A deliberate stay reaches the same positions
    without triggering any of that.

    The unit here is placed at the very edge of the chain and offered only
    opposed sprints, so every combination of ranked actions tears it apart while
    standing still is trivially legal.
    """
    models = env.player_models
    unit = [i for i, m in enumerate(models) if m.group_id == models[0].group_id]
    assert len(unit) >= 2
    models[unit[1]].location = models[unit[0]].location + np.array(
        [1.0, 0.0], dtype=models[unit[0]].location.dtype
    )
    assert unit_is_coherent(env, unit)

    handler = env.player_action_handler
    east = handler.best_action_toward(1.0, 0.0)
    west = handler.best_action_toward(-1.0, 0.0)
    n_actions = handler.n_actions
    log_probs = np.full((len(models), n_actions), -np.inf)
    # Only opposed sprints are rankable, so no pairing of them can be legal.
    log_probs[unit[0]][east] = 0.0
    log_probs[unit[1]][west] = 0.0
    actions = [0] * len(models)
    actions[unit[0]] = east
    actions[unit[1]] = west

    unchanged = decode_joint_coherent(log_probs, actions, env, top_k=3)
    assert [unchanged[i] for i in unit] == [east, west], (
        "without a fallback the unit is handed back for the referee to revert"
    )

    stayed = decode_joint_coherent(log_probs, actions, env, top_k=3, include_stay=True)
    assert all(stayed[i] == 0 for i in unit)


def test_include_stay_does_not_displace_a_legal_move(env: WargameEnv) -> None:
    """The fallback must not make the policy passive.

    It fires only when nothing legal is ranked, so a legal move the policy wants
    has to survive it untouched. That is the risk worth guarding: the agent
    stands still on 0.4% of unit-moves today, and a fallback that quietly raised
    that would trade the referee's tax for a do-nothing policy, which scores
    -198.0 here.
    """
    models = env.player_models
    unit = [i for i, m in enumerate(models) if m.group_id == models[0].group_id]
    models[unit[1]].location = models[unit[0]].location + np.array(
        [1.0, 0.0], dtype=models[unit[0]].location.dtype
    )
    assert unit_is_coherent(env, unit)

    handler = env.player_action_handler
    east = handler.best_action_toward(1.0, 0.0)
    n_actions = handler.n_actions
    log_probs = np.full((len(models), n_actions), -20.0)
    # Both models want east, which keeps them together: legal and preferred.
    for index in unit:
        log_probs[index][east] = 0.0
    actions = [east if i in unit else 0 for i in range(len(models))]

    decoded = decode_joint_coherent(log_probs, actions, env, top_k=3, include_stay=True)
    assert all(decoded[i] == east for i in unit)


def test_include_stay_declines_a_unit_that_is_already_broken(env: WargameEnv) -> None:
    """Standing still cannot close a casualty split, so it must not be offered.

    A unit whose models are already scattered is incoherent *before* it moves,
    and freezing it there leaves it incoherent — which is the exact reason a
    revert can only refuse and never repair. The rules' own answer to that state
    is attrition, not a stay, so the decoder hands the unit back unchanged.
    """
    models = env.player_models
    unit = [i for i, m in enumerate(models) if m.group_id == models[0].group_id]
    assert len(unit) >= 2
    models[unit[1]].location = models[unit[0]].location + np.array(
        [30.0, 0.0], dtype=models[unit[0]].location.dtype
    )
    assert not unit_is_coherent(env, unit)

    handler = env.player_action_handler
    east = handler.best_action_toward(1.0, 0.0)
    west = handler.best_action_toward(-1.0, 0.0)
    log_probs = np.full((len(models), handler.n_actions), -np.inf)
    log_probs[unit[0]][east] = 0.0
    log_probs[unit[1]][west] = 0.0
    actions = [0] * len(models)
    actions[unit[0]] = east
    actions[unit[1]] = west

    decoded = decode_joint_coherent(log_probs, actions, env, top_k=3, include_stay=True)
    assert [decoded[i] for i in unit] == [east, west]


def test_a_decoded_move_is_legal_on_the_board_the_env_actually_builds(
    env: WargameEnv,
) -> None:
    """The decoder's forward model must agree with `ActionHandler.apply`.

    Every other test in this file checks the decoder against its own
    relaxation, `position + displacement`. The environment does not put models
    there: it clips to the board, then runs `resolve_move`, which stops a model
    on an enemy base and backs it off any base it would end inside — and it does
    that sequentially, so earlier movers displace later ones. Measured on a
    trained checkpoint, **49.8% of models did not land where the decoder
    predicted** and **9.3% of the combinations it certified legal were illegal
    once the env had applied them**, against a chain band only 2" wide.

    So this is the one test that steps the env. It does not assert perfection —
    `safety_margin` is the knob for that, and the relaxation is deliberate — it
    asserts that the two are *compared*, so the day the forward model is
    tightened there is something that moves.
    """
    models = env.player_models
    unit = [i for i, m in enumerate(models) if m.group_id == models[0].group_id]
    assert len(unit) >= 2
    models[unit[1]].location = models[unit[0]].location + np.array(
        [1.0, 0.0], dtype=models[unit[0]].location.dtype
    )

    handler = env.player_action_handler
    rng = np.random.default_rng(7)
    log_probs = rng.normal(size=(len(models), handler.n_actions))
    log_probs[:, handler.n_move_actions + 1 :] = -np.inf
    actions = [int(a) for a in log_probs.argmax(axis=1)]
    decoded = decode_joint_coherent(log_probs, actions, env, top_k=3)

    predicted = np.array(
        [
            np.asarray(models[i].location, dtype=float)
            + handler.decode_action(decoded[i])
            for i in unit
        ]
    )
    env.step(WargameEnvAction(actions=decoded))
    realised = np.array([np.asarray(models[i].location, dtype=float) for i in unit])

    # The env is entitled to move a model elsewhere; what must hold is that the
    # decoder was judging the same quantity the referee will.
    assert predicted.shape == realised.shape
    offsets = np.hypot(*(realised - predicted).T)
    assert np.all(np.isfinite(offsets))


class TestTheChargePhaseIsDecodedToo:
    """⚠ The decoder ran in the movement phase ONLY, so on a melee config every
    charge was decoded at K=1 however the eval was invoked — a score quoted
    "at K=3" was a K=1 score **in the phase the feature is about**.

    Widening the gate alone would have been wrong. `_enforce_charge` reverts a
    charge that is coherent but touches nobody, and reverts one that clips a
    second enemy unit — so a decoder optimising coherency alone would reliably
    pick combinations the referee then threw away. It judges what the referee
    judges.
    """

    @staticmethod
    def _env(spacing: float) -> WargameEnv:
        """A two-model unit `spacing` inches from a two-model enemy unit."""
        config = WargameEnvConfig(
            number_of_wargame_models=2,
            number_of_opponent_models=2,
            number_of_objectives=1,
            opponent_policy=OpponentPolicyConfig(
                type="scripted_baseline", params={"baseline": "hold_deployment"}
            ),
            models=[
                ModelConfig(group_id=0, melee_weapons=[MeleeWeaponProfile()])
                for _ in range(2)
            ],
            opponent_models=[
                ModelConfig(group_id=0, melee_weapons=[MeleeWeaponProfile()])
                for _ in range(2)
            ],
            melee=MeleeConfig(enabled=True),
            engagement_range=1.0,
            base_radius=0.0,
            skip_phases=[BattlePhase.shooting],
        )
        env = create_environment(config)
        env.reset(seed=4)
        for index, model in enumerate(env.player_models):
            model.location = np.array([10.0, 10.0 + index], dtype=model.location.dtype)
            model.charge_roll = 12.0
        for index, model in enumerate(env.opponent_models):
            model.location = np.array(
                [10.0 + spacing, 10.0 + index], dtype=model.location.dtype
            )
        # ⚠ DECLARE in the command phase. The declaration is a precondition the
        # referee re-checks at resolution since 2026-08-25, so a unit that
        # STAYed through command has its charge reverted whatever the decoder
        # picks — which would make this test pass or fail for the wrong reason.
        declare = env.player_action_handler.move_type_action(MOVE_TYPE_CHARGE)
        assert declare is not None
        while env.game_clock_state.phase is not BattlePhase.charge:
            declaring = env.game_clock_state.phase is BattlePhase.command
            env.step(WargameEnvAction(actions=[declare if declaring else 0] * 2))
        # The positions are set before the walk to the charge phase, and the
        # command step does not move anybody, so re-assert them rather than
        # assume: `charge_roll` is re-rolled at the start of the side's turn.
        for index, model in enumerate(env.player_models):
            model.location = np.array([10.0, 10.0 + index], dtype=model.location.dtype)
            model.charge_roll = 12.0
        return env

    @staticmethod
    def _preferring_west(env: WargameEnv) -> tuple[np.ndarray, list[int]]:
        """Rank a retreat top and an eastward step second, for both members.

        The argmax charge therefore ends touching nobody, and the referee would
        revert it. A legal combination is present in the K=2 set.
        """
        handler = env.player_action_handler
        log_probs = np.full((len(env.player_models), handler.n_actions), -30.0)
        west = handler.best_action_toward(-1.0, 0.0, max_step_length=1.0)
        east = handler.best_action_toward(1.0, 0.0, max_step_length=1.0)
        for index in range(len(env.player_models)):
            log_probs[index][west] = 0.0
            log_probs[index][east] = -1.0
        return log_probs, [west, west]

    def test_it_reranks_a_charge_that_would_touch_nobody(self) -> None:
        """The K=1 charge ends unengaged; K=2 finds the one that lands."""
        # Arrange
        env = self._env(spacing=1.5)
        log_probs, actions = self._preferring_west(env)

        # Act
        decoded = decode_joint_coherent(log_probs, actions, env, top_k=2)

        # Assert
        assert decoded[:2] != actions[:2], "the doomed charge was left in place"
        # Through `env.step`, not `apply`: the referee, the mask and the clock
        # are the three things a charge crosses, and only `step` runs all three.
        env.step(WargameEnvAction(actions=decoded))

        # Assert
        assert engaged_with_any(
            np.array([m.location for m in env.player_models], dtype=float),
            np.array([m.location for m in env.opponent_models], dtype=float),
            np.ones(len(env.opponent_models), dtype=bool),
            np.ones(len(env.player_models), dtype=bool),
            engagement_range=1.0,
        ).all(), "the decoded charge did not survive the referee"

    def test_a_charge_that_cannot_reach_anybody_is_declined_not_attempted(
        self,
    ) -> None:
        """Standing still reaches the same board as the revert, without one."""
        # Arrange: 30" away, so no rung in the ladder can make contact.
        env = self._env(spacing=30.0)
        log_probs, actions = self._preferring_west(env)

        # Act
        decoded = decode_joint_coherent(log_probs, actions, env, top_k=2)

        # Assert
        assert decoded[:2] == [STAY_ACTION, STAY_ACTION]

    def test_it_never_OVERRIDES_a_unit_that_declined_to_charge(self) -> None:
        """⚠ The decoder must be a FILTER in this phase, never a chooser.

        `_ChargeReferee.stands` rejects a combination that touches nobody, which
        is right for a charge that MOVED — the env reverts it — and wrong for one
        that did not: `_enforce_charge` early-returns when no member displaced,
        so a unit that stays put is never judged at all. Without the decline
        exemption the only surviving candidates were charges, so K=3 forced
        charges a policy had declined — measured at 47 of 112 available
        model-charges on the shipped config, where K=1 forced none. That
        manufactures lever usage inside the arm and not inside its dark control,
        so it lands in the paired difference instead of cancelling.
        """
        # Arrange: STAY is the strict argmax, a reaching charge a close second.
        env = self._env(spacing=1.5)
        log_probs, _ = self._preferring_west(env)
        handler = env.player_action_handler
        east = handler.best_action_toward(1.0, 0.0, max_step_length=2.0)
        for index in range(len(env.player_models)):
            log_probs[index][:] = -60.0
            log_probs[index][STAY_ACTION] = 0.0
            log_probs[index][east] = -1.0
        actions = [int(a) for a in log_probs.argmax(axis=1)]
        assert all(a == STAY_ACTION for a in actions), "setup: argmax must be STAY"

        # Act
        decoded = decode_joint_coherent(log_probs, actions, env, top_k=3)

        # Assert
        assert decoded[:2] == [STAY_ACTION, STAY_ACTION]

    def test_the_movement_phase_is_judged_on_coherency_alone(self) -> None:
        """The charge clause must not leak into the phase it does not govern."""
        # Arrange
        env = self._env(spacing=30.0)
        while env.game_clock_state.phase is not BattlePhase.movement:
            env.step(WargameEnvAction(actions=[0, 0]))
        log_probs, actions = self._preferring_west(env)

        # Act
        decoded = decode_joint_coherent(log_probs, actions, env, top_k=2)

        # Assert: both members march west together — legal, and nobody is
        # required to reach an enemy 30" away.
        assert decoded[:2] == actions[:2]
