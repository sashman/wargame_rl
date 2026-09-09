"""The reward paid per decision: who a step pays, what a close pays, and
what the env's step effect reports.

Every payment rule in `envs/per_model/reward_timing.py` is pinned against a
direct calculator call or a hand computation, and the two exact classes --
`vp_gain` and the terminal bonus -- are pinned to the phase facade's episode
totals on bridge-identical play. State terms are documented approximations
(scaled from the close board) and are not asserted equal across facades.
"""

from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np
import pytest

from tests.per_model_seats import (
    charging_driver,
    shooting_charging_driver,
    small_config,
)
from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.domain.kernel.entities import alive_mask_for
from wargame_rl.wargame.envs.domain.sequencing.activation import (
    CHARGE_TARGET_DECLINE,
    MoveDeclaration,
)
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_ownership_from_norms_offset,
)
from wargame_rl.wargame.envs.per_model import PerModelAction, PerModelEnv, StepKind
from wargame_rl.wargame.envs.per_model.env import StepEffect
from wargame_rl.wargame.envs.per_model.random_seat import random_legal_action
from wargame_rl.wargame.envs.per_model.reward_timing import (
    DELTA_GLOBALS,
    EVENT_ACTION_TERMS,
    POTENTIAL_ACTION_TERMS,
    REFUSED_TERMS,
    STATE_GLOBALS,
    STATE_TERMS,
    PerStepReward,
    classify,
)
from wargame_rl.wargame.envs.per_model.scripted import ScriptedSeat
from wargame_rl.wargame.envs.per_model.types import PerModelObservation
from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator
from wargame_rl.wargame.envs.reward.calculators.closest_objective_v2 import (
    ClosestObjectiveV2Calculator,
)
from wargame_rl.wargame.envs.reward.calculators.registry import CALCULATOR_REGISTRY
from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import BattlePhase, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

COMBAT_SEED = 4242


def _terms(
    *,
    terminate_on_success: bool = False,
    extra: list[RewardCalculatorConfig] | None = None,
) -> RewardPhaseConfig:
    """One term of every payment class, the golden config's weights."""
    return RewardPhaseConfig(
        name="retimed",
        reward_calculators=[
            RewardCalculatorConfig(
                type="closest_objective_v2",
                weight=1.0,
                params={"progress_scale": 6.0, "fallback_to_nearest": True},
            ),
            RewardCalculatorConfig(
                type="objective_hold", weight=1.25, params={"crowding_exponent": 1.0}
            ),
            RewardCalculatorConfig(
                type="model_kills", weight=1.0, params={"bonus_per_kill": 2.0}
            ),
            RewardCalculatorConfig(type="group_cohesion", weight=0.3),
            RewardCalculatorConfig(type="vp_gain", weight=2.0),
            RewardCalculatorConfig(type="objective_coverage", weight=0.3),
            *(extra or []),
        ],
        success_criteria=SuccessCriteriaConfig(type="player_ahead_on_vp"),
        terminate_on_success=terminate_on_success,
        terminal_success_bonus=2.0,
    )


def _config(phase: RewardPhaseConfig | None = None, **kwargs: Any) -> WargameEnvConfig:
    base = small_config(**kwargs)
    return WargameEnvConfig(
        **{**base.model_dump(), "reward_phases": [phase or _terms()]}
    )


def _context(env: PerModelEnv, phase: BattlePhase | None) -> StepContext:
    return StepContext(
        distance_cache=compute_distances(
            env.wargame_models,
            env.objectives,
            compute_model_model=True,
            alive_mask=alive_mask_for(env.wargame_models),
        ),
        current_turn=env.current_turn,
        max_turns=env.max_turns,
        board_width=env.board_width,
        board_height=env.board_height,
        action_phase=phase,
    )


def _walk_to(
    env: PerModelEnv,
    observation: PerModelObservation,
    kind: StepKind,
    phase: BattlePhase,
) -> PerModelObservation:
    """Drive with the charging script until a point of `kind` in `phase`."""
    while not (
        observation.decision.kind is kind and observation.decision.phase is phase
    ):
        observation, _, done, _, _ = env.step(
            charging_driver(env, observation.decision)
        )
        assert not done
    return observation


# ---------------------------------------------------------------- classes


def test_every_registered_calculator_has_exactly_one_payment_class() -> None:
    classes = (
        POTENTIAL_ACTION_TERMS,
        EVENT_ACTION_TERMS,
        STATE_TERMS,
        DELTA_GLOBALS,
        STATE_GLOBALS,
        REFUSED_TERMS,
    )
    for name, calculator in CALCULATOR_REGISTRY.items():
        matches = [c for c in classes if issubclass(calculator, c)]
        assert len(matches) == 1, f"{name} is in {len(matches)} classes"


def test_an_unclassified_calculator_is_refused_by_name() -> None:
    class Mystery(PerModelRewardCalculator):
        def calculate(self, model_idx: int, model: Any, view: Any, ctx: Any) -> float:
            return 0.0

    manager = RewardPhaseManager.from_configs([_terms()])
    manager.current_phase.per_model_calculators.append(("mystery", Mystery()))
    with pytest.raises(ValueError, match="mystery"):
        classify(manager)


def test_the_legacy_closest_objective_is_refused() -> None:
    phase = RewardPhaseConfig(
        name="legacy",
        reward_calculators=[RewardCalculatorConfig(type="closest_objective")],
        success_criteria=SuccessCriteriaConfig(type="player_ahead_on_vp"),
    )
    with pytest.raises(ValueError, match="closest_objective_v2"):
        PerStepReward(PerModelEnv(_config(phase)))


def test_a_curriculum_config_is_refused() -> None:
    base = small_config()
    config = WargameEnvConfig(
        **{**base.model_dump(), "reward_phases": [_terms(), _terms()]}
    )
    with pytest.raises(ValueError, match="single reward phase"):
        PerStepReward(PerModelEnv(config))


# -------------------------------------------------------------- act steps


def test_an_act_step_pays_the_actor_its_term_over_the_alive_count() -> None:
    env = PerModelEnv(_config())
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=1, options={"combat_seed": COMBAT_SEED})
    retimer.reset()
    observation = _walk_to(env, observation, StepKind.act, BattlePhase.movement)
    point = observation.decision
    actor = int(np.flatnonzero(point.selector_mask)[0])
    legal = np.flatnonzero(point.action_mask[actor])
    action = PerModelAction.act(actor, int(legal[-1]))
    name, calculator = retimer.classes.potential[0]
    twin = copy.deepcopy(calculator)

    _, _, terminated, _, info = env.step(action)
    effect: StepEffect = info["effect"]
    n_alive = int(alive_mask_for(env.wargame_models).sum())
    expected = (
        twin.weight
        * twin.calculate(
            actor,
            env.wargame_models[actor],
            cast(Any, env),
            _context(env, BattlePhase.movement),
        )
        / n_alive
    )
    payment = retimer.on_step(observation, action, effect, terminated)

    assert effect.acted == (actor,)
    assert not payment.is_close
    assert set(payment.breakdown) <= {name}
    assert payment.reward == pytest.approx(expected)
    assert payment.breakdown.get(name, 0.0) == pytest.approx(expected)


def test_a_skip_declaration_pays_every_consumed_member() -> None:
    env = PerModelEnv(_config())
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=2, options={"combat_seed": COMBAT_SEED})
    retimer.reset()
    point = observation.decision
    assert point.kind is StepKind.open and point.phase is BattlePhase.movement
    opener = int(np.flatnonzero(point.selector_mask)[0])
    unit = int(env.wargame_models[opener].group_id)
    members = env.player_seat.unit_members(unit)
    action = PerModelAction.open(opener, int(MoveDeclaration.stationary))

    _, _, terminated, _, info = env.step(action)
    effect: StepEffect = info["effect"]
    payment = retimer.on_step(observation, action, effect, terminated)

    assert sorted(effect.acted) == sorted(members)
    assert len(members) == 3
    # Nobody moved, so the potential pays each member its anchoring step:
    # the term is present for the unit and the close bundle is not.
    assert "closest_objective_v2" in payment.breakdown
    assert "vp_gain" not in payment.breakdown


def test_a_target_that_names_a_unit_pays_nothing_and_a_decline_pays_the_unit() -> None:
    config = _config(melee=True, opponent_x=22)
    for decline in (False, True):
        env = PerModelEnv(config)
        retimer = PerStepReward(env)
        observation, _ = env.reset(seed=3, options={"combat_seed": COMBAT_SEED})
        retimer.reset()
        observation = _walk_to(env, observation, StepKind.target, BattlePhase.charge)
        point = observation.decision
        model = int(np.flatnonzero(point.selector_mask)[0])
        unit = int(env.wargame_models[model].group_id)
        if decline:
            action = PerModelAction.target(model, CHARGE_TARGET_DECLINE)
        else:
            target = int(np.flatnonzero(point.target_mask[model])[0])
            action = PerModelAction.target(model, target)
        _, _, terminated, _, info = env.step(action)
        effect: StepEffect = info["effect"]
        payment = retimer.on_step(observation, action, effect, terminated)
        if decline:
            assert sorted(effect.acted) == sorted(env.player_seat.unit_members(unit))
            assert "closest_objective_v2" in payment.breakdown
        else:
            assert effect.acted == ()
            assert payment.reward == 0.0
            assert payment.breakdown == {}


def test_a_shooting_units_kills_are_paid_on_its_last_members_step() -> None:
    """The volley resolves at the unit's close, so the kills land on the step
    of whichever member acted last, naming every attacker that killed."""
    config = _config(opponent_x=22, baseline="squad_march")
    seen_kill = False
    for seed in range(1, 12):
        env = PerModelEnv(config)
        retimer = PerStepReward(env)
        observation, _ = env.reset(seed=seed, options={"combat_seed": seed})
        retimer.reset()
        done = False
        kills_total = 0
        while not done:
            point = observation.decision
            action = shooting_charging_driver(env, point)
            before = observation
            observation, _, done, _, info = env.step(action)
            effect: StepEffect = info["effect"]
            payment = retimer.on_step(before, action, effect, done)
            kills = sum(effect.kills_by_model.values())
            kills_total += kills
            if kills and action.kind is StepKind.act:
                seen_kill = True
                n_alive = int(alive_mask_for(env.wargame_models).sum())
                expected = 2.0 * kills / n_alive
                assert payment.breakdown["model_kills"] == pytest.approx(expected)
                shooters = {
                    int(env.wargame_models[i].group_id) for i in effect.kills_by_model
                }
                assert shooters == {int(env.wargame_models[action.model].group_id)}
        dead_opponents = int((~alive_mask_for(env.opponent_models)).sum())
        assert kills_total == dead_opponents
    assert seen_kill


# ------------------------------------------------------------ close steps


def test_the_close_pays_vp_gain_the_whole_turns_net_delta() -> None:
    env = PerModelEnv(_config(baseline="squad_march_take"))
    retimer = PerStepReward(env)
    rng = np.random.default_rng(5)
    observation, _ = env.reset(seed=5, options={"combat_seed": COMBAT_SEED})
    retimer.reset()
    previous = (env.player_vp, env.opponent_vp)
    done = False
    closes = 0
    while not done:
        point = observation.decision
        action = random_legal_action(point, rng)
        before = observation
        observation, _, done, _, info = env.step(action)
        payment = retimer.on_step(before, action, info["effect"], done)
        if payment.is_close:
            closes += 1
            now = (env.player_vp, env.opponent_vp)
            net = (now[0] - previous[0]) - (now[1] - previous[1])
            expected = 2.0 * net / env.config.mission.per_round_cap
            assert payment.breakdown.get("vp_gain", 0.0) == pytest.approx(expected)
            previous = now
        else:
            assert "vp_gain" not in payment.breakdown
    assert closes == env.n_rounds


def test_state_terms_at_the_close_are_scaled_by_the_phases_per_round() -> None:
    env = PerModelEnv(_config())
    retimer = PerStepReward(env)
    rng = np.random.default_rng(6)
    observation, _ = env.reset(seed=6, options={"combat_seed": COMBAT_SEED})
    retimer.reset()
    assert retimer.phases_per_round == 2
    while observation.decision.kind is not StepKind.close_turn:
        action = random_legal_action(observation.decision, rng)
        observation, _, _, _, _ = env.step(action)
    player = compute_distances(
        env.wargame_models,
        env.objectives,
        alive_mask=alive_mask_for(env.wargame_models),
    )
    opponent = compute_distances(
        env.opponent_models,
        env.objectives,
        alive_mask=alive_mask_for(env.opponent_models),
    )
    held, _ = objective_ownership_from_norms_offset(
        player.model_obj_norms_offset, opponent.model_obj_norms_offset, player.obj_radii
    )
    coverage = float(held.sum()) / len(env.objectives)
    action = PerModelAction.close_turn()
    _, _, terminated, _, info = env.step(action)
    payment = retimer.on_step(observation, action, info["effect"], terminated)
    assert payment.is_close
    assert payment.breakdown["objective_coverage"] == pytest.approx(0.3 * coverage * 2)
    assert "group_cohesion" in payment.breakdown
    assert "objective_hold" in payment.breakdown


def _phase_facade_totals(
    config: WargameEnvConfig, name: str, seed: int
) -> dict[str, float]:
    env = WargameEnv(config)
    policy = build_baseline_policy(name)
    observation, _ = env.reset(seed=seed, options={"combat_seed": COMBAT_SEED})
    totals: dict[str, float] = {}
    done = False
    while not done:
        action = policy.select_action(
            env.wargame_models, env, action_mask=observation.action_mask
        )
        observation, _, done, _, _ = env.step(action)
        for key, value in env.last_reward_breakdown.items():
            totals[key] = totals.get(key, 0.0) + value
    return totals


def _per_model_totals(
    config: WargameEnvConfig, name: str, seed: int
) -> dict[str, float]:
    env = PerModelEnv(config)
    policy = build_baseline_policy(name)
    shoots = type(policy).select_shooting is not BaselinePolicy.select_shooting
    planner = ScriptedSeat(policy, shoots=shoots)
    env.set_player_planner(planner)
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=seed, options={"combat_seed": COMBAT_SEED})
    retimer.reset()
    done = False
    while not done:
        point = observation.decision
        if point.kind is StepKind.close_turn:
            action = PerModelAction.close_turn()
        else:
            action = planner.choose(point, env.player_seat)
        before = observation
        observation, _, done, _, info = env.step(action)
        retimer.on_step(before, action, info["effect"], done)
    return retimer.episode_breakdown


@pytest.mark.parametrize("seed", [700001, 700002])
def test_the_exact_classes_conserve_their_episode_totals_across_facades(
    seed: int,
) -> None:
    """`vp_gain` telescopes and the terminal bonus is paid once, so on
    bridge-identical play their episode totals are the phase facade's to the
    last bit. `model_kills` joins them on an episode with no player losses --
    the phase facade divides by the alive count AT SETTLE, after the
    opponent's turn, so a shooter that then dies drops its own credit."""
    config = _config(opponent_x=22, baseline="squad_march")
    old = _phase_facade_totals(config, "squad_march_take", seed)
    new = _per_model_totals(config, "squad_march_take", seed)
    assert new["vp_gain"] == pytest.approx(old["vp_gain"], abs=1e-12)
    assert new.get("terminal_success_bonus", 0.0) == pytest.approx(
        old.get("terminal_success_bonus", 0.0), abs=1e-12
    )
    assert new.get("model_kills", 0.0) == pytest.approx(
        old.get("model_kills", 0.0), abs=1e-12
    )
    assert old.get("model_kills", 0.0) > 0.0


def test_a_terminating_decision_step_pays_its_terms_and_the_close_bundle() -> None:
    """`_advance` can terminate on a decision step (a mid-turn settle under
    `terminate_on_success` or player elimination); the retimer then pays the
    step's own action terms AND the close bundle, and marks it closing. The
    shipped criteria settle at phase boundaries and land on `close_turn`, so
    the rule is pinned on the retimer directly."""
    env = PerModelEnv(_config(baseline="squad_march"))
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=9, options={"combat_seed": COMBAT_SEED})
    retimer.reset()
    observation = _walk_to(env, observation, StepKind.act, BattlePhase.movement)
    point = observation.decision
    actor = int(np.flatnonzero(point.selector_mask)[0])
    legal = np.flatnonzero(point.action_mask[actor])
    action = PerModelAction.act(actor, int(legal[-1]))
    _, _, terminated, _, info = env.step(action)
    assert not terminated
    payment = retimer.on_step(observation, action, info["effect"], terminated=True)
    assert payment.is_close
    assert "closest_objective_v2" in payment.breakdown
    assert "vp_gain" in payment.breakdown
    assert "objective_coverage" in payment.breakdown
    assert retimer.closes == 1
    assert retimer.last_context is not None and retimer.last_context.is_terminated


# ----------------------------------------------------------------- effects


def test_a_close_turn_reports_the_strike_back_kills_of_the_opponents_turn() -> None:
    """A kill resolved in the opponent's turn (a fight the player wins on the
    opponent's activation) lands on the close, not on any decision."""
    env = PerModelEnv(_config(melee=True, opponent_x=22))
    observation, _ = env.reset(seed=4, options={"combat_seed": COMBAT_SEED})
    done = False
    on_close = 0
    on_decisions = 0
    while not done:
        action = charging_driver(env, observation.decision)
        observation, _, done, _, info = env.step(action)
        kills = sum(info["effect"].kills_by_model.values())
        if action.kind is StepKind.close_turn:
            on_close += kills
            assert info["effect"].acted == ()
        else:
            on_decisions += kills
    dead = int((~alive_mask_for(env.opponent_models)).sum())
    assert on_close + on_decisions == dead


def test_attaching_a_retimer_leaves_the_windows_bit_identical() -> None:
    """The retimer has its own calculator instances, so the env's settled
    windows -- the bridge quantity -- do not move when one is attached."""

    def settled(with_retimer: bool) -> list[float]:
        env = PerModelEnv(_config(opponent_x=22))
        retimer = PerStepReward(env) if with_retimer else None
        rng = np.random.default_rng(7)
        observation, _ = env.reset(seed=7, options={"combat_seed": COMBAT_SEED})
        if retimer is not None:
            retimer.reset()
        rewards: list[float] = []
        done = False
        while not done:
            action = random_legal_action(observation.decision, rng)
            before = observation
            observation, reward, done, _, info = env.step(action)
            if retimer is not None:
                retimer.on_step(before, action, info["effect"], done)
            rewards.append(reward)
        return rewards

    assert settled(True) == settled(False)


def test_closest_objective_v2_recomputes_for_distinct_contexts_sharing_a_turn() -> None:
    """The memo used to key on `(current_turn, id(distance_cache))`; a freed
    context's id recycled under the per-model facade and served stale counts."""
    env = PerModelEnv(_config())
    env.reset(seed=8, options={"combat_seed": COMBAT_SEED})
    calculator = ClosestObjectiveV2Calculator()
    view = cast(Any, env)
    first = _context(env, BattlePhase.movement)
    _, counts_before, _ = calculator._objective_presence_masks(view, first)
    objective = env.objectives[0]
    for model in env.wargame_models:
        model.location = np.array(objective.location, dtype=float)
    second = _context(env, BattlePhase.movement)
    assert second.current_turn == first.current_turn
    _, counts_after, _ = calculator._objective_presence_masks(view, second)
    assert counts_after[0] == len(env.wargame_models)
    assert counts_before[0] != counts_after[0]
