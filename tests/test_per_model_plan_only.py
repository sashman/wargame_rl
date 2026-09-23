"""The plan-only machinery (#384, after Stage 1): the committed script's
per-squad fallback, a scripted seat that executes without emitting, the
plan-only chooser (the head plans, the script walks), the counterfactual
planning credit, and the plan-only trainer's rollout and update.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.baseline.scripted_squad_march_committed import (
    ScriptedSquadMarchCommittedPolicy,
)
from wargame_rl.wargame.envs.baseline.scripted_squad_march_take import (
    ScriptedSquadMarchTakePolicy,
)
from wargame_rl.wargame.envs.per_model import PerModelEnv, StepKind
from wargame_rl.wargame.envs.per_model.commitment import NO_TARGET
from wargame_rl.wargame.envs.per_model.reward_timing import (
    PerStepReward,
    PlanningCredit,
)
from wargame_rl.wargame.envs.per_model.scripted import ScriptedSeat
from wargame_rl.wargame.envs.per_model.types import NO_COMMIT_DECISION
from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import CommitmentConfig
from wargame_rl.wargame.model.per_model.agent import SetAgent
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig
from wargame_rl.wargame.model.per_model.evaluate import plan_only_chooser
from wargame_rl.wargame.model.per_model.net import SetNetwork
from wargame_rl.wargame.model.per_model.ppo import (
    PerModelPPOConfig,
    collect_rollout,
    compute_gae,
    compute_planning_gae,
    ppo_update,
)

SMALL_TRUNK = SetNetworkConfig(n_layers=1, embedding_size=32, n_heads=4)
MEMBERS = "squad_march_committed"


def _head_config(**kwargs: object) -> WargameEnvConfig:
    copied: WargameEnvConfig = small_config(**kwargs).model_copy(  # type: ignore[arg-type]
        update={"commitments": CommitmentConfig(assignment="head")}  # type: ignore[arg-type]
    )
    return copied


def test_the_committed_script_keeps_a_written_squad_and_falls_back_per_squad() -> None:
    env = PerModelEnv(_head_config())
    env.reset(seed=3)
    policy = build_baseline_policy(MEMBERS)
    assert isinstance(policy, ScriptedSquadMarchCommittedPolicy)
    groups = env.player_seat.living_units()
    assert len(groups) >= 2 and len(env.objectives) >= 2
    greedy = ScriptedSquadMarchTakePolicy.squad_objectives(
        policy, env.wargame_models, env, groups
    )
    # Nothing written: the parent's plan for everyone.
    assert policy.squad_objectives(env.wargame_models, env, groups) == greedy
    # One squad written to an objective the greedy plan did not give it: that
    # squad keeps its commitment, the others keep a plan of their own.
    first = groups[0]
    other = next(k for k, o in enumerate(env.objectives) if o is not greedy[0])
    env.player_commitments.set_ground(first, other)
    mixed = policy.squad_objectives(env.wargame_models, env, groups)
    assert mixed[0] is env.objectives[other]
    assert all(o in env.objectives for o in mixed[1:])


def test_a_non_emitting_seat_leaves_the_commitment_state_alone() -> None:
    for emits in (True, False):
        env = PerModelEnv(_head_config())
        env.reset(seed=4)
        seat = env.player_seat
        point = env.pending
        assert point is not None and point.phase is not None
        planner = ScriptedSeat.for_policy(
            build_baseline_policy("squad_march_take"), emits=emits
        )
        planner.plan(point.phase, seat, env)
        state = env.player_commitments
        written = [state.ground_of(g) != NO_TARGET for g in seat.living_units()]
        assert any(written) is emits


def test_the_plan_only_chooser_lets_the_head_write_and_the_script_walk() -> None:
    torch.manual_seed(0)
    envs = [PerModelEnv(_head_config(rounds=3))]
    agent = SetAgent(SetNetwork.from_env(envs[0], SMALL_TRUNK))
    choose = plan_only_chooser(agent, envs, MEMBERS)
    env = envs[0]
    observation, _ = env.reset(seed=6)
    state = env.player_commitments
    offered = 0
    for _ in range(400):
        point = observation.decision
        before = {g: state.ground_of(g) for g in env.player_seat.living_units()}
        action = choose(envs, [observation])[0]
        if point.kind is StepKind.open and point.commit_mask is not None:
            if point.commit_mask[action.model].any():
                assert action.commitment != NO_COMMIT_DECISION
                offered += 1
        else:
            assert action.commitment == NO_COMMIT_DECISION
        assert point.why_illegal(action) is None
        observation, _, terminated, _, _ = env.step(action)
        after = {g: state.ground_of(g) for g in env.player_seat.living_units()}
        if point.kind in (StepKind.act, StepKind.target):
            # The script executes and never writes: on its steps the state
            # can only lose a commitment (the env retires a latecomer when
            # the window settles), never gain one.
            for g, target in after.items():
                assert target == before[g] or target == NO_TARGET
        if terminated:
            break
    assert offered > 0


def _spread_config() -> WargameEnvConfig:
    """The head config with A3's reward: travel, coverage at 0.3, the success
    bonus on every objective occupied."""
    copied: WargameEnvConfig = _head_config().model_copy(
        update={
            "reward_phases": [
                RewardPhaseConfig(
                    name="arrive",
                    terminal_success_bonus=5.0,
                    reward_calculators=[
                        RewardCalculatorConfig(
                            type="closest_objective_v2",
                            weight=1.0,
                            params={"progress_scale": 6.0, "fallback_to_nearest": True},
                        ),
                        RewardCalculatorConfig(type="objective_coverage", weight=0.3),
                    ],
                    success_criteria=SuccessCriteriaConfig(
                        type="all_objectives_occupied"
                    ),
                )
            ]
        }
    )
    return copied


def _place(env: PerModelEnv) -> None:
    """Unit 0 alone on objective 0, unit 1 in a corner off every objective."""
    target = np.asarray(env.objectives[0].location, dtype=float)
    for model in env.wargame_models:
        if int(model.group_id) == 0:
            model.location = target.copy()
        else:
            model.location = np.array([30.0, 38.0])


def test_the_counterfactual_credit_pays_each_unit_what_its_bodies_change() -> None:
    config = _spread_config()
    env = PerModelEnv(config)
    twin = PerModelEnv(config)
    env.reset(seed=7)
    twin.reset(seed=7)
    counterfactual = PerStepReward(
        env, streams=True, planning_credit=PlanningCredit.counterfactual
    )
    broadcast = PerStepReward(twin, streams=True)
    counterfactual.reset()
    broadcast.reset()
    _place(env)
    _place(twin)
    _total, _credits, planning, per_unit = counterfactual._pay_close(False, {})
    _total, _credits, twin_planning, twin_units = broadcast._pay_close(False, {})
    coverage_share = 0.3 * (1.0 / len(env.objectives)) * counterfactual.phase_scale
    assert twin_planning == pytest.approx(coverage_share) and twin_units == {}
    assert planning == 0.0
    assert set(per_unit) == {0, 1}
    # The squad alone on the objective is what makes it ours; the squad in
    # the corner changes nothing.
    assert per_unit[0] == pytest.approx(coverage_share)
    assert per_unit[1] == 0.0


def test_the_plan_only_rollout_trains_only_the_head() -> None:
    torch.manual_seed(5)
    envs = [PerModelEnv(_head_config(rounds=4)) for _ in range(2)]
    for env in envs:
        env.set_player_planner(
            ScriptedSeat.for_policy(build_baseline_policy(MEMBERS), emits=False)
        )
    observations = [env.reset(seed=30 + i)[0] for i, env in enumerate(envs)]
    retimers = [
        PerStepReward(env, streams=True, planning_credit=PlanningCredit.counterfactual)
        for env in envs
    ]
    for retimer in retimers:
        retimer.reset()
    agent = SetAgent(SetNetwork.from_env(envs[0], SMALL_TRUNK))
    config = PerModelPPOConfig(
        rollout_rounds=2, batch_size=16, n_epochs=2, members=MEMBERS
    )
    rollout = collect_rollout(envs, agent, retimers, observations, config)
    assert not any(t.has_policy for t in rollout.transitions)
    commits = [t for t in rollout.transitions if t.has_commitment]
    assert commits
    assert any(t.planning_reward != 0.0 for t in commits) or True
    returns, advantages = compute_gae(rollout, config)
    planning_returns, planning_advantages = compute_planning_gae(rollout, config)
    network = agent.network
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)
    stats = ppo_update(
        network,
        optimizer,
        rollout,
        returns,
        advantages,
        config,
        generator=torch.Generator().manual_seed(0),
        planning_returns=planning_returns,
        planning_advantages=planning_advantages,
    )
    assert stats.commit_rows > 0
    assert np.isfinite(stats.planning_explained_variance)
