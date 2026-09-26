"""The commitment head, #384 Stage 1: the decision at a unit's first open of
the turn, the agent's draw kept apart from the member's, the two reward
streams, the planning buffer and its semi-Markov return, the update's second
surrogate, the checkpoint loader, and the scripted bar under the head.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from tests.per_model_seats import random_legal_action, small_config
from wargame_rl.wargame.envs.per_model import PerModelEnv, StepKind
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.types import (
    COMMIT_KEEP,
    NO_COMMIT_DECISION,
    PerModelAction,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import CommitmentConfig
from wargame_rl.wargame.model.per_model.agent import NO_DRAW, SetAgent
from wargame_rl.wargame.model.per_model.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig
from wargame_rl.wargame.model.per_model.net import SetNetwork
from wargame_rl.wargame.model.per_model.ppo import (
    PerModelPPOConfig,
    Rollout,
    Transition,
    collect_rollout,
    compute_gae,
    compute_planning_gae,
    ppo_update,
)

SMALL_TRUNK = SetNetworkConfig(n_layers=1, embedding_size=32, n_heads=4)


def _weight(module: object) -> torch.Tensor:
    return cast(nn.Linear, module).weight


def _with_writer(config: WargameEnvConfig, assignment: str) -> WargameEnvConfig:
    copied: WargameEnvConfig = config.model_copy(
        update={"commitments": CommitmentConfig(assignment=assignment)}  # type: ignore[arg-type]
    )
    return copied


def _head_config(**kwargs: object) -> WargameEnvConfig:
    return _with_writer(small_config(**kwargs), "head")  # type: ignore[arg-type]


def _greedy_config() -> WargameEnvConfig:
    return _with_writer(small_config(), "greedy")


def _first_open(env: PerModelEnv) -> tuple[int, int]:
    """A selectable model at the pending open point and a legal declaration."""
    point = env.pending
    assert point is not None and point.kind is StepKind.open
    model = int(np.flatnonzero(point.selector_mask)[0])
    declaration = int(np.flatnonzero(point.declaration_mask[model])[0])
    return model, declaration


def test_the_head_writer_offers_a_decision_at_a_units_first_open_and_writes_it() -> (
    None
):
    env = PerModelEnv(_head_config())
    env.reset(seed=1)
    state = env.player_commitments
    point = env.pending
    assert point is not None and point.kind is StepKind.open
    assert point.commit_mask is not None
    for index in np.flatnonzero(point.selector_mask):
        # Every objective is offered; KEEP is not, on an empty slot (a unit
        # with no plan must name one, #384).
        assert point.commit_mask[index, 1:].all() and not point.commit_mask[index, 0]
    assert not state.any_set(), "nothing is written at deployment"

    model, declaration = _first_open(env)
    group = int(env.wargame_models[model].group_id)
    # A commitment the point does not offer is illegal: an unknown column, and
    # KEEP while the slot is empty; an objective is legal.
    assert point.why_illegal(PerModelAction.open(model, declaration, 99)) is not None
    assert (
        point.why_illegal(PerModelAction.open(model, declaration, COMMIT_KEEP))
        is not None
    )
    env.step(PerModelAction.open(model, declaration, 2))
    assert state.ground_of(group) == 2

    # The other unit is still offered its decision; the same unit is not
    # until the next turn.
    rng = np.random.default_rng(1)
    seen_other = False
    for _ in range(200):
        point = env.pending
        assert point is not None
        if point.kind is StepKind.open:
            for index in np.flatnonzero(point.selector_mask):
                g = int(env.wargame_models[int(index)].group_id)
                offered = point.offers_commitment(int(index))
                assert offered == (g != group) or env.current_turn > 1
                seen_other |= offered and g != group
        _, _, terminated, _, _ = env.step(random_legal_action(point, rng))
        if terminated:
            break
    assert seen_other

    # An open with no decision is legal (a scripted seat writes the state
    # itself), and KEEP leaves the slot as it stands.
    env.reset(seed=2)
    model, declaration = _first_open(env)
    group = int(env.wargame_models[model].group_id)
    env.step(PerModelAction.open(model, declaration))
    assert env.player_commitments.ground_of(group) < 0


def test_the_agent_draws_the_commitment_only_where_offered() -> None:
    torch.manual_seed(0)
    env = PerModelEnv(_head_config())
    observation, _ = env.reset(seed=3)
    agent = SetAgent(SetNetwork.from_env(env, SMALL_TRUNK))
    decision = agent.act(env, observation, generator=torch.Generator().manual_seed(0))
    assert decision.has_commitment and decision.commit_log_prob <= 0.0
    assert decision.action.commitment != NO_COMMIT_DECISION
    assert decision.has_policy
    env.step(decision.action)
    group = int(env.wargame_models[decision.model].group_id)
    expected = (
        -1 if decision.action.commitment == COMMIT_KEEP else decision.action.commitment
    )
    assert env.player_commitments.ground_of(group) == expected

    plain = PerModelEnv(_greedy_config())
    observation, _ = plain.reset(seed=3)
    decision = agent.act(plain, observation, generator=torch.Generator().manual_seed(0))
    assert decision.commit_column == NO_DRAW and decision.commit_log_prob == 0.0
    assert decision.action.commitment == NO_COMMIT_DECISION


def test_streams_route_the_close_outcome_to_the_planning_scalar() -> None:
    """Identical actions on twin envs: with streams the close's outcome terms
    come back as `planning` and leave `reward`; without them they are inside
    `reward`. The episode total is the same either way."""
    single = PerModelEnv(_head_config())
    split = PerModelEnv(_head_config())
    single_pay = PerStepReward(single)
    split_pay = PerStepReward(split, streams=True)
    obs_a, _ = single.reset(seed=5)
    obs_b, _ = split.reset(seed=5)
    single_pay.reset()
    split_pay.reset()
    rng = np.random.default_rng(5)
    planning_seen = 0.0
    for _ in range(300):
        action = random_legal_action(obs_a.decision, rng)
        nxt_a, _, term_a, _, info_a = single.step(action)
        nxt_b, _, term_b, _, info_b = split.step(action)
        pay_a = single_pay.on_step(obs_a, action, info_a["effect"], term_a)
        pay_b = split_pay.on_step(obs_b, action, info_b["effect"], term_b)
        assert pay_a.planning == 0.0
        assert pay_a.reward == pytest.approx(pay_b.reward + pay_b.planning)
        if not pay_b.is_close:
            assert pay_b.planning == 0.0
        planning_seen += abs(pay_b.planning)
        obs_a, obs_b = nxt_a, nxt_b
        if term_a:
            break
    assert single_pay.episode_reward == pytest.approx(split_pay.episode_reward)


def _head_envs(n: int = 2) -> tuple[list[PerModelEnv], list[PerStepReward], list]:
    envs = [PerModelEnv(_head_config(rounds=4)) for _ in range(n)]
    observations = [env.reset(seed=10 + i)[0] for i, env in enumerate(envs)]
    retimers = [PerStepReward(env, streams=True) for env in envs]
    for retimer in retimers:
        retimer.reset()
    return envs, retimers, observations


def test_a_rollout_under_the_head_carries_commitment_spans() -> None:
    torch.manual_seed(1)
    envs, retimers, observations = _head_envs()
    agent = SetAgent(SetNetwork.from_env(envs[0], SMALL_TRUNK))
    config = PerModelPPOConfig(rollout_rounds=2)
    rollout = collect_rollout(
        envs,
        agent,
        retimers,
        observations,
        config,
        generator=torch.Generator().manual_seed(1),
    )
    commits = [t for t in rollout.transitions if t.has_commitment]
    assert commits, "every unit commits at its first open of the turn"
    assert all(t.unit >= 0 and t.commit_log_prob <= 0.0 for t in commits)
    # One commitment per unit per turn: a unit's spans are ordered by row.
    per_env_unit = {(t.env_index, t.unit) for t in commits}
    assert len(per_env_unit) == 2 * 2
    # The spans still open at the cut carry a bootstrap for their unit.
    assert len(rollout.planning_bootstrap) == 2
    for env_index in range(2):
        open_units = {t.unit for t in commits if t.env_index == env_index}
        assert open_units <= set(rollout.planning_bootstrap[env_index]) | set()
    returns, advantages = compute_planning_gae(rollout, config)
    assert returns.shape[0] == rollout.n_steps
    non_commit = torch.tensor([not t.has_commitment for t in rollout.transitions])
    assert torch.all(returns[non_commit] == 0.0) and torch.all(
        advantages[non_commit] == 0.0
    )


def test_the_planning_gae_runs_along_each_units_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three commitment steps of one unit with rewards 1, 2, 3, values 0, a
    bootstrap of 10 and gamma 0.5, lambda 1: returns 1 + .5(2 + .5(3 + .5 x 10))."""
    env = PerModelEnv(_head_config())
    observation, _ = env.reset(seed=7)
    agent = SetAgent(SetNetwork.from_env(env, SMALL_TRUNK))
    tokens = agent.observe(env, observation)

    def transition(reward: float, commit: bool, done: bool = False) -> Transition:
        return Transition(
            tokens=tokens,
            model=0,
            column=0,
            head=tokens.head,
            phase=None,
            value=0.0,
            log_prob=0.0,
            reward=0.0,
            done=False,
            is_close=False,
            env_index=0,
            unit=0 if commit else -1,
            commit_column=0 if commit else NO_DRAW,
            planning_value=0.0,
            planning_reward=reward,
            planning_done=done,
        )

    rollout = Rollout(
        transitions=[
            transition(1.0, True),
            transition(0.0, False),
            transition(2.0, True),
            transition(3.0, True),
        ],
        n_envs=1,
        bootstrap=[0.0],
        observations=[observation],
        closes=0,
        planning_bootstrap=({0: 10.0},),
    )
    config = PerModelPPOConfig(planning_gamma=0.5, gae_lambda=1.0)
    returns, _ = compute_planning_gae(rollout, config)
    assert returns[3].item() == pytest.approx(3.0 + 0.5 * 10.0)
    assert returns[2].item() == pytest.approx(2.0 + 0.5 * returns[3].item())
    assert returns[0].item() == pytest.approx(1.0 + 0.5 * returns[2].item())
    assert returns[1].item() == 0.0
    # An episode that ended inside the last span bootstraps from nothing.
    rollout.transitions[3] = transition(3.0, True, done=True)
    returns, _ = compute_planning_gae(rollout, config)
    assert returns[3].item() == pytest.approx(3.0)


def test_an_update_with_the_planning_stream_moves_the_commitment_head() -> None:
    torch.manual_seed(2)
    envs, retimers, observations = _head_envs()
    agent = SetAgent(SetNetwork.from_env(envs[0], SMALL_TRUNK))
    config = PerModelPPOConfig(rollout_rounds=2, batch_size=16, n_epochs=2)
    rollout = collect_rollout(envs, agent, retimers, observations, config)
    returns, advantages = compute_gae(rollout, config)
    planning_returns, planning_advantages = compute_planning_gae(rollout, config)
    network = agent.network
    head_before = _weight(network.commit_keep_head).detach().clone()
    value_before = _weight(network.planning_value_head[0]).detach().clone()
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
    assert not torch.equal(head_before, _weight(network.commit_keep_head))
    assert not torch.equal(value_before, _weight(network.planning_value_head[0]))
    assert np.isfinite(stats.planning_explained_variance)

    # Without commitment rows the commitment head receives no gradient.
    plain = [PerModelEnv(_greedy_config()) for _ in range(2)]
    plain_obs = [env.reset(seed=20 + i)[0] for i, env in enumerate(plain)]
    plain_pay = [PerStepReward(env) for env in plain]
    for pay in plain_pay:
        pay.reset()
    rollout = collect_rollout(plain, agent, plain_pay, plain_obs, config)
    returns, advantages = compute_gae(rollout, config)
    head_before = _weight(network.commit_keep_head).detach().clone()
    stats = ppo_update(
        network,
        optimizer,
        rollout,
        returns,
        advantages,
        config,
        generator=torch.Generator().manual_seed(0),
    )
    assert stats.commit_rows == 0
    assert torch.equal(head_before, _weight(network.commit_keep_head))


def test_a_checkpoint_without_the_stage1_heads_loads_with_fresh_ones(
    tmp_path: Path,
) -> None:
    env = PerModelEnv(small_config())
    network = SetNetwork.from_env(env, SMALL_TRUNK)
    path = tmp_path / "old.pt"
    save_checkpoint(
        path,
        network,
        ppo_config=PerModelPPOConfig(),
        env_config=env.config.model_dump(mode="json"),
        rounds=0,
        seed=0,
        revision="test",
    )
    payload = torch.load(path, weights_only=True)
    stripped = {
        k: v
        for k, v in payload["state_dict"].items()
        if not k.startswith(("commit_", "planning_value_head."))
    }
    assert len(stripped) < len(payload["state_dict"])
    payload["state_dict"] = stripped
    torch.save(payload, path)
    loaded = load_checkpoint(path)
    assert torch.equal(
        loaded.network.displacement_head.weight, network.displacement_head.weight
    )
    # Any other absent key is still refused.
    del payload["state_dict"]["displacement_head.weight"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        load_checkpoint(path)


def test_the_scripted_bar_writes_its_plan_under_the_head_writer() -> None:
    from scripts.scenario_overrides import load_env_config
    from wargame_rl.wargame.scoring import evaluate_spec
    from wargame_rl.wargame.selectors import build_per_model_chooser

    config = load_env_config("configs/experiments/curriculum/a3_head.yaml")
    config.render_mode = None
    env = PerModelEnv(config)
    observation, _ = env.reset(seed=700000)
    chooser = build_per_model_chooser("squad_march_take", [env], seed=0)
    for _ in range(6):
        observation, _, terminated, _, _ = env.step(
            chooser.choose([env], [observation])[0]
        )
        if terminated:
            break
    state = env.player_commitments
    assert all(state.ground_of(g) >= 0 for g in env.player_seat.living_units())
    result = evaluate_spec("squad_march_take", config, [700000, 700001], "bar")
    assert result.success_rate == 1.0
