"""The set agent: a network-driven episode is legal from the first decision to the last.

The agent raises on any decoded decision the point refuses, so an episode
that terminates is the proof the token masks, the heads and the decoder
agree at every step the network happened to take. The exhaustive case goes
further: every column every head offers, decoded, is a decision the point
accepts -- including the fight step, where column 0 must be absent, and the
target step, where column 0 is the decline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.per_model_seats import shooting_charging_driver, small_config
from wargame_rl.wargame.envs.domain.sequencing.activation import CHARGE_TARGET_DECLINE
from wargame_rl.wargame.envs.per_model import PerModelEnv, StepKind
from wargame_rl.wargame.envs.per_model.tokens import Head
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config.terrain import RandomTerrainConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.per_model import SetAgent, SetNetwork, SetNetworkConfig
from wargame_rl.wargame.model.per_model.agent import _head_logits

SMALL_TRUNK = SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)


def _agent(env: PerModelEnv, *, greedy: bool = False, seed: int = 0) -> SetAgent:
    torch.manual_seed(seed)
    network = SetNetwork.from_env(env, SMALL_TRUNK)
    network.eval()
    return SetAgent(network, greedy=greedy)


def _play(env: PerModelEnv, agent: SetAgent, seed: int) -> list[str]:
    generator = torch.Generator().manual_seed(seed)
    observation, _ = env.reset(seed=seed)
    trace = []
    done = False
    while not done:
        decision = agent.act(env, observation, generator=generator)
        trace.append(repr(decision.action))
        observation, _r, done, _t, _i = env.step(decision.action)
    return trace


@pytest.mark.parametrize("melee", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
def test_a_network_driven_episode_terminates_legally(melee: bool, greedy: bool) -> None:
    env = PerModelEnv(
        small_config(melee=melee, opponent_x=22 if melee else 56, rounds=2)
    )
    agent = _agent(env, greedy=greedy)
    trace = _play(env, agent, seed=1)
    assert env.current_turn == env.max_turns
    assert len(trace) == env.episode_step
    opens = sum(1 for action in trace if "kind=<StepKind.open" in action)
    assert sum(agent.declaration_counts.values()) == opens > 0


def test_seeded_sampling_is_reproducible() -> None:
    config = small_config(rounds=1)
    first = _play(PerModelEnv(config), _agent(PerModelEnv(config)), seed=4)
    second = _play(PerModelEnv(config), _agent(PerModelEnv(config)), seed=4)
    assert first == second
    assert len(first) > 3


def test_every_column_every_head_offers_decodes_to_a_legal_decision() -> None:
    """One decision point per (kind, phase) the charging driver reaches; every
    finite column of the active head, decoded, is accepted by the point."""
    env = PerModelEnv(small_config(melee=True, opponent_x=22, rounds=2))
    agent = _agent(env)
    observation, _ = env.reset(seed=5)
    seen: dict[tuple[StepKind, BattlePhase | None], int] = {}
    done = False
    while not done:
        point = observation.decision
        key = (point.kind, point.phase)
        if key not in seen and point.kind is not StepKind.close_turn:
            seen[key] = _check_every_column(env, agent, observation)
        observation, _r, done, _t, _i = env.step(shooting_charging_driver(env, point))
    assert (StepKind.target, BattlePhase.charge) in seen
    assert (StepKind.act, BattlePhase.fight) in seen
    assert (StepKind.act, BattlePhase.shooting) in seen
    assert all(count > 0 for count in seen.values())


def _check_every_column(env: PerModelEnv, agent: SetAgent, observation: object) -> int:
    from wargame_rl.wargame.model.per_model.batch import collate

    point = observation.decision  # type: ignore[attr-defined]
    tokens = agent.observe(env, observation)  # type: ignore[arg-type]
    batch = collate([tokens])
    output = agent.network(batch)
    checked = 0
    for model in np.flatnonzero(point.selector_mask):
        heads = agent.network.heads(output, batch, torch.tensor([int(model)]))
        logits = _head_logits(heads, tokens.head)[0]
        for column in np.flatnonzero(torch.isfinite(logits).numpy()):
            action = SetAgent._decode(
                point.kind, tokens, env.player_seat, int(model), int(column)
            )
            assert point.why_illegal(action) is None, point.why_illegal(action)
            if point.kind is StepKind.target and column == 0:
                assert action.value == CHARGE_TARGET_DECLINE
            checked += 1
        if point.phase is BattlePhase.fight and tokens.head is Head.unit_pointer:
            assert not torch.isfinite(logits[0]), "a striker cannot decline"
    return checked


def test_the_scenario_cache_is_rebuilt_for_a_new_episode() -> None:
    config = WargameEnvConfig(
        **{
            **small_config(rounds=1).model_dump(),
            "random_terrain": RandomTerrainConfig(
                count=3, min_size=4, max_size=6, mirror=False
            ).model_dump(),
        }
    )
    env = PerModelEnv(config)
    agent = _agent(env)
    observation, _ = env.reset(seed=1)
    agent.act(env, observation)
    first = agent.scenario_for(env, env.player_seat)
    observation, _ = env.reset(seed=2)
    agent.act(env, observation)
    second = agent.scenario_for(env, env.player_seat)
    assert first.episode_id + 1 == second.episode_id == env.episode_id
    assert not np.array_equal(first.terrain_rows, second.terrain_rows)


def test_the_closing_step_needs_no_head_and_carries_a_value() -> None:
    env = PerModelEnv(small_config(rounds=1))
    agent = _agent(env, greedy=True)
    observation, _ = env.reset(seed=1)
    done = False
    closes = 0
    while not done:
        decision = agent.act(env, observation)
        if observation.decision.kind is StepKind.close_turn:
            closes += 1
            assert decision.log_prob == 0.0
            assert np.isfinite(decision.value)
        observation, _r, done, _t, _i = env.step(decision.action)
    assert closes == 1
