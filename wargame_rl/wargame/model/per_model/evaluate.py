"""Scoring a per-model checkpoint with the same instruments as everything else.

`evaluate_per_model` drives one env sequentially; `evaluate_per_model_batched`
is the per-model form of batched evaluation — the whole-phase
`_run_episodes_batched` runs lockstep waves because every episode is exactly
``max_turns`` steps, and that assumption **breaks** under the per-model step,
where steps per phase vary with the alive count. Here waves run until the
last env finishes, with finished envs simply absent from the batch.

Both aggregate through `episode_metrics` / `aggregate_result` — the exact
arithmetic `evaluate_selector` uses — so a per-model score is directly
comparable to the scripted bar measured through the whole-phase facade, which
the bridge test ties to this one bit-for-bit.
"""

from __future__ import annotations

import torch

from wargame_rl.wargame.envs.baseline.evaluate import (
    BaselineResult,
    EpisodeMetrics,
    aggregate_result,
    episode_metrics,
)
from wargame_rl.wargame.envs.per_model.facade import PerModelEnv
from wargame_rl.wargame.envs.per_model.types import PerModelObservation
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.per_model.agent import SetAgent
from wargame_rl.wargame.model.per_model.net import SetNetwork


def evaluate_per_model(
    env: PerModelEnv,
    agent: SetAgent,
    seeds: list[int],
    name: str,
    combat_seeds: list[int] | None = None,
) -> BaselineResult:
    """Greedily play one episode per seed and aggregate the outcome.

    The per-model counterpart of `evaluate_selector`: identical seeds give
    identical layouts, and the metrics come from the same extraction.
    """
    if combat_seeds is not None and len(combat_seeds) != len(seeds):
        raise ValueError(
            f"combat_seeds must match seeds in length: "
            f"{len(combat_seeds)} != {len(seeds)}"
        )
    metrics: list[EpisodeMetrics] = []
    for index, seed in enumerate(seeds):
        options = None if combat_seeds is None else {"combat_seed": combat_seeds[index]}
        observation, _ = env.reset(seed=seed, options=options)
        terminated = False
        while not terminated:
            decision = agent.act(env, observation, greedy=True)
            observation, _reward, terminated, _truncated, _ = env.step(decision.action)
        metrics.append(episode_metrics(env))
    return aggregate_result(name, metrics)


def evaluate_per_model_batched(
    config: WargameEnvConfig,
    network: SetNetwork,
    seeds: list[int],
    name: str,
    n_parallel: int = 8,
    combat_seeds: list[int] | None = None,
    device: torch.device | None = None,
) -> BaselineResult:
    """The batched form: one forward pass decides a step for every env in flight.

    Episodes finish at different step counts, so a wave holds whichever envs
    still owe a decision; a finished env picks up the next unplayed seed or
    drops out. Metrics land in seed order, so the result pairs against any
    other run of the same seeds.
    """
    if combat_seeds is not None and len(combat_seeds) != len(seeds):
        raise ValueError(
            f"combat_seeds must match seeds in length: "
            f"{len(combat_seeds)} != {len(seeds)}"
        )
    agent = SetAgent(network, device=device)
    n_envs = min(n_parallel, len(seeds))
    envs = [PerModelEnv(config, build_info=False) for _ in range(n_envs)]
    metrics: list[EpisodeMetrics | None] = [None] * len(seeds)

    next_seed = 0
    active: list[tuple[PerModelEnv, PerModelObservation, int]] = []

    def start(env: PerModelEnv) -> tuple[PerModelEnv, PerModelObservation, int] | None:
        nonlocal next_seed
        if next_seed >= len(seeds):
            return None
        index = next_seed
        next_seed += 1
        options = None if combat_seeds is None else {"combat_seed": combat_seeds[index]}
        observation, _ = env.reset(seed=seeds[index], options=options)
        return (env, observation, index)

    for env in envs:
        started = start(env)
        if started is not None:
            active.append(started)

    while active:
        actions = agent.act_batched(
            [(env, observation) for env, observation, _index in active]
        )
        survivors: list[tuple[PerModelEnv, PerModelObservation, int]] = []
        for (env, _observation, seed_index), action in zip(
            active, actions, strict=True
        ):
            observation, _reward, terminated, _truncated, _ = env.step(action)
            if not terminated:
                survivors.append((env, observation, seed_index))
                continue
            metrics[seed_index] = episode_metrics(env)
            restarted = start(env)
            if restarted is not None:
                survivors.append(restarted)
        active = survivors

    finished = [m for m in metrics if m is not None]
    assert len(finished) == len(seeds)
    return aggregate_result(name, finished)
