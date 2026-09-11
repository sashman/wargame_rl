"""Greedy, seeded evaluation of a set network over the per-model facade.

A thin client of `envs/per_model/evaluate.py`: the agent's `act_batch` IS a
batch chooser, so the runner in the env layer does the stepping, the reading
and the aggregation, and this module only flips the agent into greedy eval
mode around it. The result is the `EvalResult` every scoring script prints.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from wargame_rl.wargame.envs.evaluation import EvalResult
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.evaluate import evaluate_per_model_chooser
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.types import PerModelAction, PerModelObservation
from wargame_rl.wargame.model.per_model.agent import SetAgent


def evaluate_per_model(
    envs: Sequence[PerModelEnv],
    agent: SetAgent,
    retimers: Sequence[PerStepReward],
    seeds: Sequence[int],
    *,
    combat_seeds: Sequence[int] | None = None,
    name: str = "set_network",
) -> EvalResult:
    """One greedy episode per seed, in seed order, over waves of `envs`."""
    was_greedy = agent.greedy
    was_training = agent.network.training
    agent.greedy = True
    agent.network.eval()

    def choose(
        envs_: Sequence[PerModelEnv], observations: Sequence[PerModelObservation]
    ) -> list[PerModelAction]:
        return [decision.action for decision in agent.act_batch(envs_, observations)]

    try:
        with torch.no_grad():
            return evaluate_per_model_chooser(
                choose,
                envs,
                seeds,
                name,
                combat_seeds=combat_seeds,
                retimers=retimers,
            )
    finally:
        agent.greedy = was_greedy
        agent.network.train(was_training)


__all__ = ["EvalResult", "evaluate_per_model"]
