"""Greedy, seeded evaluation of a set network over the per-model facade.

A thin client of `envs/per_model/evaluate.py`: the agent's `act_batch` IS a
batch chooser, so the runner in the env layer does the stepping, the reading
and the aggregation, and this module only flips the agent into greedy eval
mode around it. The result is the `EvalResult` every scoring script prints.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.evaluation import EvalResult
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.evaluate import evaluate_per_model_chooser
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.scripted import ScriptedSeat
from wargame_rl.wargame.envs.per_model.types import (
    NO_COMMIT_DECISION,
    BatchChooser,
    PerModelAction,
    PerModelObservation,
    StepKind,
)
from wargame_rl.wargame.model.per_model.agent import SetAgent


def plan_only_chooser(
    agent: SetAgent,
    envs: Sequence[PerModelEnv],
    members: str,
    *,
    generator: torch.Generator | None = None,
) -> BatchChooser:
    """`agent`'s HEAD plans, the scripted `members` policy walks (#384).

    Installs a non-emitting scripted seat over `members` (normally
    `squad_march_committed`, which reads the state) on every env, and answers
    each decision with the network's commitment draw where one is offered and
    the script's choice everywhere else -- the unit's declaration on that same
    open step, every act, every target -- re-planning the script before each
    decision so a commitment written this turn is the plan it executes.
    """
    for env in envs:
        if not env.config.commitments.policy_writes:
            raise ValueError(
                "the plan-only chooser needs `commitments.assignment: head`; "
                "nothing else offers the head a decision"
            )
        env.set_player_planner(
            ScriptedSeat.for_policy(build_baseline_policy(members), emits=False)
        )

    def choose(
        envs_: Sequence[PerModelEnv], observations: Sequence[PerModelObservation]
    ) -> list[PerModelAction]:
        with torch.no_grad():
            decisions = agent.act_batch(envs_, observations, generator=generator)
        actions: list[PerModelAction] = []
        for env, observation, decision in zip(envs_, observations, decisions):
            point = observation.decision
            if point.kind is StepKind.close_turn:
                actions.append(PerModelAction.close_turn())
                continue
            seat = env.player_seat
            planner = seat.adapter
            if planner is None:
                raise RuntimeError("the env has no scripted seat installed")
            if point.phase is not None:
                planner.plan(point.phase, seat, env)
            commitment = decision.action.commitment
            if point.kind is StepKind.open and commitment != NO_COMMIT_DECISION:
                model = decision.action.model
                actions.append(
                    PerModelAction.open(
                        model, planner.declaration_for(point, model, seat), commitment
                    )
                )
                continue
            actions.append(planner.choose(point, seat))
        return actions

    return choose


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


__all__ = [
    "EvalResult",
    "evaluate_per_model",
    "plan_only_chooser",
]
