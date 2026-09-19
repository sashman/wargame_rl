"""Behaviour cloning into the set network (#331): a scripted chooser's decisions
recorded as the transitions the PPO update reads, fitted by maximum likelihood.

The clone is the instrument the curriculum's D rungs measure with: D1 asks
whether the set network can *hold* an ordered plan the reward alone did not
teach (C3 / C3b), D2 whether PPO can improve a policy that starts with one.

Two things make this a per-model clone rather than the whole-army one in
`scripts/behaviour_clone.py`. The demonstration is a **decision** -- which
model acts and which column of which head it takes -- so the target is the
pair the selector and one head draw, and the loss is the joint log-prob
`evaluate_transitions` already computes for PPO. And the script's action has
to be mapped *back* into those columns: `action_to_column` is the inverse of
`SetAgent._decode`, and the recorder checks the round trip on every step, so
a map that drifted from the decoder would fail loudly rather than teach the
wrong column.

⚠ The critic is not fitted here; a clone's value head is at initialisation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wargame_rl.wargame.envs.domain.sequencing.activation import CHARGE_TARGET_DECLINE
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.seat import Seat
from wargame_rl.wargame.envs.per_model.tokens import (
    Head,
    TokenObservation,
    TokenScenario,
    build_tokens,
)
from wargame_rl.wargame.envs.per_model.types import PerModelAction, StepKind
from wargame_rl.wargame.model.per_model.agent import NO_DRAW, SetAgent
from wargame_rl.wargame.model.per_model.batch import collate
from wargame_rl.wargame.model.per_model.checkpoint import save_checkpoint
from wargame_rl.wargame.model.per_model.net import SetNetwork
from wargame_rl.wargame.model.per_model.ppo import (
    PerModelPPOConfig,
    Rollout,
    Transition,
    compute_gae,
    evaluate_transitions,
)

# Disjoint from every other seed band: evaluation 700000+, in-run 500000+,
# baselines 10000+, tuning 900000+. The whole-army clone uses the same band.
CLONE_SEED_BASE = 800_000

BatchChooser = Any  # (envs, observations) -> list[PerModelAction]


def _unit_column(group: int, groups: np.ndarray) -> int:
    hits = np.flatnonzero(groups == group)
    if hits.size != 1:
        raise ValueError(f"enemy unit {group} is not a unit-pointer column")
    return int(hits[0]) + 1


def action_to_column(
    action: PerModelAction, tokens: TokenObservation, seat: Seat
) -> int:
    """The head column that names `action` at this decision: the inverse of
    `SetAgent._decode`."""
    kind = action.kind
    if kind is StepKind.open:
        return int(action.value)
    if kind is StepKind.target:
        if action.value == CHARGE_TARGET_DECLINE:
            return 0
        return _unit_column(int(action.value), tokens.opponent_unit_groups)
    if tokens.head is Head.unit_pointer:
        if action.value == STAY_ACTION:
            return 0
        shooting = seat.handler.shooting_slice
        if shooting is None:
            raise ValueError("no shooting slice to read a target from")
        return _unit_column(
            int(action.value) - shooting.start, tokens.opponent_unit_groups
        )
    if action.value == STAY_ACTION:
        return 0
    movement = seat.handler.movement_slice
    if movement.start <= action.value < movement.end:
        return int(action.value) - movement.start + 1
    advance = seat.handler.advance_slice
    if advance is None or not (advance.start <= action.value < advance.end):
        raise ValueError(f"action {action.value} names no displacement column")
    return int(action.value) - advance.start + 1 + movement.size


def compact_tokens(tokens: TokenObservation) -> TokenObservation:
    """The token observation with its float arrays at half precision.

    A recorded decision is dominated by the (P, P, 16) relation block -- on a
    24-body army about 37 KB of the ~50 KB a step costs -- and 1,200 games of
    A5 held 16 GB of float32 demonstrations. `collate` copies every array into
    float32 buffers, so the fit reads them back widened; three decimal digits
    is more than a quantised move column needs.
    """
    return replace(
        tokens,
        players=tokens.players.astype(np.float16),
        context=tokens.context.astype(np.float16),
        self_relations=tokens.self_relations.astype(np.float16),
        cross_relations=tokens.cross_relations.astype(np.float16),
    )


def record_demonstrations(
    env: PerModelEnv,
    choose: BatchChooser,
    n_episodes: int,
    *,
    seed_base: int = CLONE_SEED_BASE,
) -> list[Transition]:
    """Play `choose` on the player seat for `n_episodes` and record every
    decision step as a `Transition` with the teacher's model and column.

    Closing steps are recorded too, without a policy (`column == NO_DRAW`),
    because the re-timed reward pays the round's state terms at the close:
    the reward, `done` and `is_close` on every row are what `value_targets`
    turns into discounted returns for a critic fit. `env_index` holds the
    episode index, so a held-out split can be made by episode. Every
    recorded policy pair is decoded back through `SetAgent._decode` and
    compared with the teacher's action, so the inverse map cannot silently
    drift.
    """
    seat = env.player_seat
    retimer = PerStepReward(env)
    transitions: list[Transition] = []
    for episode in range(n_episodes):
        observation, _ = env.reset(seed=seed_base + episode)
        retimer.reset()
        scenario = TokenScenario.for_episode(env, seat)
        done = False
        while not done:
            point = observation.decision
            tokens = build_tokens(env, seat, observation, scenario)
            if point.kind is StepKind.close_turn:
                action = PerModelAction.close_turn()
                model = column = NO_DRAW
            else:
                action = choose([env], [observation])[0]
                column = action_to_column(action, tokens, seat)
                model = int(action.model)
                decoded = SetAgent._decode(point.kind, tokens, seat, model, column)
                if decoded != action:
                    raise RuntimeError(
                        f"the inverse map does not round-trip: {action} -> column "
                        f"{column} -> {decoded}"
                    )
            before = observation
            observation, _r, done, _t, info = env.step(action)
            payment = retimer.on_step(before, action, info["effect"], done)
            transitions.append(
                Transition(
                    tokens=compact_tokens(tokens),
                    model=model,
                    column=column,
                    head=tokens.head,
                    phase=point.phase,
                    value=0.0,
                    log_prob=0.0,
                    reward=payment.reward,
                    done=done,
                    is_close=payment.is_close,
                    env_index=episode,
                )
            )
    return transitions


def value_targets(
    transitions: Sequence[Transition], gamma: float = 0.9
) -> torch.Tensor:
    """The discounted return from every recorded row to the end of its
    episode, in the transitions' order: `compute_gae` at `gae_lambda` 1.0 on
    a rollout whose envs are the episodes and whose bootstrap is zero."""
    n_episodes = max(t.env_index for t in transitions) + 1
    rollout = Rollout(
        transitions=list(transitions),
        n_envs=n_episodes,
        bootstrap=[0.0] * n_episodes,
        observations=[],
        closes=sum(t.is_close for t in transitions),
        episodes=[],
        breakdown={},
    )
    returns, _ = compute_gae(rollout, PerModelPPOConfig(gamma=gamma, gae_lambda=1.0))
    return returns


def fit_critic(
    network: SetNetwork,
    transitions: Sequence[Transition],
    targets: torch.Tensor,
    *,
    epochs: int,
    seed: int,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> list[float]:
    """Fit ONLY the value head to `targets` by MSE, the trunk and the policy
    heads frozen, so the policy the clone plays is bit-identical before and
    after. Returns the mean loss per epoch. The whole-phase record: PPO from
    a clone with a cold critic destroys it; this is the direct test's
    instrument on the per-model facade."""
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    for parameter in network.value_head.parameters():
        parameter.requires_grad_(True)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(network.value_head.parameters(), lr=lr)
    device = network.device
    losses: list[float] = []
    try:
        for _ in range(epochs):
            network.train()
            order = torch.randperm(len(transitions), generator=generator).tolist()
            total = 0.0
            for start in range(0, len(order), batch_size):
                rows = order[start : start + batch_size]
                evaluated = evaluate_transitions(
                    network, [transitions[i] for i in rows]
                )
                loss = torch.nn.functional.mse_loss(
                    evaluated.values, targets[rows].to(device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(rows)
            losses.append(total / max(1, len(order)))
    finally:
        for parameter in network.parameters():
            parameter.requires_grad_(True)
        network.eval()
    return losses


@torch.no_grad()
def explained_variance(
    network: SetNetwork, transitions: Sequence[Transition], targets: torch.Tensor
) -> float:
    """1 - Var(target - value) / Var(target) over `transitions`."""
    network.eval()
    values = torch.cat(
        [
            evaluate_transitions(network, list(transitions[i : i + 256])).values.cpu()
            for i in range(0, len(transitions), 256)
        ]
    )
    variance = float(targets.var().item())
    if variance == 0.0:
        return 0.0
    return 1.0 - float((targets - values).var().item()) / variance


def fit_clone(
    network: SetNetwork,
    transitions: Sequence[Transition],
    *,
    epochs: int,
    seed: int,
    batch_size: int = 64,
    lr: float = 3e-4,
    held_out: Sequence[Transition] | None = None,
    log: Any = None,
) -> list[float]:
    """Maximise the joint log-prob of the teacher's decisions; returns the mean
    negative log-prob per epoch. `held_out` is scored every epoch when given."""
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    losses: list[float] = []
    for epoch in range(epochs):
        network.train()
        order = torch.randperm(len(transitions), generator=generator).tolist()
        total = 0.0
        count = 0
        for start in range(0, len(order), batch_size):
            batch = [transitions[i] for i in order[start : start + batch_size]]
            evaluated = evaluate_transitions(network, batch)
            loss = -evaluated.log_probs[evaluated.has_policy].mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(batch)
            count += len(batch)
        losses.append(total / max(count, 1))
        if log is not None:
            report = match_report(network, held_out) if held_out else {}
            log(epoch + 1, losses[-1], report)
    network.eval()
    return losses


@torch.no_grad()
def match_report(
    network: SetNetwork, transitions: Sequence[Transition], batch_size: int = 128
) -> dict[str, float]:
    """Greedy agreement with the teacher on `transitions`: the selector's model,
    each head's column given the teacher's model, and the joint decision."""
    network.eval()
    device = network.device
    selector_hits = joint_hits = 0
    head_hits: dict[Head, int] = {}
    head_counts: dict[Head, int] = {}
    for start in range(0, len(transitions), batch_size):
        chunk = list(transitions[start : start + batch_size])
        batch = collate([t.tokens for t in chunk], device=device)
        output = network(batch)
        models = torch.tensor(
            [max(t.model, 0) for t in chunk], dtype=torch.int64, device=device
        )
        chosen_models = output.selector_logits.float().argmax(dim=-1)
        heads = network.heads(output, batch, models)
        for row, t in enumerate(chunk):
            if t.column == NO_DRAW:
                continue
            logits = {
                Head.declaration: heads.declaration,
                Head.displacement: heads.displacement,
                Head.unit_pointer: heads.unit,
            }[t.head][row]
            column_hit = int(logits.float().argmax().item()) == t.column
            model_hit = int(chosen_models[row].item()) == t.model
            selector_hits += model_hit
            joint_hits += model_hit and column_hit
            head_hits[t.head] = head_hits.get(t.head, 0) + int(column_hit)
            head_counts[t.head] = head_counts.get(t.head, 0) + 1
    n = max(1, sum(head_counts.values()))
    report = {"selector": selector_hits / n, "joint": joint_hits / n}
    for head, count in head_counts.items():
        report[head.name] = head_hits[head] / count
    return report


def save_clone(
    path: Path,
    network: SetNetwork,
    *,
    env_config: dict[str, Any],
    seed: int,
    revision: str,
    teacher: str,
    n_episodes: int,
    epochs: int,
    match: dict[str, float],
) -> None:
    """Write the clone as a per-model checkpoint at zero rounds, with its
    provenance beside it in `<path>.clone.json`."""
    save_checkpoint(
        path,
        network,
        ppo_config=PerModelPPOConfig(),
        env_config=env_config,
        rounds=0,
        seed=seed,
        revision=revision,
    )
    provenance = {
        "teacher": teacher,
        "n_episodes": n_episodes,
        "epochs": epochs,
        "seed": seed,
        "seed_base": CLONE_SEED_BASE,
        "held_out_match": match,
        "revision": revision,
    }
    path.with_suffix(".clone.json").write_text(json.dumps(provenance, indent=2))


__all__ = [
    "CLONE_SEED_BASE",
    "action_to_column",
    "explained_variance",
    "fit_clone",
    "fit_critic",
    "match_report",
    "record_demonstrations",
    "save_clone",
    "value_targets",
]
