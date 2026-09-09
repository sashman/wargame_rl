"""Single-action PPO over per-model decision steps (issue #286).

One decision is one transition with one scalar reward, one value and one
importance ratio -- ordinary PPO. What is not ordinary is the clock:

- The discount and the GAE decay apply only across a CLOSING step (a
  `close_turn`, or any step that terminates the episode). Every decision
  within a turn is the same instant, so `gamma` counts rounds, not steps,
  and the horizon does not drift as models die or shrink with the army.
  A per-step gamma was rejected in the #283 design for exactly that.
- The rollout budget is in ROUNDS: `rollout_rounds` closing steps per env.
  Envs run in lockstep -- each always has one pending decision -- so one
  batched forward serves them all, and they are never reset between
  rollouts: a budget shorter than the episode must still visit its later
  rounds.
- A closing step has no policy factor. It enters the value loss only, and
  is excluded from the policy loss, the entropies, the ratio statistics and
  the advantage normalisation.

⚠ `gamma` 0.9 and `gae_lambda` 0.95 are the phase facade's values, measured
at TWO steps per round; the rollout budget is a first guess. All three are
#288's to calibrate in the new unit before any race number is quoted.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, model_validator

from wargame_rl.wargame.envs.per_model.env import PerModelEnv, StepEffect
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.tokens import Head, TokenObservation
from wargame_rl.wargame.envs.per_model.types import PerModelObservation, StepKind
from wargame_rl.wargame.envs.types import BattlePhase
from wargame_rl.wargame.model.per_model.agent import NO_DRAW, SetAgent
from wargame_rl.wargame.model.per_model.batch import collate
from wargame_rl.wargame.model.per_model.net import SetNetwork

# Log-probs are clamped here before `p * log p`: the naive
# `where(finite, p * log_p, 0)` leaks NaN through the unselected branch's
# local gradient, since autograd differentiates both branches.
_LOG_PROB_FLOOR = -30.0
# The training-time start-state augmentation, opt-in per reset. The shipped
# collector passes it; an eval never does.
_AUGMENT_START = {"augment_start": True}


class PerModelPPOConfig(BaseModel):
    """PPO over decision steps, with time constants in rounds."""

    gamma: float = 0.9
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    vf_coef: float = 0.3
    ent_coef: float = 0.03
    # The selector's own coefficient; None means `ent_coef`. Two knobs so the
    # order can be kept exploring while the action sharpens (#283).
    selector_ent_coef: float | None = None
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    batch_size: int = 128
    # Closing steps per env per update.
    rollout_rounds: int = 16
    # 0 auto-detects: 8 on a usable GPU, 4 on the CPU, clamped by affinity.
    num_rollout_envs: int = 0

    @model_validator(mode="after")
    def _positive(self) -> PerModelPPOConfig:
        if self.rollout_rounds < 1:
            raise ValueError("rollout_rounds must be at least 1")
        if self.batch_size < 1 or self.n_epochs < 1:
            raise ValueError("batch_size and n_epochs must be at least 1")
        return self

    @property
    def resolved_selector_ent_coef(self) -> float:
        return (
            self.ent_coef if self.selector_ent_coef is None else self.selector_ent_coef
        )


def auto_num_rollout_envs(device: torch.device) -> int:
    """The shipped loop's heuristic: 8 on CUDA, 4 on CPU, clamped by affinity."""
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 1
    max_envs = 8 if device.type == "cuda" else 4
    return max(1, min(max_envs, cpu_count))


@dataclass(frozen=True)
class Transition:
    """One decision step, as the update reads it."""

    tokens: TokenObservation
    model: int
    column: int
    head: Head
    phase: BattlePhase | None
    value: float
    log_prob: float
    reward: float
    done: bool
    is_close: bool
    env_index: int

    @property
    def has_policy(self) -> bool:
        return self.column >= 0


@dataclass(frozen=True)
class EpisodeOutcome:
    """A rollout episode that finished, read the moment it did."""

    env_index: int
    player_vp: int
    opponent_vp: int
    reward: float
    decision_steps: int
    rounds: int


@dataclass
class Rollout:
    """What `collect_rollout` returns: time-major transitions and the rest."""

    transitions: list[Transition]
    n_envs: int
    bootstrap: list[float]
    observations: list[PerModelObservation]
    closes: int
    episodes: list[EpisodeOutcome] = field(default_factory=list)
    # Per-term reward paid over the rollout, summed across envs.
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.transitions)

    @property
    def steps_per_env(self) -> int:
        return len(self.transitions) // max(1, self.n_envs)


OnEpisodeEnd = Callable[[int, PerModelEnv, PerStepReward], None]


def collect_rollout(
    envs: Sequence[PerModelEnv],
    agent: SetAgent,
    retimers: Sequence[PerStepReward],
    observations: Sequence[PerModelObservation],
    config: PerModelPPOConfig,
    *,
    generator: torch.Generator | None = None,
    on_episode_end: OnEpisodeEnd | None = None,
) -> Rollout:
    """Play every env in lockstep until `rollout_rounds x n_envs` turns close.

    `observations` are the envs' pending decisions -- the ones the previous
    rollout returned, or the reset observations for the first. An env that
    terminates is reset inline (with the start augmentation, as the shipped
    collector does) after `on_episode_end` has read its outcome.
    """
    if not (len(envs) == len(retimers) == len(observations)):
        raise ValueError("one retimer and one observation per env")
    n_envs = len(envs)
    budget = config.rollout_rounds * n_envs
    current = list(observations)
    transitions: list[Transition] = []
    episodes: list[EpisodeOutcome] = []
    breakdown: dict[str, float] = {}
    closes = 0
    decision_counts = [0] * n_envs
    round_counts = [0] * n_envs
    last_done = [False] * n_envs
    while closes < budget:
        decisions = agent.act_batch(envs, current, generator=generator)
        for index, (env, retimer, decision) in enumerate(
            zip(envs, retimers, decisions)
        ):
            before = current[index]
            observation, _, terminated, _, info = env.step(decision.action)
            effect: StepEffect = info["effect"]
            payment = retimer.on_step(before, decision.action, effect, terminated)
            for key, value in payment.breakdown.items():
                breakdown[key] = breakdown.get(key, 0.0) + value
            if decision.has_policy:
                decision_counts[index] += 1
            if payment.is_close:
                closes += 1
                round_counts[index] += 1
            transitions.append(
                Transition(
                    tokens=decision.tokens,
                    model=decision.model,
                    column=decision.column,
                    head=decision.tokens.head,
                    phase=before.decision.phase,
                    value=decision.value,
                    log_prob=decision.log_prob,
                    reward=payment.reward,
                    done=terminated,
                    is_close=payment.is_close,
                    env_index=index,
                )
            )
            last_done[index] = terminated
            if terminated:
                episodes.append(
                    EpisodeOutcome(
                        env_index=index,
                        player_vp=int(env.player_vp),
                        opponent_vp=int(env.opponent_vp),
                        reward=retimer.episode_reward,
                        decision_steps=decision_counts[index],
                        rounds=round_counts[index],
                    )
                )
                if on_episode_end is not None:
                    on_episode_end(index, env, retimer)
                observation, _ = env.reset(options=dict(_AUGMENT_START))
                retimer.reset()
                decision_counts[index] = 0
                round_counts[index] = 0
            current[index] = observation
    bootstrap = [0.0] * n_envs
    live = [i for i in range(n_envs) if not last_done[i]]
    if live:
        values = _values(agent, [envs[i] for i in live], [current[i] for i in live])
        for slot, index in enumerate(live):
            bootstrap[index] = values[slot]
    return Rollout(
        transitions=transitions,
        n_envs=n_envs,
        bootstrap=bootstrap,
        observations=current,
        closes=closes,
        episodes=episodes,
        breakdown=breakdown,
    )


def _values(
    agent: SetAgent,
    envs: Sequence[PerModelEnv],
    observations: Sequence[PerModelObservation],
) -> list[float]:
    tokens = [agent.observe(env, obs) for env, obs in zip(envs, observations)]
    batch = collate(tokens, device=agent.network.device)
    with torch.no_grad():
        output = agent.network(batch)
    return [float(v) for v in output.value.float().cpu().tolist()]


def compute_gae(
    rollout: Rollout, config: PerModelPPOConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns and advantages, flat in the rollout's time-major order.

    Per env, backwards. The discount and decay between step t and t+1 are
    `gamma` / `gae_lambda` when t is a closing step -- a round boundary lies
    between them -- and 1.0 otherwise: every decision within a turn is the
    same instant. A rollout cut mid-turn bootstraps `r + V(next)` at 1.0.
    """
    n = rollout.n_steps
    n_envs = rollout.n_envs
    rewards = torch.tensor([t.reward for t in rollout.transitions], dtype=torch.float32)
    values = torch.tensor([t.value for t in rollout.transitions], dtype=torch.float32)
    dones = torch.tensor([t.done for t in rollout.transitions], dtype=torch.float32)
    closes = torch.tensor(
        [t.is_close for t in rollout.transitions], dtype=torch.float32
    )
    advantages = torch.zeros(n, dtype=torch.float32)
    gamma = torch.where(closes > 0, config.gamma, 1.0)
    lam = torch.where(closes > 0, config.gae_lambda, 1.0)
    for env_index in range(n_envs):
        rows = list(range(env_index, n, n_envs))
        running = 0.0
        next_value = rollout.bootstrap[env_index]
        for row in reversed(rows):
            not_done = 1.0 - float(dones[row])
            delta = (
                float(rewards[row])
                + float(gamma[row]) * next_value * not_done
                - float(values[row])
            )
            running = delta + float(gamma[row]) * float(lam[row]) * not_done * running
            advantages[row] = running
            next_value = float(values[row])
    return advantages + values, advantages


@dataclass
class Evaluated:
    """A batch of transitions re-scored under the current weights."""

    log_probs: torch.Tensor  # (B,), 0 on rows without a policy
    selector_entropy: torch.Tensor  # (B,)
    head_entropy: torch.Tensor  # (B,)
    values: torch.Tensor  # (B,)
    has_policy: torch.Tensor  # (B,) bool


def evaluate_transitions(
    network: SetNetwork, transitions: Sequence[Transition]
) -> Evaluated:
    """Recompute the joint log-prob, entropies and value of each transition.

    Pinned to reproduce the sampled log-prob exactly at unchanged weights.
    The selector's softmax runs over policy rows only: a closing row's
    selector is all `-inf` and would poison the batch with NaN.
    """
    device = network.device
    batch = collate([t.tokens for t in transitions], device=device)
    has_policy = torch.tensor([t.has_policy for t in transitions], device=device)
    models = torch.tensor(
        [max(t.model, 0) for t in transitions], dtype=torch.int64, device=device
    )
    columns = torch.tensor(
        [max(t.column, 0) for t in transitions], dtype=torch.int64, device=device
    )
    output = network(batch)
    n = len(transitions)
    log_probs = torch.zeros(n, device=device)
    selector_entropy = torch.zeros(n, device=device)
    head_entropy = torch.zeros(n, device=device)
    if bool(has_policy.any()):
        rows = torch.nonzero(has_policy).squeeze(-1)
        selector = F.log_softmax(output.selector_logits[rows].float(), dim=-1)
        log_probs[rows] = selector.gather(1, models[rows, None]).squeeze(-1)
        selector_entropy[rows] = _entropy(selector)
    heads = network.heads(output, batch, models)
    for head, logits in (
        (Head.declaration, heads.declaration),
        (Head.displacement, heads.displacement),
        (Head.unit_pointer, heads.unit),
    ):
        mask = torch.tensor(
            [t.has_policy and t.head is head for t in transitions], device=device
        )
        if not bool(mask.any()):
            continue
        rows = torch.nonzero(mask).squeeze(-1)
        head_log_probs = F.log_softmax(logits[rows].float(), dim=-1)
        log_probs[rows] = log_probs[rows] + head_log_probs.gather(
            1, columns[rows, None]
        ).squeeze(-1)
        head_entropy[rows] = _entropy(head_log_probs)
    return Evaluated(
        log_probs=log_probs,
        selector_entropy=selector_entropy,
        head_entropy=head_entropy,
        values=output.value.float(),
        has_policy=has_policy,
    )


def _entropy(log_probs: torch.Tensor) -> torch.Tensor:
    clamped = log_probs.clamp(min=_LOG_PROB_FLOOR)
    return -(clamped.exp() * clamped).sum(dim=-1)


@dataclass
class UpdateStats:
    """What one update did, in the whole-phase trainer's vocabulary."""

    train_loss: float
    policy_loss: float
    value_loss: float
    entropy_loss: float
    clip_fraction: float
    approx_kl: float
    explained_variance: float
    grad_norm: float
    grad_clipped_fraction: float
    n_minibatches: int


def check_trainable(network: SetNetwork) -> None:
    """Refuse a network whose rollouts and updates would not see one policy."""
    if network.config.dropout != 0.0:
        raise ValueError(
            "the per-model PPO loop needs dropout 0: rollouts sample in train "
            "mode, so the sampled and recomputed log-probs would come from "
            "different masks and every ratio would be silently wrong"
        )


def ppo_update(
    network: SetNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    config: PerModelPPOConfig,
    *,
    generator: torch.Generator | None = None,
) -> UpdateStats:
    """`n_epochs` passes of clipped-surrogate minibatches over the rollout."""
    check_trainable(network)
    network.train()
    device = network.device
    transitions = rollout.transitions
    n = len(transitions)
    has_policy = torch.tensor([t.has_policy for t in transitions])
    old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32)
    old_values = torch.tensor([t.value for t in transitions], dtype=torch.float32)
    policy_rows = has_policy.nonzero().squeeze(-1)
    normalised = advantages.clone()
    if policy_rows.numel() > 1:
        policy_adv = advantages[policy_rows]
        normalised = (advantages - policy_adv.mean()) / (policy_adv.std() + 1e-8)
    explained = _explained_variance(returns, old_values)

    totals = {
        "loss": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "entropy": 0.0,
        "clip": 0.0,
        "kl": 0.0,
        "grad": 0.0,
        "clipped": 0.0,
    }
    n_minibatches = 0
    selector_coef = config.resolved_selector_ent_coef
    for _ in range(config.n_epochs):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, config.batch_size):
            rows = permutation[start : start + config.batch_size]
            evaluated = evaluate_transitions(network, [transitions[i] for i in rows])
            policy = evaluated.has_policy
            n_policy = max(1, int(policy.sum().item()))
            ratio = torch.exp(evaluated.log_probs - old_log_probs[rows].to(device))
            adv = normalised[rows].to(device)
            surrogate = torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - config.eps_clip, 1 + config.eps_clip) * adv,
            )
            policy_loss = -(surrogate * policy).sum() / n_policy
            value_loss = F.mse_loss(evaluated.values, returns[rows].to(device))
            head_entropy = (evaluated.head_entropy * policy).sum() / n_policy
            selector_entropy = (evaluated.selector_entropy * policy).sum() / n_policy
            entropy_loss = -(
                config.ent_coef * head_entropy + selector_coef * selector_entropy
            )
            loss = policy_loss + config.vf_coef * value_loss + entropy_loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), config.max_grad_norm
                ).item()
            )
            optimizer.step()
            with torch.no_grad():
                log_ratio = evaluated.log_probs - old_log_probs[rows].to(device)
                clipped = ((ratio - 1).abs() > config.eps_clip).float()
                totals["clip"] += float((clipped * policy).sum().item()) / n_policy
                totals["kl"] += (
                    float((((ratio - 1) - log_ratio) * policy).sum().item()) / n_policy
                )
            totals["loss"] += float(loss.item())
            totals["policy"] += float(policy_loss.item())
            totals["value"] += float(value_loss.item())
            totals["entropy"] += float(entropy_loss.item())
            totals["grad"] += grad_norm
            totals["clipped"] += float(grad_norm > config.max_grad_norm)
            n_minibatches += 1
    count = max(1, n_minibatches)
    return UpdateStats(
        train_loss=totals["loss"] / count,
        policy_loss=totals["policy"] / count,
        value_loss=totals["value"] / count,
        entropy_loss=totals["entropy"] / count,
        clip_fraction=totals["clip"] / count,
        approx_kl=totals["kl"] / count,
        explained_variance=explained,
        grad_norm=totals["grad"] / count,
        grad_clipped_fraction=totals["clipped"] / count,
        n_minibatches=n_minibatches,
    )


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    variance = float(returns.var().item()) if returns.numel() > 1 else 0.0
    if variance == 0.0:
        return 0.0
    residual = float((returns - values).var().item())
    return 1.0 - residual / variance


def rollout_entropy(
    network: SetNetwork, rollout: Rollout, batch_size: int
) -> tuple[dict[str, float], float]:
    """The sampled policy's entropies over the rollout, by phase, plus the
    selector's -- one no-grad pass at the weights that played it."""
    network.eval()
    by_phase: dict[str, list[float]] = {}
    selector: list[float] = []
    transitions = rollout.transitions
    with torch.no_grad():
        for start in range(0, len(transitions), batch_size):
            chunk = transitions[start : start + batch_size]
            evaluated = evaluate_transitions(network, chunk)
            head = evaluated.head_entropy.cpu().tolist()
            sel = evaluated.selector_entropy.cpu().tolist()
            for transition, h, s in zip(chunk, head, sel):
                if not transition.has_policy:
                    continue
                key = transition.phase.value if transition.phase else "none"
                by_phase.setdefault(key, []).append(float(h))
                selector.append(float(s))
    return (
        {key: float(np.mean(values)) for key, values in by_phase.items()},
        float(np.mean(selector)) if selector else 0.0,
    )


__all__ = [
    "NO_DRAW",
    "EpisodeOutcome",
    "Evaluated",
    "PerModelPPOConfig",
    "Rollout",
    "StepKind",
    "Transition",
    "UpdateStats",
    "auto_num_rollout_envs",
    "check_trainable",
    "collect_rollout",
    "compute_gae",
    "evaluate_transitions",
    "ppo_update",
    "rollout_entropy",
]
