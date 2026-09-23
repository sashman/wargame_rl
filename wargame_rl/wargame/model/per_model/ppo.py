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
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator

from wargame_rl.wargame.envs.per_model.env import PerModelEnv, StepEffect
from wargame_rl.wargame.envs.per_model.reward_timing import Credit, PerStepReward
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

    gamma: float = Field(default=0.9, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    eps_clip: float = Field(default=0.2, gt=0.0)
    vf_coef: float = Field(default=0.3, ge=0.0)
    ent_coef: float = Field(default=0.03, ge=0.0)
    # The selector's own coefficient; None means `ent_coef`. Two knobs so the
    # order can be kept exploring while the action sharpens (#283).
    selector_ent_coef: float | None = Field(default=None, ge=0.0)
    # A KL anchor to the run's starting weights (#332): `kl_ref_coef` weights
    # the per-decision drift estimator; `kl_ref_target` (nats per decision)
    # makes the coefficient adaptive by the phase facade's rule. 0.0 builds no
    # reference network and adds no term, so a run without it is unchanged.
    kl_ref_coef: float = Field(default=0.0, ge=0.0)
    kl_ref_target: float = Field(default=0.0, ge=0.0)
    # The planning stream (#384 D6): the discount per commitment step (one
    # per turn per unit, so near one) and the planning value head's weight.
    # Read only when a rollout carries commitment decisions.
    planning_gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    planning_vf_coef: float = Field(default=0.3, ge=0.0)
    # Who a payment reaches (`reward_timing.Credit`): `mean` is the bridge
    # accounting, `actor` pays each model its own term and its own state
    # credit on its own step. A knob of the run, so a resume keeps it.
    credit: Credit = Credit.mean
    lr: float = Field(default=3e-4, gt=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    n_epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=128, ge=1)
    # Closing steps per env per update.
    rollout_rounds: int = Field(default=16, ge=1)
    # 0 auto-detects: 8 on a usable GPU, 4 on the CPU, clamped by affinity.
    # The driver writes the resolved count back, so a checkpoint carries it.
    num_rollout_envs: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_budgets(self) -> PerModelPPOConfig:
        if self.selector_ent_coef is not None and self.selector_ent_coef < 0:
            raise ValueError("selector_ent_coef must be non-negative")
        return self

    @property
    def resolved_selector_ent_coef(self) -> float:
        return (
            self.ent_coef if self.selector_ent_coef is None else self.selector_ent_coef
        )


# The adaptive anchor's coefficient band, the phase facade's constants.
KL_COEF_MIN = 1e-4
KL_COEF_MAX = 1e4


def adapt_kl_coef(coef: float, measured_drift: float, target: float) -> float:
    """Schulman's KL-penalty rule: halve the coefficient when the policy stays
    closer than `target` by 1.5x, double it when it drifts further by 1.5x;
    a no-op at `target == 0.0`. The wide band keeps the coefficient from
    oscillating faster than the policy can answer it."""
    if target <= 0.0:
        return coef
    if measured_drift < target / 1.5:
        return max(coef / 2.0, KL_COEF_MIN)
    if measured_drift > target * 1.5:
        return min(coef * 2.0, KL_COEF_MAX)
    return coef


def affinity_cpu_count() -> int:
    """The CPUs this process may run on -- the count that clamps the env count.

    ⚠ A launcher that pins a run (`taskset`, a cgroup) pins this too, and
    with it the rounds per update. The calibration sweep of 2026-09-07 was
    launched that way and every cell trained on ONE env.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def device_max_rollout_envs(device: torch.device) -> int:
    """The shipped loop's ceiling: 8 on CUDA, 4 on the CPU."""
    return 8 if device.type == "cuda" else 4


def auto_num_rollout_envs(device: torch.device) -> int:
    """The shipped loop's heuristic: the device ceiling, clamped by affinity."""
    return max(1, min(device_max_rollout_envs(device), affinity_cpu_count()))


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
    # The commitment decision on an `open` step (#384 Stage 1): the unit it
    # belongs to, the drawn column and its log-prob, the planning value at
    # the step, the planning reward accumulated from this step to the unit's
    # next commitment step, and whether the episode ended inside that span.
    unit: int = -1
    commit_column: int = NO_DRAW
    commit_log_prob: float = 0.0
    planning_value: float = 0.0
    planning_reward: float = 0.0
    planning_done: bool = False

    @property
    def has_policy(self) -> bool:
        return self.column >= 0

    @property
    def has_commitment(self) -> bool:
        return self.commit_column >= 0


@dataclass(frozen=True)
class EpisodeOutcome:
    """A rollout episode that finished, read the moment it did."""

    env_index: int
    player_vp: int
    opponent_vp: int
    reward: float
    decision_steps: int
    rounds: int
    # The phase's success criteria on the terminating board, and how many
    # squads the episode began with on objectives (0: from deployment).
    success: bool = False
    start_groups: int = 0


# How many squads the next inline reset starts on objectives; 0 is the plain
# augmented start. Called once per reset, so a schedule can draw.
StartGroups = Callable[[], int]


@dataclass(frozen=True)
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
    # Per env, how many squads the episode in progress started on objectives.
    start_groups: tuple[int, ...] = ()
    # Per env, the planning value per unit at the rollout's cut, for the
    # commitment spans still open there (#384 D6).
    planning_bootstrap: tuple[dict[int, float], ...] = ()

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
    start_groups: StartGroups | None = None,
    initial_start_groups: Sequence[int] | None = None,
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
    # Per env, how many squads its CURRENT episode started on objectives.
    started_on = (
        list(initial_start_groups) if initial_start_groups is not None else [0] * n_envs
    )
    # Per env, the transition on which each model last acted THIS turn: where
    # a close's per-model credit (`Credit.actor`) lands. Cleared at the close.
    acted_at: list[dict[int, int]] = [{} for _ in range(n_envs)]
    # Per env, each unit's OPEN commitment span: the transition of its latest
    # commitment step, which collects the planning rewards paid until the
    # unit's next commitment step (#384 D6).
    spans: list[dict[int, int]] = [{} for _ in range(n_envs)]
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
            unit = -1
            if decision.has_commitment:
                unit = int(env.player_seat.models[decision.model].group_id)
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
                    unit=unit,
                    commit_column=decision.commit_column,
                    commit_log_prob=decision.commit_log_prob,
                    planning_value=decision.planning_value,
                )
            )
            for model in effect.actor_set:
                acted_at[index][model] = len(transitions) - 1
            if payment.credits:
                _land_credits(transitions, acted_at[index], payment.credits)
            if payment.planning != 0.0:
                _land_planning(transitions, spans[index], payment.planning)
            if decision.has_commitment:
                spans[index][unit] = len(transitions) - 1
            if payment.is_close:
                acted_at[index] = {}
            last_done[index] = terminated
            if terminated:
                for at in spans[index].values():
                    transitions[at] = replace(transitions[at], planning_done=True)
                spans[index] = {}
                episodes.append(
                    EpisodeOutcome(
                        env_index=index,
                        player_vp=int(env.player_vp),
                        opponent_vp=int(env.opponent_vp),
                        reward=retimer.episode_reward,
                        decision_steps=decision_counts[index],
                        rounds=round_counts[index],
                        success=retimer.succeeded(),
                        start_groups=started_on[index],
                    )
                )
                if on_episode_end is not None:
                    on_episode_end(index, env, retimer)
                started_on[index] = int(start_groups()) if start_groups else 0
                options: dict[str, Any] = dict(_AUGMENT_START)
                if started_on[index] > 0:
                    options["start_groups"] = started_on[index]
                observation, _ = env.reset(options=options)
                retimer.reset()
                decision_counts[index] = 0
                round_counts[index] = 0
            current[index] = observation
    bootstrap = [0.0] * n_envs
    planning_bootstrap: list[dict[int, float]] = [{} for _ in range(n_envs)]
    live = [i for i in range(n_envs) if not last_done[i]]
    if live:
        values = _values(agent, [envs[i] for i in live], [current[i] for i in live])
        for slot, index in enumerate(live):
            bootstrap[index] = values[slot]
        open_spans = [i for i in live if spans[i]]
        if open_spans:
            planning = _planning_values(
                agent, [envs[i] for i in open_spans], [current[i] for i in open_spans]
            )
            for slot, index in enumerate(open_spans):
                planning_bootstrap[index] = planning[slot]
    return Rollout(
        transitions=transitions,
        n_envs=n_envs,
        bootstrap=bootstrap,
        observations=current,
        closes=closes,
        episodes=episodes,
        breakdown=breakdown,
        start_groups=tuple(started_on),
        planning_bootstrap=tuple(planning_bootstrap),
    )


def _land_planning(
    transitions: list[Transition], spans: dict[int, int], planning: float
) -> None:
    """Add a close's planning scalar to every unit's open commitment span: the
    outcome is the army's, and each unit's decision is credited with it until
    that unit decides again (#384 D2, D6)."""
    for at in spans.values():
        transitions[at] = replace(
            transitions[at],
            planning_reward=transitions[at].planning_reward + planning,
        )


def _planning_values(
    agent: SetAgent,
    envs: Sequence[PerModelEnv],
    observations: Sequence[PerModelObservation],
) -> list[dict[int, float]]:
    """Per env, the planning value at `observations` for each living unit,
    read at the unit's first living member: the bootstrap of a commitment
    span cut by the rollout."""
    tokens = [agent.observe(env, obs) for env, obs in zip(envs, observations)]
    batch = collate(tokens, device=agent.network.device)
    results: list[dict[int, float]] = [{} for _ in envs]
    leaders: list[list[tuple[int, int]]] = []
    for env in envs:
        seat = env.player_seat
        alive = seat.alive()
        rows: list[tuple[int, int]] = []
        for group in seat.living_units():
            members = [i for i in seat.unit_members(group) if alive[i]]
            if members:
                rows.append((int(group), int(members[0])))
        leaders.append(rows)
    width = max((len(rows) for rows in leaders), default=0)
    with torch.no_grad():
        output = agent.network(batch)
        for slot in range(width):
            index = torch.tensor(
                [rows[slot][1] if slot < len(rows) else 0 for rows in leaders],
                dtype=torch.int64,
                device=agent.network.device,
            )
            values = agent.network.commitment(output, batch, index).planning_value
            for row, rows in enumerate(leaders):
                if slot < len(rows):
                    results[row][rows[slot][0]] = float(values[row].item())
    return results


def _land_credits(
    transitions: list[Transition],
    acted_at: dict[int, int],
    credits: dict[int, float],
) -> None:
    """Add each model's close credit to the transition it acted on this turn;
    a model that took no step this turn is credited on the close itself (the
    last transition), so nothing paid is lost. Within a turn the discount is
    1.0, so moving a payment earlier leaves every earlier step's return as it
    was and takes it out of the returns of the steps after it."""
    last = len(transitions) - 1
    for model, value in credits.items():
        at = acted_at.get(model, last)
        transitions[at] = replace(
            transitions[at], reward=transitions[at].reward + value
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
    step_gamma = torch.where(closes > 0, config.gamma, 1.0)
    step_lambda = torch.where(closes > 0, config.gae_lambda, 1.0)
    # Group by the transition's own env index: the stored field is the source
    # of truth, not the lockstep stride it happens to equal.
    rows_by_env: dict[int, list[int]] = {index: [] for index in range(n_envs)}
    for row, transition in enumerate(rollout.transitions):
        rows_by_env[transition.env_index].append(row)
    for env_index, rows in rows_by_env.items():
        running = 0.0
        next_value = rollout.bootstrap[env_index]
        for row in reversed(rows):
            not_done = 1.0 - float(dones[row])
            delta = (
                float(rewards[row])
                + float(step_gamma[row]) * next_value * not_done
                - float(values[row])
            )
            running = (
                delta
                + float(step_gamma[row]) * float(step_lambda[row]) * not_done * running
            )
            advantages[row] = running
            next_value = float(values[row])
    return advantages + values, advantages


def compute_planning_gae(
    rollout: Rollout, config: PerModelPPOConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Planning returns and advantages, flat in the rollout's order, non-zero
    only on commitment rows (#384 D6).

    A unit's commitment steps form their own trajectory: each step's reward
    is the planning scalar collected until the unit's next commitment step,
    the discount `planning_gamma` applies once per step of that chain, and
    the last step bootstraps from the planning value at the rollout's cut
    (0 when the episode ended inside the span).
    """
    n = rollout.n_steps
    returns = torch.zeros(n, dtype=torch.float32)
    advantages = torch.zeros(n, dtype=torch.float32)
    chains: dict[tuple[int, int], list[int]] = {}
    for row, transition in enumerate(rollout.transitions):
        if transition.has_commitment:
            chains.setdefault((transition.env_index, transition.unit), []).append(row)
    for (env_index, unit), rows in chains.items():
        last = rollout.transitions[rows[-1]]
        bootstrap = (
            rollout.planning_bootstrap[env_index] if rollout.planning_bootstrap else {}
        )
        next_value = 0.0 if last.planning_done else float(bootstrap.get(unit, 0.0))
        running = 0.0
        for row in reversed(rows):
            transition = rollout.transitions[row]
            not_done = 0.0 if transition.planning_done else 1.0
            delta = (
                transition.planning_reward
                + config.planning_gamma * next_value * not_done
                - transition.planning_value
            )
            running = (
                delta + config.planning_gamma * config.gae_lambda * not_done * running
            )
            advantages[row] = running
            returns[row] = running + transition.planning_value
            next_value = transition.planning_value
    return returns, advantages


@dataclass(frozen=True)
class Evaluated:
    """A batch of transitions re-scored under the current weights."""

    log_probs: torch.Tensor  # (B,), 0 on rows without a policy
    selector_entropy: torch.Tensor  # (B,)
    head_entropy: torch.Tensor  # (B,)
    values: torch.Tensor  # (B,)
    has_policy: torch.Tensor  # (B,) bool
    # The commitment head's re-scored rows (#384 Stage 1), 0 elsewhere.
    commit_log_probs: torch.Tensor  # (B,)
    commit_entropy: torch.Tensor  # (B,)
    planning_values: torch.Tensor  # (B,)
    has_commitment: torch.Tensor  # (B,) bool


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
    has_commitment = torch.tensor(
        [t.has_commitment for t in transitions], device=device
    )
    commit_log_probs = torch.zeros(n, device=device)
    commit_entropy = torch.zeros(n, device=device)
    planning_values = torch.zeros(n, device=device)
    if bool(has_commitment.any()):
        rows = torch.nonzero(has_commitment).squeeze(-1)
        commit_columns = torch.tensor(
            [max(t.commit_column, 0) for t in transitions],
            dtype=torch.int64,
            device=device,
        )
        commitment = network.commitment(output, batch, models)
        commit_lp = F.log_softmax(commitment.logits[rows].float(), dim=-1)
        commit_log_probs[rows] = commit_lp.gather(
            1, commit_columns[rows, None]
        ).squeeze(-1)
        commit_entropy[rows] = _entropy(commit_lp)
        planning_values[rows] = commitment.planning_value[rows].float()
    return Evaluated(
        log_probs=log_probs,
        selector_entropy=selector_entropy,
        head_entropy=head_entropy,
        values=output.value.float(),
        has_policy=has_policy,
        commit_log_probs=commit_log_probs,
        commit_entropy=commit_entropy,
        planning_values=planning_values,
        has_commitment=has_commitment,
    )


def _masked_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Row-wise `KL(p || q)` over masked categoricals: `-inf` columns (masked
    on both sides, since both score the same tokens) contribute nothing."""
    finite = torch.isfinite(log_p)
    p = torch.where(finite, log_p.exp(), torch.zeros_like(log_p))
    diff = torch.where(finite, log_p - log_q, torch.zeros_like(log_p))
    return (p * diff).sum(dim=-1)


def drift_to_reference(
    network: SetNetwork, reference: SetNetwork, transitions: Sequence[Transition]
) -> torch.Tensor:
    """Per-transition `KL(policy || reference)` -- the selector's plus the
    step's head's, over the full masked distributions -- with the gradient
    through `network`; 0 on rows without a policy. The estimator the
    whole-phase anchor uses, so a target is nats per decision as there it
    is nats per model."""
    device = network.device
    batch = collate([t.tokens for t in transitions], device=device)
    has_policy = torch.tensor([t.has_policy for t in transitions], device=device)
    models = torch.tensor(
        [max(t.model, 0) for t in transitions], dtype=torch.int64, device=device
    )
    output = network(batch)
    with torch.no_grad():
        ref_output = reference(batch)
    n = len(transitions)
    kl = torch.zeros(n, device=device)
    if not bool(has_policy.any()):
        return kl
    rows = torch.nonzero(has_policy).squeeze(-1)
    kl[rows] = _masked_kl(
        F.log_softmax(output.selector_logits[rows].float(), dim=-1),
        F.log_softmax(ref_output.selector_logits[rows].float(), dim=-1),
    )
    heads = network.heads(output, batch, models)
    with torch.no_grad():
        ref_heads = reference.heads(ref_output, batch, models)
    for head, logits, ref_logits in (
        (Head.declaration, heads.declaration, ref_heads.declaration),
        (Head.displacement, heads.displacement, ref_heads.displacement),
        (Head.unit_pointer, heads.unit, ref_heads.unit),
    ):
        mask = torch.tensor(
            [t.has_policy and t.head is head for t in transitions], device=device
        )
        if not bool(mask.any()):
            continue
        head_rows = torch.nonzero(mask).squeeze(-1)
        kl[head_rows] = kl[head_rows] + _masked_kl(
            F.log_softmax(logits[head_rows].float(), dim=-1),
            F.log_softmax(ref_logits[head_rows].float(), dim=-1),
        )
    return kl


def _entropy(log_probs: torch.Tensor) -> torch.Tensor:
    clamped = log_probs.clamp(min=_LOG_PROB_FLOOR)
    return -(clamped.exp() * clamped).sum(dim=-1)


@dataclass(frozen=True)
class UpdateStats:
    """What one update did, in the whole-phase trainer's vocabulary, plus
    the pipeline-health panel (`docs/metrics.md` § The per-model health
    panel): the advantage moments BEFORE normalisation, the return and value
    moments the critic is fitting, and the tails of the importance ratio."""

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
    # Over policy rows, before normalisation: the scale PPO is about to
    # divide away. An abs-max far above the std is one transition steering
    # the update; a std near zero is a reward that never varied.
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_abs_max: float = 0.0
    # Over every row (closing rows carry a value target too).
    return_mean: float = 0.0
    return_std: float = 0.0
    value_mean: float = 0.0
    value_std: float = 0.0
    # The importance ratio's tails over the update's policy rows. Both inside
    # `1 ± eps_clip` on the first minibatch by construction; a 99th percentile
    # far outside it by the last is a trust region that no longer binds.
    ratio_p01: float = 1.0
    ratio_p99: float = 1.0
    # The KL anchor (#332): the mean per-decision drift estimator against the
    # reference over the update's policy rows, and the coefficient it was
    # weighted by. Both 0.0 when no reference is attached.
    kl_ref: float = 0.0
    kl_ref_coef: float = 0.0
    # The planning stream (#384 Stage 1), over the update's commitment rows;
    # all 0.0 when the rollout carried none.
    commit_rows: int = 0
    planning_explained_variance: float = 0.0
    planning_return_mean: float = 0.0
    planning_advantage_std: float = 0.0
    commit_clip_fraction: float = 0.0
    commit_entropy: float = 0.0


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
    reference: SetNetwork | None = None,
    kl_ref_coef: float = 0.0,
    planning_returns: torch.Tensor | None = None,
    planning_advantages: torch.Tensor | None = None,
) -> UpdateStats:
    """`n_epochs` passes of clipped-surrogate minibatches over the rollout.

    With `reference` and `kl_ref_coef > 0` the loss carries `kl_ref_coef`
    times the mean per-decision `KL(policy || reference)` over the policy
    rows -- the selector's and the step's head's full masked distributions
    (`drift_to_reference`), the whole-phase anchor's estimator. A
    taken-action estimator was tried first and rejected: heavy-tailed on
    the actions the policy stops liking (11.7 nats on an update that
    started AT the reference), it escalated the coefficient without binding.
    """
    check_trainable(network)
    anchored = reference is not None and kl_ref_coef > 0.0
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
    panel = _panel_moments(advantages[policy_rows], returns, old_values)
    ratios: list[torch.Tensor] = []
    # The planning stream (#384): its own surrogate over the commitment rows,
    # its own normalisation, its own value loss and entropy bonus.
    has_commitment = torch.tensor([t.has_commitment for t in transitions])
    commit_rows = has_commitment.nonzero().squeeze(-1)
    planning_on = (
        planning_returns is not None
        and planning_advantages is not None
        and commit_rows.numel() > 0
    )
    planning_stats: dict[str, float] = {}
    commit_ratios: list[torch.Tensor] = []
    if planning_on:
        assert planning_returns is not None and planning_advantages is not None
        old_commit_log_probs = torch.tensor(
            [t.commit_log_prob for t in transitions], dtype=torch.float32
        )
        old_planning_values = torch.tensor(
            [t.planning_value for t in transitions], dtype=torch.float32
        )
        plan_adv = planning_advantages[commit_rows]
        normalised_planning = planning_advantages.clone()
        if commit_rows.numel() > 1:
            normalised_planning = (planning_advantages - plan_adv.mean()) / (
                plan_adv.std() + 1e-8
            )
        planning_stats = {
            "commit_rows": float(commit_rows.numel()),
            "planning_explained_variance": _explained_variance(
                planning_returns[commit_rows], old_planning_values[commit_rows]
            ),
            "planning_return_mean": float(planning_returns[commit_rows].mean().item()),
            "planning_advantage_std": float(plan_adv.std().item())
            if commit_rows.numel() > 1
            else 0.0,
        }

    totals = {
        "loss": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "entropy": 0.0,
        "clip": 0.0,
        "kl": 0.0,
        "grad": 0.0,
        "clipped": 0.0,
        "kl_ref": 0.0,
        "commit_clip": 0.0,
        "commit_entropy": 0.0,
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
            advantage = normalised[rows].to(device)
            surrogate = torch.min(
                ratio * advantage,
                torch.clamp(ratio, 1 - config.eps_clip, 1 + config.eps_clip)
                * advantage,
            )
            policy_loss = -(surrogate * policy).sum() / n_policy
            value_loss = F.mse_loss(evaluated.values, returns[rows].to(device))
            head_entropy = (evaluated.head_entropy * policy).sum() / n_policy
            selector_entropy = (evaluated.selector_entropy * policy).sum() / n_policy
            entropy_loss = -(
                config.ent_coef * head_entropy + selector_coef * selector_entropy
            )
            loss = policy_loss + config.vf_coef * value_loss + entropy_loss
            if planning_on:
                assert planning_returns is not None
                commit = evaluated.has_commitment
                n_commit = max(1, int(commit.sum().item()))
                commit_ratio = torch.exp(
                    evaluated.commit_log_probs - old_commit_log_probs[rows].to(device)
                )
                plan_advantage = normalised_planning[rows].to(device)
                plan_surrogate = torch.min(
                    commit_ratio * plan_advantage,
                    torch.clamp(commit_ratio, 1 - config.eps_clip, 1 + config.eps_clip)
                    * plan_advantage,
                )
                plan_policy_loss = -(plan_surrogate * commit).sum() / n_commit
                plan_value_loss = (
                    (evaluated.planning_values - planning_returns[rows].to(device)) ** 2
                    * commit
                ).sum() / n_commit
                plan_entropy = (evaluated.commit_entropy * commit).sum() / n_commit
                loss = (
                    loss
                    + plan_policy_loss
                    + config.planning_vf_coef * plan_value_loss
                    - config.ent_coef * plan_entropy
                )
                with torch.no_grad():
                    if bool(commit.any()):
                        commit_ratios.append(
                            commit_ratio[commit].detach().float().cpu()
                        )
                        totals["commit_clip"] += (
                            float(
                                (
                                    ((commit_ratio - 1).abs() > config.eps_clip).float()
                                    * commit
                                )
                                .sum()
                                .item()
                            )
                            / n_commit
                        )
                        totals["commit_entropy"] += float(plan_entropy.item())
            if anchored:
                assert reference is not None
                drift = drift_to_reference(
                    network, reference, [transitions[i] for i in rows]
                )
                kl_ref_term = (drift * policy).sum() / n_policy
                loss = loss + kl_ref_coef * kl_ref_term
                totals["kl_ref"] += float(kl_ref_term.item())
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
                ratios.append(ratio[policy].detach().float().cpu())
            totals["loss"] += float(loss.item())
            totals["policy"] += float(policy_loss.item())
            totals["value"] += float(value_loss.item())
            totals["entropy"] += float(entropy_loss.item())
            totals["grad"] += grad_norm
            totals["clipped"] += float(grad_norm > config.max_grad_norm)
            n_minibatches += 1
    count = max(1, n_minibatches)
    ratio_p01, ratio_p99 = _ratio_tails(ratios)
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
        ratio_p01=ratio_p01,
        ratio_p99=ratio_p99,
        kl_ref=totals["kl_ref"] / count,
        kl_ref_coef=kl_ref_coef if anchored else 0.0,
        commit_rows=int(planning_stats.get("commit_rows", 0.0)),
        planning_explained_variance=planning_stats.get(
            "planning_explained_variance", 0.0
        ),
        planning_return_mean=planning_stats.get("planning_return_mean", 0.0),
        planning_advantage_std=planning_stats.get("planning_advantage_std", 0.0),
        commit_clip_fraction=totals["commit_clip"] / count,
        commit_entropy=totals["commit_entropy"] / count,
        **panel,
    )


def _panel_moments(
    policy_advantages: torch.Tensor, returns: torch.Tensor, values: torch.Tensor
) -> dict[str, float]:
    """The advantage, return and value moments, read before the update."""

    def std(tensor: torch.Tensor) -> float:
        return float(tensor.std().item()) if tensor.numel() > 1 else 0.0

    def mean(tensor: torch.Tensor) -> float:
        return float(tensor.mean().item()) if tensor.numel() > 0 else 0.0

    return {
        "advantage_mean": mean(policy_advantages),
        "advantage_std": std(policy_advantages),
        "advantage_abs_max": (
            float(policy_advantages.abs().max().item())
            if policy_advantages.numel() > 0
            else 0.0
        ),
        "return_mean": mean(returns),
        "return_std": std(returns),
        "value_mean": mean(values),
        "value_std": std(values),
    }


def _ratio_tails(ratios: list[torch.Tensor]) -> tuple[float, float]:
    """The 1st and 99th percentile of the importance ratio over the update."""
    if not ratios:
        return 1.0, 1.0
    pooled = torch.cat(ratios)
    if pooled.numel() == 0:
        return 1.0, 1.0
    quantiles = torch.quantile(pooled, torch.tensor([0.01, 0.99]))
    return float(quantiles[0].item()), float(quantiles[1].item())


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    variance = float(returns.var().item()) if returns.numel() > 1 else 0.0
    if variance == 0.0:
        return 0.0
    residual = float((returns - values).var().item())
    return 1.0 - residual / variance


@dataclass(frozen=True)
class RolloutEntropy:
    """The sampled policy's entropies over a rollout, in raw nats.

    `by_phase` is the whole-phase trainer's split; `by_head` is this
    facade's -- the declaration, displacement and unit-pointer heads have
    different widths (4, ~97, a handful), so one phase's mean mixes a
    four-way choice with a ninety-seven-way one, and a policy that has
    collapsed one head hides behind the other.
    """

    by_phase: dict[str, float]
    by_head: dict[str, float]
    selector: float


def rollout_entropy(
    network: SetNetwork, rollout: Rollout, batch_size: int
) -> RolloutEntropy:
    """The sampled policy's entropies over the rollout, by phase and by head,
    plus the selector's -- one no-grad pass at the weights that played it."""
    was_training = network.training
    network.eval()
    by_phase: dict[str, list[float]] = {}
    by_head: dict[str, list[float]] = {}
    selector_entropies: list[float] = []
    transitions = rollout.transitions
    try:
        with torch.no_grad():
            for start in range(0, len(transitions), batch_size):
                chunk = transitions[start : start + batch_size]
                evaluated = evaluate_transitions(network, chunk)
                head_entropies = evaluated.head_entropy.cpu().tolist()
                chunk_selector = evaluated.selector_entropy.cpu().tolist()
                for transition, head_entropy, selector_entropy in zip(
                    chunk, head_entropies, chunk_selector
                ):
                    if not transition.has_policy:
                        continue
                    key = transition.phase.value if transition.phase else "none"
                    by_phase.setdefault(key, []).append(float(head_entropy))
                    by_head.setdefault(transition.head.name, []).append(
                        float(head_entropy)
                    )
                    selector_entropies.append(float(selector_entropy))
    finally:
        network.train(was_training)
    return RolloutEntropy(
        by_phase={key: float(np.mean(values)) for key, values in by_phase.items()},
        by_head={key: float(np.mean(values)) for key, values in by_head.items()},
        selector=float(np.mean(selector_entropies)) if selector_entropies else 0.0,
    )


__all__ = [
    "NO_DRAW",
    "EpisodeOutcome",
    "Evaluated",
    "PerModelPPOConfig",
    "Rollout",
    "RolloutEntropy",
    "StepKind",
    "Transition",
    "UpdateStats",
    "affinity_cpu_count",
    "auto_num_rollout_envs",
    "check_trainable",
    "device_max_rollout_envs",
    "collect_rollout",
    "compute_gae",
    "evaluate_transitions",
    "ppo_update",
    "rollout_entropy",
]
