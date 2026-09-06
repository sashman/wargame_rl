"""Single-scalar PPO over per-model steps (issue #286).

Standard single-action PPO — one reward, one value ``V(s)``, one importance
ratio per step — where a step is one model's action (or the turn-closing
step, which carries no decision and enters only the value loss). A step's
log-prob is the sum of its factors (selection + declaration-on-opening-steps
+ action) and all factors share the step's advantage.

**Every time constant is per round.** ``gamma`` and the GAE decay apply only
across a turn-closing step; every model step within a turn is the same
instant, and the rollout budget is expressed in rounds. A per-step gamma was
rejected in the design: steps per round shrink as models die, so the horizon
would drift within an episode and differ between scenarios.

⚠ ``gamma``, ``gae_lambda`` and the rollout budget were measured at two steps
per round in the whole-phase world; the defaults here are starting points and
must be re-measured in the new unit before any race number is quoted
(`model/ppo/config.py`'s own gamma comment says the same).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from wargame_rl.wargame.envs.per_model.facade import PerModelEnv
from wargame_rl.wargame.envs.per_model.observation import (
    TokenObservation,
    build_token_observation,
)
from wargame_rl.wargame.envs.per_model.types import PerModelObservation, StepKind
from wargame_rl.wargame.model.per_model.agent import (
    SetAgent,
    StepDecision,
    _combined_movement_logits,
)
from wargame_rl.wargame.model.per_model.batch import TokenBatch, collate
from wargame_rl.wargame.model.per_model.net import SetNetwork


class PerModelPPOConfig(BaseModel):
    """PPO hyperparameters for the per-model step. Defaults are unmeasured."""

    model_config = ConfigDict(extra="forbid")

    gamma: float = Field(default=0.9, gt=0.0, le=1.0)
    """Per-ROUND discount, applied only across a turn-closing step. The 0.9
    carries over the whole-phase world's measured preference as a starting
    point only — its own comment says to retest when the reward's time
    structure changes, and it has."""
    gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    clip_epsilon: float = Field(default=0.2, gt=0.0)
    value_coef: float = Field(default=0.5, ge=0.0)
    entropy_coef_selector: float = Field(default=0.01, ge=0.0)
    """Entropy bonus on the SELECTION factor — kept separate so exploring the
    order can be kept alive while the action sharpens. Defaults equal to the
    action coefficient, per the design."""
    entropy_coef_action: float = Field(default=0.01, ge=0.0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    n_update_epochs: int = Field(default=4, ge=1)
    minibatch_size: int = Field(default=64, ge=1)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    rollout_rounds: int = Field(default=8, ge=1)
    """The rollout budget, in ROUNDS (turn cycles), not steps."""


@dataclass(slots=True)
class Transition:
    """One env step as the buffer keeps it: ``(T,)`` scalars, no model axis."""

    observation: TokenObservation
    decision: StepDecision
    reward: float
    done: bool
    is_close: bool


def collect_rollout(
    env: PerModelEnv,
    agent: SetAgent,
    n_rounds: int,
    *,
    start_observation: PerModelObservation | None = None,
    generator: torch.Generator | None = None,
    on_episode_end: Callable[[PerModelEnv], None] | None = None,
) -> tuple[list[Transition], float, PerModelObservation]:
    """Play until `n_rounds` turn cycles have closed; episodes reset inline.

    ``start_observation`` continues the episode already in flight on ``env``;
    ``None`` resets. A training loop MUST pass the observation the previous
    rollout returned — resetting every rollout means a budget shorter than the
    episode never visits the later rounds of the game at all.

    Returns the transitions, the bootstrap value of the state after the last
    one (0.0 when that transition ended an episode), and that state's
    observation, to hand to the next rollout. ``on_episode_end`` is called with
    the env at each episode end, before the inline reset — the one moment the
    finished episode's outcome is still readable.
    """
    transitions: list[Transition] = []
    if start_observation is None:
        observation, _ = env.reset()
    else:
        observation = start_observation
    closes = 0
    while closes < n_rounds:
        decision = agent.act(env, observation, generator=generator)
        token_observation = build_token_observation(env, observation)
        is_close = observation.kind is StepKind.turn_close
        next_observation, reward, done, _tr, _ = env.step(decision.action)
        transitions.append(
            Transition(
                observation=token_observation,
                decision=decision,
                reward=float(reward),
                done=done,
                is_close=is_close,
            )
        )
        if is_close:
            closes += 1
        if done:
            if on_episode_end is not None:
                on_episode_end(env)
            next_observation, _ = env.reset()
        observation = next_observation
    if transitions[-1].done:
        return transitions, 0.0, observation
    bootstrap = agent.act(env, observation, greedy=True).value
    return transitions, bootstrap, observation


def compute_gae(
    transitions: list[Transition],
    bootstrap_value: float,
    config: PerModelPPOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """(advantages, returns) with the discount applied only across closes.

    The discount between step t and t+1 is ``gamma`` when step t is a
    turn-closing step (a round boundary lies between them) and 1.0 otherwise
    — every model step within a turn is the same instant. The GAE decay
    follows the same clock.
    """
    n = len(transitions)
    advantages = np.zeros(n, dtype=np.float64)
    running = 0.0
    next_value = bootstrap_value
    for t in range(n - 1, -1, -1):
        transition = transitions[t]
        not_done = 0.0 if transition.done else 1.0
        step_gamma = config.gamma if transition.is_close else 1.0
        step_lambda = config.gae_lambda if transition.is_close else 1.0
        delta = (
            transition.reward
            + step_gamma * next_value * not_done
            - transition.decision.value
        )
        running = delta + step_gamma * step_lambda * not_done * running
        advantages[t] = running
        next_value = transition.decision.value
    values = np.array([t.decision.value for t in transitions], dtype=np.float64)
    return advantages, advantages + values


@dataclass(slots=True)
class EvaluationOutput:
    """Recomputed quantities for a minibatch, with grad."""

    log_probs: torch.Tensor  # (B,) — 0 where the step carried no decision
    values: torch.Tensor  # (B,)
    selector_entropy: torch.Tensor  # (B,)
    action_entropy: torch.Tensor  # (B,) — declaration + head factors
    has_decision: torch.Tensor  # (B,) bool


def evaluate_transitions(
    network: SetNetwork, transitions: list[Transition]
) -> EvaluationOutput:
    """Recompute each stored step's joint log-prob under the current weights.

    Closing steps carry no decision: their log-prob and entropies are zero and
    ``has_decision`` is False, so they enter only the value loss.
    """
    device = next(network.parameters()).device
    batch = collate([t.observation for t in transitions], device=device)
    output = network(batch)
    n = len(transitions)
    device = output.value.device
    log_probs = torch.zeros(n, device=device)
    selector_entropy = torch.zeros(n, device=device)
    action_entropy = torch.zeros(n, device=device)
    has_decision = torch.zeros(n, dtype=torch.bool, device=device)

    model_rows = [
        (row, t)
        for row, t in enumerate(transitions)
        if t.decision.action.model_index is not None
    ]
    if not model_rows:
        return EvaluationOutput(
            log_probs, output.value, selector_entropy, action_entropy, has_decision
        )
    rows = torch.tensor([row for row, _ in model_rows], device=device)
    chosen = torch.tensor(
        [t.decision.action.model_index for _, t in model_rows], device=device
    )
    # Softmaxed over the decision rows only: a closing step's selector row is
    # all -inf and would poison the whole batch with NaN gradients.
    selector_log_probs = torch.log_softmax(output.selector_logits[rows], dim=-1)
    position_index = torch.arange(len(model_rows), device=device)
    log_probs[rows] = selector_log_probs[position_index, chosen]
    selector_entropy[rows] = _entropy_rows(selector_log_probs)
    has_decision[rows] = True

    from wargame_rl.wargame.model.per_model.net import SetNetworkOutput

    decision_output = SetNetworkOutput(
        player_latents=output.player_latents[rows],
        game_latent=output.game_latent[rows],
        selector_logits=output.selector_logits[rows],
        value=output.value[rows],
    )
    logits = network.action_logits(decision_output, _subset(batch, rows), chosen)
    for position, (row, transition) in enumerate(model_rows):
        decision = transition.decision
        if decision.declaration_sampled:
            factor_log_probs = torch.log_softmax(logits.declaration[position], dim=-1)
            declared = decision.action.declaration or 0
            log_probs[row] = log_probs[row] + factor_log_probs[declared]
            action_entropy[row] = action_entropy[row] + _entropy(factor_log_probs)
        if decision.head == "target":
            assert decision.head_choice is not None
            factor_log_probs = torch.log_softmax(logits.target[position], dim=-1)
            log_probs[row] = log_probs[row] + factor_log_probs[decision.head_choice]
            action_entropy[row] = action_entropy[row] + _entropy(factor_log_probs)
        elif decision.head == "movement":
            assert decision.head_choice is not None
            combined = _combined_movement_logits(
                logits.displacement[position],
                logits.advance[position],
                decision.advance_offered,
            )
            factor_log_probs = torch.log_softmax(combined, dim=-1)
            log_probs[row] = log_probs[row] + factor_log_probs[decision.head_choice]
            action_entropy[row] = action_entropy[row] + _entropy(factor_log_probs)

    return EvaluationOutput(
        log_probs, output.value, selector_entropy, action_entropy, has_decision
    )


def _subset(batch: TokenBatch, rows: torch.Tensor) -> TokenBatch:
    """Index every tensor field of a TokenBatch by batch row."""
    return TokenBatch(
        phase_index=batch.phase_index[rows],
        player_tokens=batch.player_tokens[rows],
        player_alive=batch.player_alive[rows],
        player_pad=batch.player_pad[rows],
        selection_mask=batch.selection_mask[rows],
        context_tokens=batch.context_tokens[rows],
        context_kinds=batch.context_kinds[rows],
        context_mask=batch.context_mask[rows],
        self_relations=batch.self_relations[rows],
        cross_relations=batch.cross_relations[rows],
        opponent_unit_rows=batch.opponent_unit_rows[rows],
        unit_pad=batch.unit_pad[rows],
        displacement_mask=batch.displacement_mask[rows],
        advance_mask=batch.advance_mask[rows],
        target_mask=batch.target_mask[rows],
        declaration_mask=batch.declaration_mask[rows],
    )


def _entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Entropy of one masked categorical, NaN-safe under autograd.

    Masked entries are -inf; clamping before the product keeps both the value
    (a clamped entry contributes ~1e-12) and its gradient finite — the naive
    `where(finite, p*logp, 0)` leaks NaN through the unselected branch's local
    gradient.
    """
    clamped = log_probs.clamp(min=-30.0)
    return -(clamped.exp() * clamped).sum()


def _entropy_rows(log_probs: torch.Tensor) -> torch.Tensor:
    """Row-wise `_entropy` for a (N, K) block."""
    clamped = log_probs.clamp(min=-30.0)
    return -(clamped.exp() * clamped).sum(-1)


def ppo_update(
    network: SetNetwork,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    bootstrap_value: float,
    config: PerModelPPOConfig,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """One PPO update over a rollout. Returns mean losses for logging."""
    device = next(network.parameters()).device
    advantages, returns = compute_gae(transitions, bootstrap_value, config)
    advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    return_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(
        [t.decision.log_prob for t in transitions], dtype=torch.float32, device=device
    )

    decision_rows = np.array(
        [t.decision.action.model_index is not None for t in transitions]
    )
    if decision_rows.any():
        live = advantage_tensor[torch.from_numpy(decision_rows).to(device)]
        mean, std = live.mean(), live.std().clamp(min=1e-8)
        advantage_tensor = (advantage_tensor - mean) / std

    n = len(transitions)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    batches = 0
    for _ in range(config.n_update_epochs):
        order = torch.randperm(n, generator=generator).tolist()
        for start in range(0, n, config.minibatch_size):
            index = order[start : start + config.minibatch_size]
            evaluated = evaluate_transitions(network, [transitions[i] for i in index])
            mask = evaluated.has_decision.float()
            ratio = torch.exp(evaluated.log_probs - old_log_probs[index])
            advantage = advantage_tensor[index]
            surrogate = ratio * advantage
            clipped = (
                torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                * advantage
            )
            policy_loss = -(
                torch.min(surrogate, clipped) * mask
            ).sum() / mask.sum().clamp(min=1.0)
            value_loss = torch.nn.functional.mse_loss(
                evaluated.values, return_tensor[index]
            )
            entropy_bonus = (
                config.entropy_coef_selector * (evaluated.selector_entropy * mask).sum()
                + config.entropy_coef_action * (evaluated.action_entropy * mask).sum()
            ) / mask.sum().clamp(min=1.0)
            loss = policy_loss + config.value_coef * value_loss - entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
            optimizer.step()

            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy_bonus.detach())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}
