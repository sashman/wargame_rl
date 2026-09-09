"""A seat over the set network: one decision per call, sampled or greedy.

The seam between the network and the facade. It builds the token
observation (caching the episode's static scenario on `env.episode_id`),
runs one forward, samples the selector and then the head the decision's
kind and phase name, and decodes the two columns into a `PerModelAction`
the decision point accepts. Sampling happens on the CPU whatever device the
network is on, so a seeded rollout is device-independent.

A decoded action the point refuses is a builder/decoder disagreement -- a
bug -- and raises rather than resampling.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.seat import Seat
from wargame_rl.wargame.envs.per_model.tokens import (
    Head,
    TokenObservation,
    TokenScenario,
    build_tokens,
    displacement_to_action,
    unit_column_to_value,
)
from wargame_rl.wargame.envs.per_model.types import (
    PerModelAction,
    PerModelObservation,
    StepKind,
)
from wargame_rl.wargame.model.per_model.batch import collate
from wargame_rl.wargame.model.per_model.net import HeadLogits, SetNetwork


@dataclass(frozen=True)
class StepDecision:
    """What one call decided: the action, its joint log-prob, the value."""

    action: PerModelAction
    log_prob: float
    value: float
    tokens: TokenObservation


class SetAgent:
    """Drives a `PerModelEnv` seat with a `SetNetwork`."""

    def __init__(self, network: SetNetwork, *, greedy: bool = False) -> None:
        self.network = network
        self.greedy = greedy
        self._scenario: TokenScenario | None = None
        # "<phase>:<declaration>" per opening step: the skip declarations gate a
        # whole unit through one logit, so their share is the first thing to
        # read when a score sits at the do-nothing fingerprint.
        self.declaration_counts: Counter[str] = Counter()

    def scenario_for(self, env: PerModelEnv, seat: Seat) -> TokenScenario:
        """The episode's static scenario, rebuilt when the episode changes."""
        if self._scenario is None or self._scenario.episode_id != env.episode_id:
            self._scenario = TokenScenario.for_episode(env, seat)
        return self._scenario

    def observe(
        self, env: PerModelEnv, observation: PerModelObservation
    ) -> TokenObservation:
        """The token observation for the player seat at this decision."""
        seat = env.player_seat
        return build_tokens(env, seat, observation, self.scenario_for(env, seat))

    def act(
        self,
        env: PerModelEnv,
        observation: PerModelObservation,
        *,
        generator: torch.Generator | None = None,
    ) -> StepDecision:
        """Decide the next step for the player seat."""
        tokens = self.observe(env, observation)
        point = observation.decision
        batch = collate([tokens], device=self.network.device)
        with torch.no_grad():
            output = self.network(batch)
            value = float(output.value[0].item())
            if point.kind is StepKind.close_turn:
                return StepDecision(PerModelAction.close_turn(), 0.0, value, tokens)
            selector = output.selector_logits[0].float().cpu()
            model, selector_log_prob = _draw(selector, self.greedy, generator)
            heads = self.network.heads(
                output, batch, torch.tensor([model], device=self.network.device)
            )
        logits = _head_logits(heads, tokens.head)[0].float().cpu()
        column, value_log_prob = _draw(logits, self.greedy, generator)
        action = self._decode(point.kind, tokens, env.player_seat, model, column)
        reason = point.why_illegal(action)
        if reason is not None:
            raise RuntimeError(
                f"the set agent decoded an illegal decision ({reason}); the token "
                "masks and the decision point disagree"
            )
        if point.kind is StepKind.open:
            phase = point.phase.value if point.phase is not None else "-"
            self.declaration_counts[f"{phase}:{column}"] += 1
        return StepDecision(action, selector_log_prob + value_log_prob, value, tokens)

    @staticmethod
    def _decode(
        kind: StepKind, tokens: TokenObservation, seat: Seat, model: int, column: int
    ) -> PerModelAction:
        if kind is StepKind.open:
            return PerModelAction.open(model, column)
        if kind is StepKind.target:
            value = unit_column_to_value(
                column, kind, seat, tokens.opponent_unit_groups
            )
            return PerModelAction.target(model, value)
        if tokens.head is Head.unit_pointer:
            value = unit_column_to_value(
                column, kind, seat, tokens.opponent_unit_groups
            )
            return PerModelAction.act(model, value)
        return PerModelAction.act(model, displacement_to_action(column, seat))


def _head_logits(heads: HeadLogits, head: Head) -> torch.Tensor:
    if head is Head.declaration:
        return heads.declaration
    if head is Head.displacement:
        return heads.displacement
    if head is Head.unit_pointer:
        return heads.unit
    raise ValueError(f"no head scores a {head.name} step")


def _draw(
    logits: torch.Tensor, greedy: bool, generator: torch.Generator | None
) -> tuple[int, float]:
    """One column from masked logits on the CPU, with its log-prob."""
    if not torch.isfinite(logits).any():
        raise RuntimeError("every column is masked; the decision point offers nothing")
    log_probs = F.log_softmax(logits, dim=-1)
    if greedy:
        index = int(torch.argmax(log_probs).item())
    else:
        probs = log_probs.exp()
        index = int(torch.multinomial(probs, 1, generator=generator).item())
    return index, float(log_probs[index].item())
