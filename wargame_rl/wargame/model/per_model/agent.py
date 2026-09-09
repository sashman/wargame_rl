"""A seat over the set network: one decision per call, sampled or greedy.

The seam between the network and the facade. It builds the token
observation (caching each env's static scenario on `env.episode_id`),
runs one forward, samples the selector and then the head the decision's
kind and phase name, and decodes the two columns into a `PerModelAction`
the decision point accepts. Sampling happens on the CPU whatever device the
network is on, so a seeded rollout is device-independent.

`act_batch` does the same for N envs in one forward: every env always has
exactly one pending decision (`close_turn` included), so lockstep rollouts
collate N observations and draw N rows. A closing row draws nothing -- its
selector is all `-inf` -- and carries only the value.

A decoded action the point refuses is a builder/decoder disagreement -- a
bug -- and raises rather than resampling.
"""

from __future__ import annotations

import weakref
from collections import Counter
from collections.abc import Sequence
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

# The column a closing step reports: nothing was drawn.
NO_DRAW = -1


@dataclass(frozen=True)
class StepDecision:
    """What one call decided: the action, its joint log-prob, the value.

    `model` and `column` are the two drawn factors (the selector's row and
    the head's column), so a training update can recompute their log-probs
    without decoding the action back; both are `NO_DRAW` on a closing step,
    which has no policy factor. `column >= 0` is "this step has a policy".
    """

    action: PerModelAction
    log_prob: float
    value: float
    tokens: TokenObservation
    model: int = NO_DRAW
    column: int = NO_DRAW

    @property
    def has_policy(self) -> bool:
        return self.column >= 0


class SetAgent:
    """Drives `PerModelEnv` seats with a `SetNetwork`."""

    def __init__(self, network: SetNetwork, *, greedy: bool = False) -> None:
        self.network = network
        self.greedy = greedy
        # Keyed on the env OBJECT, weakly: an `id(env)` key would serve a
        # freed env's scenario to a new one allocated at the same address
        # with the same `episode_id` -- the recycled-id defect, one level up.
        self._scenarios: weakref.WeakKeyDictionary[PerModelEnv, TokenScenario] = (
            weakref.WeakKeyDictionary()
        )
        # "<phase>:<declaration>" per opening step: the skip declarations gate a
        # whole unit through one logit, so their share is the first thing to
        # read when a score sits at the do-nothing fingerprint.
        self.declaration_counts: Counter[str] = Counter()

    def scenario_for(self, env: PerModelEnv, seat: Seat) -> TokenScenario:
        """The env's static scenario for this episode, rebuilt when it changes."""
        scenario = self._scenarios.get(env)
        if scenario is None or scenario.episode_id != env.episode_id:
            scenario = TokenScenario.for_episode(env, seat)
            self._scenarios[env] = scenario
        return scenario

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
        return self.act_batch([env], [observation], generator=generator)[0]

    def act_batch(
        self,
        envs: Sequence[PerModelEnv],
        observations: Sequence[PerModelObservation],
        *,
        generator: torch.Generator | None = None,
    ) -> list[StepDecision]:
        """Decide the next step for every env, in one forward."""
        if len(envs) != len(observations):
            raise ValueError("one observation per env")
        tokens = [self.observe(env, obs) for env, obs in zip(envs, observations)]
        device = self.network.device
        batch = collate(tokens, device=device)
        closing = [obs.decision.kind is StepKind.close_turn for obs in observations]
        with torch.no_grad():
            output = self.network(batch)
            values = output.value.float().cpu()
            selectors = output.selector_logits.float().cpu()
            models = [NO_DRAW] * len(envs)
            selector_log_probs = [0.0] * len(envs)
            for row, is_closing in enumerate(closing):
                if is_closing:
                    continue
                models[row], selector_log_probs[row] = _draw(
                    selectors[row], self.greedy, generator
                )
            # A closing row's model index is a placeholder: its heads are read
            # and discarded, since it draws no column.
            index = torch.tensor(
                [max(m, 0) for m in models], dtype=torch.int64, device=device
            )
            heads = self.network.heads(output, batch, index)
        decisions: list[StepDecision] = []
        for row, (env, observation) in enumerate(zip(envs, observations)):
            value = float(values[row].item())
            if closing[row]:
                decisions.append(
                    StepDecision(PerModelAction.close_turn(), 0.0, value, tokens[row])
                )
                continue
            logits = _head_logits(heads, tokens[row].head)[row].float().cpu()
            column, column_log_prob = _draw(logits, self.greedy, generator)
            point = observation.decision
            model = models[row]
            action = self._decode(
                point.kind, tokens[row], env.player_seat, model, column
            )
            reason = point.why_illegal(action)
            if reason is not None:
                raise RuntimeError(
                    f"the set agent decoded an illegal decision ({reason}); the "
                    "token masks and the decision point disagree"
                )
            if point.kind is StepKind.open:
                phase = point.phase.value if point.phase is not None else "-"
                self.declaration_counts[f"{phase}:{column}"] += 1
            decisions.append(
                StepDecision(
                    action,
                    selector_log_probs[row] + column_log_prob,
                    value,
                    tokens[row],
                    model=model,
                    column=column,
                )
            )
        return decisions

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
