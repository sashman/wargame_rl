"""A seat over the set network: one decision per call, sampled or greedy.

The seam between the network and the facade. It builds the token
observation (caching each env's static scenario on `env.episode_id`),
runs one forward, samples the selector and then the head the decision's
kind and phase name, and decodes the two columns into a `PerModelAction`
the decision point accepts. Sampling happens on the CPU whatever device the
network is on, so a seeded rollout is device-independent.

`act_batch` does the same for N envs in one forward: every env always has
exactly one pending decision (`close_turn` included), so lockstep rollouts
collate N observations and draw N rows. A `close_turn` row draws nothing --
its selector is all `-inf` -- and carries only the value.

A decoded action the point refuses is a builder/decoder disagreement -- a
bug -- and raises rather than resampling.
"""

from __future__ import annotations

import weakref
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

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
    COMMIT_KEEP,
    NO_COMMIT_DECISION,
    PerModelAction,
    PerModelObservation,
    StepKind,
)
from wargame_rl.wargame.model.per_model.batch import collate
from wargame_rl.wargame.model.per_model.net import HeadLogits, SetNetwork

# The column a `close_turn` point reports: nothing was drawn there.
NO_DRAW = -1


@dataclass(frozen=True)
class StepDecision:
    """What one call decided: the action, its joint log-prob, the value.

    `model` and `column` are the two drawn factors (the selector's row and
    the head's column), so a training update can recompute their log-probs
    without decoding the action back; both are `NO_DRAW` on a `close_turn`
    point, where nothing is drawn. `column >= 0` is "this step has a policy"
    -- and that is not the same as "not a closing step": a decision step that
    terminates the episode is a closing step for the discount and still
    carries its drawn factors.
    """

    action: PerModelAction
    log_prob: float
    value: float
    tokens: TokenObservation
    model: int = NO_DRAW
    column: int = NO_DRAW
    # The commitment decision drawn on an `open` step (#384 Stage 1): its
    # column (0 KEEP, 1 + k objective k), its own log-prob -- never added to
    # the member's, the two streams are trained apart -- and the planning
    # value at the step. `NO_DRAW` when the step offered no decision.
    commit_column: int = NO_DRAW
    commit_log_prob: float = 0.0
    planning_value: float = 0.0

    @property
    def has_policy(self) -> bool:
        return self.column >= 0

    @property
    def has_commitment(self) -> bool:
        return self.commit_column >= 0


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
            # A `close_turn` row's model index is a placeholder: its heads are
            # read and discarded, since it draws no column.
            index = torch.tensor(
                [max(m, 0) for m in models], dtype=torch.int64, device=device
            )
            heads = self.network.heads(output, batch, index)
            offers = [
                not closing[row]
                and models[row] >= 0
                and bool(tokens[row].commit_mask[models[row]].any())
                for row in range(len(envs))
            ]
            commitments = (
                self.network.commitment(output, batch, index) if any(offers) else None
            )
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
            commit_column, commit_log_prob, planning_value = NO_DRAW, 0.0, 0.0
            if commitments is not None and offers[row]:
                commit_logits = commitments.logits[row].float().cpu()
                commit_column, commit_log_prob = _draw(
                    commit_logits, self.greedy, generator
                )
                planning_value = float(commitments.planning_value[row].item())
            action = self._decode(
                point.kind,
                tokens[row],
                env.player_seat,
                model,
                column,
                commit_column=commit_column,
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
                    commit_column=commit_column,
                    commit_log_prob=commit_log_prob,
                    planning_value=planning_value,
                )
            )
        return decisions

    def plan_batch(
        self,
        envs: Sequence[PerModelEnv],
        observations: Sequence[PerModelObservation],
        models: Sequence[int],
        *,
        generator: torch.Generator | None = None,
    ) -> list[int]:
        """This network's commitment draw for the unit of `models[row]` on each
        env's pending open point (#384: the frozen planner), NO_DRAW where the
        point offers that model no decision. Nothing else is decided here; the
        executor's own decision supplies the unit and the declaration."""
        if not (len(envs) == len(observations) == len(models)):
            raise ValueError("one observation and one model per env")
        tokens = [self.observe(env, obs) for env, obs in zip(envs, observations)]
        offers = [
            models[row] >= 0
            and observations[row].decision.kind is StepKind.open
            and bool(tokens[row].commit_mask[models[row]].any())
            for row in range(len(envs))
        ]
        if not any(offers):
            return [NO_DRAW] * len(envs)
        device = self.network.device
        batch = collate(tokens, device=device)
        index = torch.tensor(
            [max(m, 0) for m in models], dtype=torch.int64, device=device
        )
        with torch.no_grad():
            output = self.network(batch)
            commitments = self.network.commitment(output, batch, index)
        columns = [NO_DRAW] * len(envs)
        for row in range(len(envs)):
            if offers[row]:
                columns[row], _ = _draw(
                    commitments.logits[row].float().cpu(), self.greedy, generator
                )
        return columns

    def replace_commitment(
        self, decision: StepDecision, seat: Seat, commit_column: int
    ) -> StepDecision:
        """`decision` with its open action's commitment redrawn from
        `commit_column` (another network's draw) and its own commitment row
        dropped, so the update trains no planning on it."""
        action = self._decode(
            decision.action.kind,
            decision.tokens,
            seat,
            decision.model,
            decision.column,
            commit_column=commit_column,
        )
        return replace(
            decision,
            action=action,
            commit_column=NO_DRAW,
            commit_log_prob=0.0,
            planning_value=0.0,
        )

    @staticmethod
    def _decode(
        kind: StepKind,
        tokens: TokenObservation,
        seat: Seat,
        model: int,
        column: int,
        *,
        commit_column: int = NO_DRAW,
    ) -> PerModelAction:
        if kind is StepKind.open:
            commitment = NO_COMMIT_DECISION
            if commit_column >= 0:
                commitment = COMMIT_KEEP if commit_column == 0 else commit_column - 1
            return PerModelAction.open(model, column, commitment)
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
