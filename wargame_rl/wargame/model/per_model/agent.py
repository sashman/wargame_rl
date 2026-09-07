"""Drives the per-model facade with a `SetNetwork`: one step, one joint sample.

The step's action is the pair (model, action) — and on a unit's opening step
the triple (model, declaration, action) — whose log-prob is the sum of the
factors' log-probs, exactly the quantity PPO's ratio needs (#286). The agent
owns the one piece of masking the network cannot: whether the advance rungs
are offered depends on the unit's declaration, which on an opening step is
sampled in the same breath.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wargame_rl.wargame.envs.per_model.facade import PerModelEnv
from wargame_rl.wargame.envs.per_model.observation import (
    TokenObservation,
    build_token_observation,
)
from wargame_rl.wargame.envs.per_model.types import (
    MoveDeclaration,
    PerModelAction,
    PerModelObservation,
    StepKind,
)
from wargame_rl.wargame.envs.types import BattlePhase
from wargame_rl.wargame.model.per_model.batch import collate
from wargame_rl.wargame.model.per_model.net import NEG_INF, SetNetwork

# Declarations that consume the unit on its opening step: the action factor is
# then trivially STAY and is not sampled.
_SKIP_DECLARATIONS: dict[BattlePhase, frozenset[int]] = {
    BattlePhase.movement: frozenset({int(MoveDeclaration.remain_stationary)}),
    BattlePhase.shooting: frozenset({1}),
    BattlePhase.charge: frozenset({0}),
}


@dataclass(slots=True)
class StepDecision:
    """One sampled step and the quantities the training loop keeps.

    The trailing fields are what `evaluate_transitions` needs to recompute the
    step's joint log-prob under new weights: which head decided the action (or
    None for declaration-only and closing steps), the choice's index within
    that head's categorical, whether a declaration factor was sampled, and —
    for the movement head — whether the advance rungs were offered, which
    depends on env state at act time and cannot be recovered later.
    """

    action: PerModelAction
    log_prob: float
    value: float
    entropy: float
    selector_entropy: float = 0.0
    head: str | None = None
    head_choice: int | None = None
    declaration_sampled: bool = False
    advance_offered: bool = False
    tokens: TokenObservation | None = None
    """The token observation the decision was sampled under. Carried so the
    rollout buffer can store it instead of rebuilding it — the build is the
    env side's most expensive call and used to run twice per step."""


class SetAgent:
    """Samples (or argmaxes) one facade step from a `SetNetwork`.

    ``declaration_counts`` tallies every sampled unit declaration as
    ``"<phase>:<option>" -> count`` — the passive-attractor instrument: the
    skip declarations gate five models through one logit, so their share is
    the first thing to read when an eval score sits at the do-nothing
    fingerprint. Callers read and reset it around whatever window they score.
    """

    def __init__(self, network: SetNetwork, device: torch.device | None = None):
        self.network = network
        self.device = device or torch.device("cpu")
        self.declaration_counts: dict[str, int] = {}

    def reset_declaration_counts(self) -> dict[str, int]:
        """Return the tallies so far and start a fresh window."""
        counts = self.declaration_counts
        self.declaration_counts = {}
        return counts

    @torch.no_grad()
    def act(
        self,
        env: PerModelEnv,
        observation: PerModelObservation,
        *,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> StepDecision:
        """Decide the next step for `observation`.

        A turn-closing step carries no decision: the action is the empty
        close, the log-prob is zero, and the value is still computed — the
        closing step is a real state PPO bootstraps through.
        """
        token_observation = build_token_observation(env, observation)
        batch = collate([token_observation], device=self.device)
        output = self.network(batch)

        if observation.kind is StepKind.turn_close:
            return StepDecision(
                action=PerModelAction(model_index=None),
                log_prob=0.0,
                value=float(output.value[0]),
                entropy=0.0,
                tokens=token_observation,
            )  # no decision: the closing step still carries a value to bootstrap

        model_index, selector_log_prob, selector_entropy = _pick(
            output.selector_logits[0], greedy, generator
        )
        logits = self.network.action_logits(
            output, batch, torch.tensor([model_index], device=self.device)
        )

        log_prob = selector_log_prob
        entropy = selector_entropy
        declaration: int | None = None
        phase = observation.phase or BattlePhase.movement
        opening = env.unit_needs_declaration(model_index)
        if opening:
            declaration, declaration_log_prob, declaration_entropy = _pick(
                logits.declaration[0], greedy, generator
            )
            log_prob += declaration_log_prob
            entropy += declaration_entropy
            key = f"{phase.value}:{declaration}"
            self.declaration_counts[key] = self.declaration_counts.get(key, 0) + 1

        action = 0
        head: str | None = None
        head_choice: int | None = None
        advance_offered = False
        skip = declaration is not None and declaration in _SKIP_DECLARATIONS.get(
            phase, frozenset()
        )
        if phase is BattlePhase.fight or skip:
            # Declaration-only steps: the action factor is trivially STAY.
            pass
        elif phase is BattlePhase.shooting:
            choice, choice_log_prob, choice_entropy = _pick(
                logits.target[0], greedy, generator
            )
            log_prob += choice_log_prob
            entropy += choice_entropy
            action = self._shooting_action(env, choice)
            head, head_choice = "target", choice
        else:
            advance_offered = self.advance_rungs_offered(env, model_index, declaration)
            combined = _combined_movement_logits(
                logits.displacement[0], logits.advance[0], advance_offered
            )
            choice, choice_log_prob, choice_entropy = _pick(combined, greedy, generator)
            log_prob += choice_log_prob
            entropy += choice_entropy
            action = self._movement_action(env, choice)
            head, head_choice = "movement", choice

        return StepDecision(
            action=PerModelAction(
                model_index=model_index, action=action, declaration=declaration
            ),
            log_prob=log_prob,
            value=float(output.value[0]),
            entropy=entropy,
            selector_entropy=selector_entropy,
            head=head,
            head_choice=head_choice,
            declaration_sampled=opening,
            advance_offered=advance_offered,
            tokens=token_observation,
        )

    @staticmethod
    def advance_rungs_offered(
        env: PerModelEnv, model_index: int, declaration: int | None
    ) -> bool:
        """Whether the advance rungs join the movement categorical this step.

        Exactly when the model's unit has declared an advance — already (the
        env's own mask then also carries them), or in this very step's
        declaration factor.
        """
        return bool(env.wargame_models[model_index].declared_advance) or (
            declaration == int(MoveDeclaration.advance)
        )

    def _movement_action(self, env: PerModelEnv, choice: int) -> int:
        handler = env.player_action_handler
        n_displacement = 1 + handler.n_move_actions
        if choice == 0:
            return 0  # STAY
        if choice < n_displacement:
            return handler.movement_slice.start + (choice - 1)
        advance_slice = handler.advance_slice
        assert advance_slice is not None
        return advance_slice.start + (choice - n_displacement)

    def _shooting_action(self, env: PerModelEnv, choice: int) -> int:
        if choice == 0:
            return 0  # hold fire = STAY
        shooting_slice = env.player_action_handler.shooting_slice
        assert shooting_slice is not None
        unit_ids = sorted(
            {int(m.group_id) for m in env.opponent_models}
        )  # the pointer's candidate order (ascending unit id)
        return shooting_slice.start + unit_ids[choice - 1]

    @torch.no_grad()
    def act_batched(
        self, pairs: list[tuple[PerModelEnv, PerModelObservation]]
    ) -> list[PerModelAction]:
        """Greedy decisions for several envs in ONE forward pass.

        The batched evaluation path: per-model episodes take different step
        counts, so waves run until the last env finishes with finished envs
        simply absent from the batch — this decides one step for every env
        still in flight. Greedy only; the sampling path stays `act`.
        """
        observations = [
            build_token_observation(env, observation) for env, observation in pairs
        ]
        batch = collate(observations, device=self.device)
        output = self.network(batch)
        chosen: list[int] = []
        for row, (_env, observation) in enumerate(pairs):
            if observation.kind is StepKind.turn_close:
                chosen.append(0)  # placeholder; the row's heads are unread
            else:
                chosen.append(int(torch.argmax(output.selector_logits[row])))
        logits = self.network.action_logits(
            output, batch, torch.tensor(chosen, device=self.device)
        )
        actions: list[PerModelAction] = []
        for row, (env, observation) in enumerate(pairs):
            if observation.kind is StepKind.turn_close:
                actions.append(PerModelAction(model_index=None))
                continue
            model_index = chosen[row]
            phase = observation.phase or BattlePhase.movement
            declaration: int | None = None
            if env.unit_needs_declaration(model_index):
                declaration = _pick(logits.declaration[row], True, None)[0]
            action = 0
            skip = declaration is not None and declaration in _SKIP_DECLARATIONS.get(
                phase, frozenset()
            )
            if phase is BattlePhase.fight or skip:
                pass
            elif phase is BattlePhase.shooting:
                choice = _pick(logits.target[row], True, None)[0]
                action = self._shooting_action(env, choice)
            else:
                combined = _combined_movement_logits(
                    logits.displacement[row],
                    logits.advance[row],
                    self.advance_rungs_offered(env, model_index, declaration),
                )
                choice = _pick(combined, True, None)[0]
                action = self._movement_action(env, choice)
            actions.append(
                PerModelAction(
                    model_index=model_index, action=action, declaration=declaration
                )
            )
        return actions


def _combined_movement_logits(
    displacement: torch.Tensor, advance: torch.Tensor, advance_offered: bool
) -> torch.Tensor:
    """STAY + movement bins, with the advance rungs appended when offered."""
    if advance.shape[-1] == 0 or not advance_offered:
        advance = torch.full_like(advance, NEG_INF)
    return torch.cat([displacement, advance], dim=-1)


def _pick(
    logits: torch.Tensor, greedy: bool, generator: torch.Generator | None
) -> tuple[int, float, float]:
    """Sample (or argmax) one masked categorical; return index, log-prob, entropy."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    finite = torch.isfinite(log_probs)
    entropy = float(-(probs[finite] * log_probs[finite]).sum())
    if greedy:
        index = int(torch.argmax(log_probs))
    else:
        # Sampling happens on CPU whatever device the network runs on: a CUDA
        # multinomial refuses a CPU generator, and a CPU draw keeps a seeded
        # rollout's action stream identical across devices.
        index = int(torch.multinomial(probs.cpu(), 1, generator=generator))
    return index, float(log_probs[index]), entropy
