"""Reward re-timing for the per-model facade (issue #286).

No term is dropped and no term's mathematics changes — only the step it is
paid on:

- **Action terms** pay on the acting model's step, computed for that model
  alone (throughput mitigation 3): the per-model calculators that price what
  a model's own action changed — progress, kills, cohesion.
- **State terms, the global terms and VP** pay once per turn cycle on the
  **turn-closing step**, so the global signal has a step of its own instead of
  being broadcast onto 25 model actions, and a term's per-round total does not
  scale with the army size (per-model state terms keep the whole-phase
  facade's mean-over-alive aggregation for exactly that reason).
- **Terminal bonuses** pay on the closing step of the last round, mirroring
  the whole-phase manager's own conditions.

Every calculator object is the phase manager's own — one set of per-episode
state, one registry, one set of weights — so the two facades cannot pay
different mathematics for the same term. A per-model calculator class this
module has not classified is refused loudly: a term silently defaulting to
the wrong step is an arm measuring something its config does not say.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext

# Per-model calculators that price the acting model's own action — paid on its
# step. Names are calculator CLASS names, the one identity a live phase object
# carries.
_ACTION_TERM_CLASSES = frozenset(
    {
        "ClosestObjectiveCalculator",
        "ClosestObjectiveV2Calculator",
        "ModelKillsCalculator",
        "GroupCohesionCalculator",
        "ChargeProgressCalculator",
        "DeclaredObjectiveProgressCalculator",
        "DeclaredTargetProgressCalculator",
    }
)
# Per-model calculators that price a STATE — paid once per turn cycle on the
# closing step, aggregated exactly as the whole-phase manager aggregates them
# (mean over alive models).
_STATE_TERM_CLASSES = frozenset(
    {
        "ObjectiveHoldCalculator",
        "DeclaredObjectiveHoldCalculator",
        "UnitCoherencyCalculator",
        "ModelsAtObjectivesCalculator",
    }
)


class PerModelRewardTimer:
    """Pays the phase manager's terms on the per-model facade's steps."""

    def __init__(self, phase_manager: RewardPhaseManager) -> None:
        self.phase_manager = phase_manager
        for phase in phase_manager.phases:
            for _name, calculator in phase.per_model_calculators:
                kind = type(calculator).__name__
                if kind not in _ACTION_TERM_CLASSES and kind not in _STATE_TERM_CLASSES:
                    raise ValueError(
                        f"Per-model calculator {kind} has no per-model-step "
                        "timing classification. Add it to _ACTION_TERM_CLASSES "
                        "or _STATE_TERM_CLASSES in per_model/reward_timing.py "
                        "— a term silently paid on the wrong step is an arm "
                        "measuring something its config does not say."
                    )

    def model_step_reward(
        self, view: BattleView, ctx: StepContext, actor: int
    ) -> tuple[float, dict[str, float]]:
        """The acting model's own reward: action terms, for the actor alone."""
        phase = self.phase_manager.current_phase
        model = view.player_models[actor]
        total = 0.0
        breakdown: dict[str, float] = {}
        if not model.is_alive:
            return 0.0, breakdown
        for name, calculator in phase.per_model_calculators:
            if type(calculator).__name__ not in _ACTION_TERM_CLASSES:
                continue
            contribution = calculator.weight * calculator.calculate(
                actor, model, view, ctx
            )
            if contribution != 0.0:
                breakdown[name] = breakdown.get(name, 0.0) + contribution
            total += contribution
        return total, breakdown

    def closing_reward(
        self, view: BattleView, ctx: StepContext
    ) -> tuple[float, dict[str, float]]:
        """The turn-closing step: state terms, globals, VP, terminal bonuses.

        `ctx` spans the whole turn cycle (its kill and damage tallies are the
        cycle's, and the view's VP deltas accumulate since the last close), so
        `vp_gain` prices the net delta of the turn as the design requires.
        """
        phase = self.phase_manager.current_phase
        total = 0.0
        breakdown: dict[str, float] = {}

        alive = [(i, m) for i, m in enumerate(view.player_models) if m.is_alive]
        if alive:
            for name, calculator in phase.per_model_calculators:
                if type(calculator).__name__ not in _STATE_TERM_CLASSES:
                    continue
                paid = sum(
                    calculator.weight * calculator.calculate(i, model, view, ctx)
                    for i, model in alive
                )
                mean_paid = paid / len(alive)
                if mean_paid != 0.0:
                    breakdown[name] = breakdown.get(name, 0.0) + mean_paid
                total += mean_paid

        for name, global_calculator in phase.global_calculators:
            contribution = global_calculator.weight * global_calculator.calculate(
                view, ctx
            )
            if contribution != 0.0:
                breakdown[name] = breakdown.get(name, 0.0) + contribution
            total += contribution

        # The terminal bonuses, exactly the whole-phase manager's conditions.
        if ctx.is_terminated and phase.terminal_success_bonus != 0.0:
            if phase.criteria.is_successful(view, ctx):
                if phase.terminate_on_success:
                    remaining = max(0.0, float(ctx.max_turns - ctx.current_turn + 1))
                    denominator = float(ctx.max_turns) if ctx.max_turns > 0 else 1.0
                    speed_scale = remaining / denominator
                else:
                    speed_scale = 1.0
                bonus = phase.terminal_success_bonus * speed_scale
                if bonus != 0.0:
                    breakdown["terminal_success_bonus"] = bonus
                total += bonus
        if ctx.is_terminated and phase.terminal_vp_bonus != 0.0:
            vp_threshold = phase.criteria.vp_threshold_for_terminal_bonus(view)
            if vp_threshold is not None and view.player_vp >= vp_threshold:
                bonus = phase.terminal_vp_bonus
                breakdown["terminal_vp_bonus"] = bonus
                total += bonus

        return total, breakdown

    def state_term_names(self) -> set[str]:
        """The current phase's close-paid per-model term names, for tests."""
        return {
            name
            for name, calculator in self.phase_manager.current_phase.per_model_calculators
            if type(calculator).__name__ in _STATE_TERM_CLASSES
        }


def kills_by_model_for_step(n_models: int, actor: int, kills: int) -> np.ndarray:
    """A `player_kills_by_model` vector attributing this step's kills to its actor."""
    vector = np.zeros(n_models, dtype=np.int64)
    if 0 <= actor < n_models:
        vector[actor] = kills
    return vector
