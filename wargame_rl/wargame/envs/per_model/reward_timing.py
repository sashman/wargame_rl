"""Reward re-timing for the per-model facade (issue #286).

No term is dropped, and the per-ROUND scalar totals are conserved against the
whole-phase facade — only the step a term is paid on changes:

- **Action terms** pay on the acting model's step, computed for that model
  alone (throughput mitigation 3) and **divided by the alive count**, because
  the whole-phase scalar means per-model terms over the alive — paid
  undivided, an action term would enter the return ``n_alive`` times heavier
  than every weight in the configs was tuned for (measured 24x on
  ``model_kills`` before this divisor existed). Members consumed by a skip
  declaration (remain stationary / hold fire / decline / fight priority) are
  paid on the unit's OPENING step — the step whose decision froze them —
  so standing still costs a unit exactly what the whole-phase facade charges.
- **State terms** (mean over alive) and the **state-like globals** pay once
  per turn cycle on the **turn-closing step**, scaled by the number of
  stepped player phases per round — the whole-phase facade pays them at
  every phase boundary, and on a non-melee config every boundary of a round
  reads the same board, so the scale factor makes the totals equal. ⚠ On a
  melee config (positions change between boundaries within one round) the
  scaled close is an approximation of the per-boundary sum, stated here
  rather than hidden.
- **Delta-like globals** (``vp_gain``, kills, losses, flips) telescope over
  the cycle, so they pay once at the close, unscaled, and are exact.
- **Terminal bonuses** pay on the closing step of the last round, mirroring
  the whole-phase manager's own conditions.

Every calculator object is the phase manager's own — one set of per-episode
state, one registry, one set of weights — so the two facades cannot pay
different mathematics for the same term. A calculator class this module has
not classified is refused loudly: a term silently paid on the wrong step, or
at the wrong scale, is an arm measuring something its config does not say.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext

# Per-model calculators that price the acting model's own action — paid on its
# step, divided by the alive count (the whole-phase scalar's mean). Names are
# calculator CLASS names, the one identity a live phase object carries.
_ACTION_TERM_CLASSES = frozenset(
    {
        "ClosestObjectiveCalculator",
        "ClosestObjectiveV2Calculator",
        "ModelKillsCalculator",
        "ChargeProgressCalculator",
        "DeclaredObjectiveProgressCalculator",
        "DeclaredTargetProgressCalculator",
    }
)
# Per-model calculators that price a STATE — paid once per turn cycle on the
# closing step, aggregated exactly as the whole-phase manager aggregates them
# (mean over alive models), scaled by the stepped phases per round.
# `group_cohesion` is here deliberately: as an action term it fined the FIRST
# mover of a coherent marching unit for a transient mid-phase gap the
# whole-phase facade never scores, teaching the selector to reorder.
_STATE_TERM_CLASSES = frozenset(
    {
        "ObjectiveHoldCalculator",
        "DeclaredObjectiveHoldCalculator",
        "UnitCoherencyCalculator",
        "GroupCohesionCalculator",
    }
)
# Globals that price a DELTA since the previous evaluation. They telescope
# over the cycle: one evaluation at the close pays exactly the whole-phase
# facade's per-round total (verified for vp_gain by
# `test_the_close_pays_the_turns_net_vp`).
_DELTA_GLOBAL_CLASSES = frozenset(
    {
        "VPGainCalculator",
        "KillingReward",
        "ModelsLostPenalty",
        "ObjectiveFlipBonusCalculator",
    }
)
# Globals that price a STATE: the whole-phase facade re-pays them at every
# phase boundary, so the close scales them like the per-model state terms.
_STATE_GLOBAL_CLASSES = frozenset(
    {
        "ObjectiveCoverageCalculator",
        "ModelsAtObjectivesCalculator",
    }
)


class PerModelRewardTimer:
    """Pays the phase manager's terms on the per-model facade's steps.

    ``stepped_phases_per_round`` is the number of player phase steps the
    whole-phase facade takes per battle round on this scenario — the number
    of times it would evaluate every state term and state-like global.
    """

    def __init__(
        self, phase_manager: RewardPhaseManager, stepped_phases_per_round: int
    ) -> None:
        if stepped_phases_per_round < 1:
            raise ValueError(
                f"stepped_phases_per_round must be >= 1, got {stepped_phases_per_round}"
            )
        self.phase_manager = phase_manager
        self.stepped_phases_per_round = stepped_phases_per_round
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
            for _name, global_calculator in phase.global_calculators:
                kind = type(global_calculator).__name__
                if (
                    kind not in _DELTA_GLOBAL_CLASSES
                    and kind not in _STATE_GLOBAL_CLASSES
                ):
                    raise ValueError(
                        f"Global calculator {kind} has no per-model-step "
                        "timing classification. Add it to _DELTA_GLOBAL_CLASSES "
                        "(telescoping delta, paid once per close) or "
                        "_STATE_GLOBAL_CLASSES (re-paid per phase boundary, "
                        "scaled at the close) in per_model/reward_timing.py."
                    )

    def model_step_reward(
        self, view: BattleView, ctx: StepContext, actors: Sequence[int]
    ) -> tuple[float, dict[str, float]]:
        """The step's action-term reward for every model this step resolved.

        ``actors`` is the acting model plus any squadmates a skip declaration
        consumed on this (opening) step. Each alive actor's terms are divided
        by the alive count, matching the whole-phase scalar's mean-over-alive.
        """
        phase = self.phase_manager.current_phase
        total = 0.0
        breakdown: dict[str, float] = {}
        alive_count = sum(1 for m in view.player_models if m.is_alive)
        if alive_count == 0:
            return 0.0, breakdown
        for actor in actors:
            model = view.player_models[actor]
            if not model.is_alive:
                continue
            for name, calculator in phase.per_model_calculators:
                if type(calculator).__name__ not in _ACTION_TERM_CLASSES:
                    continue
                contribution = (
                    calculator.weight
                    * calculator.calculate(actor, model, view, ctx)
                    / alive_count
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
        scale = float(self.stepped_phases_per_round)
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
                mean_paid = paid / len(alive) * scale
                if mean_paid != 0.0:
                    breakdown[name] = breakdown.get(name, 0.0) + mean_paid
                total += mean_paid

        for name, global_calculator in phase.global_calculators:
            contribution = global_calculator.weight * global_calculator.calculate(
                view, ctx
            )
            if type(global_calculator).__name__ in _STATE_GLOBAL_CLASSES:
                contribution *= scale
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
