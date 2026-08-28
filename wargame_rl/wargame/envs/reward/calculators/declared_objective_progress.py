"""Pay a model for closing on the objective ITS SQUAD DECLARED.

The learning form of `baseline/reallocation.py`, and the design the record's
post-mortems converge on:

- The four dead travel terms (`closest_objective_v2` and family) paid approach
  toward targets a HEURISTIC imposed — the agent could neither see nor choose
  the assignment. `charge_progress`'s post-mortem: *"the declaration gate is
  the entire difference."* Here the target is the agent's own command-phase
  declaration (`ActionHandler.declare_objectives`), observable as a one-hot on
  the model token, so the term pays the execution of a commitment the policy
  chose itself.
- The DELTA form, not the level: the level was farmed on `melee-shaping-v4`
  (an annuity on a repeatable state), and the delta failed for CHARGES only
  because the referee's revert restores positions. Ordinary movement never
  reverts, so distance-closed-per-step pays every step of a real march,
  hovering pays nothing, and there is no revert to blank it.

Distances come from `ctx.distance_cache.model_obj_norms_offset` — the SCORING
definition, measured from the base edge to the objective's outline. Three
implementations of "on an objective" once disagreed here on 7.6% of slots;
this term computes no geometry of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class DeclaredObjectiveProgressCalculator(PerModelRewardCalculator):
    """Distance closed toward the squad's own declared objective, per step."""

    def __init__(
        self, weight: float = 1.0, value: float = 0.25, span: float = 6.0
    ) -> None:
        """
        Args:
            weight: Scales the whole term, as for every calculator.
            value: The most a model can earn in one movement step — paid for
                closing a full `span` of distance. 0.25 matches the
                charge-progress arms so magnitudes stay comparable.
            span: The distance, in board units, whose closure earns `value` —
                the maximum normal move, so one full-speed step toward the
                declared objective pays exactly `value`.
        """
        super().__init__(weight=weight)
        if value < 0.0:
            raise ValueError(f"value must be >= 0, got {value}")
        if span <= 0.0:
            raise ValueError(f"span must be > 0, got {span}")
        self.value = value
        self.span = span
        self._prev_gaps: np.ndarray | None = None
        self._cached_ctx: StepContext | None = None
        self._cached_progress: np.ndarray | None = None

    def reset_episode(self) -> None:
        """Drop the per-step cache and the previous step's gaps."""
        self._prev_gaps = None
        self._cached_ctx = None
        self._cached_progress = None

    def _progress(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """Per-model payment fraction for this step."""
        if ctx is self._cached_ctx and self._cached_progress is not None:
            return self._cached_progress
        models = view.player_models
        progress = np.zeros(len(models), dtype=float)
        gaps = np.asarray(ctx.distance_cache.model_obj_norms_offset, dtype=float)
        prev_gaps = self._prev_gaps
        self._prev_gaps = gaps
        # ⚠ `ctx.action_phase`, never the live clock — the clock has already
        # advanced past the phase that acted (the charge_progress defect).
        if (
            ctx.action_phase is BattlePhase.movement
            and prev_gaps is not None
            and prev_gaps.shape == gaps.shape
        ):
            for index, model in enumerate(models):
                declared = int(getattr(model, "declared_objective", -1))
                if not model.is_alive or declared < 0:
                    continue
                if declared >= gaps.shape[1]:
                    continue
                closed = float(prev_gaps[index, declared]) - float(
                    gaps[index, declared]
                )
                progress[index] = float(np.clip(closed / self.span, 0.0, 1.0))
        self._cached_ctx = ctx
        self._cached_progress = progress
        return progress

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        """Return this model's declared-objective progress payment."""
        if not model.is_alive:
            return 0.0
        return float(self.value) * float(self._progress(view, ctx)[model_idx])
