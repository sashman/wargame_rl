"""Pay a squad for HOLDING the objective it declared — §34's v12-lite, the whole term.

The stripped survivor of two adversarial panels (§34): a CONSTANT pot, no
mission query, no observation columns. It completes the incentive symmetry the
march term broke: `declared_objective_progress` pays *getting to* the
commitment and nothing for keeping it, so a zero-travel commitment (the home
objective — §36: worth +10.4 to the best script) earns nothing through the one
channel measured to steer allocation (§33). This term pays the *kept*
commitment.

Pot semantics, the measured-good form (`crowding_exponent` a=1): each objective
pays its pot ONCE per step, split among the alive models standing on it whose
squad DECLARED it. Spreading declared holders over two points doubles income;
piling a second squad onto a declared point dilutes it. A holder that never
declared is not paid here (it already collects `objective_hold`); a declarer
not standing on its commitment is not paid (that is the march term's job).

Distances are the scoring definition via `ctx.distance_cache` — this term
computes no geometry of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class DeclaredObjectiveHoldCalculator(PerModelRewardCalculator):
    """A constant pot per objective, split among its declaring holders."""

    def __init__(self, weight: float = 1.0, pot: float = 0.25) -> None:
        """
        Args:
            weight: Scales the whole term, as for every calculator.
            pot: What one held-and-declared objective pays per step, in total,
                however many declaring models stand on it. 0.25 matches the
                march term's per-step ceiling so keeping a commitment is worth
                what one full-speed step toward it was.
        """
        super().__init__(weight=weight)
        if pot < 0.0:
            raise ValueError(f"pot must be >= 0, got {pot}")
        self.pot = pot
        self._cached_ctx: StepContext | None = None
        self._cached_pay: np.ndarray | None = None

    def reset_episode(self) -> None:
        """Drop the per-step cache."""
        self._cached_ctx = None
        self._cached_pay = None

    def _payments(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """Per-model payment for this step."""
        if ctx is self._cached_ctx and self._cached_pay is not None:
            return self._cached_pay
        models = view.player_models
        pay = np.zeros(len(models), dtype=float)
        gaps = np.asarray(ctx.distance_cache.model_obj_norms_offset, dtype=float)
        radii = np.asarray(ctx.distance_cache.obj_radii, dtype=float)
        holders: dict[int, list[int]] = {}
        for index, model in enumerate(models):
            declared = int(getattr(model, "declared_objective", -1))
            if not model.is_alive or declared < 0:
                continue
            if declared >= gaps.shape[1]:
                continue
            if gaps[index, declared] <= radii[declared]:
                holders.setdefault(declared, []).append(index)
        for members in holders.values():
            share = self.pot / len(members)
            for index in members:
                pay[index] = share
        self._cached_ctx = ctx
        self._cached_pay = pay
        return pay

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        """Return this model's declared-hold payment."""
        if not model.is_alive:
            return 0.0
        return float(self._payments(view, ctx)[model_idx])
