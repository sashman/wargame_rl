"""Pay a model for closing on the enemy unit ITS SQUAD DECLARED.

The hunt form of `declared_objective_progress`, and the charge-teaching design
§33 licenses: the dark pair measured the objective term CAUSAL at +30.4 ± 6.7
(6/6 seeds), so the recipe — a command-phase commitment priced by a
movement-phase execution delta — is proven, and this term aims it at the bar's
remaining edge (it stands ~6 charges/episode against the agent's ~2, and the
measured value of a standing charge is the shooting shield).

Why the MOVEMENT phase and not the charge phase: the charge-phase forms are
both measured dead — the level is an annuity (§17, farmed while hovering) and
the charge-move delta is blanked because the referee's revert restores the
positions the delta would price (§18). Ordinary movement never reverts, so the
march INTO charge range pays every real step, and the charge itself stays
priced by what it wins (the approach mask converts arrival into standing
charges without any charge-phase payment).

The gap is edge-to-edge to the NEAREST ALIVE member of the declared group —
charge-range semantics — clipped at zero exactly as the objective form is:
v10's farm screen measured that form clean (top unit 25–33% of cap against a
60% tripwire), and §31 S1 re-arms the same screen for this term at landing.
A declared group with no alive member pays nothing (and legality masks its
re-declaration off).
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


class DeclaredTargetProgressCalculator(PerModelRewardCalculator):
    """Distance closed toward the squad's own declared enemy unit, per step."""

    def __init__(
        self, weight: float = 1.0, value: float = 0.25, span: float = 6.0
    ) -> None:
        """
        Args:
            weight: Scales the whole term, as for every calculator.
            value: The most a model can earn in one movement step — paid for
                closing a full `span` of distance. 0.25 matches
                `declared_objective_progress` so the two commitments are
                priced identically and neither outbids the other by fiat.
            span: The distance whose closure earns `value` — the maximum
                normal move.
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

    @staticmethod
    def _group_gaps(view: BattleView) -> np.ndarray:
        """`(n_models, n_groups)` edge-to-edge gap to each enemy group's nearest alive member.

        The full matrix, not just the declared column, so a re-declaration
        re-anchors against a gap that was measured last step — the same
        no-windfall property `declared_objective_progress` gets from the
        distance cache holding every objective.
        """
        models = view.player_models
        enemies = view.opponent_models
        n_groups = 1 + max((int(m.group_id) for m in enemies), default=-1)
        gaps = np.full((len(models), max(n_groups, 1)), np.inf, dtype=float)
        if not enemies:
            return gaps
        locs = np.array([m.location for m in models], dtype=float)
        radii = np.array([float(m.base_radius) for m in models], dtype=float)
        for group in range(n_groups):
            member_locs = [
                (m.location, float(m.base_radius))
                for m in enemies
                if m.is_alive and int(m.group_id) == group
            ]
            if not member_locs:
                continue
            enemy_locs = np.array([loc for loc, _ in member_locs], dtype=float)
            enemy_radii = np.array([r for _, r in member_locs], dtype=float)
            dists = np.linalg.norm(locs[:, None, :] - enemy_locs[None, :, :], axis=2)
            edge = dists - radii[:, None] - enemy_radii[None, :]
            gaps[:, group] = np.maximum(edge.min(axis=1), 0.0)
        return gaps

    def _progress(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """Per-model payment fraction for this step."""
        if ctx is self._cached_ctx and self._cached_progress is not None:
            return self._cached_progress
        models = view.player_models
        progress = np.zeros(len(models), dtype=float)
        gaps = self._group_gaps(view)
        prev_gaps = self._prev_gaps
        self._prev_gaps = gaps
        if (
            ctx.action_phase is BattlePhase.movement
            and prev_gaps is not None
            and prev_gaps.shape == gaps.shape
        ):
            for index, model in enumerate(models):
                declared = int(getattr(model, "declared_target", -1))
                if not model.is_alive or declared < 0:
                    continue
                if declared >= gaps.shape[1]:
                    continue
                previous = float(prev_gaps[index, declared])
                current = float(gaps[index, declared])
                if not np.isfinite(previous) or not np.isfinite(current):
                    continue
                closed = previous - current
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
        """Return this model's declared-target progress payment."""
        if not model.is_alive:
            return 0.0
        return float(self.value) * float(self._progress(view, ctx)[model_idx])
