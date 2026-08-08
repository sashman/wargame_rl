from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class GroupCohesionCalculator(PerModelRewardCalculator):
    """Negative reward proportional to distance beyond ``max_distance``
    from the closest same-group model.

    Returns 0 when the model is within range or is alone in its group.
    """

    def __init__(
        self,
        weight: float = 1.0,
        group_max_distance: float | None = None,
        violation_penalty: float | None = None,
    ) -> None:
        super().__init__(weight=weight)
        self.group_max_distance = group_max_distance
        self.violation_penalty = violation_penalty
        # The whole (n_models,) vector is computed to read one element, once per
        # model, so the per-step cost was quadratic in the army for no reason.
        # Keyed on `ctx` identity: a fresh StepContext is built every step and
        # held by the env, the same key `objective_hold` uses. Its own field
        # rather than a shared one -- a key shared between two quantities
        # computed at different points in a step freezes the later one.
        self._cached_ctx: StepContext | None = None
        self._cached_min_distances: np.ndarray | None = None

    def reset_episode(self) -> None:
        """Drop the per-step cohesion-distance cache."""
        self._cached_ctx = None
        self._cached_min_distances = None

    def _min_distances(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """Per-model distance to the nearest live same-group model."""
        if ctx is self._cached_ctx and self._cached_min_distances is not None:
            return self._cached_min_distances
        group_ids = np.array([m.group_id for m in view.player_models], dtype=np.intp)
        alive = alive_mask_for(view.player_models)
        self._cached_min_distances = ctx.distance_cache.min_distances_to_same_group(
            group_ids, alive_mask=alive
        )
        self._cached_ctx = ctx
        return self._cached_min_distances

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        cache = ctx.distance_cache
        if cache.model_model_norms is None:
            return 0.0

        min_dist = float(self._min_distances(view, ctx)[model_idx])
        max_distance = float(
            self.group_max_distance if self.group_max_distance is not None else 10.0
        )
        if min_dist <= max_distance:
            return 0.0

        excess = min_dist - max_distance
        penalty = float(
            self.violation_penalty if self.violation_penalty is not None else -10.0
        )
        return penalty * excess

    @property
    def needs_model_model_distances(self) -> bool:
        return True
