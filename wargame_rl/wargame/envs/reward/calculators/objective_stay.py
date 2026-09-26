"""Per-model reward for ENDING a step inside an objective, the objective's
pot split among its occupants -- paid to the mover on its own step."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.env_components.distance_cache import (
    objective_counts_from_norms_offset,
)
from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel

DEFAULT_CROWDING_EXPONENT = 1.0


class ObjectiveStayCalculator(PerModelRewardCalculator):
    """Pays a model for standing inside an objective at the end of its step.

    Built for the per-model curriculum's five-objective half-step (#340),
    where every trained per-model policy was measured to stand still on
    0-2% of its decisions on an objective and to walk out of it on 60-81%,
    because the reward paid the same for staying as for leaving to within a
    few thousandths: the travel term is zero inside an objective, the
    coverage term is a broadcast mean at the turn close, and the bonus is
    terminal. ``objective_hold`` prices the same standing, but on the
    per-model facade it is a STATE term -- paid at the close as the army
    mean, so the body that stayed and the body that left are paid alike.
    This term is classed as an action term there: it is computed for the
    MOVER, on its own step, from the board after its move, so a body that
    ends inside an objective is paid and one that walks out is not.

    The objective pays a pot divided by ``occupants ** crowding_exponent``
    (the ``objective_hold`` mechanism): at the default exponent 1.0 a body
    alone on an objective earns the whole value and three squadmates a
    third each, so spreading onto an empty objective strictly raises income
    and stacking does not multiply it. Enemy models are ignored; the term
    prices presence, not control. Returns unweighted.
    """

    def __init__(
        self,
        weight: float = 1.0,
        crowding_exponent: float = DEFAULT_CROWDING_EXPONENT,
        cap_per_commitment: float | None = None,
    ) -> None:
        super().__init__(weight=weight)
        if crowding_exponent < 0.0:
            raise ValueError(
                f"crowding_exponent must be >= 0, got {crowding_exponent}: a "
                "negative exponent would pay *more* for crowding."
            )
        if cap_per_commitment is not None and cap_per_commitment <= 0.0:
            raise ValueError(
                f"cap_per_commitment must be > 0, got {cap_per_commitment}"
            )
        self.crowding_exponent = crowding_exponent
        # The two-stream reward's second constraint (#384, 2026-09-24): paid
        # per step, a near objective reached early collects more holding
        # steps than a far one, so a near plan out-pays a far one for the
        # members. With a cap, the term pays at most this much (in its own
        # unweighted units) per model per commitment -- a fresh budget when
        # the model's committed objective changes -- so the maximum a
        # commitment can pay is the same whatever the plan. None: uncapped,
        # every recorded number.
        self.cap_per_commitment = cap_per_commitment
        self._paid: dict[int, tuple[int, float]] = {}

    def reset_episode(self) -> None:
        """Clear the per-commitment budgets (called by the env on reset)."""
        self._paid.clear()

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        cache = ctx.distance_cache
        norms = cache.model_obj_norms_offset
        if norms.size == 0:
            return 0.0
        inside = np.flatnonzero(norms[model_idx] <= cache.obj_radii)
        # The commitment layer (#384): with a committed objective, only ending
        # inside THAT objective pays; a body on someone else's earns nothing.
        committed = ctx.committed_objective
        if committed is not None and int(committed[model_idx]) >= 0:
            inside = inside[inside == int(committed[model_idx])]
        if inside.size == 0:
            return 0.0
        objective = int(inside[0])
        counts = objective_counts_from_norms_offset(norms, cache.obj_radii)
        occupants = max(1, int(counts[objective]))
        value = float(1.0 / float(occupants) ** self.crowding_exponent)
        if self.cap_per_commitment is None:
            return value
        key = int(committed[model_idx]) if committed is not None else objective
        previous = self._paid.get(model_idx)
        paid = previous[1] if previous is not None and previous[0] == key else 0.0
        value = min(value, max(0.0, self.cap_per_commitment - paid))
        self._paid[model_idx] = (key, paid + value)
        return value
