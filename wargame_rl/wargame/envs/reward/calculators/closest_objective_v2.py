from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ClosestObjectiveV2Calculator(PerModelRewardCalculator):
    """Closest-objective shaping that targets only net-VP-impactful objectives.

    A target objective is considered net-positive for the player when this model's
    arrival improves objective state:
    - neutral -> player-controlled
    - opponent-controlled -> contested
    - contested -> player-controlled

    Control state here is based on in-range model counts:
    - player-controlled: player_count >= opponent_count + 1
    - opponent-controlled: opponent_count >= player_count + 1
    - contested: both present, neither side has +1 margin
    - neutral: neither side present

    Each objective is assigned to at most one player group for "move closer"
    shaping on a step. Other groups cannot earn shaping reward for that objective.
    If no net-positive target exists for a model, this calculator returns 0 unless
    over-stack penalty applies.
    """

    def __init__(
        self,
        weight: float = 1.0,
        best_distance_bonus_scale: float | None = None,
        overstack_penalty_per_extra: float = 0.05,
        non_improvement_penalty_slope: float = 2.0,
        non_improvement_penalty_base: float = 0.3,
        progress_scale: float = 0.0,
        fallback_to_nearest: bool = False,
        contest_deficit: int = 1,
    ) -> None:
        super().__init__(weight=weight)
        if contest_deficit < 1:
            raise ValueError(
                "contest_deficit must be >= 1; 1 reproduces the one-model gate "
                f"this calculator has always used, got {contest_deficit}"
            )
        self.best_distance_bonus_scale = (
            0.0 if best_distance_bonus_scale is None else best_distance_bonus_scale
        )
        self.overstack_penalty_per_extra = overstack_penalty_per_extra
        self.non_improvement_penalty_slope = non_improvement_penalty_slope
        self.non_improvement_penalty_base = non_improvement_penalty_base
        # When > 0, use a smooth linear potential pull toward the assigned objective
        # (reward = progress_scale * distance_closed) instead of the legacy flat
        # penalty + cubic best-distance bonus. This is the recommended "go to the
        # right objective" shaping.
        self.progress_scale = progress_scale
        # When True, a model with no VP-relevant/assigned target falls back to its
        # nearest objective (so e.g. extra models, when models > objectives, are
        # still guided to an objective). Keep False for strict de-stacking.
        self.fallback_to_nearest = fallback_to_nearest
        # How far behind on an objective we may be and still be paid to walk at
        # it. The gate below asks whether an arrival IMPROVES the control label;
        # at the historical value of 1 that silently means "one more model must
        # flip it", so an objective the opponent holds by two or more can never
        # be a candidate and the travel reward never points at it. Measured on
        # the v3.0 lineage, that excluded 43.4% of all non-candidate objectives,
        # left 68% of units unassigned, and dropped them through
        # `fallback_to_nearest` onto the point they were already standing on --
        # i.e. the agent was never paid to attack. Raising this to a unit's
        # worth of models lets a whole unit's arrival count.
        self.contest_deficit = contest_deficit
        self._last_breakdown: dict[int, dict[str, float]] = {}
        self._target_obj_idx: dict[int, int] = {}
        self._previous_target_distance: dict[int, float] = {}
        self._best_target_distance: dict[int, float] = {}
        self._cached_step_key: tuple[int, int] | None = None
        self._cached_player_in_range: np.ndarray | None = None
        self._cached_player_counts: np.ndarray | None = None
        self._cached_opponent_counts: np.ndarray | None = None
        self._cached_group_assignment: dict[int, int] | None = None
        self._cached_candidate_mask: np.ndarray | None = None

    @staticmethod
    def _normalized_distance(ctx: StepContext, distance_to_objective: float) -> float:
        max_diagonal = float((ctx.board_width**2 + ctx.board_height**2) ** 0.5)
        return distance_to_objective / max_diagonal

    def _penalty_for_non_improvement(
        self, previous_model_distance: float, current_distance: float
    ) -> float:
        distance_delta = float(current_distance - previous_model_distance)
        if distance_delta >= 0:
            return -(
                self.non_improvement_penalty_slope * abs(distance_delta)
                + self.non_improvement_penalty_base
            )
        return 0.0

    def _best_distance_bonus(
        self, previous_best: float | None, current_distance: float
    ) -> float:
        if previous_best is None or current_distance >= previous_best:
            return 0.0
        if self.best_distance_bonus_scale == 0.0:
            return 0.0
        improvement = float(previous_best - current_distance)
        return self.best_distance_bonus_scale * (improvement**3)

    def _clear_model_state(self, model_idx: int) -> None:
        self._target_obj_idx.pop(model_idx, None)
        self._previous_target_distance.pop(model_idx, None)
        self._best_target_distance.pop(model_idx, None)

    def reset_episode(self) -> None:
        """Clear per-episode state (called by the env on reset)."""
        self._target_obj_idx.clear()
        self._previous_target_distance.clear()
        self._best_target_distance.clear()
        self._cached_step_key = None
        self._cached_player_in_range = None
        self._cached_player_counts = None
        self._cached_opponent_counts = None
        self._cached_group_assignment = None
        self._cached_candidate_mask = None

    def _objective_presence_masks(
        self,
        view: BattleView,
        ctx: StepContext,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        step_key = (ctx.current_turn, id(ctx.distance_cache))
        if self._cached_step_key != step_key:
            self._cached_group_assignment = None
            self._cached_candidate_mask = None
        if (
            self._cached_step_key == step_key
            and self._cached_player_in_range is not None
            and self._cached_player_counts is not None
            and self._cached_opponent_counts is not None
        ):
            return (
                self._cached_player_in_range,
                self._cached_player_counts,
                self._cached_opponent_counts,
            )

        cache = ctx.distance_cache
        player_in_range = cache.model_obj_norms_offset <= cache.obj_radii
        player_counts = np.sum(player_in_range, axis=0)

        n_obj = len(view.objectives)
        if view.opponent_models:
            opponent_cache = ctx.opponent_distances(view)
            opponent_in_range = (
                opponent_cache.model_obj_norms_offset <= opponent_cache.obj_radii
            )
            opponent_counts = np.sum(opponent_in_range, axis=0)
        else:
            opponent_counts = np.zeros(n_obj, dtype=int)

        self._cached_step_key = step_key
        self._cached_player_in_range = player_in_range
        self._cached_player_counts = player_counts
        self._cached_opponent_counts = opponent_counts
        return player_in_range, player_counts, opponent_counts

    @staticmethod
    def _state_label(player_count: int, opponent_count: int) -> str:
        if player_count <= 0 and opponent_count <= 0:
            return "neutral"
        if player_count >= opponent_count + 1:
            return "player"
        if opponent_count >= player_count + 1:
            return "opponent"
        return "contested"

    def _is_positive_transition(
        self,
        player_count: int,
        opponent_count: int,
    ) -> bool:
        """Would arriving here improve the control label within reach?

        `contest_deficit` is how many models we are allowed to imagine arriving.
        At 1 this is exactly the historical test -- one more model must move the
        label -- and every recorded number was measured under it.
        """
        current = self._state_label(player_count, opponent_count)
        for arrivals in range(1, self.contest_deficit + 1):
            next_state = self._state_label(player_count + arrivals, opponent_count)
            if (
                (current == "neutral" and next_state == "player")
                or (current == "opponent" and next_state == "contested")
                or (current == "opponent" and next_state == "player")
                or (current == "contested" and next_state == "player")
            ):
                return True
        return False

    def _compute_group_assignment(
        self,
        view: BattleView,
        cache: DistanceCache,
        candidate_mask: np.ndarray,
        step_key: tuple[int, int],
    ) -> dict[int, int]:
        if (
            self._cached_step_key == step_key
            and self._cached_group_assignment is not None
        ):
            return self._cached_group_assignment

        group_ids = np.array([m.group_id for m in view.player_models], dtype=int)
        n_obj = candidate_mask.shape[1]
        assignment: dict[int, int] = {}
        for obj_idx in range(n_obj):
            model_idxs = np.flatnonzero(candidate_mask[:, obj_idx])
            if model_idxs.size == 0:
                continue
            best_group: int | None = None
            best_distance: float | None = None
            for idx in model_idxs:
                group_id = int(group_ids[idx])
                dist = float(cache.model_obj_norms_offset[idx, obj_idx])
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_group = group_id
            if best_group is not None:
                assignment[obj_idx] = best_group

        self._cached_group_assignment = assignment
        return assignment

    def _candidate_mask(
        self,
        player_in_range: np.ndarray,
        player_counts: np.ndarray,
        opponent_counts: np.ndarray,
        step_key: tuple[int, int],
    ) -> np.ndarray:
        """``(n_models, n_objectives)``: would this model's arrival improve control?

        The mask does not depend on which model is being scored, so it is built
        once per step rather than once per model. Each row was previously
        ``idx_outside & idx_positive`` where ``idx_positive[obj]`` is only ever
        set where ``idx_outside[obj]`` holds, and ``_is_positive_transition``
        reads only the two objective-level counts — so the whole thing reduces
        to ``(~player_in_range) & positive_transition``, evaluated per objective.

        That the mask is model-independent is not a new assumption: the group
        assignment derived from it is already memoised per step, which is only
        correct because the mask is the same for every model.
        """
        if (
            self._cached_step_key == step_key
            and self._cached_candidate_mask is not None
        ):
            return self._cached_candidate_mask

        positive_transition = np.array(
            [
                self._is_positive_transition(
                    int(player_counts[obj_idx]), int(opponent_counts[obj_idx])
                )
                for obj_idx in range(player_in_range.shape[1])
            ],
            dtype=bool,
        )
        mask: np.ndarray = (~player_in_range) & positive_transition[np.newaxis, :]
        self._cached_candidate_mask = mask
        return mask

    def _overstack_penalty(
        self,
        model_idx: int,
        player_in_range: np.ndarray,
        player_counts: np.ndarray,
        opponent_counts: np.ndarray,
    ) -> float:
        if self.overstack_penalty_per_extra <= 0:
            return 0.0
        penalty = 0.0
        for obj_idx in range(player_in_range.shape[1]):
            if not bool(player_in_range[model_idx, obj_idx]):
                continue
            p_count = int(player_counts[obj_idx])
            o_count = int(opponent_counts[obj_idx])
            if p_count >= o_count + 2:
                extra = p_count - (o_count + 1)
                penalty -= self.overstack_penalty_per_extra * float(extra)
        return penalty

    def _choose_target_objective(
        self,
        model_idx: int,
        view: BattleView,
        step_key: tuple[int, int],
        cache: DistanceCache,
        player_in_range: np.ndarray,
        player_counts: np.ndarray,
        opponent_counts: np.ndarray,
    ) -> int | None:
        model_distances = cache.model_obj_norms_offset[model_idx]
        n_obj = model_distances.shape[0]
        # Objective-to-group assignment: one objective can reward only one group,
        # so the mask covers every model, not just this one.
        candidate_mask = self._candidate_mask(
            player_in_range, player_counts, opponent_counts, step_key
        )

        assignment = self._compute_group_assignment(
            view, cache, candidate_mask, step_key
        )
        model_group = int(view.player_models[model_idx].group_id)
        allowed = np.array(
            [assignment.get(obj_idx, None) == model_group for obj_idx in range(n_obj)],
            dtype=bool,
        )
        candidates = candidate_mask[model_idx] & allowed
        if not np.any(candidates):
            if self.fallback_to_nearest:
                # No VP-relevant/assigned target: still pull this model toward its
                # nearest objective (handles extra models when models > objectives).
                return int(np.argmin(model_distances))
            return None

        candidate_indices = np.flatnonzero(candidates)
        closest_idx = int(
            candidate_indices[np.argmin(model_distances[candidate_indices])]
        )
        return closest_idx

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        cache = ctx.distance_cache
        step_key = (ctx.current_turn, id(ctx.distance_cache))

        player_in_range, player_counts, opponent_counts = (
            self._objective_presence_masks(view, ctx)
        )
        target_obj_idx = self._choose_target_objective(
            model_idx=model_idx,
            view=view,
            step_key=step_key,
            cache=cache,
            player_in_range=player_in_range,
            player_counts=player_counts,
            opponent_counts=opponent_counts,
        )
        overstack_penalty = self._overstack_penalty(
            model_idx=model_idx,
            player_in_range=player_in_range,
            player_counts=player_counts,
            opponent_counts=opponent_counts,
        )

        if target_obj_idx is None:
            self._clear_model_state(model_idx)
            self._last_breakdown[model_idx] = {
                "target_obj_idx": -1.0,
                "target_switched": 0.0,
                "distance_delta": 0.0,
                "base_penalty": 0.0,
                "best_distance_bonus": 0.0,
                "overstack_penalty": overstack_penalty,
            }
            return overstack_penalty

        distance_to_target = float(
            cache.model_obj_norms_offset[model_idx, target_obj_idx]
        )
        normalized_distance = self._normalized_distance(ctx, distance_to_target)

        previous_target = self._target_obj_idx.get(model_idx)
        if previous_target != target_obj_idx:
            self._target_obj_idx[model_idx] = target_obj_idx
            self._previous_target_distance[model_idx] = normalized_distance
            self._best_target_distance[model_idx] = normalized_distance
            self._last_breakdown[model_idx] = {
                "target_obj_idx": float(target_obj_idx),
                "target_switched": 1.0,
                "distance_delta": 0.0,
                "base_penalty": 0.0,
                "best_distance_bonus": 0.0,
                "overstack_penalty": overstack_penalty,
            }
            return overstack_penalty

        previous = self._previous_target_distance.get(model_idx)
        self._previous_target_distance[model_idx] = normalized_distance

        best_prev = self._best_target_distance.get(model_idx)
        if best_prev is None or normalized_distance < best_prev:
            self._best_target_distance[model_idx] = normalized_distance

        if self.progress_scale > 0.0:
            # Linear potential pull toward the assigned objective: positive when
            # closing distance, negative when receding. Smooth, policy-invariant.
            distance_delta = (
                0.0 if previous is None else float(normalized_distance - previous)
            )
            progress = (
                0.0 if previous is None else self.progress_scale * (-distance_delta)
            )
            self._last_breakdown[model_idx] = {
                "target_obj_idx": float(target_obj_idx),
                "target_switched": 0.0,
                "distance_delta": distance_delta,
                "base_penalty": 0.0,
                "best_distance_bonus": 0.0,
                "progress": progress,
                "overstack_penalty": overstack_penalty,
            }
            return progress + overstack_penalty

        bonus = self._best_distance_bonus(best_prev, normalized_distance)

        if previous is None:
            self._last_breakdown[model_idx] = {
                "target_obj_idx": float(target_obj_idx),
                "target_switched": 0.0,
                "distance_delta": 0.0,
                "base_penalty": 0.0,
                "best_distance_bonus": bonus,
                "overstack_penalty": overstack_penalty,
            }
            return bonus + overstack_penalty

        distance_delta = float(normalized_distance - previous)
        base_penalty = self._penalty_for_non_improvement(
            float(previous), normalized_distance
        )
        self._last_breakdown[model_idx] = {
            "target_obj_idx": float(target_obj_idx),
            "target_switched": 0.0,
            "distance_delta": distance_delta,
            "base_penalty": base_penalty,
            "best_distance_bonus": bonus,
            "overstack_penalty": overstack_penalty,
        }
        return base_penalty + bonus + overstack_penalty

    def get_last_breakdown(self, model_idx: int) -> dict[str, float]:
        return self._last_breakdown.get(model_idx, {})
