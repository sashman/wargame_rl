"""Event types for the append-only match event stream.

Events form an ordered log of a complete match. A ResetEvent anchors the
initial state; StepEvents record per-step deltas. Periodic anchor snapshots
(full GameStateSnapshot embedded in a StepEvent) allow efficient seek.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from wargame_rl.wargame.envs.state.snapshot import (
    ClockSnapshot,
    CombatResultSnapshot,
    GameStateSnapshot,
    ModelSnapshot,
    ObjectiveSnapshot,
    RewardSnapshot,
)


class ModelDelta(BaseModel):
    """Per-model state change within a single step."""

    idx: int
    location: list[int] | None = None
    previous_location: list[int] | None = None
    alive: bool | None = None
    current_wounds: int | None = None
    advanced_this_turn: bool | None = None
    distances_to_objectives: list[float] | None = None
    at_objective: list[bool] | None = None
    closest_objective_idx: int | None = None
    closest_objective_distance: float | None = None


class StateDelta(BaseModel):
    """Granular state change between two consecutive steps.

    Only populated fields represent actual changes from the previous state.
    """

    step: int
    clock: ClockSnapshot | None = None
    action_phase: str | None = None
    objectives: list[ObjectiveSnapshot] | None = None
    player_model_deltas: list[ModelDelta] = Field(default_factory=list)
    opponent_model_deltas: list[ModelDelta] = Field(default_factory=list)
    player_vp: int | None = None
    opponent_vp: int | None = None
    player_vp_delta: int | None = None
    opponent_vp_delta: int | None = None
    objective_control: list[str] | None = None
    player_actions: list[int] | None = None
    opponent_actions: list[int] | None = None
    player_action_descriptions: list[str] | None = None
    player_combat_results: list[CombatResultSnapshot] | None = None
    opponent_combat_results: list[CombatResultSnapshot] | None = None
    reward: RewardSnapshot | None = None
    is_terminated: bool | None = None
    is_truncated: bool | None = None
    player_alive_count: int | None = None
    opponent_alive_count: int | None = None
    player_total_wounds: int | None = None
    opponent_total_wounds: int | None = None


class ResetEvent(BaseModel):
    """Marks episode start with a full state snapshot as the anchor."""

    type: Literal["reset"] = "reset"
    snapshot: GameStateSnapshot


class StepEvent(BaseModel):
    """Records a single env step as either a delta or a full anchor snapshot."""

    type: Literal["step"] = "step"
    delta: StateDelta
    anchor: GameStateSnapshot | None = None


MatchEvent = Annotated[
    ResetEvent | StepEvent,
    Field(discriminator="type"),
]


def compute_delta(
    previous: GameStateSnapshot,
    current: GameStateSnapshot,
) -> StateDelta:
    """Compute a StateDelta representing changes from previous to current."""
    delta = StateDelta(step=current.step)

    if current.clock != previous.clock:
        delta.clock = current.clock
    if current.action_phase != previous.action_phase:
        delta.action_phase = current.action_phase

    # Objective occupancy changes as models move even though location/radius are
    # static, so the whole list must be diffed — not just the objective markers.
    if current.objectives != previous.objectives:
        delta.objectives = current.objectives

    if current.player_vp != previous.player_vp:
        delta.player_vp = current.player_vp
    if current.opponent_vp != previous.opponent_vp:
        delta.opponent_vp = current.opponent_vp
    if current.player_vp_delta != previous.player_vp_delta:
        delta.player_vp_delta = current.player_vp_delta
    if current.opponent_vp_delta != previous.opponent_vp_delta:
        delta.opponent_vp_delta = current.opponent_vp_delta

    if current.objective_control != previous.objective_control:
        delta.objective_control = current.objective_control

    if current.player_actions != previous.player_actions:
        delta.player_actions = current.player_actions
    if current.opponent_actions != previous.opponent_actions:
        delta.opponent_actions = current.opponent_actions
    if current.player_action_descriptions != previous.player_action_descriptions:
        delta.player_action_descriptions = current.player_action_descriptions

    if current.player_combat_results != previous.player_combat_results:
        delta.player_combat_results = current.player_combat_results
    if current.opponent_combat_results != previous.opponent_combat_results:
        delta.opponent_combat_results = current.opponent_combat_results

    if current.reward != previous.reward:
        delta.reward = current.reward

    if current.is_terminated != previous.is_terminated:
        delta.is_terminated = current.is_terminated
    if current.is_truncated != previous.is_truncated:
        delta.is_truncated = current.is_truncated

    if current.player_alive_count != previous.player_alive_count:
        delta.player_alive_count = current.player_alive_count
    if current.opponent_alive_count != previous.opponent_alive_count:
        delta.opponent_alive_count = current.opponent_alive_count
    if current.player_total_wounds != previous.player_total_wounds:
        delta.player_total_wounds = current.player_total_wounds
    if current.opponent_total_wounds != previous.opponent_total_wounds:
        delta.opponent_total_wounds = current.opponent_total_wounds

    for i, (prev_m, cur_m) in enumerate(
        zip(previous.player_models, current.player_models)
    ):
        md = _compute_model_delta(i, prev_m, cur_m)
        if md is not None:
            delta.player_model_deltas.append(md)

    for i, (prev_m, cur_m) in enumerate(
        zip(previous.opponent_models, current.opponent_models)
    ):
        md = _compute_model_delta(i, prev_m, cur_m)
        if md is not None:
            delta.opponent_model_deltas.append(md)

    return delta


def apply_delta(
    snapshot: GameStateSnapshot,
    delta: StateDelta,
) -> GameStateSnapshot:
    """Apply a StateDelta to a snapshot, producing the next state."""
    updates: dict[str, object] = {"step": delta.step}

    if delta.clock is not None:
        updates["clock"] = delta.clock
    if delta.action_phase is not None:
        updates["action_phase"] = delta.action_phase
    if delta.objectives is not None:
        updates["objectives"] = delta.objectives
    if delta.player_vp is not None:
        updates["player_vp"] = delta.player_vp
    if delta.opponent_vp is not None:
        updates["opponent_vp"] = delta.opponent_vp
    if delta.player_vp_delta is not None:
        updates["player_vp_delta"] = delta.player_vp_delta
    if delta.opponent_vp_delta is not None:
        updates["opponent_vp_delta"] = delta.opponent_vp_delta
    if delta.objective_control is not None:
        updates["objective_control"] = delta.objective_control
    if delta.player_actions is not None:
        updates["player_actions"] = delta.player_actions
    if delta.opponent_actions is not None:
        updates["opponent_actions"] = delta.opponent_actions
    if delta.player_action_descriptions is not None:
        updates["player_action_descriptions"] = delta.player_action_descriptions
    if delta.player_combat_results is not None:
        updates["player_combat_results"] = delta.player_combat_results
    if delta.opponent_combat_results is not None:
        updates["opponent_combat_results"] = delta.opponent_combat_results
    if delta.reward is not None:
        updates["reward"] = delta.reward
    if delta.is_terminated is not None:
        updates["is_terminated"] = delta.is_terminated
    if delta.is_truncated is not None:
        updates["is_truncated"] = delta.is_truncated
    if delta.player_alive_count is not None:
        updates["player_alive_count"] = delta.player_alive_count
    if delta.opponent_alive_count is not None:
        updates["opponent_alive_count"] = delta.opponent_alive_count
    if delta.player_total_wounds is not None:
        updates["player_total_wounds"] = delta.player_total_wounds
    if delta.opponent_total_wounds is not None:
        updates["opponent_total_wounds"] = delta.opponent_total_wounds

    if delta.player_model_deltas:
        p_models = list(snapshot.player_models)
        for md in delta.player_model_deltas:
            p_models[md.idx] = _apply_model_delta(p_models[md.idx], md)
        updates["player_models"] = p_models

    if delta.opponent_model_deltas:
        o_models = list(snapshot.opponent_models)
        for md in delta.opponent_model_deltas:
            o_models[md.idx] = _apply_model_delta(o_models[md.idx], md)
        updates["opponent_models"] = o_models

    result: GameStateSnapshot = snapshot.model_copy(update=updates)
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_model_delta(
    idx: int,
    prev: ModelSnapshot,
    cur: ModelSnapshot,
) -> ModelDelta | None:
    """Compute delta for a single model. Returns None if unchanged."""
    changes: dict[str, Any] = {}
    if cur.location != prev.location:
        changes["location"] = cur.location
    if cur.previous_location != prev.previous_location:
        changes["previous_location"] = cur.previous_location
    if cur.alive != prev.alive:
        changes["alive"] = cur.alive
    if cur.current_wounds != prev.current_wounds:
        changes["current_wounds"] = cur.current_wounds
    if cur.advanced_this_turn != prev.advanced_this_turn:
        changes["advanced_this_turn"] = cur.advanced_this_turn
    if cur.distances_to_objectives != prev.distances_to_objectives:
        changes["distances_to_objectives"] = cur.distances_to_objectives
    if cur.at_objective != prev.at_objective:
        changes["at_objective"] = cur.at_objective
    if cur.closest_objective_idx != prev.closest_objective_idx:
        changes["closest_objective_idx"] = cur.closest_objective_idx
    if cur.closest_objective_distance != prev.closest_objective_distance:
        changes["closest_objective_distance"] = cur.closest_objective_distance

    if not changes:
        return None
    return ModelDelta(idx=idx, **changes)


def _apply_model_delta(model: ModelSnapshot, md: ModelDelta) -> ModelSnapshot:
    """Apply a ModelDelta to a ModelSnapshot."""
    updates: dict[str, Any] = {}
    if md.location is not None:
        updates["location"] = md.location
    if md.previous_location is not None:
        updates["previous_location"] = md.previous_location
    if md.alive is not None:
        updates["alive"] = md.alive
    if md.current_wounds is not None:
        updates["current_wounds"] = md.current_wounds
    if md.advanced_this_turn is not None:
        updates["advanced_this_turn"] = md.advanced_this_turn
    if md.distances_to_objectives is not None:
        updates["distances_to_objectives"] = md.distances_to_objectives
    if md.at_objective is not None:
        updates["at_objective"] = md.at_objective
    if md.closest_objective_idx is not None:
        updates["closest_objective_idx"] = md.closest_objective_idx
    if md.closest_objective_distance is not None:
        updates["closest_objective_distance"] = md.closest_objective_distance
    result: ModelSnapshot = model.model_copy(update=updates)
    return result
