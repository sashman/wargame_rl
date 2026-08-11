"""Canonical game-state snapshot: pure-Python Pydantic models with JSON export.

All fields use native Python types (no numpy) so the snapshot is trivially
serialisable to JSON, MessagePack, or any other format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

from wargame_rl.wargame.envs.domain.shooting import (
    DefenderStats,
    PairedShootingResult,
    expected_damage,
    wound_roll_threshold,
)
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_ownership_from_norms_offset,
)
from wargame_rl.wargame.envs.types.config import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase, GameState

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
    from wargame_rl.wargame.envs.domain.terrain import Terrain
    from wargame_rl.wargame.envs.types.config import ModelConfig


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class WeaponSnapshot(BaseModel):
    """Weapon stat block (mirrors WeaponProfile with native types)."""

    weapon_range: int = Field(validation_alias="range")
    attacks: int
    ballistic_skill: int
    strength: int
    ap: int
    damage: int

    model_config = {"populate_by_name": True}


class ModelSnapshot(BaseModel):
    """State of a single model (unit) on the board."""

    location: list[float]
    previous_location: list[float] | None
    base_radius: float = 0.0
    group_id: int
    alive: bool
    current_wounds: int
    max_wounds: int
    toughness: int
    save: int
    advanced_this_turn: bool
    weapons: list[WeaponSnapshot]
    distances_to_objectives: list[float]
    at_objective: list[bool]
    closest_objective_idx: int | None
    closest_objective_distance: float | None


class ObjectiveSnapshot(BaseModel):
    """State of an objective marker."""

    location: list[float]
    radius_size: float
    player_models_in_range: list[int]
    opponent_models_in_range: list[int]
    area: list[list[float]] | None = None
    """Outline vertices when the objective *is* a piece of ground.

    Present rather than derived, because an area objective's `location` is only
    its centroid: a replay reconstructed from the centroid alone would draw a
    marker where the rules have a shape, and score control by a radius of 0.
    """


class ClockSnapshot(BaseModel):
    """Game timing position."""

    game_phase: str
    battle_round: int | None
    active_player: str | None
    battle_phase: str | None


class CombatResultSnapshot(BaseModel):
    """One model-vs-model shooting resolution with analytical fields."""

    attacker_idx: int
    target_idx: int
    hits: int
    wounds: int
    unsaved: int
    damage_dealt: int
    expected_damage: float
    hit_probability: float
    wound_probability: float
    killed: bool = False
    """Whether this shot took the target to zero wounds.

    Added in schema 2.3 so a replay can draw a killing shot as one. It cannot be
    recovered afterwards -- several attackers may fire on the same target in a
    phase and only one made the kill -- so it is recorded at resolution time.
    ``False`` on earlier recordings, where a kill replays as an ordinary hit.
    """


class RewardSnapshot(BaseModel):
    """Reward breakdown for the last step."""

    total: float | None
    breakdown: dict[str, float]
    phase_name: str
    phase_index: int
    episode_total: float | None = None
    """Running sum of every step's ``total`` so far this episode.

    Added in schema 2.2 so a replay can show the cumulative reward without
    re-summing the log; ``None`` on earlier recordings.
    """


class GameStateSnapshot(BaseModel):
    """Complete, serialisable snapshot of game state at one point in time.

    ``clock`` describes the state *after* the step completed, so its
    ``battle_phase`` is the phase that will execute next. ``action_phase`` is the
    phase the reported actions were executed in — use that, not ``clock``, when
    attributing ``player_actions`` to a phase.
    """

    schema_version: str = "2.3"
    step: int
    max_steps: int
    clock: ClockSnapshot
    action_phase: str | None = None
    n_rounds: int
    board_width: int
    board_height: int
    player_models: list[ModelSnapshot]
    opponent_models: list[ModelSnapshot]
    objectives: list[ObjectiveSnapshot]
    deployment_zone: list[int]
    opponent_deployment_zone: list[int]
    terrain_footprints: list[list[list[float]]] | None = None
    """Outline vertices of each terrain piece, one ``[[x, y], ...]`` per footprint.

    Static per episode, so it is recorded on the reset snapshot and every anchor
    (never in a delta — see ``build_snapshot``). ``None`` on pre-2.1 recordings,
    which carried no terrain; a replay of those draws no ruins.
    """
    skip_phases: list[str] | None = None
    """Battle phases the config auto-advances, so a replay can dim them.

    Static per episode, so it rides on the full snapshots exactly as
    ``terrain_footprints`` does. ``None`` on pre-2.2 recordings.
    """
    player_vp: int
    opponent_vp: int
    player_vp_delta: int
    opponent_vp_delta: int
    objective_control: list[str]
    player_actions: list[int] | None
    opponent_actions: list[int] | None
    player_action_descriptions: list[str] | None
    player_combat_results: list[CombatResultSnapshot]
    opponent_combat_results: list[CombatResultSnapshot]
    reward: RewardSnapshot
    is_terminated: bool
    is_truncated: bool
    player_alive_count: int
    opponent_alive_count: int
    player_total_wounds: int
    opponent_total_wounds: int
    mission_type: str
    mission_params: dict[str, int | float | str]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _model_to_snapshot(
    model: WargameModel,
    config: ModelConfig | None,
    obj_distances: list[float] | None = None,
    obj_in_range: list[bool] | None = None,
) -> ModelSnapshot:
    """Convert a domain WargameModel to a snapshot."""
    weapons: list[WeaponSnapshot] = []
    if config is not None:
        for w in config.weapons:
            weapons.append(
                WeaponSnapshot(
                    weapon_range=w.range,
                    attacks=w.attacks,
                    ballistic_skill=w.ballistic_skill,
                    strength=w.strength,
                    ap=w.ap,
                    damage=w.damage,
                )
            )

    dists = obj_distances or []
    at_obj = obj_in_range or []
    if dists and model.is_alive:
        closest_idx = int(np.argmin(dists))
        closest_dist = float(dists[closest_idx])
    else:
        closest_idx = None
        closest_dist = None

    return ModelSnapshot(
        location=model.location.tolist(),
        base_radius=model.base_radius,
        previous_location=(
            model.previous_location.tolist()
            if model.previous_location is not None
            else None
        ),
        group_id=model.group_id,
        alive=model.is_alive,
        current_wounds=model.stats["current_wounds"],
        max_wounds=model.stats["max_wounds"],
        toughness=model.stats["toughness"],
        save=model.stats["save"],
        advanced_this_turn=model.advanced_this_turn,
        weapons=weapons,
        distances_to_objectives=dists,
        at_objective=at_obj,
        closest_objective_idx=closest_idx,
        closest_objective_distance=closest_dist,
    )


def _objective_to_snapshot(
    obj: WargameObjective,
    player_in_range: list[int],
    opponent_in_range: list[int],
) -> ObjectiveSnapshot:
    return ObjectiveSnapshot(
        location=obj.location.tolist(),
        radius_size=obj.radius_size,
        player_models_in_range=player_in_range,
        opponent_models_in_range=opponent_in_range,
        area=obj.area.vertices.tolist() if obj.area is not None else None,
    )


def _clock_to_snapshot(state: GameState) -> ClockSnapshot:
    return ClockSnapshot(
        game_phase=state.game_phase.value,
        battle_round=state.battle_round,
        active_player=state.active_player.value if state.active_player else None,
        battle_phase=state.phase.value if state.phase else None,
    )


def _combat_result_to_snapshot(
    paired: PairedShootingResult,
    attacker_configs: list[ModelConfig] | None,
    targets: list[WargameModel],
) -> CombatResultSnapshot:
    """Convert a PairedShootingResult to a snapshot with analytical fields."""
    r = paired.result
    exp_dmg = 0.0
    p_hit = 0.0
    p_wound = 0.0

    has_weapon = (
        attacker_configs is not None
        and paired.attacker_idx < len(attacker_configs)
        and attacker_configs[paired.attacker_idx].weapons
    )
    has_target = paired.target_idx < len(targets)

    if has_weapon and has_target:
        weapon = attacker_configs[paired.attacker_idx].weapons[0]  # type: ignore[index]
        target = targets[paired.target_idx]
        defender = DefenderStats(
            toughness=target.stats["toughness"],
            save=target.stats["save"],
        )
        exp_dmg = expected_damage(weapon, defender)
        p_hit = (7 - weapon.ballistic_skill) / 6.0
        threshold = wound_roll_threshold(weapon.strength, defender.toughness)
        p_wound = (7 - threshold) / 6.0

    return CombatResultSnapshot(
        attacker_idx=paired.attacker_idx,
        target_idx=paired.target_idx,
        killed=paired.killed,
        hits=r.hits,
        wounds=r.wounds,
        unsaved=r.unsaved,
        damage_dealt=r.damage_dealt,
        expected_damage=exp_dmg,
        hit_probability=p_hit,
        wound_probability=p_wound,
    )


@dataclass
class _SpatialData:
    """Intermediate spatial analysis used to populate multiple snapshot fields."""

    objective_control: list[str]
    player_obj_dists: list[list[float]]
    player_at_obj: list[list[bool]]
    opponent_obj_dists: list[list[float]]
    opponent_at_obj: list[list[bool]]
    player_in_range_per_obj: list[list[int]]
    opponent_in_range_per_obj: list[list[int]]


def _compute_spatial_data(
    player_models: list[WargameModel],
    opponent_models: list[WargameModel],
    objectives: list[WargameObjective],
) -> _SpatialData:
    """Compute all spatial relationships between models and objectives."""
    n_obj = len(objectives)
    n_player = len(player_models)
    n_opp = len(opponent_models)

    empty = _SpatialData(
        objective_control=[],
        player_obj_dists=[[] for _ in range(n_player)],
        player_at_obj=[[] for _ in range(n_player)],
        opponent_obj_dists=[[] for _ in range(n_opp)],
        opponent_at_obj=[[] for _ in range(n_opp)],
        player_in_range_per_obj=[],
        opponent_in_range_per_obj=[],
    )
    if not objectives:
        return empty

    p_cache = compute_distances(player_models, objectives)
    o_cache = (
        compute_distances(opponent_models, objectives) if opponent_models else None
    )

    p_norms = p_cache.model_obj_norms_offset
    radii = p_cache.obj_radii
    p_in_range = p_norms <= radii

    if o_cache is not None:
        o_norms = o_cache.model_obj_norms_offset
        o_in_range = o_norms <= radii
        p_ctrl, o_ctrl = objective_ownership_from_norms_offset(p_norms, o_norms, radii)
    else:
        o_norms = np.zeros((0, n_obj), dtype=np.float64)
        o_in_range = np.zeros((0, n_obj), dtype=bool)
        p_ctrl = np.any(p_in_range, axis=0)
        o_ctrl = np.zeros_like(p_ctrl, dtype=bool)

    control: list[str] = []
    for pi, oi in zip(p_ctrl, o_ctrl):
        if pi:
            control.append("player")
        elif oi:
            control.append("opponent")
        else:
            control.append("none")

    p_dists = [p_norms[i].tolist() for i in range(n_player)]
    p_at = [p_in_range[i].tolist() for i in range(n_player)]
    o_dists = [o_norms[i].tolist() for i in range(n_opp)]
    o_at = [o_in_range[i].tolist() for i in range(n_opp)]

    p_in_range_per_obj: list[list[int]] = []
    o_in_range_per_obj: list[list[int]] = []
    for j in range(n_obj):
        p_in_range_per_obj.append([i for i in range(n_player) if p_in_range[i, j]])
        o_in_range_per_obj.append(
            [i for i in range(n_opp) if o_in_range[i, j]] if n_opp > 0 else []
        )

    return _SpatialData(
        objective_control=control,
        player_obj_dists=p_dists,
        player_at_obj=p_at,
        opponent_obj_dists=o_dists,
        opponent_at_obj=o_at,
        player_in_range_per_obj=p_in_range_per_obj,
        opponent_in_range_per_obj=o_in_range_per_obj,
    )


COMPASS_LABELS_16 = [
    "E",
    "ENE",
    "NE",
    "NNE",
    "N",
    "NNW",
    "NW",
    "WNW",
    "W",
    "WSW",
    "SW",
    "SSW",
    "S",
    "SSE",
    "SE",
    "ESE",
]


def describe_action(
    action: int,
    n_angles: int,
    n_speed_bins: int,
    shooting_slice_start: int | None,
    shooting_slice_end: int | None,
) -> str:
    """Decode a raw action integer into a human-readable description."""
    if action == 0:
        return "Stay"
    if (
        shooting_slice_start is not None
        and shooting_slice_end is not None
        and shooting_slice_start <= action < shooting_slice_end
    ):
        target_idx = action - shooting_slice_start
        return f"Shoot at opponent {target_idx}"
    move_idx = action - 1
    angle_idx = move_idx // n_speed_bins
    speed_idx = move_idx % n_speed_bins
    speed_label = speed_idx + 1
    if n_angles <= 16:
        step = 16 // n_angles
        direction = COMPASS_LABELS_16[angle_idx * step]
    else:
        direction = f"angle {angle_idx}"
    return f"Move {direction} at speed {speed_label}"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_snapshot(
    *,
    config: WargameEnvConfig,
    step: int,
    max_steps: int,
    clock_state: GameState,
    n_rounds: int,
    player_models: list[WargameModel],
    opponent_models: list[WargameModel],
    objectives: list[WargameObjective],
    deployment_zone: np.ndarray,
    opponent_deployment_zone: np.ndarray,
    player_vp: int,
    opponent_vp: int,
    player_vp_delta: int,
    opponent_vp_delta: int,
    player_shooting_results: list[PairedShootingResult],
    opponent_shooting_results: list[PairedShootingResult],
    player_action: object | None,
    opponent_action: object | None,
    last_reward: float | None,
    reward_breakdown: dict[str, float],
    phase_name: str,
    phase_index: int,
    is_terminated: bool,
    is_truncated: bool,
    n_angles: int = 16,
    n_speed_bins: int = 6,
    shooting_slice_start: int | None = None,
    shooting_slice_end: int | None = None,
    action_phase: str | None = None,
    terrain: "Terrain | None" = None,
    episode_reward: float | None = None,
) -> GameStateSnapshot:
    """Build a complete game-state snapshot from env internals."""
    player_configs = config.models
    opponent_configs = config.opponent_models

    # Terrain is static for the episode, so it rides on every full snapshot
    # (reset + anchors) and is intentionally left out of deltas — apply_delta
    # preserves it from the anchor. Do NOT move this into StateDelta.
    terrain_footprints: list[list[list[float]]] | None = (
        [fp.polygon.vertices.tolist() for fp in terrain.footprints]
        if terrain is not None and terrain.footprints
        else None
    )

    spatial = _compute_spatial_data(player_models, opponent_models, objectives)

    p_snaps = [
        _model_to_snapshot(
            m,
            player_configs[i] if player_configs and i < len(player_configs) else None,
            obj_distances=spatial.player_obj_dists[i],
            obj_in_range=spatial.player_at_obj[i],
        )
        for i, m in enumerate(player_models)
    ]
    o_snaps = [
        _model_to_snapshot(
            m,
            (
                opponent_configs[i]
                if opponent_configs and i < len(opponent_configs)
                else None
            ),
            obj_distances=spatial.opponent_obj_dists[i]
            if i < len(spatial.opponent_obj_dists)
            else [],
            obj_in_range=spatial.opponent_at_obj[i]
            if i < len(spatial.opponent_at_obj)
            else [],
        )
        for i, m in enumerate(opponent_models)
    ]

    obj_snaps = [
        _objective_to_snapshot(
            o,
            player_in_range=spatial.player_in_range_per_obj[j],
            opponent_in_range=spatial.opponent_in_range_per_obj[j],
        )
        for j, o in enumerate(objectives)
    ]
    clock = _clock_to_snapshot(clock_state)

    p_combat = [
        _combat_result_to_snapshot(r, player_configs, opponent_models)
        for r in player_shooting_results
    ]
    o_combat = [
        _combat_result_to_snapshot(r, opponent_configs, player_models)
        for r in opponent_shooting_results
    ]

    p_actions: list[int] | None = None
    p_action_descs: list[str] | None = None
    if player_action is not None and hasattr(player_action, "actions"):
        p_actions = list(player_action.actions)
        p_action_descs = [
            describe_action(
                a,
                n_angles,
                n_speed_bins,
                shooting_slice_start,
                shooting_slice_end,
            )
            for a in p_actions
        ]

    o_actions: list[int] | None = None
    if opponent_action is not None and hasattr(opponent_action, "actions"):
        o_actions = list(opponent_action.actions)

    dz: list[int] = deployment_zone.tolist()
    odz: list[int] = opponent_deployment_zone.tolist()

    p_alive = sum(1 for m in player_models if m.is_alive)
    o_alive = sum(1 for m in opponent_models if m.is_alive)
    p_wounds = sum(m.stats["current_wounds"] for m in player_models if m.is_alive)
    o_wounds = sum(m.stats["current_wounds"] for m in opponent_models if m.is_alive)

    mission_params: dict[str, Any] = dict(config.mission.params)

    return GameStateSnapshot(
        step=step,
        max_steps=max_steps,
        clock=clock,
        action_phase=action_phase,
        n_rounds=n_rounds,
        board_width=config.board_width,
        board_height=config.board_height,
        player_models=p_snaps,
        opponent_models=o_snaps,
        objectives=obj_snaps,
        deployment_zone=dz,
        opponent_deployment_zone=odz,
        terrain_footprints=terrain_footprints,
        skip_phases=[phase.value for phase in config.skip_phases],
        player_vp=player_vp,
        opponent_vp=opponent_vp,
        player_vp_delta=player_vp_delta,
        opponent_vp_delta=opponent_vp_delta,
        objective_control=spatial.objective_control,
        player_actions=p_actions,
        opponent_actions=o_actions,
        player_action_descriptions=p_action_descs,
        player_combat_results=p_combat,
        opponent_combat_results=o_combat,
        reward=RewardSnapshot(
            total=last_reward,
            breakdown=dict(reward_breakdown),
            phase_name=phase_name,
            phase_index=phase_index,
            episode_total=episode_reward,
        ),
        is_terminated=is_terminated,
        is_truncated=is_truncated,
        player_alive_count=p_alive,
        opponent_alive_count=o_alive,
        player_total_wounds=p_wounds,
        opponent_total_wounds=o_wounds,
        mission_type=config.mission.type,
        mission_params=mission_params,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_snapshot(
    snapshot: GameStateSnapshot,
    config: WargameEnvConfig,
) -> list[str]:
    """Return a list of validation errors. Empty list means the snapshot is valid."""
    errors: list[str] = []

    if len(snapshot.player_models) != config.number_of_wargame_models:
        errors.append(
            f"Expected {config.number_of_wargame_models} player models, "
            f"got {len(snapshot.player_models)}"
        )
    if len(snapshot.opponent_models) != config.number_of_opponent_models:
        errors.append(
            f"Expected {config.number_of_opponent_models} opponent models, "
            f"got {len(snapshot.opponent_models)}"
        )
    if len(snapshot.objectives) != config.number_of_objectives:
        errors.append(
            f"Expected {config.number_of_objectives} objectives, "
            f"got {len(snapshot.objectives)}"
        )

    bw, bh = config.board_width, config.board_height
    for side, models in [
        ("player", snapshot.player_models),
        ("opponent", snapshot.opponent_models),
    ]:
        for i, m in enumerate(models):
            x, y = m.location
            # Inclusive on both ends: the board is continuous, so `bw` is a
            # coordinate a model on the far edge legitimately has, not one past
            # the last cell index.
            if x < 0 or x > bw or y < 0 or y > bh:
                errors.append(
                    f"{side} model {i} location ({x}, {y}) out of bounds "
                    f"[0, {bw}] x [0, {bh}]"
                )
            if m.current_wounds < 0 or m.current_wounds > m.max_wounds:
                errors.append(
                    f"{side} model {i} current_wounds={m.current_wounds} "
                    f"not in [0, {m.max_wounds}]"
                )

    for i, o in enumerate(snapshot.objectives):
        x, y = o.location
        if x < 0 or x > bw or y < 0 or y > bh:
            errors.append(
                f"objective {i} location ({x}, {y}) out of bounds [0, {bw}] x [0, {bh}]"
            )

    clock = snapshot.clock
    if clock.game_phase == "battle":
        if clock.battle_round is not None:
            n_rounds = config.number_of_battle_rounds
            if clock.battle_round < 1 or clock.battle_round > n_rounds:
                errors.append(
                    f"battle_round={clock.battle_round} not in [1, {n_rounds}]"
                )
        if clock.battle_phase is not None:
            valid_phases = {p.value for p in BattlePhase}
            if clock.battle_phase not in valid_phases:
                errors.append(f"invalid battle_phase '{clock.battle_phase}'")

    if snapshot.step < 0 or snapshot.step > snapshot.max_steps:
        errors.append(f"step={snapshot.step} not in [0, {snapshot.max_steps}]")

    return errors


# ---------------------------------------------------------------------------
# Encoder protocol + JSON implementation
# ---------------------------------------------------------------------------


@runtime_checkable
class SnapshotEncoder(Protocol):
    """Protocol for snapshot serialisation."""

    def encode(self, snapshot: GameStateSnapshot) -> str: ...

    def content_type(self) -> str: ...


class JsonEncoder:
    """JSON encoder for GameStateSnapshot."""

    def encode(self, snapshot: GameStateSnapshot) -> str:
        """Serialise a snapshot to a JSON string."""
        return str(snapshot.model_dump_json())

    def content_type(self) -> str:
        return "application/json"
