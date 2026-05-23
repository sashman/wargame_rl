"""Canonical game-state snapshot: pure-Python Pydantic models with JSON export.

All fields use native Python types (no numpy) so the snapshot is trivially
serialisable to JSON, MessagePack, or any other format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

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

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
    from wargame_rl.wargame.envs.types.config import ModelConfig, WargameEnvConfig
    from wargame_rl.wargame.envs.types.game_timing import GameState


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class WeaponSnapshot(BaseModel):
    """Weapon stat block (mirrors WeaponProfile with native types)."""

    weapon_range: int = Field(alias="range")
    attacks: int
    ballistic_skill: int
    strength: int
    ap: int
    damage: int

    model_config = {"populate_by_name": True}


class ModelSnapshot(BaseModel):
    """State of a single model (unit) on the board."""

    location: list[int]
    previous_location: list[int] | None
    group_id: int
    alive: bool
    current_wounds: int
    max_wounds: int
    toughness: int
    save: int
    advanced_this_turn: bool
    weapons: list[WeaponSnapshot]


class ObjectiveSnapshot(BaseModel):
    """State of an objective marker."""

    location: list[int]
    radius_size: int


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


class RewardSnapshot(BaseModel):
    """Reward breakdown for the last step."""

    total: float | None
    breakdown: dict[str, float]
    phase_name: str
    phase_index: int


class GameStateSnapshot(BaseModel):
    """Complete, serialisable snapshot of game state at one point in time."""

    schema_version: str = "1.0"
    step: int
    max_steps: int
    clock: ClockSnapshot
    n_rounds: int
    board_width: int
    board_height: int
    player_models: list[ModelSnapshot]
    opponent_models: list[ModelSnapshot]
    objectives: list[ObjectiveSnapshot]
    deployment_zone: list[int]
    opponent_deployment_zone: list[int]
    player_vp: int
    opponent_vp: int
    player_vp_delta: int
    opponent_vp_delta: int
    objective_control: list[str]
    player_actions: list[int] | None
    opponent_actions: list[int] | None
    player_combat_results: list[CombatResultSnapshot]
    opponent_combat_results: list[CombatResultSnapshot]
    reward: RewardSnapshot
    is_terminated: bool
    is_truncated: bool


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _model_to_snapshot(
    model: WargameModel,
    config: ModelConfig | None,
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
    return ModelSnapshot(
        location=model.location.tolist(),
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
    )


def _objective_to_snapshot(obj: WargameObjective) -> ObjectiveSnapshot:
    return ObjectiveSnapshot(
        location=obj.location.tolist(),
        radius_size=obj.radius_size,
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
        hits=r.hits,
        wounds=r.wounds,
        unsaved=r.unsaved,
        damage_dealt=r.damage_dealt,
        expected_damage=exp_dmg,
        hit_probability=p_hit,
        wound_probability=p_wound,
    )


def _compute_objective_control(
    player_models: list[WargameModel],
    opponent_models: list[WargameModel],
    objectives: list[WargameObjective],
) -> list[str]:
    """Compute per-objective ownership strings: 'player', 'opponent', or 'none'."""
    if not objectives:
        return []

    p_cache = compute_distances(player_models, objectives)
    o_cache = (
        compute_distances(opponent_models, objectives) if opponent_models else None
    )

    if o_cache is not None:
        p_ctrl, o_ctrl = objective_ownership_from_norms_offset(
            p_cache.model_obj_norms_offset,
            o_cache.model_obj_norms_offset,
            p_cache.obj_radii,
        )
    else:
        p_ctrl = np.any(p_cache.model_obj_norms_offset <= p_cache.obj_radii, axis=0)
        o_ctrl = np.zeros_like(p_ctrl, dtype=bool)

    result: list[str] = []
    for pi, oi in zip(p_ctrl, o_ctrl):
        if pi:
            result.append("player")
        elif oi:
            result.append("opponent")
        else:
            result.append("none")
    return result


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
) -> GameStateSnapshot:
    """Build a complete game-state snapshot from env internals."""
    player_configs = config.models
    opponent_configs = config.opponent_models

    p_snaps = [
        _model_to_snapshot(
            m,
            player_configs[i] if player_configs and i < len(player_configs) else None,
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
        )
        for i, m in enumerate(opponent_models)
    ]

    obj_snaps = [_objective_to_snapshot(o) for o in objectives]
    clock = _clock_to_snapshot(clock_state)

    p_combat = [
        _combat_result_to_snapshot(r, player_configs, opponent_models)
        for r in player_shooting_results
    ]
    o_combat = [
        _combat_result_to_snapshot(r, opponent_configs, player_models)
        for r in opponent_shooting_results
    ]

    obj_control = _compute_objective_control(player_models, opponent_models, objectives)

    p_actions: list[int] | None = None
    if player_action is not None and hasattr(player_action, "actions"):
        p_actions = list(player_action.actions)

    o_actions: list[int] | None = None
    if opponent_action is not None and hasattr(opponent_action, "actions"):
        o_actions = list(opponent_action.actions)

    dz: list[int] = deployment_zone.tolist()
    odz: list[int] = opponent_deployment_zone.tolist()

    return GameStateSnapshot(
        step=step,
        max_steps=max_steps,
        clock=clock,
        n_rounds=n_rounds,
        board_width=config.board_width,
        board_height=config.board_height,
        player_models=p_snaps,
        opponent_models=o_snaps,
        objectives=obj_snaps,
        deployment_zone=dz,
        opponent_deployment_zone=odz,
        player_vp=player_vp,
        opponent_vp=opponent_vp,
        player_vp_delta=player_vp_delta,
        opponent_vp_delta=opponent_vp_delta,
        objective_control=obj_control,
        player_actions=p_actions,
        opponent_actions=o_actions,
        player_combat_results=p_combat,
        opponent_combat_results=o_combat,
        reward=RewardSnapshot(
            total=last_reward,
            breakdown=dict(reward_breakdown),
            phase_name=phase_name,
            phase_index=phase_index,
        ),
        is_terminated=is_terminated,
        is_truncated=is_truncated,
    )


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
