"""Build observations and info from battle state (BattleView).

Extracted so observation shape or content can be varied without touching step/reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.domain.value_objects import POSITION_DTYPE
from wargame_rl.wargame.envs.env_components.actions import ActionRegistry
from wargame_rl.wargame.envs.env_components.shooting_masks import compute_shooting_masks
from wargame_rl.wargame.envs.types import (
    WargameEnvInfo,
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
    WargameTerrainObservation,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.types.terrain_observation import TERRAIN_VERTEX_BUDGET

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameObjective
    from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache
    from wargame_rl.wargame.envs.types.config import ModelConfig
    from wargame_rl.wargame.envs.wargame_model import WargameModel


def update_distances_to_objectives(
    wargame_models: list[WargameModel],
    objectives: list[WargameObjective],
    distance_cache: DistanceCache | None = None,
) -> None:
    """Update each model's distances_to_objectives from current locations. Mutates models."""
    # No `.astype(int)` here. The vector to the objective is the single most
    # informative feature the policy has, and truncating it to whole units threw
    # away sub-unit steering on a board where a move is now any real length.
    if distance_cache is not None:
        deltas = distance_cache.model_obj_deltas.astype(POSITION_DTYPE)
        for i, model in enumerate(wargame_models):
            model.distances_to_objectives = deltas[i]
        return

    for model in wargame_models:
        model.distances_to_objectives = np.array(
            [model.location - obj.location for obj in objectives],
            dtype=POSITION_DTYPE,
        )


def _models_to_obs(
    models: list[WargameModel],
    max_groups: int,
    model_configs: list[ModelConfig] | None = None,
) -> list[WargameModelObservation]:
    result: list[WargameModelObservation] = []
    for i, m in enumerate(models):
        w_attacks = 0
        w_bs = 0
        w_str = 0
        w_ap = 0
        w_dmg = 0
        toughness = 0
        save = 0
        if model_configs is not None and i < len(model_configs):
            cfg = model_configs[i]
            toughness = cfg.toughness
            save = cfg.save
            if cfg.weapons:
                w = cfg.weapons[0]
                w_attacks = w.attacks
                w_bs = w.ballistic_skill
                w_str = w.strength
                w_ap = w.ap
                w_dmg = w.damage
        result.append(
            WargameModelObservation(
                location=m.location,
                distances_to_objectives=m.distances_to_objectives,
                group_id=m.group_id,
                max_groups=max_groups,
                alive=1.0 if m.is_alive else 0.0,
                current_wounds=int(m.stats["current_wounds"]),
                max_wounds=int(m.stats["max_wounds"]),
                weapon_attacks=w_attacks,
                weapon_ballistic_skill=w_bs,
                weapon_strength=w_str,
                weapon_ap=w_ap,
                weapon_damage=w_dmg,
                toughness=toughness,
                save_stat=save,
            )
        )
    return result


def _terrain_to_obs(
    view: BattleView,
) -> list[WargameTerrainObservation]:
    """Build terrain observations: padded outline vertices, plus a vertex count.

    Vertices are normalised to [-1, 1] by the board half-dimensions and padded to
    `TERRAIN_VERTEX_BUDGET` by repeating the last one, so pieces with different
    vertex counts stack into one array — which observation batching requires.

    This is the first encoding that can tell an outline from its bounding box,
    and therefore the first honest test of whether the agent can use terrain: the
    whole cover line of work was run against four numbers that made an L-shaped
    ruin and a solid block identical.

    A piece with more vertices than the budget is a config error, not something
    to silently truncate — dropping vertices would quietly shrink a ruin the
    sight trace is still using at full size.
    """
    half_w = view.board_width / 2.0
    half_h = view.board_height / 2.0
    result: list[WargameTerrainObservation] = []
    for fp in view.terrain.footprints:
        if fp.n_vertices > TERRAIN_VERTEX_BUDGET:
            raise ValueError(
                f"terrain piece has {fp.n_vertices} vertices, over the "
                f"observation budget of {TERRAIN_VERTEX_BUDGET}. Raise "
                "TERRAIN_VERTEX_BUDGET (which changes the network's input "
                "width, so existing checkpoints will fail to load) or simplify "
                "the outline."
            )
        padded = fp.polygon.padded_to(TERRAIN_VERTEX_BUDGET)
        normalised = np.empty(2 * TERRAIN_VERTEX_BUDGET + 1, dtype=np.float32)
        normalised[0 : 2 * TERRAIN_VERTEX_BUDGET : 2] = (padded[:, 0] - half_w) / half_w
        normalised[1 : 2 * TERRAIN_VERTEX_BUDGET : 2] = (padded[:, 1] - half_h) / half_h
        normalised[-1] = fp.n_vertices / TERRAIN_VERTEX_BUDGET
        result.append(WargameTerrainObservation(outline=normalised))
    return result


def _objectives_to_obs(
    view: BattleView, with_control: bool
) -> list[WargameEnvObjectiveObservation]:
    """Objective observations, optionally carrying per-objective control state.

    Counts are of *alive* models inside each disc, normalised by the static
    army sizes so the feature stays O(1) rather than shrinking with force size.
    Both sides use their own establishment as the divisor, which keeps
    "half my army is here" and "half of theirs is here" on the same scale.
    """
    if not with_control:
        return [
            WargameEnvObjectiveObservation(location=obj.location)
            for obj in view.objectives
        ]

    player_locations = np.array(
        [m.location for m in view.player_models if m.is_alive], dtype=float
    )
    opponent_locations = np.array(
        [m.location for m in view.opponent_models if m.is_alive], dtype=float
    )
    n_player = max(1, view.config.number_of_wargame_models)
    n_opponent = max(1, view.config.number_of_opponent_models)
    board_diagonal = float(np.hypot(view.board_width, view.board_height)) or 1.0

    def inside(locations: np.ndarray, objective: WargameObjective) -> int:
        """Count models controlling this objective, whichever kind it is.

        An area objective has radius 0, so a distance-to-centre test would count
        only models standing exactly on the centroid — a control feature that
        reads zero forever while the reward keyed on it pays out. The membership
        rule has to follow the objective's own shape.
        """
        if locations.size == 0:
            return 0
        if objective.area is not None:
            return int(objective.area.contains_points(locations).sum())
        centre = np.asarray(objective.location, dtype=float)
        return int(
            (
                np.linalg.norm(locations - centre, axis=1)
                <= float(objective.radius_size)
            ).sum()
        )

    observations = []
    for objective in view.objectives:
        # An area's "radius" is reported as the radius of a disc with the same
        # area, so the feature keeps meaning "how big is this objective" across
        # both kinds rather than collapsing to zero for one of them.
        if objective.area is not None:
            extent = float(np.sqrt(objective.area.area / np.pi))
        else:
            extent = float(objective.radius_size)
        observations.append(
            WargameEnvObjectiveObservation(
                location=objective.location,
                player_count=inside(player_locations, objective) / n_player,
                opponent_count=inside(opponent_locations, objective) / n_opponent,
                radius=extent / board_diagonal,
            )
        )
    return observations


def build_observation(
    view: BattleView,
    distance_cache: DistanceCache | None = None,
    action_registry: ActionRegistry | None = None,
) -> WargameEnvObservation:
    """Build the Gym observation from battle state (BattleView)."""
    if distance_cache is not None:
        update_distances_to_objectives(
            view.player_models, view.objectives, distance_cache
        )
    if view.opponent_models:
        update_distances_to_objectives(view.opponent_models, view.objectives, None)

    action_mask: np.ndarray | None = None
    if action_registry is not None:
        phase = view.game_clock_state.phase or BattlePhase.movement
        player_alive = alive_mask_for(view.player_models)
        action_mask = action_registry.get_model_action_masks(
            phase, len(view.player_models), alive_mask=player_alive
        )
        if (
            action_registry.has_slice("shooting")
            and phase == BattlePhase.shooting
            and view.opponent_models
        ):
            shooting_slice = action_registry.slice_for("shooting")
            opponent_alive = alive_mask_for(view.opponent_models)
            player_positions = np.array([m.location for m in view.player_models])
            opponent_positions = np.array([m.location for m in view.opponent_models])
            player_ranges = view.player_max_ranges
            player_advanced = np.array(
                [m.advanced_this_turn for m in view.player_models]
            )
            shooting_validity = compute_shooting_masks(
                player_positions,
                opponent_positions,
                player_alive,
                opponent_alive,
                player_ranges,
                view.line_of_sight_matrix,
                player_advanced=player_advanced,
                engagement_range=view.rules_quantities.engagement_range,
                base_diameter=2.0 * view.rules_quantities.base_radius,
            )
            action_mask[:, shooting_slice.start : shooting_slice.end] &= (
                shooting_validity
            )

    clock = view.game_clock_state
    phase = clock.phase or BattlePhase.movement
    battle_phase_index = list(BattlePhase).index(phase)
    battle_round = clock.battle_round if clock.battle_round is not None else 1
    max_groups = view.config.max_groups
    objectives_obs = _objectives_to_obs(view, view.config.observe_objective_control)
    terrain_obs = _terrain_to_obs(view)
    return WargameEnvObservation(
        current_turn=view.current_turn,
        wargame_models=_models_to_obs(
            view.player_models,
            max_groups,
            model_configs=view.config.models,
        ),
        objectives=objectives_obs,
        board_width=view.board_width,
        board_height=view.board_height,
        opponent_models=_models_to_obs(
            view.opponent_models,
            max_groups,
            model_configs=view.config.opponent_models,
        ),
        terrain=terrain_obs,
        action_mask=action_mask,
        battle_round=battle_round,
        battle_phase_index=battle_phase_index,
        n_rounds=view.n_rounds,
        player_vp=view.player_vp,
        opponent_vp=view.opponent_vp,
        player_vp_delta=view.player_vp_delta,
    )


def build_info(view: BattleView) -> WargameEnvInfo:
    """Build the Gym info dict from battle state (BattleView)."""
    dz = view.deployment_zone
    odz = view.opponent_deployment_zone
    deployment_zone = (int(dz[0]), int(dz[1]), int(dz[2]), int(dz[3]))
    opponent_deployment_zone = (int(odz[0]), int(odz[1]), int(odz[2]), int(odz[3]))
    max_groups = view.config.max_groups
    objectives_obs = [
        WargameEnvObjectiveObservation(location=obj.location) for obj in view.objectives
    ]
    return WargameEnvInfo(
        current_turn=view.current_turn,
        wargame_models=_models_to_obs(
            view.player_models, max_groups, model_configs=view.config.models
        ),
        objectives=objectives_obs,
        opponent_models=_models_to_obs(
            view.opponent_models, max_groups, model_configs=view.config.opponent_models
        ),
        deployment_zone=deployment_zone,
        opponent_deployment_zone=opponent_deployment_zone,
        player_vp=view.player_vp,
        opponent_vp=view.opponent_vp,
        player_vp_delta=view.player_vp_delta,
        opponent_vp_delta=view.opponent_vp_delta,
    )
