"""Factory for creating Battle aggregates from config."""

from __future__ import annotations

from typing import Any

import numpy as np

from wargame_rl.wargame.envs.types import WargameEnvConfig

from .battle import Battle
from .entities import WargameModel, WargameObjective
from .rules_quantities import RulesQuantities, resolve_rules_quantities
from .terrain import Footprint, Terrain
from .value_objects import BoardDimensions, DeploymentZone

__all__ = [
    "create_objectives",
    "create_opponent_models",
    "create_wargame_models",
    "from_config",
    "resolve_rules_quantities",
]


def _build_models(
    n: int,
    model_configs: list[Any] | None,
    n_objectives: int,
    max_groups: int,
    quantities: RulesQuantities,
) -> list[WargameModel]:
    """Build a list of WargameModel instances (player or opponent)."""
    result: list[WargameModel] = []
    increment = max(1, n // max_groups)
    for i in range(n):
        base_radius = quantities.base_radius
        if model_configs is not None:
            mc = model_configs[i]
            group_id = mc.group_id
            max_wounds = mc.max_wounds
            toughness = mc.toughness
            save = mc.save
            if mc.base_radius is not None:
                base_radius = quantities.scale.to_units(mc.base_radius)
        else:
            group_id = i // increment
            max_wounds = 100
            toughness = 3
            save = 4
        result.append(
            WargameModel(
                location=np.zeros(2, dtype=float),
                stats={
                    "max_wounds": max_wounds,
                    "current_wounds": max_wounds,
                    "toughness": toughness,
                    "save": save,
                },
                group_id=group_id,
                distances_to_objectives=np.zeros([n_objectives, 2], dtype=float),
                base_radius=base_radius,
            )
        )
    return result


def _build_objectives(
    config: WargameEnvConfig, quantities: RulesQuantities
) -> list[WargameObjective]:
    """Build the list of objectives from config."""
    result: list[WargameObjective] = []
    for i in range(config.number_of_objectives):
        override = (
            config.objectives[i].radius_size if config.objectives is not None else None
        )
        radius = (
            quantities.objective_radius
            if override is None
            else quantities.scale.to_units(override)
        )

        result.append(
            WargameObjective(
                location=np.zeros(2, dtype=float),
                radius_size=radius,
            )
        )
    return result


def from_config(config: WargameEnvConfig) -> Battle:
    """Create a Battle from environment config."""
    board_dimensions = BoardDimensions(
        width=config.board_width, height=config.board_height
    )
    board_width = config.board_width
    board_height = config.board_height
    n_objectives = config.number_of_objectives
    quantities = resolve_rules_quantities(config)

    player_models = _build_models(
        config.number_of_wargame_models,
        config.models,
        n_objectives,
        config.max_groups,
        quantities,
    )
    opponent_models = _build_models(
        config.number_of_opponent_models,
        config.opponent_models,
        n_objectives,
        config.max_groups,
        quantities,
    )
    objectives = _build_objectives(config, quantities)

    if config.deployment_zone is not None:
        t = config.deployment_zone
        deployment_zone = DeploymentZone(x_min=t[0], y_min=t[1], x_max=t[2], y_max=t[3])
    else:
        deployment_zone = DeploymentZone(
            x_min=0, y_min=0, x_max=board_width // 3, y_max=board_height
        )

    if config.opponent_deployment_zone is not None:
        t = config.opponent_deployment_zone
        opponent_deployment_zone = DeploymentZone(
            x_min=t[0], y_min=t[1], x_max=t[2], y_max=t[3]
        )
    else:
        opponent_deployment_zone = DeploymentZone(
            x_min=board_width * 2 // 3,
            y_min=0,
            x_max=board_width,
            y_max=board_height,
        )

    footprints = [
        Footprint.from_cell_rect(*tp.footprint) for tp in (config.terrain or [])
    ]
    terrain = Terrain(footprints, blocking_mask=config.blocking_mask)

    return Battle(
        board_dimensions=board_dimensions,
        player_models=player_models,
        opponent_models=opponent_models,
        objectives=objectives,
        deployment_zone=deployment_zone,
        opponent_deployment_zone=opponent_deployment_zone,
        terrain=terrain,
    )


def create_wargame_models(config: WargameEnvConfig) -> list[WargameModel]:
    """Build the list of player wargame models from config (for tests / backward compat)."""
    return _build_models(
        config.number_of_wargame_models,
        config.models,
        config.number_of_objectives,
        config.max_groups,
        resolve_rules_quantities(config),
    )


def create_opponent_models(config: WargameEnvConfig) -> list[WargameModel]:
    """Build the list of opponent models from config (for tests / backward compat)."""
    return _build_models(
        config.number_of_opponent_models,
        config.opponent_models,
        config.number_of_objectives,
        config.max_groups,
        resolve_rules_quantities(config),
    )


def create_objectives(config: WargameEnvConfig) -> list[WargameObjective]:
    """Build the list of objectives from config (for tests / backward compat)."""
    return _build_objectives(config, resolve_rules_quantities(config))
