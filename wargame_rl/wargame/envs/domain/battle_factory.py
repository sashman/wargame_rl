"""Factory for creating Battle aggregates from config."""

from __future__ import annotations

from typing import Any

import numpy as np

from wargame_rl.wargame.envs.types import WargameEnvConfig

from .battle import Battle
from .entities import WargameModel, WargameObjective
from .value_objects import BoardDimensions, DeploymentZone


def _build_models(
    n: int,
    model_configs: list[Any] | None,
    n_objectives: int,
    max_groups: int,
) -> list[WargameModel]:
    """Build a list of WargameModel instances (player or opponent)."""
    result: list[WargameModel] = []
    increment = max(1, n // max_groups)
    for i in range(n):
        if model_configs is not None:
            mc = model_configs[i]
            group_id = mc.group_id
            max_wounds = mc.max_wounds
        else:
            group_id = i // increment
            max_wounds = 100
        result.append(
            WargameModel(
                location=np.zeros(2, dtype=int),
                stats={"max_wounds": max_wounds, "current_wounds": max_wounds},
                group_id=group_id,
                distances_to_objectives=np.zeros([n_objectives, 2], dtype=int),
            )
        )
    return result


def _build_objectives(config: WargameEnvConfig) -> list[WargameObjective]:
    """Build the list of objectives from config."""
    result: list[WargameObjective] = []
    for i in range(config.number_of_objectives):
        if (
            config.objectives is not None
            and config.objectives[i].radius_size is not None
        ):
            radius = config.objectives[i].radius_size
        else:
            radius = config.objective_radius_size

        result.append(
            WargameObjective(
                location=np.zeros(2, dtype=int),
                radius_size=radius,  # type: ignore[arg-type]
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

    player_models = _build_models(
        config.number_of_wargame_models,
        config.models,
        n_objectives,
        config.max_groups,
    )
    opponent_models = _build_models(
        config.number_of_opponent_models,
        config.opponent_models,
        n_objectives,
        config.max_groups,
    )
    objectives = _build_objectives(config)

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

    return Battle(
        board_dimensions=board_dimensions,
        player_models=player_models,
        opponent_models=opponent_models,
        objectives=objectives,
        deployment_zone=deployment_zone,
        opponent_deployment_zone=opponent_deployment_zone,
    )


def create_wargame_models(config: WargameEnvConfig) -> list[WargameModel]:
    """Build the list of player wargame models from config (for tests / backward compat)."""
    return _build_models(
        config.number_of_wargame_models,
        config.models,
        config.number_of_objectives,
        config.max_groups,
    )


def create_opponent_models(config: WargameEnvConfig) -> list[WargameModel]:
    """Build the list of opponent models from config (for tests / backward compat)."""
    return _build_models(
        config.number_of_opponent_models,
        config.opponent_models,
        config.number_of_objectives,
        config.max_groups,
    )


def create_objectives(config: WargameEnvConfig) -> list[WargameObjective]:
    """Build the list of objectives from config (for tests / backward compat)."""
    return _build_objectives(config)
