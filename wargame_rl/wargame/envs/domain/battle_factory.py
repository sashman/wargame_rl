"""Factory for creating Battle aggregates from config."""

from __future__ import annotations

from typing import Any

import numpy as np

from wargame_rl.wargame.envs.types import WargameEnvConfig

from .battle import Battle
from .entities import WargameModel, WargameObjective
from .rules_quantities import resolve_rules_quantities
from .terrain import Footprint, Terrain
from .value_objects import (
    POSITION_DTYPE,
    BoardDimensions,
    DeploymentZone,
    zero_position,
)


def group_span(n_models: int, max_groups: int) -> int:
    """How many models share a group id, given the army size and the cap.

    Mirrors the split `_build_models` performs. Exposed so that anything sized
    per *unit* -- the shooting action space, its mask -- derives the same number
    the models were built with rather than assuming one.
    """
    return max(1, n_models // max_groups)


def n_groups_for(n_models: int, max_groups: int) -> int:
    """The number of units an army of `n_models` actually splits into.

    **Not `max_groups`.** The split is `group_id = i // (n // max_groups)`, so a
    7-model army with a cap of 5 gets an increment of 1 and ends up with *seven*
    units, one per model. Anything sized on the unit count has to ask, not
    assume: an action space built from `max_groups` would silently be unable to
    name three of those units.
    """
    if n_models <= 0:
        return 0
    span = group_span(n_models, max_groups)
    return (n_models + span - 1) // span


def unit_count(
    n_models: int, max_groups: int, model_configs: list[Any] | None = None
) -> int:
    """How many shooting-target slots an army needs.

    **The action index *is* the group id**, so the slice has to be wide enough
    for the highest id in play, not merely as wide as the number of units. Those
    differ whenever `model_configs` name their own groups: two models both
    declared `group_id: 0` are one unit, while the count-based split would have
    made two.

    Deriving this from the counts alone was a real bug -- a config with explicit
    per-model groups got a slice sized for a split it does not use, leaving
    actions that name no unit at all.
    """
    if model_configs:
        return max(int(cfg.group_id) for cfg in model_configs) + 1
    return n_groups_for(n_models, max_groups)


def _build_models(
    n: int,
    model_configs: list[Any] | None,
    n_objectives: int,
    max_groups: int,
    base_radius: float = 0.0,
) -> list[WargameModel]:
    """Build a list of WargameModel instances (player or opponent)."""
    result: list[WargameModel] = []
    increment = group_span(n, max_groups)
    for i in range(n):
        if model_configs is not None:
            mc = model_configs[i]
            group_id = mc.group_id
            max_wounds = mc.max_wounds
            toughness = mc.toughness
            save = mc.save
        else:
            group_id = i // increment
            max_wounds = 100
            toughness = 3
            save = 4
        result.append(
            WargameModel(
                location=zero_position(),
                stats={
                    "max_wounds": max_wounds,
                    "current_wounds": max_wounds,
                    "toughness": toughness,
                    "save": save,
                },
                group_id=group_id,
                base_radius=base_radius,
                distances_to_objectives=np.zeros(
                    [n_objectives, 2], dtype=POSITION_DTYPE
                ),
            )
        )
    return result


def _build_objectives(config: WargameEnvConfig) -> list[WargameObjective]:
    """Build the list of objectives from config."""
    result: list[WargameObjective] = []
    for i in range(config.number_of_objectives):
        objective_config = (
            config.objectives[i] if config.objectives is not None else None
        )
        area = objective_config.to_polygon() if objective_config is not None else None
        if area is not None:
            # An area is not placed: its outline is its position, and its
            # location is the centroid so anything steering at an objective
            # still has a point to aim at.
            objective = WargameObjective(location=zero_position(), radius_size=0.0)
            objective.set_area(area)
            result.append(objective)
            continue

        radius = (
            objective_config.radius_size
            if objective_config is not None and objective_config.radius_size is not None
            else float(config.objective_radius_size)
        )
        result.append(
            WargameObjective(location=zero_position(), radius_size=float(radius))
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

    base_radius = resolve_rules_quantities(config).base_radius
    player_models = _build_models(
        config.number_of_wargame_models,
        config.models,
        n_objectives,
        config.max_groups,
        base_radius,
    )
    opponent_models = _build_models(
        config.number_of_opponent_models,
        config.opponent_models,
        n_objectives,
        config.max_groups,
        base_radius,
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

    # `to_polygon` is where a config's chosen form -- inclusive cell rectangle or
    # explicit outline -- becomes the one shape the domain holds.
    footprints = [Footprint(tp.to_polygon()) for tp in (config.terrain or [])]
    terrain = Terrain(footprints)

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
        resolve_rules_quantities(config).base_radius,
    )


def create_opponent_models(config: WargameEnvConfig) -> list[WargameModel]:
    """Build the list of opponent models from config (for tests / backward compat)."""
    return _build_models(
        config.number_of_opponent_models,
        config.opponent_models,
        config.number_of_objectives,
        config.max_groups,
        resolve_rules_quantities(config).base_radius,
    )


def create_objectives(config: WargameEnvConfig) -> list[WargameObjective]:
    """Build the list of objectives from config (for tests / backward compat)."""
    return _build_objectives(config)
