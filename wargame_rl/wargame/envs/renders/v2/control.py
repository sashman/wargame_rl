"""Domain reads the v2 renderer needs, returned as plain data.

The legacy renderer computed objective ownership and the debug LOS verdict inside
its draw methods, mixing domain calls into drawing. v2 keeps drawing
(`scene`/`backend`) domain-free by doing those reads here and handing the results
to `build_scene` as data. This is the one place v2 touches the domain, and it
reuses the exact functions the legacy renderer used so the results match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_ownership_from_norms_offset,
)
from wargame_rl.wargame.envs.renders.v2.scene import Control


def compute_objective_control(view: BattleView) -> tuple[Control, ...]:
    """Ownership per objective, mirroring the legacy ``_draw_target`` body.

    Reproduced exactly, including the empty-opponent branch that feeds a
    ``(0, n_obj)`` norms array so a board with no opponents still resolves.
    """
    objectives = view.objectives
    if not objectives:
        return ()

    player_models = view.player_models
    opponent_models = view.opponent_models
    n_obj = len(objectives)

    player_alive = alive_mask_for(player_models)
    player_cache = compute_distances(player_models, objectives, alive_mask=player_alive)
    if opponent_models:
        opponent_alive = alive_mask_for(opponent_models)
        opponent_cache = compute_distances(
            opponent_models, objectives, alive_mask=opponent_alive
        )
        opponent_norms = opponent_cache.model_obj_norms_offset
    else:
        opponent_norms = np.zeros((0, n_obj), dtype=np.float64)

    player_controls, opponent_controls = objective_ownership_from_norms_offset(
        player_cache.model_obj_norms_offset,
        opponent_norms,
        player_cache.obj_radii,
    )

    result: list[Control] = []
    for i in range(n_obj):
        if player_controls[i]:
            result.append(Control.PLAYER)
        elif opponent_controls[i]:
            result.append(Control.OPPONENT)
        else:
            result.append(Control.NEUTRAL)
    return tuple(result)


@dataclass(frozen=True)
class LosResult:
    """The debug sight line and whether it is clear."""

    clear: bool
    a: tuple[float, float]
    b: tuple[float, float]


def probe_debug_los(view: BattleView) -> LosResult | None:
    """First alive player to first alive opponent; ``None`` if either is absent."""
    player_alive = alive_mask_for(view.player_models)
    p_idx = next((i for i, ok in enumerate(player_alive) if ok), None)
    if p_idx is None or not view.opponent_models:
        return None
    opponent_alive = alive_mask_for(view.opponent_models)
    o_idx = next((i for i, ok in enumerate(opponent_alive) if ok), None)
    if o_idx is None:
        return None

    pm = view.player_models[p_idx]
    om = view.opponent_models[o_idx]
    a = (float(pm.location[0]), float(pm.location[1]))
    b = (float(om.location[0]), float(om.location[1]))
    clear = view.has_line_of_sight_between_points(a[0], a[1], b[0], b[1])
    return LosResult(clear=clear, a=a, b=b)
