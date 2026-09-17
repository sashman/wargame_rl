"""Baseline: one armed squad clears the blocker, then the unarmed squads march."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_shoot import (
    ScriptedSquadMarchShootPolicy,
)
from wargame_rl.wargame.envs.baseline.scripted_squad_march_take import (
    ScriptedSquadMarchTakePolicy,
)
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.kernel.entities import WargameModel
    from wargame_rl.wargame.envs.wargame import WargameEnv

# How far inside its weapon's reach an armed squad stops. Its members are
# spread up to a chain's width around the centroid the vector is steered on, so
# a stop exactly at the reach leaves the trailing member a shot short.
STANDOFF_MARGIN = 2.0
# How far OUTSIDE the enemy's reach an unarmed squad waits: the same chain's
# width, on the other side of the line.
WAIT_MARGIN = 2.0


class ScriptedEscortPolicy(ScriptedSquadMarchTakePolicy):
    """A plan with an order to it: shoot the blocker off the point, then march.

    The curriculum's C3 rung (#340) asks for sequencing -- one of our squads is
    armed and an enemy unit stands on a point firing at whoever approaches, so
    the unarmed squads have to wait for the armed one to clear it. No other
    baseline waits for anything; `squad_march_take` walks every squad at its
    point on turn one and the unarmed ones die on the way in. This is the
    reference solution the rung is read against (#330), and what D1 clones.

    While any enemy is alive: every **armed** squad steers, as one body on
    the squad-march vector, at the centroid of the nearest live enemy unit and
    stops a little inside its own weapon reach; every **unarmed** squad walks
    its `squad_march_take` vector but stops a little outside the enemy's
    reach, and waits there. Once the last enemy is dead, or when no squad is
    armed at all, every squad plays `squad_march_take` unchanged. The target rule is
    `squad_march_shoot`'s (nearest valid unit under the env's own mask), so on
    the seat with one enemy unit the armed squad fires at the blocker the
    moment it is in reach.

    Both seats: it reads only what the mirror provides (`player_max_ranges`,
    `opponent_models`, `player_action_handler`), as its parent does.
    """

    select_shooting = ScriptedSquadMarchShootPolicy.select_shooting

    def select_movement(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """Armed squads close on the blocker, unarmed squads wait out of its
        reach, until it dies."""
        enemies = [m for m in env.opponent_models if m.is_alive]
        reach = np.asarray(env.player_max_ranges, dtype=float)
        enemy_reach = np.asarray(env.opponent_max_ranges, dtype=float)
        armed_groups = {
            m.group_id
            for i, m in enumerate(models)
            if m.is_alive and i < reach.size and reach[i] > 0.0
        }
        if not enemies or not armed_groups:
            return super().select_movement(models, env)

        actions = list(super().select_movement(models, env).actions)
        speeds = env.player_action_handler.move_speeds
        enemy_locations = np.array([m.location for m in enemies], dtype=float)
        enemy_groups = np.array([m.group_id for m in enemies], dtype=int)
        enemy_alive = [i for i, m in enumerate(env.opponent_models) if m.is_alive]
        danger = float(
            max(
                (enemy_reach[i] for i in enemy_alive if i < enemy_reach.size),
                default=0.0,
            )
        )
        objectives = self.squad_objectives(
            models, env, sorted({m.group_id for m in models})
        )

        for squad_index, group_id in enumerate(sorted({m.group_id for m in models})):
            member_indices = [
                i for i, m in enumerate(models) if m.group_id == group_id and m.is_alive
            ]
            if not member_indices:
                continue
            centroid = np.mean(
                [models[i].location for i in member_indices], axis=0, dtype=float
            )
            max_step = float(
                min(speeds[i] for i in member_indices) if speeds.size else 0.0
            )
            if group_id not in armed_groups:
                # Walk at the point, but never into the blocker's reach: the
                # step is capped by the room left before the waiting line.
                nearest_enemy = float(
                    np.linalg.norm(enemy_locations - centroid, axis=1).min()
                )
                room = max(0.0, nearest_enemy - (danger + WAIT_MARGIN))
                objective = objectives[squad_index]
                lead = np.asarray(objective.location, dtype=float) - centroid
                lead_distance = float(np.linalg.norm(lead))
                step = min(max_step, lead_distance, room)
                for i in member_indices:
                    actions[i] = (
                        STAY_ACTION
                        if step <= 0.0
                        else env.player_action_handler.best_action_toward(
                            float(lead[0]),
                            float(lead[1]),
                            max_step_length=step,
                            model_idx=i,
                        )
                    )
                continue
            # The nearest enemy UNIT by its closest model, as the target rule
            # measures it; the squad steers at that unit's centroid.
            distances = np.linalg.norm(enemy_locations - centroid, axis=1)
            unit = int(enemy_groups[int(np.argmin(distances))])
            target = np.mean(enemy_locations[enemy_groups == unit], axis=0)
            lead = target - centroid
            lead_distance = float(np.linalg.norm(lead))
            standoff = float(min(reach[i] for i in member_indices)) - STANDOFF_MARGIN
            step = min(max_step, max(0.0, lead_distance - max(standoff, 0.0)))
            for i in member_indices:
                actions[i] = (
                    STAY_ACTION
                    if step <= 0.0
                    else env.player_action_handler.best_action_toward(
                        float(lead[0]),
                        float(lead[1]),
                        max_step_length=step,
                        model_idx=i,
                    )
                )
        return WargameEnvAction(actions=actions)


register_baseline("scripted_escort", ScriptedEscortPolicy)
