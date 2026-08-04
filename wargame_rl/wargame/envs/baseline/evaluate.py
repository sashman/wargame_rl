"""Run a baseline policy over fixed seeds and report what it achieved.

Shared by the `measure-baselines` script and by training, so the bar logged
next to a learned policy is produced by exactly the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy
    from wargame_rl.wargame.envs.wargame import WargameEnv


@dataclass(frozen=True)
class BaselineResult:
    """Aggregate outcome of running a baseline over a set of episodes."""

    name: str
    n_episodes: int
    final_fraction_at_objectives: float
    win_rate: float
    player_vp: float
    opponent_vp: float
    worst_cohesion_gap: float

    @property
    def vp_margin(self) -> float:
        """Mean VP lead over the opponent — the phase-invariant scoreboard."""
        return self.player_vp - self.opponent_vp


def _worst_cohesion_gap(env: WargameEnv) -> float:
    """Largest distance from any alive model to its nearest living squadmate.

    Uses the same helper the `group_cohesion` calculator does, so the number is
    directly comparable to a phase's `group_max_distance`.
    """
    models = env.wargame_models
    alive = alive_mask_for(models)
    if not alive.any():
        return 0.0
    cache = compute_distances(models, env.objectives, compute_model_model=True)
    group_ids = np.array([m.group_id for m in models], dtype=np.intp)
    distances = cache.min_distances_to_same_group(group_ids, alive_mask=alive)
    return float(distances[alive].max())


def evaluate_baseline(
    policy: BaselinePolicy,
    env: WargameEnv,
    seeds: list[int],
) -> BaselineResult:
    """Run `policy` on `env` once per seed and aggregate the outcome.

    Episodes are seeded so two baselines are compared on identical layouts —
    objective placement dominates episode variance, so resampling would make
    the comparison mostly a question of which maps each policy drew.
    """
    name = type(policy).__name__
    fractions: list[float] = []
    wins: list[float] = []
    player_vps: list[float] = []
    opponent_vps: list[float] = []
    cohesion_gaps: list[float] = []

    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy.select_action(env.wargame_models, env)
            observation, _reward, terminated, truncated, _info = env.step(action)

        alive = alive_mask_for(env.wargame_models)
        cache = compute_distances(env.wargame_models, env.objectives, alive_mask=alive)
        at_objective = np.atleast_1d(
            (cache.model_obj_norms_offset <= cache.obj_radii).any(axis=1)
        )
        fractions.append(
            float(at_objective[alive].mean()) if alive.any() else 0.0,
        )
        wins.append(1.0 if env.player_vp > env.opponent_vp else 0.0)
        player_vps.append(float(env.player_vp))
        opponent_vps.append(float(env.opponent_vp))
        cohesion_gaps.append(_worst_cohesion_gap(env))

    return BaselineResult(
        name=name,
        n_episodes=len(seeds),
        final_fraction_at_objectives=float(np.mean(fractions)),
        win_rate=float(np.mean(wins)),
        player_vp=float(np.mean(player_vps)),
        opponent_vp=float(np.mean(opponent_vps)),
        worst_cohesion_gap=float(np.mean(cohesion_gaps)),
    )
