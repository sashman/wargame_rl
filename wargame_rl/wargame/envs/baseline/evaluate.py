"""Run a baseline policy over fixed seeds and report what it achieved.

Shared by the `measure-baselines` script and by training, so the bar logged
next to a learned policy is produced by exactly the same code path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.state import EventLogExporter, JsonMatchCodec
from wargame_rl.wargame.envs.types import (
    WargameEnvAction,
    WargameEnvConfig,
    WargameEnvObservation,
)
from wargame_rl.wargame.envs.wargame import WargameEnv

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy

# Anything that can drive the player's models for one step. Scripted baselines
# and learned checkpoints both reduce to this, so they can be scored and
# recorded by identical code rather than by two loops that drift apart.
ActionSelector: TypeAlias = Callable[
    [WargameEnvObservation, WargameEnv], WargameEnvAction
]


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


def selector_for(policy: BaselinePolicy) -> ActionSelector:
    """Adapt a scripted baseline to the `ActionSelector` calling convention."""

    def select(observation: WargameEnvObservation, env: WargameEnv) -> WargameEnvAction:
        return policy.select_action(
            env.wargame_models, env, action_mask=observation.action_mask
        )

    return select


def record_episode(
    select: ActionSelector,
    config: WargameEnvConfig,
    seed: int,
    output_path: Path,
) -> Path:
    """Run one episode with event recording and write the log.

    Reference traces are what give a per-step metric a scale: an agent's
    `oscillation_rate` of 0.3 means nothing until a known-good policy's is on
    the same chart. `just analyze-compare <agent> <baseline>` consumes these.

    One episode rather than the whole seed set, because `EventLog.record_reset`
    replaces the event list — the log only ever holds the most recent episode.
    """
    exporter = EventLogExporter()
    env = WargameEnv(config, renderer=None, state_exporters=[exporter])
    try:
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = select(observation, env)
            observation, _reward, terminated, truncated, _info = env.step(action)
    finally:
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(JsonMatchCodec().encode(exporter.log))
    return output_path


def record_baseline_episode(
    policy: BaselinePolicy,
    config: WargameEnvConfig,
    seed: int,
    output_path: Path,
) -> Path:
    """Record one episode driven by a scripted baseline."""
    return record_episode(selector_for(policy), config, seed, output_path)


def evaluate_baseline(
    policy: BaselinePolicy,
    env: WargameEnv,
    seeds: list[int],
) -> BaselineResult:
    """Run a scripted baseline on `env` once per seed and aggregate the outcome."""
    return evaluate_selector(selector_for(policy), env, seeds, type(policy).__name__)


def evaluate_selector(
    select: ActionSelector,
    env: WargameEnv,
    seeds: list[int],
    name: str,
) -> BaselineResult:
    """Run `select` on `env` once per seed and aggregate the outcome.

    Episodes are seeded so two policies are compared on identical layouts —
    objective placement dominates episode variance, so resampling would make
    the comparison mostly a question of which maps each policy drew. A learned
    checkpoint scored through here is therefore directly comparable to the
    baseline table, because it is the same code.
    """
    fractions: list[float] = []
    wins: list[float] = []
    player_vps: list[float] = []
    opponent_vps: list[float] = []
    cohesion_gaps: list[float] = []

    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            # The observation's mask already encodes range, line of sight,
            # target-alive and engagement-range validity, so a shooting
            # baseline plays by exactly the rules the learned policy does.
            action = select(observation, env)
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
