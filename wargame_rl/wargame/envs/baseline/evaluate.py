"""Run a baseline policy over fixed seeds and report what it achieved.

Shared by the `measure-baselines` script and by training, so the bar logged
next to a learned policy is produced by exactly the same code path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from wargame_rl.wargame.envs.evaluation import (
    EvalResult,
    format_optional_metric,
    mean_of_measured,
    paired_difference,
    read_end_of_episode,
    standard_error,
)
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


# The result value object and its helpers live in the evaluation kernel
# (`envs/evaluation/`) so the per-model facade's runner can produce the same
# value without importing this module, which imports the phase facade. The
# old names stay importable from here.
BaselineResult = EvalResult


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
    combat_seeds: list[int] | None = None,
) -> BaselineResult:
    """Run a scripted baseline on `env` once per seed and aggregate the outcome."""
    return evaluate_selector(
        selector_for(policy),
        env,
        seeds,
        type(policy).__name__,
        combat_seeds=combat_seeds,
    )


def evaluate_selector(
    select: ActionSelector,
    env: WargameEnv,
    seeds: list[int],
    name: str,
    combat_seeds: list[int] | None = None,
) -> BaselineResult:
    """Run `select` on `env` once per seed and aggregate the outcome.

    Episodes are seeded so two policies are compared on identical layouts —
    objective placement dominates episode variance, so resampling would make
    the comparison mostly a question of which maps each policy drew. A learned
    checkpoint scored through here is therefore directly comparable to the
    baseline table, because it is the same code.

    `combat_seeds` (same length as `seeds`) drives the dice independently of the
    layout, so a fixed set of maps can be replayed under different rolls. That
    is the only way to tell a policy's spread apart from the dice's.
    """
    if combat_seeds is not None and len(combat_seeds) != len(seeds):
        raise ValueError(
            f"combat_seeds must match seeds in length: "
            f"{len(combat_seeds)} != {len(seeds)}"
        )
    fractions: list[float] = []
    wins: list[float] = []
    player_vps: list[float] = []
    opponent_vps: list[float] = []
    cohesion_gaps: list[float] = []
    survivals: list[float] = []
    exposures: list[float | None] = []
    proximities: list[float | None] = []
    firepower: list[float | None] = []
    held: list[float] = []
    coherency: list[float | None] = []
    models_out: list[float | None] = []
    opponent_coherency: list[float | None] = []
    opponent_models_out: list[float | None] = []

    for index, seed in enumerate(seeds):
        options = None if combat_seeds is None else {"combat_seed": combat_seeds[index]}
        observation, _ = env.reset(seed=seed, options=options)
        terminated = truncated = False
        while not (terminated or truncated):
            # The observation's mask already encodes range, line of sight,
            # target-alive and engagement-range validity, so a shooting
            # baseline plays by exactly the rules the learned policy does.
            action = select(observation, env)
            observation, _reward, terminated, truncated, _info = env.step(action)

        end = read_end_of_episode(
            env.wargame_models, env.opponent_models, env.objectives
        )
        fractions.append(end.at_objectives)
        wins.append(1.0 if env.player_vp > env.opponent_vp else 0.0)
        player_vps.append(float(env.player_vp))
        opponent_vps.append(float(env.opponent_vp))
        cohesion_gaps.append(end.worst_cohesion_gap)
        survivals.append(end.fraction_alive)
        exposures.append(env.exposure_rate)
        proximities.append(env.terrain_proximity)
        firepower.append(env.firepower_ratio)
        # Intent first: under `enforce_move` the realised rate is 1.000 however
        # the policy played, so reading it would report the referee. With no
        # referee `intended_*` is None and the two are the same board anyway.
        coherency.append(
            env.intended_coherency_rate
            if env.intended_coherency_rate is not None
            else env.coherency_rate
        )
        models_out.append(
            env.intended_models_out_of_coherency
            if env.intended_models_out_of_coherency is not None
            else env.models_out_of_coherency
        )
        opponent_coherency.append(
            env.opponent_intended_coherency_rate
            if env.opponent_intended_coherency_rate is not None
            else env.opponent_coherency_rate
        )
        opponent_models_out.append(
            env.opponent_intended_models_out_of_coherency
            if env.opponent_intended_models_out_of_coherency is not None
            else env.opponent_models_out_of_coherency
        )
        held.append(end.objectives_held)

    return BaselineResult(
        name=name,
        n_episodes=len(seeds),
        final_fraction_at_objectives=float(np.mean(fractions)),
        win_rate=float(np.mean(wins)),
        player_vp=float(np.mean(player_vps)),
        opponent_vp=float(np.mean(opponent_vps)),
        worst_cohesion_gap=float(np.mean(cohesion_gaps)),
        final_fraction_alive=float(np.mean(survivals)),
        # Stays None when the config did not measure it — averaging an unmeasured
        # metric to a number would invent data.
        exposure_rate=mean_of_measured(exposures),
        terrain_proximity=mean_of_measured(proximities),
        firepower_ratio=mean_of_measured(firepower),
        objectives_held=float(np.mean(held)),
        coherency_rate=mean_of_measured(coherency),
        models_out_of_coherency=mean_of_measured(models_out),
        opponent_coherency_rate=mean_of_measured(opponent_coherency),
        opponent_models_out_of_coherency=mean_of_measured(opponent_models_out),
        vp_margin_per_episode=tuple(
            player - opponent for player, opponent in zip(player_vps, opponent_vps)
        ),
        objectives_held_per_episode=tuple(held),
        win_per_episode=tuple(wins),
    )


__all__ = [
    "ActionSelector",
    "BaselineResult",
    "EvalResult",
    "evaluate_baseline",
    "evaluate_selector",
    "format_optional_metric",
    "mean_of_measured",
    "paired_difference",
    "record_baseline_episode",
    "record_episode",
    "selector_for",
    "standard_error",
]
