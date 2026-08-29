"""Run a baseline policy over fixed seeds and report what it achieved.

Shared by the `measure-baselines` script and by training, so the bar logged
next to a learned policy is produced by exactly the same code path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    final_fraction_alive: float
    # None unless the config sets `track_exposure`. Read together: a policy that
    # is merely out of range keeps proximity high, one using ruins pulls it down.
    exposure_rate: float | None
    terrain_proximity: float | None
    # (enemies we can shoot) - (our models they can shoot), per shooting phase.
    # The exchange-ratio measure: exposure alone cannot tell manoeuvre from
    # hiding, because both lower it.
    firepower_ratio: float | None
    # Mean count of objectives the player *controls* at episode end -- strictly
    # more player models than opponent models inside the disc, the same rule VP
    # scores on.
    #
    # This is not derivable from `final_fraction_at_objectives`, which is the
    # fraction of *alive* models standing on *any* objective and therefore
    # cannot tell 15 models on one point from 5 each on three. Both read ~0.95
    # while one scores 5 VP a round and the other 15. Measuring occupancy
    # without this is how three experimental rounds were aimed at a deficit that
    # was mostly measurement noise.
    objectives_held: float
    # **The rules-legality column, reported unconditionally.** Share of the
    # player's unit-movement-phases in coherency (`docs/rules/03-moving.md`
    # § Coherency), and the mean models outside their unit's coherent body.
    #
    # Always present, never opt-in, because a score quoted without it is a score
    # that may have been earned by illegal moves. Coherency is *measured* on
    # every config and *enforced* on almost none, so silence here reads as
    # compliance and is not.
    #
    # This is the **policy's own** figure: it prefers `intended_coherency_rate`,
    # falling back to the realised rate only when nothing is enforcing, where the
    # two are identical by construction. Under `coherency.enforce_move` the
    # realised rate is 1.000 whatever the policy does -- a metric sampled after a
    # corrective wrapper measures the wrapper -- and reading it that way is what
    # published a policy intending 0.630 as 1.000.
    #
    # Read the pair together: a unit shot down to one model is coherent by
    # definition, so a rising rate can mean the units died. `models_out` has no
    # such failure mode, since a dead model contributes nothing to it.
    coherency_rate: float | None = None
    models_out_of_coherency: float | None = None
    # The same two columns for the OPPONENT force. A rated leg seats entrant B
    # there and nothing else measured it, so an entrant that never took the
    # player seat came back with the coherency column blank -- a score without
    # the claim that the moves earning it were legal, which is the one thing
    # this column exists to carry. Every other consumer ignores them, exactly as
    # it ignores `exposure_rate` on a config that does not track it.
    opponent_coherency_rate: float | None = None
    opponent_models_out_of_coherency: float | None = None
    # Per-episode values, in seed order, kept so a result can carry an error bar
    # and so two results measured on the same seeds can be paired. The loop
    # already builds these lists; discarding them is why no figure in this
    # repo's reports has ever had one. Default empty, so a hand-built
    # `BaselineResult` in a test stays valid.
    vp_margin_per_episode: tuple[float, ...] = ()
    objectives_held_per_episode: tuple[float, ...] = ()
    win_per_episode: tuple[float, ...] = ()

    @property
    def vp_margin(self) -> float:
        """Mean VP lead over the opponent — the phase-invariant scoreboard."""
        return self.player_vp - self.opponent_vp

    @property
    def vp_margin_se(self) -> float | None:
        """Standard error of the mean `vp_margin`, or None below two episodes.

        Per-episode `vp_margin` has a standard deviation of 45–50 on the 25v25
        scenarios, so n=30 carries an SE of ~8–9 — larger than most arm
        differences ever measured here. Reporting the mean without this is what
        made a string of noise-level gaps read as effects.
        """
        return standard_error(self.vp_margin_per_episode)


def format_optional_metric(value: float | None, decimals: int = 3) -> str:
    """Render a metric that may not have been measured.

    `exposure_rate` and `terrain_proximity` are None unless the config sets
    `track_exposure`. Printing them as `0.000` would read as "never exposed",
    so an unmeasured value is shown as a dash instead.
    """
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def standard_error(values: Sequence[float]) -> float | None:
    """Standard error of the mean, or None when fewer than two samples."""
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def paired_difference(
    treatment: BaselineResult, control: BaselineResult
) -> tuple[float, float | None]:
    """Mean and SE of the per-episode `vp_margin` difference, treatment first.

    Pairing is the whole point: layout variance dwarfs most effects here, and
    it cancels exactly when both policies played the same seeds. An unpaired
    read of one such comparison said +8.0 where the paired read said
    +1.7 ± 5.7.

    Raises:
        ValueError: If the two results did not cover the same episode count —
            differencing across different layout sets is meaningless.
    """
    left = treatment.vp_margin_per_episode
    right = control.vp_margin_per_episode
    if len(left) != len(right) or not left:
        raise ValueError(
            f"paired difference needs equal, non-empty episode counts: "
            f"{len(left)} != {len(right)}"
        )
    differences = [a - b for a, b in zip(left, right)]
    return float(np.mean(differences)), standard_error(differences)


def mean_of_measured(values: list[float | None]) -> float | None:
    """Mean over the episodes that measured the metric, or None if none did."""
    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return float(np.mean(measured))


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
        survivals.append(float(alive.mean()))
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

        # Control is a strict count comparison, so an objective with equal
        # numbers on it scores for nobody.
        opponent_alive = alive_mask_for(env.opponent_models)
        if env.opponent_models:
            opponent_norms = compute_distances(
                env.opponent_models, env.objectives, alive_mask=opponent_alive
            ).model_obj_norms_offset
            opponent_counts = (opponent_norms <= cache.obj_radii).sum(axis=0)
        else:
            opponent_counts = np.zeros(len(env.objectives), dtype=int)
        player_counts = (cache.model_obj_norms_offset[alive] <= cache.obj_radii).sum(
            axis=0
        )
        held.append(float((player_counts > opponent_counts).sum()))

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
