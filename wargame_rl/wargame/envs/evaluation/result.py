"""The outcome of evaluating one policy over a set of seeded episodes.

`EvalResult` was `BaselineResult` in `envs/baseline/evaluate.py`, which still
exports it under that name; it moved here because the per-model facade's
runner produces the same value and may not import the phase facade's
module. The three per-model-only readouts (reward, decisions, success) are
optional and `None` where the facade did not measure them, the convention
`exposure_rate` already follows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvalResult:
    """Aggregate outcome of running a policy over a set of episodes."""

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
    # repo's reports has ever had one. Default empty, so a hand-built result in
    # a test stays valid.
    vp_margin_per_episode: tuple[float, ...] = ()
    objectives_held_per_episode: tuple[float, ...] = ()
    win_per_episode: tuple[float, ...] = ()
    # Measured only by the per-model runner, which carries the re-timed reward
    # and the success criterion beside the game: the episode's reward, its
    # count of decision steps, and whether the phase's criterion held at the
    # end. None on the phase facade, which reads those off its own env.
    episode_rewards: tuple[float, ...] | None = None
    decisions_per_episode: tuple[int, ...] | None = None
    success_per_episode: tuple[bool, ...] | None = None

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

    @property
    def mean_reward(self) -> float | None:
        """Mean episode reward, or None where the runner did not measure it."""
        return _mean_or_none(self.episode_rewards)

    @property
    def max_reward(self) -> float | None:
        return None if not self.episode_rewards else float(max(self.episode_rewards))

    @property
    def min_reward(self) -> float | None:
        return None if not self.episode_rewards else float(min(self.episode_rewards))

    @property
    def mean_decisions(self) -> float | None:
        """Mean decision steps per episode -- the per-model facade's unit."""
        return _mean_or_none(self.decisions_per_episode)

    @property
    def success_rate(self) -> float | None:
        return _mean_or_none(
            None
            if self.success_per_episode is None
            else [float(s) for s in self.success_per_episode]
        )


def _mean_or_none(values: Sequence[float] | Sequence[int] | None) -> float | None:
    if values is None or len(values) == 0:
        return None
    return float(np.mean(values))


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
    treatment: EvalResult, control: EvalResult
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


__all__ = [
    "EvalResult",
    "format_optional_metric",
    "mean_of_measured",
    "paired_difference",
    "standard_error",
]
