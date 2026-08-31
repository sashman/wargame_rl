"""The rating a training run logs about itself.

Two numbers, on two different scales, and the tests that keep them apart:

- `eval/elo` inverts the Elo curve on the eval games already played, against
  the config's own opponent pinned at zero. It is a **monotone transform of
  `eval/vp_margin`** and adds no information about that pairing -- what it adds
  is a bounded scale and a unit.
- `self_play/learner_elo` is a ladder against the pool the learner actually
  played, which is the thing a self-play run is supposed to move and which a
  margin against one fixed opponent cannot see.

⚠ Neither is a Bradley-Terry rating. `test_the_in_run_rating_is_not_a_fitted_one`
is the guard: if these ever become comparable to `just elo-table`, it is by
someone fitting them, not by relabelling them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from wargame_rl.wargame.model.common.self_play import OpponentScheduler, SelfPlayConfig
from wargame_rl.wargame.rating.elo import ELO_SCALE, rating_from_score, win_probability
from wargame_rl.wargame.rating.score import margin_score

WEIGHTS = {"layer.weight": torch.zeros(2, 2)}


def _scheduler(tmp_path: Path, **overrides: object) -> OpponentScheduler:
    settings = SelfPlayConfig(enabled=True, **overrides)  # type: ignore[arg-type]
    return OpponentScheduler(settings, tmp_path, seed=0)


class TestInvertingTheEloCurve:
    """`rating_from_score` is the exact inverse of `win_probability`."""

    @pytest.mark.parametrize("rating", [-800.0, -137.0, 0.0, 55.5, 400.0])
    def test_a_rating_survives_a_round_trip(self, rating: float) -> None:
        assert rating_from_score(win_probability(rating, 0.0)) == pytest.approx(
            rating, abs=1e-6
        )

    def test_an_even_score_is_the_opponents_own_rating(self) -> None:
        """`s = 0.5` has to land exactly on the opponent, with no special case,
        for the same reason `margin_score(0) = 0.5` exactly: a draw is a draw."""
        assert rating_from_score(0.5, opponent_rating=250.0) == pytest.approx(250.0)

    @pytest.mark.parametrize("score", [0.0, 1.0])
    def test_the_saturated_ends_clamp_rather_than_diverge(self, score: float) -> None:
        """A clean sweep inverts to infinity, and a mean over a few blowouts
        reaches it in float long before a policy is infinitely good."""
        rating = rating_from_score(score, max_points=ELO_SCALE)
        assert abs(rating) == pytest.approx(ELO_SCALE)

    def test_an_advantage_is_discounted_not_credited(self) -> None:
        """The seat term explains part of the score, so it comes OFF the rating
        the score implies -- crediting it would pay the policy for its chair."""
        assert rating_from_score(0.75, advantage=100.0) < rating_from_score(0.75)

    def test_a_bad_clamp_is_refused_at_the_call(self) -> None:
        with pytest.raises(ValueError, match="max_points"):
            rating_from_score(0.6, max_points=0.0)


class TestTheLadderAgainstThePool:
    """`self_play/learner_elo` moves with results and nothing else."""

    def test_it_starts_level_with_its_anchor(self, tmp_path: Path) -> None:
        assert _scheduler(tmp_path).learner_rating == 0.0

    def test_winning_raises_it_and_losing_lowers_it(self, tmp_path: Path) -> None:
        winner = _scheduler(tmp_path / "won")
        loser = _scheduler(tmp_path / "lost")
        winner.record_outcomes({"squad_march_take": [80.0, 60.0, 95.0]})
        loser.record_outcomes({"squad_march_take": [-80.0, -60.0, -95.0]})

        assert winner.learner_rating > 0.0
        assert loser.learner_rating < 0.0
        assert winner.learner_rating == pytest.approx(-loser.learner_rating)

    def test_a_zero_k_factor_pins_it(self, tmp_path: Path) -> None:
        """The no-op control: the metric can be turned off without turning off
        self-play, so a run can be compared against one that never rated."""
        scheduler = _scheduler(tmp_path, elo_k_factor=0.0)
        scheduler.record_outcomes({"squad_march_take": [200.0] * 10})

        assert scheduler.learner_rating == 0.0

    def test_the_update_does_not_depend_on_the_order_games_finished(
        self, tmp_path: Path
    ) -> None:
        """The load-bearing one. Elo updates do not commute, and an epoch's
        games are played in parallel across rollout envs -- so a sequential
        update would make the rating depend on which env happened to finish
        first, which is not a fact about the policy."""
        forward = _scheduler(tmp_path / "forward")
        backward = _scheduler(tmp_path / "backward")
        margins = [90.0, -40.0, 15.0, -75.0, 120.0]
        forward.record_outcomes({"squad_march_take": margins})
        backward.record_outcomes({"squad_march_take": list(reversed(margins))})

        assert forward.learner_rating == pytest.approx(backward.learner_rating)

    def test_an_epoch_with_no_finished_episode_leaves_it_alone(
        self, tmp_path: Path
    ) -> None:
        scheduler = _scheduler(tmp_path)
        scheduler.record_outcomes({"squad_march_take": [50.0]})
        rated = scheduler.learner_rating
        scheduler.record_outcomes({"squad_march_take": []})

        assert scheduler.learner_rating == rated

    def test_a_snapshot_inherits_the_rating_the_learner_had(
        self, tmp_path: Path
    ) -> None:
        """What makes the pool a LADDER rather than a bag: a later self that
        beats an earlier one gains points against a fixed reference."""
        scheduler = _scheduler(tmp_path, snapshot_every_n_epochs=1)
        scheduler.record_outcomes({"squad_march_take": [120.0] * 5})
        earned = scheduler.learner_rating
        scheduler.snapshot(1, WEIGHTS)

        frozen = next(e for e in scheduler.pool.entries if e.name == "epoch_1")
        assert frozen.rating == pytest.approx(earned)
        assert earned > 0.0

    def test_beating_a_stronger_opponent_pays_more(self, tmp_path: Path) -> None:
        """The property that makes it a rating rather than a running mean."""
        scheduler = _scheduler(tmp_path, snapshot_every_n_epochs=1)
        scheduler.snapshot(1, WEIGHTS)
        scheduler.rate({"epoch_1": 400.0}, learner_rating=0.0)

        against_strong = _scheduler(tmp_path / "strong", snapshot_every_n_epochs=1)
        against_strong.snapshot(1, WEIGHTS)
        against_strong.rate({"epoch_1": 400.0}, learner_rating=0.0)
        against_strong.record_outcomes({"epoch_1": [70.0]})

        against_anchor = _scheduler(tmp_path / "anchor")
        against_anchor.record_outcomes({"squad_march_take": [70.0]})

        assert against_strong.learner_rating > against_anchor.learner_rating

    def test_the_in_run_rating_is_not_a_fitted_one(self, tmp_path: Path) -> None:
        """A ladder anchored at one scripted policy is not the published scale.

        The pool's own anchor is the origin BY CONSTRUCTION -- it never moves,
        whatever it beats -- so these numbers say 'how far past my own history
        am I', not 'where do I sit among rated entrants'.
        """
        scheduler = _scheduler(tmp_path)
        scheduler.record_outcomes({"squad_march_take": [150.0] * 20})

        assert scheduler.pool.anchor.rating in (None, 0.0)


class TestTheEvalRating:
    """`eval/elo` is the eval margin on a win-probability scale."""

    def test_it_is_monotone_in_the_margin(self) -> None:
        """It carries no NEW information about the pairing, and the test says so
        rather than implying otherwise by only checking a sign."""
        margins = np.array([-120.0, -30.0, 0.0, 30.0, 120.0])
        ratings = [
            rating_from_score(float(margin_score(np.array([m]))[0])) for m in margins
        ]

        assert ratings == sorted(ratings)

    def test_a_tie_reads_level_with_the_opponent(self) -> None:
        assert rating_from_score(
            float(margin_score(np.array([0.0, 0.0])).mean())
        ) == pytest.approx(0.0)

    def test_it_saturates_where_the_margin_does_not(self) -> None:
        """The reason to log it beside `vp_margin` rather than instead of it:
        per-episode margin sd is 45-50, so one blowout moves a mean margin in a
        way it cannot move a mean win probability."""
        steady = np.array([40.0, 40.0, 40.0, 40.0])
        blowout = np.array([40.0, 40.0, 40.0, 400.0])

        margin_shift = float(blowout.mean() - steady.mean())
        rating_shift = rating_from_score(
            float(margin_score(blowout).mean())
        ) - rating_from_score(float(margin_score(steady).mean()))

        assert margin_shift > 80.0
        assert rating_shift < margin_shift
