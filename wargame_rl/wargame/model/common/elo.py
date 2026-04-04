"""Simple Elo rating utilities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EloRatingSystem:
    """Stateful Elo ratings for a set of named players."""

    initial_rating: float = 1000.0
    k_factor: float = 32.0
    ratings: dict[str, float] = field(default_factory=dict)

    def ensure_player(self, name: str) -> None:
        self.ratings.setdefault(name, float(self.initial_rating))

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return float(1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)))

    def update(
        self, player_a: str, player_b: str, score_a: float
    ) -> tuple[float, float]:
        """Update both players from one result.

        Args:
            player_a: First player name
            player_b: Second player name
            score_a: Observed score for player_a in {0.0, 0.5, 1.0}
        """
        self.ensure_player(player_a)
        self.ensure_player(player_b)

        ra = self.ratings[player_a]
        rb = self.ratings[player_b]
        expected_a = self.expected_score(ra, rb)
        expected_b = 1.0 - expected_a
        score_b = 1.0 - score_a

        new_ra = ra + self.k_factor * (score_a - expected_a)
        new_rb = rb + self.k_factor * (score_b - expected_b)
        self.ratings[player_a] = new_ra
        self.ratings[player_b] = new_rb
        return new_ra, new_rb

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)
