"""Score state (e.g. Victory Points) exposed in the environment observation."""

from __future__ import annotations

from dataclasses import dataclass

from gymnasium import spaces

# Max value for VP in the observation space. Discrete(n) has n elements (0 to n-1),
# so we use MAX + 1 to allow 0..MAX.
MAX_VP_FOR_SPACE = 1000


@dataclass
class EnvScoreState:
    """Player and opponent score (e.g. Victory Points) at the current step."""

    player_vp: int = 0
    opponent_vp: int = 0

    @staticmethod
    def to_space() -> spaces.Dict:
        """Gymnasium space for this score state. Each field is a single scalar (Discrete)."""
        return spaces.Dict(
            {
                "player_vp": spaces.Discrete(
                    MAX_VP_FOR_SPACE + 1
                ),  # 0..MAX_VP_FOR_SPACE
                "opponent_vp": spaces.Discrete(MAX_VP_FOR_SPACE + 1),
            }
        )
