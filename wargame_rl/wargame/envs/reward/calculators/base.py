from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class PerModelRewardCalculator(ABC):
    """Computes a scalar reward contribution for a single model each step."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    @abstractmethod
    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float: ...

    @property
    def needs_model_model_distances(self) -> bool:
        """Override to return True if this calculator uses model-model norms."""
        return False


class GlobalRewardCalculator(ABC):
    """Computes a scalar reward contribution once per step (not per model)."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    @abstractmethod
    def calculate(self, view: BattleView, ctx: StepContext) -> float: ...

    @property
    def needs_model_model_distances(self) -> bool:
        """Override to return True if this calculator uses model-model norms."""
        return False
