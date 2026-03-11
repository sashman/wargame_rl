from __future__ import annotations

from abc import ABC, abstractmethod

from wargame_rl.wargame.envs.domain.battle_view import BattleView


class Renderer(ABC):
    @abstractmethod
    def setup(self, view: BattleView) -> None:
        pass

    @abstractmethod
    def render(self, view: BattleView) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
