from __future__ import annotations

from abc import ABC, abstractmethod

from wargame_rl.wargame.envs import wargame


class Renderer(ABC):
    @abstractmethod
    def setup(self, env: wargame.WargameEnv):
        pass

    @abstractmethod
    def render(self, env: wargame.WargameEnv):
        pass

    @abstractmethod
    def close(self):
        pass
