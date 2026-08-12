"""One fixed layout — the terrain and, optionally, the objectives that go with it.

A `MapLayout` is what a map file becomes once it has been read and resolved: the
value `place_for_episode` installs. Loading and validating the pool of them is
the env layer's job (`envs/map_pool.py`), so nothing in `domain/` touches the
filesystem; the domain only ever sees a layout that is already a set of
footprints and objective descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass

from wargame_rl.wargame.envs.domain.terrain import Terrain
from wargame_rl.wargame.envs.types.config import ObjectiveConfig


@dataclass(frozen=True)
class MapLayout:
    """A named layout drawn from a `map_pool`.

    `objectives` is None for a map that carries terrain alone, which then scores
    under the scenario's own objective placement — on this layout's ruins, since
    the terrain is installed first. A map that does carry them replaces the
    scenario's outright, count included.
    """

    name: str
    terrain: Terrain
    objectives: tuple[ObjectiveConfig, ...] | None = None

    @property
    def n_objectives(self) -> int:
        """Objectives this layout brings, 0 when it defers to the scenario's."""
        return 0 if self.objectives is None else len(self.objectives)

    @property
    def n_pieces(self) -> int:
        return len(self.terrain.footprints)
