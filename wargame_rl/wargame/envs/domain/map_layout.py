"""One fixed layout — the terrain and, optionally, the objectives that go with it.

A `MapLayout` is what a map file becomes once it has been read and resolved: the
value `place_for_episode` installs. Loading and validating the pool of them is
the env layer's job (`envs/map_pool.py`), so nothing in `domain/` touches the
filesystem; the domain only ever sees a layout that is already a set of
footprints and objective descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from wargame_rl.wargame.envs.domain.terrain import Footprint, Terrain
from wargame_rl.wargame.envs.types.config import ObjectiveConfig
from wargame_rl.wargame.envs.types.geometry import Polygon


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

    def mirrored(
        self,
        board_width: float,
        board_height: float,
        flip_x: bool,
        flip_y: bool,
    ) -> "MapLayout":
        """This layout reflected about the board's mid-lines.

        Four orientations from one map, and they are *free* data: a reflected
        table is a layout the network has never seen drawn from exactly the
        distribution the real ones came from. That matters here because the
        measured defect is generalisation, not capability -- the same trained
        checkpoint scored -1.4 vp on tables it had trained on and -23.8 on
        tables it had not, and 36 layouts is a small distribution.

        Both axes are offered because the real tables are laid out symmetrically
        about both mid-lines. Reflecting in x swaps which side of the table each
        deployment zone faces, which is legitimate on a scenario whose two zones
        mirror each other; reflecting in y leaves the zones alone entirely.

        Vertex order is reversed when exactly one axis flips, because a single
        reflection reverses a polygon's winding. Nothing here reads winding
        today -- the crossing-number test is indifferent to it -- so this is
        hygiene rather than a fix, and it keeps a mirrored outline comparable to
        an authored one.
        """
        if not flip_x and not flip_y:
            return self

        def mirror_points(points: np.ndarray) -> np.ndarray:
            out = np.array(points, dtype=float, copy=True)
            if flip_x:
                out[:, 0] = board_width - out[:, 0]
            if flip_y:
                out[:, 1] = board_height - out[:, 1]
            return out[::-1] if flip_x != flip_y else out

        footprints = [
            Footprint(Polygon(mirror_points(piece.polygon.vertices)))
            for piece in self.terrain.footprints
        ]
        objectives = None
        if self.objectives is not None:
            objectives = tuple(
                self._mirror_objective(
                    objective, board_width, board_height, flip_x, flip_y
                )
                for objective in self.objectives
            )
        suffix = ("x" if flip_x else "") + ("y" if flip_y else "")
        return MapLayout(
            name=f"{self.name}:{suffix}",
            terrain=Terrain(footprints=footprints),
            objectives=objectives,
        )

    @staticmethod
    def _mirror_objective(
        objective: ObjectiveConfig,
        board_width: float,
        board_height: float,
        flip_x: bool,
        flip_y: bool,
    ) -> ObjectiveConfig:
        """One objective reflected — an outline if it has one, else its centre."""
        if objective.area is not None:
            points = np.array(objective.area, dtype=float)
            if flip_x:
                points[:, 0] = board_width - points[:, 0]
            if flip_y:
                points[:, 1] = board_height - points[:, 1]
            if flip_x != flip_y:
                points = points[::-1]
            mirrored_area = [(float(x), float(y)) for x, y in points]
            return cast(
                ObjectiveConfig, objective.model_copy(update={"area": mirrored_area})
            )
        update: dict[str, float] = {}
        if flip_x and objective.x is not None:
            update["x"] = board_width - objective.x
        if flip_y and objective.y is not None:
            update["y"] = board_height - objective.y
        if not update:
            return objective
        return cast(ObjectiveConfig, objective.model_copy(update=update))
