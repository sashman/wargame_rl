"""Load a set of fixed maps and draw one per episode.

The env layer's half of the `map_pool` mode: this reads the map files, resolves
each into a `MapLayout`, checks the whole pool against the board and the
observation budgets, and hands out one layout at a time. `place_for_episode`
takes the drawn layout as a plain domain value, so nothing under `domain/`
touches the filesystem.

Everything expensive happens once, at env construction — every file is parsed,
every polygon built, every check run. A draw is then an index into a list, which
matters because a training run resets tens of thousands of times.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from numpy.random import Generator
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.domain.map_layout import MapLayout
from wargame_rl.wargame.envs.domain.terrain import Footprint, Terrain
from wargame_rl.wargame.envs.types.config import (
    MapPoolConfig,
    TerrainMapConfig,
    WargameEnvConfig,
)


def _load_layouts(spec: MapPoolConfig) -> list[MapLayout]:
    """Parse every requested map file into a layout, in a stable order.

    Sorted by name rather than by directory order so a pool draws the same
    sequence from the same seed on any machine — `Path.glob` does not promise an
    order, and a run that depends on one is reproducible only by accident.
    """
    directory = Path(spec.directory)
    if not directory.is_dir():
        raise ValueError(f"map_pool.directory '{spec.directory}' is not a directory")

    layouts: dict[str, MapLayout] = {}
    for path in sorted(directory.glob("*.yaml")):
        terrain_map = parse_yaml_raw_as(TerrainMapConfig, path.read_text())
        layouts[terrain_map.name] = MapLayout(
            name=terrain_map.name,
            terrain=Terrain([Footprint(p.to_polygon()) for p in terrain_map.terrain]),
            objectives=(
                None
                if terrain_map.objectives is None
                else tuple(terrain_map.objectives)
            ),
        )

    if spec.names is None:
        return [layouts[name] for name in sorted(layouts)]

    missing = [name for name in spec.names if name not in layouts]
    if missing:
        raise ValueError(
            f"map_pool names not found in '{spec.directory}': {missing}. "
            f"Available: {sorted(layouts)}"
        )
    return [layouts[name] for name in sorted(spec.names)]


def _validate_pool(layouts: Sequence[MapLayout], config: WargameEnvConfig) -> None:
    """Reject a pool the scenario cannot actually run, before an epoch is spent.

    Two of these are observation-shape checks and they are the reason the pool
    mode needs the budgets at all: layouts that differ in objective or piece
    count produce tensors of different widths, which neither collate into one
    batch nor share one network. The third is a board check, which catches a map
    authored for a different table size — the env config's own terrain validator
    never sees these pieces, since they arrive from the pool rather than from
    `terrain`.
    """
    if not layouts:
        raise ValueError("map_pool selected no maps")

    for kind, counts, budget in (
        (
            "objective",
            {layout.n_objectives for layout in layouts if layout.objectives},
            config.objective_budget,
        ),
        (
            "terrain piece",
            {layout.n_pieces for layout in layouts},
            config.terrain_budget,
        ),
    ):
        if not counts:
            continue
        if budget is None:
            if len(counts) > 1:
                raise ValueError(
                    f"map_pool mixes {kind} counts {sorted(counts)}, which produce "
                    f"tensors of different widths. Set the matching budget "
                    f"(at least {max(counts)}) or use maps of one size"
                )
        elif max(counts) > budget:
            raise ValueError(
                f"map_pool has a layout with {max(counts)} {kind}s, over the "
                f"budget of {budget}"
            )

    for layout in layouts:
        for footprint in layout.terrain.footprints:
            x0, y0, x1, y1 = footprint.polygon.bounds
            if x0 < 0 or y0 < 0 or x1 > config.board_width or y1 > config.board_height:
                raise ValueError(
                    f"map '{layout.name}' has a piece at {footprint.polygon.bounds} "
                    f"outside the board "
                    f"({config.board_width}x{config.board_height})"
                )


class MapPool:
    """The layouts a run draws from, resolved and checked once at construction."""

    def __init__(self, layouts: Sequence[MapLayout]) -> None:
        self._layouts = list(layouts)

    @classmethod
    def from_config(cls, config: WargameEnvConfig) -> "MapPool | None":
        """Build the pool this config asks for, or None when it asks for none."""
        if config.map_pool is None:
            return None
        layouts = _load_layouts(config.map_pool)
        _validate_pool(layouts, config)
        return cls(layouts)

    def draw(self, rng: Generator) -> MapLayout:
        """One layout, uniformly at random.

        Uniform rather than a shuffled cycle: a cycle would make the layout a
        function of the episode index, so two runs at different rollout widths
        would see different sequences from the same seed, and an evaluation wave
        stepping in lockstep would see every env on the same map at once.
        """
        return self._layouts[int(rng.integers(len(self._layouts)))]

    @property
    def names(self) -> list[str]:
        return [layout.name for layout in self._layouts]

    def __len__(self) -> int:
        return len(self._layouts)
