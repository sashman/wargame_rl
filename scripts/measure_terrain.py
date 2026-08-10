"""Terrain-layout statistics for a `random_terrain` config.

Batch 1/2 ran on 7 near-square ruins covering 9.6% of the board and found the
agent ignored them entirely. Coverage alone does not explain that; the geometry
does. Exposure is "at least one enemy can see me", so hiding means breaking
*every* sightline at once — while stepping one cell out of weapon range breaks
all of them for free. One blob cannot compete with that.

The number that matters is therefore not coverage but **how much of the board is
genuinely out of sight of a squad that could shoot it**. Measured that way, the
batch-1/2 profile leaves 5.8% and 29 pieces of 3-7 leaves 18.6%: count dominates
size, because hiding needs ruins in many directions rather than one big one.

Tune a terrain profile here, in seconds, rather than after a thousand epochs.

Usage: just measure-terrain <env_config> [n_layouts]
"""

from __future__ import annotations

import statistics
import sys

import numpy as np
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
from wargame_rl.wargame.envs.domain.sight import line_of_sight_matrix
from wargame_rl.wargame.envs.domain.terrain import Terrain
from wargame_rl.wargame.envs.domain.terrain_placement import generate_terrain
from wargame_rl.wargame.envs.domain.value_objects import BoardDimensions
from wargame_rl.wargame.envs.types.config import WargameEnvConfig

SAMPLES_PER_LAYOUT = 400
SEED = 20_000


def _weapon_range(config: WargameEnvConfig) -> float:
    """Longest player weapon range, which sets the scale a ruin has to work at."""
    if not config.models:
        return 12.0
    ranges = [w.range for m in config.models if m.weapons for w in m.weapons]
    return float(max(ranges)) if ranges else 12.0


def _sight_clear(
    terrain: Terrain,
    starts: np.ndarray,
    ends: np.ndarray,
    sample_step: float,
) -> np.ndarray:
    """``(N,)`` clear/blocked for N paired sightlines, in one vectorised pass."""
    matrix = line_of_sight_matrix(
        starts,
        ends,
        terrain,
        None,
        sample_step=sample_step,
        candidates=np.eye(len(starts), dtype=bool),
    )
    return np.diagonal(matrix).copy()


def _coverage(terrain: Terrain, board: BoardDimensions) -> float:
    """Fraction of board area inside some footprint (footprints never overlap).

    The *polygon's* area, not its bounding box. An inscribed hexagon fills only
    ~65% of the box it was drawn in, so billing the box overstated real terrain
    by half — and the 37 x 3-6 profile was tuned against that inflated number,
    landing at 0.159 actual coverage while its header claimed 0.233. Areas, not
    cell counts: a footprint is a continuous shape, so its extent is `x1 - x0`
    rather than `x1 - x0 + 1`; the `+ 1` lives in `Footprint.from_cell_rect`.
    """
    area = sum(fp.polygon.area for fp in terrain.footprints)
    return area / (board.width * board.height)


def _blocked_fraction(
    terrain: Terrain,
    board: BoardDimensions,
    weapon_range: float,
    rng: np.random.Generator,
    sample_step: float,
) -> float:
    """Fraction of random within-range sightlines that terrain blocks.

    Endpoints are drawn uniformly over the board and paired at a separation
    within weapon range, which is the population of shots the game actually
    resolves.
    """
    starts = np.column_stack(
        [
            rng.uniform(0.0, board.width, SAMPLES_PER_LAYOUT),
            rng.uniform(0.0, board.height, SAMPLES_PER_LAYOUT),
        ]
    )
    angles = rng.uniform(0.0, 2.0 * np.pi, SAMPLES_PER_LAYOUT)
    distances = rng.uniform(1.0, weapon_range, SAMPLES_PER_LAYOUT)
    ends = np.column_stack(
        [
            np.clip(starts[:, 0] + distances * np.cos(angles), 0.0, board.width),
            np.clip(starts[:, 1] + distances * np.sin(angles), 0.0, board.height),
        ]
    )
    clear = _sight_clear(terrain, starts, ends, sample_step)
    return float((~clear).mean())


ENEMY_SQUAD_SIZE = 8
ENEMY_BAND_FRACTION = 0.35


def _hideable_fraction(
    terrain: Terrain,
    board: BoardDimensions,
    weapon_range: float,
    rng: np.random.Generator,
    sample_step: float,
) -> float:
    """Fraction of cells hidden from every enemy that could shoot them.

    This is the quantity cover actually needs. Exposure is "at least one enemy
    can see me", so a ruin that breaks three sightlines out of twenty-five buys
    nothing — hiding means breaking all of them at once.

    The probe is a squad-sized enemy group in a band, not a line spanning the
    whole board: forces here deploy in five groups and converge on objectives,
    so a uniformly-spread enemy is a worst case that no scenario produces and
    would understate what cover is worth.
    """
    hidden = 0
    samples = 0
    band = max(1, int(board.height * ENEMY_BAND_FRACTION))

    for _ in range(SAMPLES_PER_LAYOUT // 4):
        enemy_x = int(rng.integers(board.width // 2, board.width))
        band_top = int(rng.integers(0, max(1, board.height - band)))
        enemy_ys = rng.integers(band_top, band_top + band, size=ENEMY_SQUAD_SIZE)

        x = float(rng.uniform(0.0, enemy_x))
        y = float(rng.uniform(0.0, board.height))
        in_range = [
            float(ey)
            for ey in enemy_ys
            if np.hypot(enemy_x - x, float(ey) - y) <= weapon_range
        ]
        if not in_range:
            continue
        samples += 1
        starts = np.tile([float(x), float(y)], (len(in_range), 1))
        ends = np.column_stack(
            [np.full(len(in_range), float(enemy_x)), np.array(in_range, dtype=float)]
        )
        if not _sight_clear(terrain, starts, ends, sample_step).any():
            hidden += 1
    return hidden / samples if samples else 0.0


def main() -> None:
    """Print layout statistics for the config's random_terrain profile."""
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    config_path = sys.argv[1]
    n_layouts = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 200

    with open(config_path) as handle:
        config = parse_yaml_raw_as(WargameEnvConfig, handle.read())

    if config.random_terrain is None:
        print(f"{config_path} has no `random_terrain` block — nothing to measure.")
        raise SystemExit(1)

    board = BoardDimensions(width=config.board_width, height=config.board_height)
    weapon_range = _weapon_range(config)
    spec = config.random_terrain
    rng = np.random.default_rng(SEED)
    sample_step = resolve_rules_quantities(config).los_sample_step

    coverages: list[float] = []
    blocked: list[float] = []
    hideable: list[float] = []
    aspect_ratios: list[float] = []

    for _ in range(n_layouts):
        terrain = generate_terrain(spec, board, rng)
        coverages.append(_coverage(terrain, board))
        blocked.append(
            _blocked_fraction(terrain, board, weapon_range, rng, sample_step)
        )
        hideable.append(
            _hideable_fraction(terrain, board, weapon_range, rng, sample_step)
        )
        aspect_ratios.extend(
            max(fp.x1 - fp.x0, fp.y1 - fp.y0) / min(fp.x1 - fp.x0, fp.y1 - fp.y0)
            for fp in terrain.footprints
        )

    print(f"\n{config_path}  ({n_layouts} layouts, {board.width}x{board.height} board)")
    print(
        f"  profile: count={spec.count} size={spec.min_size}-{spec.max_size} "
        f"gap={spec.min_gap} mirror={spec.mirror} | weapon range {weapon_range:g}\n"
    )
    print(f"  board coverage             {statistics.fmean(coverages):.3f}")
    print(f"  sightlines blocked         {statistics.fmean(blocked):.3f}")
    print(f"  cells hidden from a squad  {statistics.fmean(hideable):.3f}")
    print(f"  mean piece aspect ratio    {statistics.fmean(aspect_ratios):.2f}")
    print(
        "\n  'cells hidden from a squad' is the one that matters: exposure is\n"
        "  'at least one enemy sees me', so terrain that breaks a few sightlines\n"
        "  out of twenty-five buys nothing at all. Batch 1/2 measured 0.058.\n"
    )


if __name__ == "__main__":
    main()
