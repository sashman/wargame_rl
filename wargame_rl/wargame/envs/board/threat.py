"""Where it is dangerous to stand -- NEXT turn, which is the only useful answer.

⚠ **The opponent moves before it shoots.** A cell behind a ruin from where an
enemy stands today is shot from beside that ruin one move later, so a map of
what bears *right now* is not a cheaper version of the planning answer -- it is
a systematically optimistic and different one, wrong in the **false-safe**
direction, which is the direction that gets models killed. `renders/v2/control.py`
already draws the current-turn region and that overlay is correct for what it
is: it answers "what can I be shot by this instant", which is a shooting-phase
question. This module answers "what happens to me if I end my turn here", which
is the movement-phase one, and they are not interchangeable.

So the default horizon is `next_turn` and sight is traced **from the ground they
can reach**, not from the ground they occupy:

    threatened(c) <=> exists model m, exists position p:
                          |p - m| <= move(m)
                        & |p - c| <= range(m)
                        & LOS(p, c)

That two-hop is affordable for one reason only: **sight depends on terrain and
nothing else here** (models do not block line of sight), so the cell-to-cell
visibility matrix is a property of the *table* and is built once per layout
rather than once per turn. See `VisibilityCache`.

WHAT THIS OVERSTATES, in all three cases deliberately and in the docstrings of
the functions concerned:

* **Coherency binds the opponent's move.** A free `move`-radius disc per model
  is not a legal set of destinations -- a 2" chain and a 9" span mean the unit
  travels roughly together. Projecting the unit centroid instead is the honest
  form and is not built.
* **Freezing.** Only ~92% of ordered inches are delivered, so `move` is an upper
  bound in practice as well as in principle.
* **An advance never extends threat.** Declaring one spends the unit's shooting,
  so the origin set is dilated by `M` and never by an advance rung even where
  `n_advance_speed_bins > 0`. "Maximum opponent move distance" naively suggests
  the top rung, and that would be wrong by up to 6" in the unsafe direction.
* **The reachable set is SAMPLED at cell centres**, and this one runs the other
  way -- it *understates*. A model may stop anywhere within its Move, but only
  cells whose centre falls inside that radius become origins, so a firing
  position half a cell beyond the last centre is not considered. The error is
  bounded by the spacing and shrinks with it; at the shipped 1" it is small
  against the overstatements above, and at a coarser spacing it is not. This is
  the reason not to draw the field coarser than the overlay already does.

WHAT IT UNDERSTATES: **cover is not applied**, so every entry is the expectation
against a target in the open -- the same choice, for a different reason, that
`expected_damage_matrix` documents. Here it is not a memoisation argument: the
three-state `visibility_matrix` is not on `BattleView` at all, and a grid cell
has no base radius, while the cover predicate is *defined* by offsetting rays by
the endpoints' radii. Cover at a cell is undefined rather than merely expensive.
⚠ The bias runs against objectives specifically, because every marker on the
real tables sits inside a terrain piece -- so this field paints the safest ground
in the game as dangerous. Read it beside `just measure-hold-hazard`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from wargame_rl.wargame.envs.board.grid import (
    DEFAULT_SPACING,
    BoardGrid,
    board_grid_for,
)
from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import WargameModel, alive_mask_for
from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
from wargame_rl.wargame.envs.domain.shooting import DefenderStats, expected_damage
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig

# Origins per chunk when building the visibility cache. Keeps the working
# distance block near a megabyte whatever the board size, which is well inside
# `domain/los.py`'s own 4M-element chunker.
_CACHE_CHUNK = 256


class ThreatHorizon(str, Enum):
    """When the danger being measured arrives."""

    next_turn = "next_turn"
    """They move, then they shoot. **This is the planning answer.**"""

    current = "current"
    """What bears from where they stand this instant.

    ⚠ **Not a planning answer**, and choosing where to stand by reading it is
    the error this module exists to remove -- it cannot see the shot taken from
    the other side of the ruin. It is kept for two jobs: it reproduces the
    shipped `compute_threat_region` exactly, which pins this module against
    already-verified behaviour, and its disagreement with `next_turn` is the
    measurement that says how large the error is.
    """


@dataclass(frozen=True, slots=True)
class ThreatField:
    """Expected loss per cell, for one side's fire, over one grid."""

    grid: BoardGrid
    casualties: np.ndarray
    """`(Q,)` expected models removed per opposing shooting phase."""

    wounds: np.ndarray
    """`(Q,)` expected wounds, before the clip to what a model has to lose."""

    shooter_count: np.ndarray
    """`(Q,)` how many enemy models bear on the cell. What a script thresholds."""

    horizon: ThreatHorizon
    reference: "ReferenceModel"
    reach: np.ndarray
    """`(n_models,)` the move allowance each shooter's origin set was built from."""

    def at(self, points: np.ndarray) -> np.ndarray:
        """`(N,)` expected casualties at each `(N, 2)` board position."""
        if points.size == 0:
            return np.zeros(0, dtype=np.float32)
        sampled: np.ndarray = self.casualties[self.grid.nearest(points)]
        return sampled

    def bands(self, quantiles: Sequence[float]) -> tuple[np.ndarray, ...]:
        """Disjoint `(Q,)` masks splitting the field at the given quantiles.

        **Disjoint, not nested.** Overlapping "above q" level sets double their
        alpha wherever they stack, which is why the renderer needs bands rather
        than thresholds. Quantiles are taken over the *threatened* cells only --
        on most boards more than half the cells are threatened by nobody, and
        including them would put every cut point at zero.
        """
        threatened = self.casualties[self.casualties > 0.0]
        if threatened.size == 0:
            return tuple(
                np.zeros(self.grid.n_cells, dtype=bool)
                for _ in range(len(quantiles) + 1)
            )
        cuts = [0.0, *np.quantile(threatened, np.asarray(quantiles)), np.inf]
        return tuple(
            (self.casualties > low) & (self.casualties <= high)
            for low, high in zip(cuts[:-1], cuts[1:], strict=True)
        )


@dataclass(frozen=True)
class VisibilityCache:
    """Cell-to-cell sight for ONE terrain layout at ONE spacing.

    Sight here depends on terrain and nothing else -- models never block it --
    so this matrix is a property of the **table**, not of the turn. It is what
    makes the two-hop affordable: build it once when a layout is installed and
    every subsequent turn of every episode on that table is a mask and an `any`.

    Gated at `max_range`: a pair further apart than the longest weapon on the
    board is never traced and reads False, which is correct for every query this
    is used for and is the difference between seconds and minutes.

    ⚠ Not symmetrised. Terrain sight *should* be symmetric and there is a known
    open defect where it is not on some endpoint pairs; tracing both directions
    costs nothing extra here (the gate is symmetric, so both orderings are in
    the candidate set anyway) and assuming symmetry would paper over the bug.
    """

    grid: BoardGrid
    visible: np.ndarray
    distances: np.ndarray
    """`(Q, Q)` float32 cell-to-cell distance. A pure function of the grid, kept
    here because every query needs it beside `visible` and recomputing it per
    shooter dominated the query cost."""

    max_range: float

    @classmethod
    def build(
        cls, view: BattleView, *, spacing: float, max_range: float
    ) -> "VisibilityCache":
        """Trace every cell pair within `max_range` on this layout.

        `max_range` is the **longest weapon range** on the board, not range plus
        move. The move is spent getting to the origin cell; the shot is taken
        from there. Gating at `range + move` traces 2.25x the pairs for nothing.
        """
        grid = board_grid_for(view, spacing)
        centres = grid.centres
        n_cells = grid.n_cells
        visible = np.zeros((n_cells, n_cells), dtype=bool)
        distances = np.zeros((n_cells, n_cells), dtype=np.float32)
        for start in range(0, n_cells, _CACHE_CHUNK):
            stop = start + _CACHE_CHUNK
            block = centres[start:stop]
            block_distances = np.linalg.norm(
                block[:, np.newaxis, :] - centres[np.newaxis, :, :], axis=2
            )
            distances[start:stop] = block_distances
            candidates = block_distances <= max_range
            if not candidates.any():
                continue
            visible[start:stop] = np.asarray(
                view.line_of_sight_matrix(block, centres, candidates)
            )
        return cls(
            grid=grid,
            visible=visible,
            distances=distances,
            max_range=float(max_range),
        )


def attacker_stat_rows(
    model_configs: list[ModelConfig] | None, n_models: int
) -> np.ndarray:
    """`(n_models, 5)` of `(attacks, skill, strength, ap, damage)`.

    ⚠ **The FIRST weapon**, matching `observation_builder`'s `cfg.weapons[0]`
    rather than `max_weapon_ranges`'s longest-range choice. The two disagree the
    day a config carries a second gun, and nothing does today. An unarmed model
    gets a zero row, which the damage arithmetic already reads as "no shooter".
    """
    rows = np.zeros((n_models, 5), dtype=np.int64)
    for index, config in enumerate(model_configs or ()):
        if index >= n_models or not config.weapons:
            continue
        weapon = config.weapons[0]
        rows[index] = (
            weapon.attacks,
            weapon.ballistic_skill,
            weapon.strength,
            weapon.ap,
            weapon.damage,
        )
    return rows


def move_reach(
    config: WargameEnvConfig, model_configs: list[ModelConfig] | None, n_models: int
) -> np.ndarray:
    """`(n_models,)` normal-move allowance in board units.

    The same resolution `ActionHandler` performs: the scenario's
    `max_move_speed` unless the model overrides it, both converted from inches
    through the scenario's scale. Recomputed rather than read off the handler
    because `env_components/` is above this layer -- `tests/test_threat_field.py`
    pins the two against each other so they cannot drift.

    ⚠ **Never an advance rung.** An advancing unit forfeits its shooting, so a
    longer move buys no threat.
    """
    quantities = resolve_rules_quantities(config)
    reach = np.full(n_models, quantities.max_move_speed, dtype=float)
    for index, model_config in enumerate(model_configs or ()):
        if index < n_models and model_config.move is not None:
            reach[index] = quantities.scale.to_units(model_config.move)
    return reach


@dataclass(frozen=True, slots=True)
class ReferenceModel:
    """The model whose loss the field is priced in.

    `DefenderStats` is what the attack sequence needs and carries only
    `(toughness, save)`. Turning wounds into *casualties* additionally needs the
    model's Wounds, so the two travel together rather than the caller being
    trusted to keep them in step.
    """

    defender: DefenderStats
    max_wounds: int

    @property
    def label(self) -> str:
        """`T3 Sv4+ W1`, for the report header."""
        return f"T{self.defender.toughness} Sv{self.defender.save}+ W{self.max_wounds}"


def reference_model(
    models: Sequence[WargameModel], model_configs: list[ModelConfig] | None
) -> ReferenceModel:
    """The `(toughness, save, wounds)` most of a force is made of.

    The field prices "what happens to **a** model of mine standing here", which
    needs one defender profile rather than a per-model answer. The mode over
    living models is the honest single choice: on a one-profile army it is that
    profile exactly, and on a mixed one it names the majority rather than
    inventing an average stat line no model has.

    Falls back to the config when nothing is alive, so a field can still be
    drawn for a wiped-out side rather than raising mid-render.
    """
    alive = [m for m in models if m.is_alive]
    if alive:
        rows = [
            (
                int(m.stats["toughness"]),
                int(m.stats["save"]),
                int(m.stats["max_wounds"]),
            )
            for m in alive
        ]
    elif model_configs:
        rows = [
            (int(c.toughness), int(c.save), int(c.max_wounds)) for c in model_configs
        ]
    else:
        return ReferenceModel(DefenderStats(toughness=3, save=4), max_wounds=1)
    values, counts = np.unique(
        np.array(rows, dtype=np.int64), axis=0, return_counts=True
    )
    toughness, save, max_wounds = values[int(np.argmax(counts))]
    return ReferenceModel(
        defender=DefenderStats(toughness=int(toughness), save=int(save)),
        max_wounds=int(max_wounds),
    )


@dataclass(frozen=True, slots=True)
class _WeaponRow:
    """A stat row viewed through the `WeaponStats` protocol."""

    attacks: int
    ballistic_skill: int
    strength: int
    ap: int
    damage: int


def _per_shooter_expectation(
    attacker_stats: np.ndarray, reference: ReferenceModel
) -> tuple[np.ndarray, np.ndarray]:
    """`(wounds, casualties)` each shooter expects against `reference`, per round.

    Casualties rather than wounds is the headline because the standing failure
    this field is pointed at is stated in `alive` -- "models lost per turn if I
    park one model here" is directly comparable to that number and a wound count
    is not. `expected_damage` does not clip damage to what a model has left and
    says so; damage does not spill between models, so a Damage-3 hit on a
    Wounds-1 model removes one model rather than three. The scale is exact
    because expected damage is linear in Damage.

    ⚠ Identically 1.0 on every shipped config (`damage: 1`, `max_wounds: 1`),
    which is exactly why it is written and tested now rather than when it first
    matters.

    Evaluated once per **distinct** stat row, the same memoisation
    `expected_damage_matrix` makes: a one-profile army costs one call.
    """
    n_models = attacker_stats.shape[0]
    if n_models == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.copy()
    unique_rows, inverse = np.unique(attacker_stats, axis=0, return_inverse=True)
    wounds = np.zeros(len(unique_rows), dtype=np.float32)
    casualties = np.zeros(len(unique_rows), dtype=np.float32)
    for index, row in enumerate(unique_rows):
        attacks, skill, strength, ap, damage = (int(value) for value in row)
        if attacks == 0 or damage == 0:
            continue
        expected = expected_damage(
            _WeaponRow(attacks, skill, strength, ap, damage), reference.defender
        )
        wounds[index] = expected
        casualties[index] = expected * (min(damage, reference.max_wounds) / damage)
    return wounds[np.ravel(inverse)], casualties[np.ravel(inverse)]


def _origin_visibility(
    view: BattleView,
    origins: np.ndarray,
    grid: BoardGrid,
    ranges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`(visible, within)` for a set of exact origins against every cell.

    One gated `line_of_sight_matrix` call, exactly as `compute_threat_region`
    makes it -- the range gate is not an optimisation here but the difference
    between a tenth of a second and three seconds, because an ungated sweep
    traces every pair on the board.
    """
    targets = grid.centres
    distances = np.linalg.norm(
        origins[:, np.newaxis, :] - targets[np.newaxis, :, :], axis=2
    )
    # `ranges > 0` is load-bearing: an unarmed model has range 0.0, and `0 <= 0`
    # would mark the cell it stands on as threatened by a model that cannot
    # shoot. The same guard `compute_threat_counts` documents.
    within = (distances <= ranges[:, np.newaxis]) & (ranges > 0)[:, np.newaxis]
    if not within.any():
        return np.zeros_like(within), within
    visible: np.ndarray = np.asarray(
        view.line_of_sight_matrix(origins, targets, within)
    )
    return visible, within


def threat_field(
    view: BattleView,
    models: Sequence[WargameModel],
    max_ranges: np.ndarray,
    attacker_stats: np.ndarray,
    reference: ReferenceModel,
    *,
    horizon: ThreatHorizon = ThreatHorizon.next_turn,
    move: np.ndarray | None = None,
    spacing: float = DEFAULT_SPACING,
    visibility: VisibilityCache | None = None,
) -> ThreatField:
    """What one side's fire costs a `reference` model standing on each cell.

    `models`, `max_ranges` and `attacker_stats` are all the **shooting** side's,
    positionally aligned; `reference` is the profile of the model being priced,
    i.e. the other side's. Passing `models` and `max_ranges` positionally rather
    than a side enum is deliberate: it is the shape `compute_threat_region`
    already takes, which is what lets the two be tested against each other.

    At `ThreatHorizon.next_turn` -- the default -- `move` is the per-model normal
    move from `move_reach`, and sight is traced from every cell a shooter can
    reach as well as from where it stands. The reachable half needs `visibility`;
    build it once per layout with `VisibilityCache.build` and pass the same one
    every turn. Without it the reachable half is skipped and a warning-free
    `current` field is returned, which would be a silent downgrade, so it raises
    instead.

    At `ThreatHorizon.current` only the shooters' exact positions are origins,
    which reproduces `compute_threat_region` cell for cell.

    **`next_turn` is a superset of `current` by construction**, because the exact
    positions are always in the origin set: moving first can add threatened
    ground and can never remove it.
    """
    grid = board_grid_for(view, spacing)
    alive = alive_mask_for(list(models))
    empty = ThreatField(
        grid=grid,
        casualties=np.zeros(grid.n_cells, dtype=np.float32),
        wounds=np.zeros(grid.n_cells, dtype=np.float32),
        shooter_count=np.zeros(grid.n_cells, dtype=np.int32),
        horizon=horizon,
        reference=reference,
        reach=np.zeros(len(models), dtype=float),
    )
    if not alive.any():
        return empty

    positions = np.array(
        [[float(m.location[0]), float(m.location[1])] for m in models], dtype=float
    )[alive]
    ranges = np.asarray(max_ranges, dtype=float)[alive]
    wounds_each, casualties_each = _per_shooter_expectation(
        np.asarray(attacker_stats, dtype=np.int64)[alive], reference
    )

    bears = _bearing_mask(
        view, grid, positions, ranges, horizon, move, alive, visibility
    )
    if bears is None:
        return empty

    reach = np.zeros(len(models), dtype=float)
    if move is not None:
        reach[alive] = np.asarray(move, dtype=float)[alive]
    return ThreatField(
        grid=grid,
        casualties=(bears * casualties_each[:, np.newaxis])
        .sum(axis=0)
        .astype(np.float32),
        wounds=(bears * wounds_each[:, np.newaxis]).sum(axis=0).astype(np.float32),
        shooter_count=bears.sum(axis=0).astype(np.int32),
        horizon=horizon,
        reference=reference,
        reach=reach,
    )


def _bearing_mask(
    view: BattleView,
    grid: BoardGrid,
    positions: np.ndarray,
    ranges: np.ndarray,
    horizon: ThreatHorizon,
    move: np.ndarray | None,
    alive: np.ndarray,
    visibility: VisibilityCache | None,
) -> np.ndarray | None:
    """`(n_alive_shooters, Q)` -- can this shooter put a shot on this cell.

    Returns None when nothing bears at all, which lets the caller hand back a
    zero field without materialising the per-shooter block.
    """
    visible, within = _origin_visibility(view, positions, grid, ranges)
    bears = visible & within
    if horizon is ThreatHorizon.current:
        return bears if bears.any() else None

    if move is None:
        raise ValueError("next_turn needs `move`; pass move_reach(...)")
    if visibility is None:
        raise ValueError(
            "next_turn needs a VisibilityCache; build it once per layout with "
            "VisibilityCache.build(view, spacing=..., max_range=...). Falling "
            "back to current-turn sight would answer a different question "
            "silently, which is the error this module exists to remove."
        )
    if visibility.grid.n_cells != grid.n_cells:
        raise ValueError(
            f"cache is {visibility.grid.n_cells} cells, grid is {grid.n_cells}; "
            "both must be built at the same spacing"
        )
    if visibility.max_range + 1e-9 < float(ranges.max()):
        raise ValueError(
            f"cache gated at {visibility.max_range} but a weapon reaches "
            f"{ranges.max()}; rebuild it at the longer range"
        )

    moves = np.asarray(move, dtype=float)[alive]
    centres = grid.centres
    for index in range(len(positions)):
        if ranges[index] <= 0.0:
            continue
        reachable = np.flatnonzero(
            np.linalg.norm(centres - positions[index], axis=1) <= moves[index]
        )
        if reachable.size == 0:
            continue
        rows = visibility.visible[reachable]
        in_range = visibility.distances[reachable] <= ranges[index]
        bears[index] |= (rows & in_range).any(axis=0)
    return bears if bears.any() else None
