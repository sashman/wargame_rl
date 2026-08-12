"""Unit coherency: the spacing rule that keeps a unit together.

The rule is [docs/rules/03-moving.md](../../../../docs/rules/03-moving.md) §
Coherency. A unit of more than one model is **in coherency** while every model
in it is:

- within ``nearest_distance`` (the rules' 2") of at least one *other* model in
  the unit -- the **chain** condition, and
- within ``furthest_distance`` (the rules' 9") of *every other* model in the
  unit -- the **spread** condition.

and the unit forms a **single connected group**. That third clause is not
implied by the first: five models in a line at 2" spacing satisfy the chain
condition, and so do two separate pairs 30" apart. Connectivity is what
distinguishes them, and it is checked on the graph whose edges are the chain
condition.

Two deliberate choices, both to match the rest of this environment:

- **Distances are base to base**, like engagement range -- ``d`` is the gap
  between the two discs, not between their centres. At ``base_radius: 0`` the
  two coincide, which is why every pre-geometry result still reproduces.
- **Dead models do not belong to their unit.** A destroyed model is off the
  board, so it neither satisfies another model's chain condition nor breaks
  anyone's spread. A unit reduced to one live model is trivially coherent --
  the rule only binds "a unit of more than one model".

This module **decides nothing and changes nothing**. It reports. Whether a
breach costs the unit models, forbids the move, or is merely priced by a reward
term is the caller's policy, and lives with the caller. Keeping the predicate
separate is what lets the same definition serve a metric, a reward and an
enforcement rule without three of them drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class UnitCoherency:
    """The coherency state of one unit at one instant.

    Every per-member array is indexed by position in ``member_indices``, which
    holds indices into the original model list. Members are the unit's *live*
    models only.
    """

    group_id: int
    member_indices: np.ndarray
    chain_ok: np.ndarray
    spread_ok: np.ndarray
    component: np.ndarray
    n_components: int
    largest_component_size: int
    max_pairwise_distance: float

    @property
    def size(self) -> int:
        """Number of live models in the unit."""
        return int(self.member_indices.size)

    @property
    def connected(self) -> bool:
        """True while the unit's chain graph is a single connected group."""
        return self.n_components <= 1

    @property
    def coherent(self) -> bool:
        """True while this unit satisfies the whole rule.

        A unit of fewer than two live models is always coherent: the rule binds
        only "a unit of more than one model".
        """
        if self.size <= 1:
            return True
        return (
            bool(self.chain_ok.all()) and bool(self.spread_ok.all()) and self.connected
        )

    @property
    def member_coherency(self) -> np.ndarray:
        """Per-member flag: is this model part of the coherent body of its unit?

        A model counts as in coherency while it satisfies both conditions *and*
        sits in the unit's largest chain component. The component clause is what
        assigns blame when a unit splits: the detached models are out, the main
        body is not. A tie between two equal-sized components leaves the lower
        component index as the body, so the result is deterministic.
        """
        if self.size <= 1:
            return np.ones(self.size, dtype=bool)
        in_body = self.component == self._body_component()
        member_coherency: np.ndarray = self.chain_ok & self.spread_ok & in_body
        return member_coherency

    def _body_component(self) -> int:
        """Index of the largest chain component, lowest index winning ties."""
        counts = np.bincount(self.component, minlength=self.n_components)
        return int(np.argmax(counts))


@dataclass(frozen=True, slots=True)
class CoherencyReport:
    """Coherency across every unit of one force at one instant."""

    units: tuple[UnitCoherency, ...]
    in_coherency: np.ndarray

    @property
    def all_coherent(self) -> bool:
        """True while every unit in the force is in coherency."""
        return all(unit.coherent for unit in self.units)

    @property
    def n_units(self) -> int:
        """Number of units with at least one live model."""
        return len(self.units)

    @property
    def n_units_coherent(self) -> int:
        """How many of those units are in coherency."""
        return sum(1 for unit in self.units if unit.coherent)

    @property
    def fraction_units_coherent(self) -> float:
        """Share of live units in coherency; 1.0 when no unit is left alive."""
        if not self.units:
            return 1.0
        return self.n_units_coherent / self.n_units

    @property
    def n_models_out_of_coherency(self) -> int:
        """Live models not part of the coherent body of their unit."""
        return int(sum(int((~unit.member_coherency).sum()) for unit in self.units))


def evaluate_coherency(
    positions: np.ndarray,
    group_ids: np.ndarray,
    alive_mask: np.ndarray,
    base_radii: np.ndarray,
    nearest_distance: float,
    furthest_distance: float,
) -> CoherencyReport:
    """Report coherency for every unit in a force.

    Args:
        positions: ``(n_models, 2)`` model locations, in board units.
        group_ids: ``(n_models,)`` unit membership.
        alive_mask: ``(n_models,)`` which models are still on the board.
        base_radii: ``(n_models,)`` base radius per model, in board units.
        nearest_distance: The chain distance, in board units (the rules' 2").
        furthest_distance: The spread distance, in board units (the rules' 9").

    Returns:
        A :class:`CoherencyReport` whose ``in_coherency`` is indexed like
        ``positions``. Dead models are reported as in coherency, so the flag
        reads as "not in breach" and a casualty never registers as a violation.
    """
    in_coherency = np.ones(positions.shape[0], dtype=bool)
    units: list[UnitCoherency] = []
    live = np.flatnonzero(alive_mask)
    for group_id in np.unique(group_ids[live]):
        members = live[group_ids[live] == group_id]
        unit = _evaluate_unit(
            group_id=int(group_id),
            member_indices=members,
            positions=positions,
            base_radii=base_radii,
            nearest_distance=nearest_distance,
            furthest_distance=furthest_distance,
        )
        units.append(unit)
        in_coherency[members] = unit.member_coherency
    return CoherencyReport(units=tuple(units), in_coherency=in_coherency)


def _evaluate_unit(
    group_id: int,
    member_indices: np.ndarray,
    positions: np.ndarray,
    base_radii: np.ndarray,
    nearest_distance: float,
    furthest_distance: float,
) -> UnitCoherency:
    """Evaluate both conditions and the chain components for a single unit."""
    gaps = base_to_base_distances(positions[member_indices], base_radii[member_indices])
    size = member_indices.size
    if size <= 1:
        return UnitCoherency(
            group_id=group_id,
            member_indices=member_indices,
            chain_ok=np.ones(size, dtype=bool),
            spread_ok=np.ones(size, dtype=bool),
            component=np.zeros(size, dtype=np.intp),
            n_components=size,
            largest_component_size=size,
            max_pairwise_distance=0.0,
        )

    # The diagonal is a model's gap to itself. Excluding it is what makes the
    # chain condition read "at least one *other* model".
    off_diagonal = ~np.eye(size, dtype=bool)
    chain_edges = (gaps <= nearest_distance) & off_diagonal
    chain_ok = np.atleast_1d(chain_edges.any(axis=1))
    spread_ok = np.atleast_1d(((gaps <= furthest_distance) | ~off_diagonal).all(axis=1))
    component, n_components = _connected_components(chain_edges)
    counts = np.bincount(component, minlength=n_components)
    return UnitCoherency(
        group_id=group_id,
        member_indices=member_indices,
        chain_ok=chain_ok,
        spread_ok=spread_ok,
        component=component,
        n_components=n_components,
        largest_component_size=int(counts.max()),
        max_pairwise_distance=float(gaps[off_diagonal].max()),
    )


def base_to_base_distances(positions: np.ndarray, base_radii: np.ndarray) -> np.ndarray:
    """Pairwise gaps between model bases, in board units.

    The gap between two discs is the centre distance less both radii, floored at
    zero so overlapping bases read as touching rather than as negative distance.
    At ``base_radius: 0`` this is exactly the centre-to-centre distance.
    """
    deltas = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    centres = np.linalg.norm(deltas, axis=-1)
    radii_sum = base_radii[:, np.newaxis] + base_radii[np.newaxis, :]
    gaps: np.ndarray = np.maximum(centres - radii_sum, 0.0)
    return gaps


def _connected_components(edges: np.ndarray) -> tuple[np.ndarray, int]:
    """Label the connected components of a small symmetric boolean graph.

    Breadth-first over an adjacency matrix rather than a union-find or a SciPy
    call: units here hold a handful of models, so the matrix form is both faster
    and shorter than the general algorithm, and it keeps the domain free of a
    dependency it would otherwise need for one function.

    Components are numbered in order of their lowest-indexed member, which is
    what makes tie-breaking on component size deterministic.
    """
    size = edges.shape[0]
    component = np.full(size, -1, dtype=np.intp)
    n_components = 0
    for start in range(size):
        if component[start] != -1:
            continue
        reached = np.zeros(size, dtype=bool)
        reached[start] = True
        frontier = reached.copy()
        while frontier.any():
            neighbours = edges[frontier].any(axis=0) & ~reached
            reached |= neighbours
            frontier = neighbours
        component[reached] = n_components
        n_components += 1
    return component, n_components
