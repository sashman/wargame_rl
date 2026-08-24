"""The frozen opponents a learner can be sampled against.

A snapshot pool is what stops self-play collapsing into a two-cycle: without
one, the learner trains only against its current self, and a policy that
exploits its immediate predecessor can lose to something it beat comfortably
fifty epochs ago. The pool is the memory that makes "did this get better" mean
something over a run rather than over a step.

Three decisions here are load-bearing, and each is a way pools go wrong:

- **The anchor is never evicted.** Entry zero is a fixed reference -- the
  initial policy, or a named scripted baseline. A pool of nothing but recent
  selves can drift as a whole, every member beating the one before it while the
  lot of them get worse against anything outside. The anchor is also what the
  rating scale is pinned to, and `rate()` refuses a table whose anchor did not
  play.
- **Eviction thins uniformly, it does not drop the oldest.** Keeping the most
  recent `k` snapshots is keeping the part of the run the learner is least
  likely to have forgotten. Thinning keeps the pool spanning the whole history.
- **Disk is a real constraint.** Checkpoints are deliberately not uploaded to
  Wandb because each run pushed ~591 MB and filled the quota; a pool multiplies
  the local footprint by its capacity. Capacity defaults low for that reason
  and not for any statistical one.

Numpy and dataclasses only -- no torch, no env, no project imports beyond the
rating package -- so the policy is testable without writing a checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from wargame_rl.wargame.rating.elo import win_probability
from wargame_rl.wargame.rating.pfsp import Mode, pfsp_weights, sample_opponent

# Low on purpose: a snapshot is a full checkpoint on local disk, and
# `checkpoints/` is the only copy of any weights this project has.
DEFAULT_CAPACITY = 8


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One frozen opponent, and enough provenance to replay the match.

    `rating` is `None` until the entry has been rated. A pool member with no
    rating is not an error -- it has simply not played a rated game yet, which
    is the normal state of a snapshot taken this epoch.
    """

    name: str
    checkpoint: str
    epoch: int
    rating: float | None = None
    is_anchor: bool = False


class SnapshotPool:
    """An ordered, capped collection of frozen opponents.

    Ordered by the epoch each snapshot was taken at, oldest first, with the
    anchor at index zero.
    """

    def __init__(self, anchor: Snapshot, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._capacity = capacity
        self._entries: list[Snapshot] = [replace(anchor, is_anchor=True)]

    @property
    def capacity(self) -> int:
        """How many snapshots the pool holds, the anchor included."""
        return self._capacity

    @property
    def entries(self) -> tuple[Snapshot, ...]:
        """Every member, oldest first, anchor at index zero."""
        return tuple(self._entries)

    @property
    def anchor(self) -> Snapshot:
        """The fixed reference, which eviction never removes."""
        return self._entries[0]

    def add(self, snapshot: Snapshot) -> None:
        """Append a snapshot, thinning to capacity if it no longer fits."""
        if snapshot.epoch < self._entries[-1].epoch:
            raise ValueError(
                f"snapshots arrive in epoch order; got epoch {snapshot.epoch} "
                f"after {self._entries[-1].epoch}"
            )
        self._entries.append(replace(snapshot, is_anchor=False))
        if len(self._entries) > self._capacity:
            self._entries = self._thinned()

    def rate(self, ratings: dict[str, float]) -> None:
        """Attach fitted ratings by entrant name, leaving unrated members alone.

        Names rather than indices, because the ratings come back from a fit over
        whatever entrants a ledger happens to hold -- which is a superset of the
        pool when scripted baselines are in the table too, and a subset of it
        when a snapshot has not played yet.
        """
        self._entries = [
            replace(entry, rating=ratings.get(entry.name, entry.rating))
            for entry in self._entries
        ]

    def sample(
        self,
        learner_rating: float,
        rng: np.random.Generator,
        mode: Mode = "hard",
        uniform_floor: float = 0.1,
        advantage: float = 0.0,
    ) -> Snapshot:
        """Draw one opponent, weighted by how hard it is for the learner.

        `advantage` is the structural terms that apply to the seating actually
        played -- `h_seat` when the learner is on the player seat -- so the `p`
        this weights on is the probability of the game about to be played rather
        than the seat-neutral comparison. They are different numbers, and using
        the wrong one on a scenario that fails seat parity would prioritise on a
        bias rather than on skill.

        An unrated snapshot is treated as an **even match** (`p = 0.5`) rather
        than skipped: it is the newest and most relevant opponent in the pool,
        and refusing to play it until it has a rating is a deadlock, since a
        rating only comes from playing it.
        """
        probability = np.array(
            [
                0.5
                if entry.rating is None
                else win_probability(learner_rating, entry.rating, advantage)
                for entry in self._entries
            ],
            dtype=np.float64,
        )
        weights = pfsp_weights(probability, mode=mode, uniform_floor=uniform_floor)
        return self._entries[sample_opponent(weights, rng)]

    def _thinned(self) -> list[Snapshot]:
        """Drop one non-anchor member, keeping the pool spread over the run.

        Removes the entry whose neighbours are **closest together in epochs** --
        the most redundant one, in the sense that the history either side of it
        is already covered. Dropping the oldest instead would turn the pool into
        a sliding window over the recent past, which is the opposite of what it
        is for.
        """
        candidates = range(1, len(self._entries) - 1)
        if not candidates:
            # Only the anchor and one other: the anchor stays, so the other goes.
            return self._entries[:-1]
        gaps = {
            index: self._entries[index + 1].epoch - self._entries[index - 1].epoch
            for index in candidates
        }
        redundant = min(gaps, key=lambda index: (gaps[index], index))
        return [
            entry for index, entry in enumerate(self._entries) if index != redundant
        ]


def spans(entries: Sequence[Snapshot]) -> tuple[int, int]:
    """The epoch range a pool covers, for logging that it has not collapsed."""
    epochs = [entry.epoch for entry in entries]
    return min(epochs), max(epochs)
