"""The snapshot pool: what it keeps, what it drops, and who it draws.

The two properties worth testing are the two ways a pool goes wrong. It can
**drift as a whole** -- every member beating the one before it while the lot get
worse against anything outside -- which the never-evicted anchor prevents. And
it can **collapse into a sliding window** over the recent past, which is where
dropping the oldest member leads and which uniform thinning prevents.

No torch and no checkpoint files: a `Snapshot` is a path and an epoch, so the
policy is testable without writing half a gigabyte to disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.rating.pool import Snapshot, SnapshotPool, spans

ANCHOR = Snapshot(name="squad_march_take", checkpoint="scripted", epoch=0)


def _pool(capacity: int = 4) -> SnapshotPool:
    return SnapshotPool(ANCHOR, capacity=capacity)


def _snapshot(epoch: int) -> Snapshot:
    return Snapshot(
        name=f"epoch_{epoch}",
        checkpoint=f"checkpoints/run/pool/{epoch}.ckpt",
        epoch=epoch,
    )


def test_the_anchor_survives_every_eviction() -> None:
    """A pool of nothing but recent selves has no floor, and the anchor is also
    what the rating scale is pinned to."""
    pool = _pool(capacity=3)

    for epoch in (25, 50, 75, 100, 125, 150):
        pool.add(_snapshot(epoch))

    assert pool.anchor == pool.entries[0]
    assert pool.anchor.name == ANCHOR.name
    assert pool.anchor.is_anchor
    assert len(pool.entries) == 3


def test_thinning_keeps_the_pool_spanning_the_run() -> None:
    """The property that separates a pool from a sliding window.

    Dropping the oldest each time would leave the last `k` snapshots, so the
    span would be `capacity * interval`. Thinning keeps the earliest and latest
    and drops from the middle, so the span stays the whole run.
    """
    pool = _pool(capacity=4)

    for epoch in (25, 50, 75, 100, 125, 150, 175, 200):
        pool.add(_snapshot(epoch))

    earliest, latest = spans(pool.entries)
    assert earliest == 0
    assert latest == 200


def test_capacity_is_honoured() -> None:
    pool = _pool(capacity=2)

    for epoch in (10, 20, 30, 40):
        pool.add(_snapshot(epoch))

    assert len(pool.entries) == 2


def test_snapshots_must_arrive_in_epoch_order() -> None:
    """Out-of-order arrival means two writers, and a pool silently reordered by
    epoch would hide that rather than report it."""
    pool = _pool()
    pool.add(_snapshot(50))

    with pytest.raises(ValueError, match="epoch order"):
        pool.add(_snapshot(25))


def test_a_capacity_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        SnapshotPool(ANCHOR, capacity=0)


def test_ratings_attach_by_name_and_leave_the_rest_alone() -> None:
    """A fit covers whatever entrants a ledger holds -- a superset of the pool
    when scripted baselines are in the table, a subset when a snapshot has not
    played yet."""
    pool = _pool()
    pool.add(_snapshot(25))
    pool.add(_snapshot(50))

    pool.rate({"epoch_25": 120.0, "an_entrant_not_in_this_pool": 999.0})

    by_name = {entry.name: entry.rating for entry in pool.entries}
    assert by_name["epoch_25"] == 120.0
    assert by_name["epoch_50"] is None


def test_sampling_prefers_the_opponents_the_learner_loses_to() -> None:
    pool = _pool()
    pool.add(_snapshot(25))
    pool.add(_snapshot(50))
    pool.rate({ANCHOR.name: -400.0, "epoch_25": 0.0, "epoch_50": 400.0})
    rng = np.random.default_rng(0)

    drawn = [pool.sample(0.0, rng, mode="hard").name for _ in range(500)]

    assert drawn.count("epoch_50") > drawn.count("epoch_25")
    assert drawn.count("epoch_25") > drawn.count(ANCHOR.name)


def test_an_unrated_snapshot_is_playable() -> None:
    """A rating only comes from playing, so refusing to play an unrated
    snapshot until it has one is a deadlock -- and it is the newest and most
    relevant opponent in the pool."""
    pool = _pool()
    pool.add(_snapshot(25))
    rng = np.random.default_rng(1)

    drawn = {pool.sample(0.0, rng).name for _ in range(200)}

    assert "epoch_25" in drawn


def test_the_seating_advantage_changes_who_is_drawn() -> None:
    """`p` for the game about to be played, not the seat-neutral comparison.

    On a scenario that fails seat parity the two are different numbers, and
    prioritising on the wrong one spends the run's games on a bias rather than
    on skill. Level and stronger opponents here; a 300-Elo seat advantage takes
    the learner from `p = 0.50 / 0.15` to `0.85 / 0.50`, which moves the `hard`
    weighting from 0.28/0.72 to 0.12/0.88.
    """
    pool = _pool()
    pool.add(_snapshot(25))
    pool.rate({ANCHOR.name: 0.0, "epoch_25": 300.0})

    def share(advantage: float) -> float:
        rng = np.random.default_rng(3)
        drawn = [
            pool.sample(0.0, rng, mode="hard", advantage=advantage).name
            for _ in range(2000)
        ]
        return drawn.count("epoch_25") / len(drawn)

    assert share(300.0) > share(0.0) + 0.1
