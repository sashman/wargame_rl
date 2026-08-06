"""Model-artifact retention: keep the newest N runs, delete the rest.

Wandb storage is finite and a 1000-epoch run uploads tens of megabytes of
checkpoints, so the store has to be bounded. The failure mode that matters is
deleting the wrong end of the list, which is unrecoverable in the cloud, so the
ordering is what these tests pin down.
"""

from __future__ import annotations

import pytest

from wargame_rl.wargame.model.common.artifact_retention import (
    prune_model_artifacts,
    select_collections_to_delete,
)


class FakeVersion:
    def __init__(self, created_at: str) -> None:
        self.created_at = created_at


class FakeCollection:
    """Stands in for `wandb.apis.public.ArtifactCollection`.

    Wandb is a paid external service, so the API surface this module touches --
    a name, version timestamps and a delete -- is faked rather than called.
    """

    def __init__(self, name: str, timestamps: list[str]) -> None:
        self.name = name
        self._versions = [FakeVersion(stamp) for stamp in timestamps]
        self.deleted = False

    def artifacts(self) -> list[FakeVersion]:
        return self._versions

    def delete(self) -> None:
        self.deleted = True


def test_deletes_the_oldest_and_keeps_the_newest() -> None:
    collections = [
        ("model-old", "2026-08-01T00:00:00"),
        ("model-newest", "2026-08-05T00:00:00"),
        ("model-middle", "2026-08-03T00:00:00"),
    ]

    doomed = select_collections_to_delete(collections, keep=2)

    assert doomed == ["model-old"]


def test_nothing_is_deleted_below_the_retention_window() -> None:
    collections = [("model-a", "2026-08-01T00:00:00")]

    assert select_collections_to_delete(collections, keep=5) == []


@pytest.mark.parametrize("keep", [0, -1])
def test_keep_zero_deletes_everything_and_negative_is_rejected(keep: int) -> None:
    collections = [("model-a", "2026-08-01T00:00:00")]

    if keep < 0:
        with pytest.raises(ValueError):
            select_collections_to_delete(collections, keep=keep)
    else:
        assert select_collections_to_delete(collections, keep=keep) == ["model-a"]


def test_a_collection_is_ranked_by_its_newest_version() -> None:
    """A run that saved a late checkpoint is newer than its first version says.

    Sorting on the oldest version would prune a run that is still being written.
    """
    stale = FakeCollection("model-stale", ["2026-08-01T00:00:00"])
    still_active = FakeCollection(
        "model-active", ["2026-08-01T00:00:00", "2026-08-09T00:00:00"]
    )

    deleted = prune_model_artifacts([stale, still_active], keep=1)

    assert deleted == ["model-stale"]
    assert stale.deleted
    assert not still_active.deleted


def test_dry_run_reports_without_deleting() -> None:
    old = FakeCollection("model-old", ["2026-08-01T00:00:00"])
    new = FakeCollection("model-new", ["2026-08-05T00:00:00"])

    deleted = prune_model_artifacts([old, new], keep=1, dry_run=True)

    assert deleted == ["model-old"]
    assert not old.deleted
    assert not new.deleted


def test_a_collection_with_no_versions_is_pruned_first() -> None:
    empty = FakeCollection("model-empty", [])
    populated = FakeCollection("model-populated", ["2026-08-01T00:00:00"])

    deleted = prune_model_artifacts([empty, populated], keep=1)

    assert deleted == ["model-empty"]
