"""Clear out Wandb model artifacts left over from past runs.

`WandbLogger(log_model=True)` used to upload every checkpoint the
`ModelCheckpoint` callback kept, once per run, as versions of a
`model-<run_id>` collection. At ~148 MB a checkpoint and four per run that
filled the project's storage quota. Uploading is now off (`get_logger`) because
nothing in this repo ever read an artifact back, so this is a cleanup tool for
the historical backlog rather than a retention policy on live runs.

Deletion is by *collection*, not by version: a collection is one run's
checkpoints, so keeping the newest N collections keeps those runs whole rather
than leaving some with a best checkpoint and no `last.ckpt`.

Nothing here deletes local files. `checkpoints/` is the recovery path if a
pruned run turns out to be worth re-scoring.
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

DEFAULT_KEEP = 5
MODEL_ARTIFACT_TYPE = "model"


class ArtifactCollection(Protocol):
    """The slice of `wandb.apis.public.ArtifactCollection` this module uses."""

    name: str

    def artifacts(self) -> Any:
        """Return the collection's versions, newest first."""
        ...

    def delete(self) -> None:
        """Delete the collection and every version in it."""
        ...


def select_collections_to_delete(
    collections: list[tuple[str, str]],
    keep: int = DEFAULT_KEEP,
) -> list[str]:
    """Return the names of the collections that fall outside the retention window.

    Args:
        collections: `(name, newest_version_created_at)` pairs. The timestamp is
            Wandb's ISO-8601 string, which sorts correctly as text.
        keep: How many of the most recent collections to retain.

    Returns:
        Names to delete, oldest first.

    Raises:
        ValueError: If `keep` is negative.
    """
    if keep < 0:
        raise ValueError(f"keep must be >= 0, got {keep}")

    newest_first = sorted(collections, key=lambda pair: pair[1], reverse=True)
    return [name for name, _ in reversed(newest_first[keep:])]


def _newest_version_timestamp(collection: ArtifactCollection) -> str:
    """Return the collection's most recent version timestamp, or "" if empty.

    An empty string sorts before every real timestamp, so a collection whose
    versions have all been deleted elsewhere is pruned first.
    """
    return max((version.created_at for version in collection.artifacts()), default="")


def prune_model_artifacts(
    collections: list[ArtifactCollection],
    keep: int = DEFAULT_KEEP,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Delete all but the `keep` most recent model-artifact collections.

    Args:
        collections: Every `model`-type collection in the project.
        keep: How many of the most recent collections to retain.
        dry_run: Report what would be deleted without deleting it.

    Returns:
        The names that were deleted (or would have been, under `dry_run`).
    """
    by_name = {collection.name: collection for collection in collections}
    timestamps = [
        (collection.name, _newest_version_timestamp(collection))
        for collection in collections
    ]
    doomed = select_collections_to_delete(timestamps, keep)

    for name in doomed:
        if dry_run:
            logger.info(f"[dry-run] would delete model artifact {name}")
            continue
        by_name[name].delete()
        logger.info(f"deleted model artifact {name}")

    return doomed
