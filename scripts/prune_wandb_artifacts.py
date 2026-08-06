"""Delete all but the most recent model artifacts in the Wandb project.

Training no longer uploads checkpoints -- nothing ever read them back, and at
~148 MB each they filled the storage quota (see `get_logger`). This is the
sweep for the backlog those uploads left behind, and the escape hatch if
`log_model` is ever turned back on.

Deletion is by artifact only. Run history, metrics and recorded videos are
untouched, and `checkpoints/` on disk still holds every run's weights.

Usage: just prune-artifacts [keep] [dry_run]
"""

from __future__ import annotations

import sys

from wargame_rl.wargame.model.common.artifact_retention import DEFAULT_KEEP
from wargame_rl.wargame.model.common.wandb import prune_wandb_model_artifacts


def main() -> None:
    """Prune the project's model artifacts down to the newest `keep`."""
    keep = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else DEFAULT_KEEP
    dry_run = len(sys.argv) > 2 and sys.argv[2].lower() in {"1", "true", "dry", "yes"}

    deleted = prune_wandb_model_artifacts(keep, dry_run=dry_run)

    verb = "Would delete" if dry_run else "Deleted"
    print(f"{verb} {len(deleted)} model artifact(s), keeping the newest {keep}.")


if __name__ == "__main__":
    main()
