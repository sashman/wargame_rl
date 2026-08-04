"""Print a compact training-run summary from Wandb history.

Usage: just run-summary <run_id> [bucket_size]

Single-epoch `success_rate` is an n_episodes-sample binomial, so this reports
rolling means over epoch buckets rather than point readings. See docs/metrics.md.
"""

from __future__ import annotations

import statistics
import sys

from wandb.apis.public import Api

ENTITY = "wargame_rl"
PROJECT = "wargame_rl"

TRACKED = [
    "success_rate",
    "reward/mean_episode_reward",
    "reward/max_episode_reward",
    "reward/components/objective_coverage",
    "reward/components/closest_objective/progress",
    "reward/components/terminal_success_bonus",
    "loss/entropy_loss_epoch",
    "loss/value_loss_epoch",
    "mean_episode_steps",
]


def main() -> None:
    run_id = sys.argv[1]
    bucket_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    api = Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    rows = list(run.scan_history())

    by_epoch: dict[int, dict[str, float]] = {}
    for row in rows:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        merged = by_epoch.setdefault(int(epoch), {})
        for key, value in row.items():
            if isinstance(value, (int, float)):
                merged[key] = value

    if not by_epoch:
        print("no epoch-indexed rows yet")
        return

    epochs = sorted(by_epoch)
    last = epochs[-1]
    runtime = max(r.get("_runtime", 0.0) for r in rows if "_runtime" in r)
    print(
        f"run={run_id} state={run.state} last_epoch={last} runtime={runtime / 60:.1f}min"
    )

    phases = {
        e: by_epoch[e]["reward_phase"] for e in epochs if "reward_phase" in by_epoch[e]
    }
    if phases:
        current = phases[max(phases)]
        print(f"reward_phase={int(current)}")
        previous = None
        for epoch in sorted(phases):
            if previous is not None and phases[epoch] != previous:
                print(
                    f"  ** PHASE ADVANCED to {int(phases[epoch])} at epoch {epoch} **"
                )
            previous = phases[epoch]

    header = f"{'epochs':>12} {'n':>3} " + " ".join(
        f"{k.split('/')[-1][:11]:>11}" for k in TRACKED
    )
    print(header)
    for start in range(0, last + 1, bucket_size):
        window = [by_epoch[e] for e in epochs if start <= e < start + bucket_size]
        if not window:
            continue
        cells = []
        for key in TRACKED:
            values = [w[key] for w in window if key in w]
            cells.append(f"{statistics.mean(values):11.4f}" if values else f"{'-':>11}")
        label = f"{start}-{start + bucket_size - 1}"
        print(f"{label:>12} {len(window):>3} " + " ".join(cells))


if __name__ == "__main__":
    main()
