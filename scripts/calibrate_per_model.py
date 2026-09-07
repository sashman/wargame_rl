"""Stage-1 calibration sweep for issue #288: 12 cells, 6 concurrent, GPU.

Grid: gamma x {0.9, 0.99}, gae_lambda x {0.9, 0.95}, rollout_rounds x {8, 16, 32}
Config: configs/golden/25v25_maps_two_mode.yaml (the training config)
Budget: 76,800 rounds/cell (75-epoch-equivalent) -- a deliberate deviation from
the drafted 300-epoch-equivalent screen, forced by measured throughput
(~3 rounds/s/cell at 6 concurrent); top cells extend to 307,200 in stage 2.
Readout: eval/vp_margin on tuning band seeds 900000+ (30 episodes, greedy).
Runs from the pinned worktree so the main tree stays editable.
"""

from __future__ import annotations

import itertools
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

PYTHON = "/home/sash/Workspace/wargame_rl/.venv/bin/python"
WORKTREE = "/home/sash/Workspace/wargame_rl_calib"
OUT_ROOT = Path(
    "/home/sash/Workspace/wargame_rl/checkpoints/per_model/calibration_stage1"
)
CONCURRENCY = 6
MAX_ROUNDS = 76_800

OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATUS = OUT_ROOT / "status.txt"
status_lock = threading.Lock()


def note(line: str) -> None:
    with status_lock:
        with STATUS.open("a") as handle:
            handle.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def run_cell(gamma: float, lam: float, rollout: int) -> bool:
    tag = f"g{int(gamma * 100)}_l{int(lam * 100)}_r{rollout}"
    note(f"START {tag}")
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    with (LOG_DIR / f"{tag}.log").open("w") as log:
        result = subprocess.run(
            [
                PYTHON,
                "train_per_model.py",
                "configs/golden/25v25_maps_two_mode.yaml",
                "--max-rounds",
                str(MAX_ROUNDS),
                "--seed",
                "1",
                "--gamma",
                str(gamma),
                "--gae-lambda",
                str(lam),
                "--rollout-rounds",
                str(rollout),
                "--eval-every-rounds",
                "7680",
                "--n-eval-episodes",
                "30",
                "--eval-seed-base",
                "900000",
                "--checkpoint-every-rounds",
                "25600",
                "--device",
                "cuda",
                "--torch-threads",
                "2",
                "--no-wandb",
                "--run-suffix",
                f"_cal_{tag}",
                "--out-root",
                str(OUT_ROOT),
            ],
            cwd=WORKTREE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    ok = result.returncode == 0
    note(f"{'DONE' if ok else 'FAILED'} {tag} rc={result.returncode}")
    return ok


def main() -> None:
    cells = list(itertools.product((0.9, 0.99), (0.9, 0.95), (8, 16, 32)))
    note(
        f"LAUNCH {len(cells)} cells, {CONCURRENCY} concurrent, {MAX_ROUNDS} rounds each"
    )
    work: queue.Queue[tuple[float, float, int]] = queue.Queue()
    for cell in cells:
        work.put(cell)
    failures = []

    def worker() -> None:
        while True:
            try:
                cell = work.get_nowait()
            except queue.Empty:
                return
            if not run_cell(*cell):
                failures.append(cell)

    threads = [threading.Thread(target=worker) for _ in range(CONCURRENCY)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    note(f"ALL DONE failed={len(failures)} {failures if failures else ''}")


if __name__ == "__main__":
    main()
