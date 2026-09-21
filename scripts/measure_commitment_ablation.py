"""The flag ablation (#384 R3): does a per-model policy READ its commitment?

Scores checkpoints on a layer-on config four ways on identical seeds, the
reward untouched (it is not computed at play), only the observation changed:

- **trained** -- the marked-target relation as the environment writes it;
- **blank** -- the relation present, no target flagged;
- **misdirect** -- each unit's flag moved to the NEXT objective (index + 1
  mod n), never the one the environment assigned and pays against;
- **nearest** -- each model's flag moved to ITS nearest objective, read off
  the relation's own offset columns. On the legibility rung the assignment
  is deliberately not the nearest, so this is the confound the rung is
  built to separate: a policy that walks to the nearest objective scores
  the same under `nearest` as under `trained`.

If success is unchanged under BLANK the members never read the flag; if
MISDIRECT or NEAREST drags them elsewhere they follow it. The objective
count is taken from the config, not inferred from the token layout.

Usage: python -m scripts.measure_commitment_ablation <config> <n> <seed_base> <ckpt...>
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.per_model import tokens
from wargame_rl.wargame.envs.per_model.commitment import CommitmentState
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.scoring import evaluate_spec

Writer = Callable[[np.ndarray, CommitmentState, np.ndarray, np.ndarray, int, int], None]

_TRAINED: Writer = tokens._write_commitment_relations


def _blank(
    cross: np.ndarray,
    commitment: CommitmentState,
    own_groups: np.ndarray,
    enemy_groups: np.ndarray,
    objective_start: int,
    enemy_unit_start: int,
) -> None:
    cross[:, :, tokens.REL_COMMIT_PRESENT] = 1.0


def _misdirect_writer(n_objectives: int) -> Writer:
    def write(
        cross: np.ndarray,
        commitment: CommitmentState,
        own_groups: np.ndarray,
        enemy_groups: np.ndarray,
        objective_start: int,
        enemy_unit_start: int,
    ) -> None:
        cross[:, :, tokens.REL_COMMIT_PRESENT] = 1.0
        if n_objectives < 2:
            return
        for i, g in enumerate(own_groups):
            ground = commitment.ground_of(int(g))
            if ground >= 0:
                wrong = (ground + 1) % n_objectives
                cross[i, objective_start + wrong, tokens.REL_COMMITTED] = 1.0

    return write


def _nearest_writer(n_objectives: int) -> Writer:
    def write(
        cross: np.ndarray,
        commitment: CommitmentState,
        own_groups: np.ndarray,
        enemy_groups: np.ndarray,
        objective_start: int,
        enemy_unit_start: int,
    ) -> None:
        cross[:, :, tokens.REL_COMMIT_PRESENT] = 1.0
        if n_objectives < 1:
            return
        columns = slice(objective_start, objective_start + n_objectives)
        dx = cross[:, columns, tokens.REL_DX]
        dy = cross[:, columns, tokens.REL_DY]
        nearest = np.argmin(dx * dx + dy * dy, axis=1)
        for i, g in enumerate(own_groups):
            if commitment.ground_of(int(g)) >= 0:
                cross[i, objective_start + int(nearest[i]), tokens.REL_COMMITTED] = 1.0

    return write


def modes(config: WargameEnvConfig) -> list[tuple[str, Writer]]:
    """The four writers, in the order the table prints them."""
    n_objectives = (
        len(config.objectives)
        if config.objectives
        else int(config.number_of_objectives)
    )
    return [
        ("trained", _TRAINED),
        ("blank", _blank),
        ("misdirect", _misdirect_writer(n_objectives)),
        ("nearest", _nearest_writer(n_objectives)),
    ]


def main(argv: list[str]) -> None:
    """`<config> <n> <seed_base> <ckpt...>`: one row per checkpoint."""
    config_path, n, seed_base, checkpoints = (
        argv[1],
        int(argv[2]),
        int(argv[3]),
        argv[4:],
    )
    config = load_env_config(config_path)
    config.render_mode = None
    if not config.commitments.enabled:
        raise SystemExit(
            f"{config_path}: the commitment layer is off; nothing to ablate"
        )
    seeds = [seed_base + i for i in range(n)]
    print(
        f"commitment ablation on {config_path} n={n} seeds {seed_base}+ (success · held · turns)"
    )
    print("| checkpoint | trained | blank | misdirect | nearest |")
    print("|---|---|---|---|---|")
    for spec in checkpoints:
        cells = []
        for name, writer in modes(config):
            tokens._write_commitment_relations = writer  # type: ignore[assignment]
            try:
                result = evaluate_spec(spec, config, seeds, name)
            finally:
                tokens._write_commitment_relations = _TRAINED  # type: ignore[assignment]
            cells.append(
                f"{result.success_rate:.2f} · {result.objectives_held:.2f} · "
                f"{result.mean_turns:.2f}"
            )
        label = spec.split("/")[-2] if "/" in spec else spec
        print(f"| {label} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main(sys.argv)
