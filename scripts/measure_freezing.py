"""How often an ordered move produces no movement at all, and why.

A model that asks to move and does not is invisible in every score this project
records: `vp_margin` sees the consequence, `coherent` sees the formation, and
nothing counts the order that evaporated. An adversarial review measured
`P(frozen | frozen)` at **0.73-0.91** against the 0.62 on record for the
coherency referee, with **28-48% of non-STAY orders producing zero
displacement** -- mostly friendly gridlock on stacked objectives, and entirely
pre-existing.

That matters most for the LONGEST moves. An advance is the longest move in the
game, so it is the most likely to be truncated or stopped outright; an advance
arm could measure "the feature does not help" when the moves never executed.

Reported per policy:

    ordered      non-STAY movement orders
    frozen       of those, ones that moved < `_STILL` inches
    truncated    ones that moved, but less than they asked for
    delivered    fraction of ordered inches that actually happened
    P(f|f)       frozen this phase given frozen last phase, same model
    P(f|moved)   frozen this phase given it moved last phase
    absorbing    P(f|f) - P(f|moved); > 0 means freezing is self-sustaining

Usage: just measure-freezing <policy|ckpt> <env_config> [n_episodes] [maps_dir]
       [decode_topk] [key=value...]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.measure_maps import build_action_selector, config_for_map, load_maps
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.domain.game_clock import BattlePhase
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.model.common.factory import create_environment

BATTLE_MOVEMENT = BattlePhase.movement

HELDOUT_SEED_BASE = 700_000
DEFAULT_MAPS_DIR = Path("configs/evaluation/maps_heldout")
# Below this a move is "frozen" rather than merely short. Well under the
# collision solver's own back-off (`_CONTACT_MARGIN` scaled by the order), so a
# move that only lost its contact margin is not miscounted as frozen.
_STILL = 1e-3


@dataclass
class Tally:
    """Per-policy movement accounting."""

    ordered: int = 0
    frozen: int = 0
    truncated: int = 0
    asked_inches: float = 0.0
    moved_inches: float = 0.0
    # (previous frozen, now frozen) transition counts, per model.
    transitions: dict[tuple[bool, bool], int] = field(default_factory=dict)

    def add_transition(self, was_frozen: bool, is_frozen: bool) -> None:
        """Count one consecutive-phase pair for one model."""
        key = (was_frozen, is_frozen)
        self.transitions[key] = self.transitions.get(key, 0) + 1

    def conditional(self, given_frozen: bool) -> float:
        """P(frozen now | frozen last) or P(frozen now | moved last)."""
        total = sum(
            n for (was, _), n in self.transitions.items() if was is given_frozen
        )
        if not total:
            return float("nan")
        hits = self.transitions.get((given_frozen, True), 0)
        return hits / total


def collect(
    policy: str,
    config: object,
    seeds: list[int],
    maps_dir: Path,
    decode_topk: int,
) -> Tally:
    """Play the policy and account for every movement order it issues."""
    tally = Tally()
    for terrain_map in load_maps(maps_dir):
        per_map = config_for_map(config, terrain_map)  # type: ignore[arg-type]
        select, _label = build_action_selector(policy, per_map, decode_topk, False)
        env = create_environment(env_config=per_map)
        handler = env.player_action_handler
        for seed in seeds:
            observation, _info = env.reset(seed=seed)
            was_frozen: dict[int, bool] = {}
            done = False
            while not done:
                phase = env.game_clock_state.phase
                before = [np.array(m.location, dtype=float) for m in env.wargame_models]
                action = select(observation, env)
                observation, _r, done, _t, _i = env.step(action)
                if phase is not BATTLE_MOVEMENT:
                    continue
                for i, model in enumerate(env.wargame_models):
                    if not model.is_alive:
                        continue
                    act = int(action.actions[i])
                    if act == STAY_ACTION:
                        continue
                    asked = float(
                        np.linalg.norm(
                            handler.decode_action(
                                act, model_idx=i, advance_roll=model.advance_roll
                            )
                        )
                    )
                    if asked <= _STILL:
                        continue
                    moved = float(np.linalg.norm(model.location - before[i]))
                    tally.ordered += 1
                    tally.asked_inches += asked
                    tally.moved_inches += moved
                    frozen = moved < _STILL
                    if frozen:
                        tally.frozen += 1
                    elif moved < asked - _STILL:
                        tally.truncated += 1
                    if i in was_frozen:
                        tally.add_transition(was_frozen[i], frozen)
                    was_frozen[i] = frozen
        env.close()
    return tally


def report(label: str, tally: Tally) -> None:
    """One line per policy, plus the absorbing-state read."""
    if not tally.ordered:
        print(f"  {label:<24} no movement orders")
        return
    frozen = 100.0 * tally.frozen / tally.ordered
    truncated = 100.0 * tally.truncated / tally.ordered
    delivered = (
        100.0 * tally.moved_inches / tally.asked_inches if tally.asked_inches else 0.0
    )
    p_ff = tally.conditional(True)
    p_fm = tally.conditional(False)
    print(
        f"  {label:<24} ordered {tally.ordered:6d}   frozen {frozen:5.1f}%   "
        f"truncated {truncated:5.1f}%   delivered {delivered:5.1f}%   "
        f"P(f|f) {p_ff:.3f}   P(f|moved) {p_fm:.3f}   "
        f"absorbing {p_ff - p_fm:+.3f}"
    )


def main() -> None:
    """Print the freeze and truncation accounting for one policy."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 and argv[3] else 20
    maps_dir = Path(argv[4]) if len(argv) > 4 and argv[4] else DEFAULT_MAPS_DIR
    decode_topk = int(argv[5]) if len(argv) > 5 and argv[5] else 1

    config = load_env_config(config_path, **overrides)
    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]

    print(f"\n{policy} on {config_path}{describe(overrides)}")
    print(
        f"  ({len(load_maps(maps_dir))} maps x {n_episodes} episodes, "
        f"seeds {seeds[0]}-{seeds[-1]}, decode_topk={decode_topk})\n"
    )
    report(policy, collect(policy, config, seeds, maps_dir, decode_topk))
    print()


if __name__ == "__main__":
    main()
