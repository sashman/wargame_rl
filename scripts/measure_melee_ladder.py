"""Score a policy on the melee ladder's four cells, and price its decode headroom.

    just measure-melee-ladder <policy|ckpt> [family] [n] [decode_topk] [charge_decode]

The four cells are the goal of [the melee teaching goal](../docs/melee-teaching-goal.md):
`vs_take`, `vs_deny`, `vs_shoot` and the **refereed head-to-head**. The goal is
**conjunctive** -- a policy has to clear `squad_march_take_charge` on all four --
so the cells are always scored together and always printed together. Reporting
whichever cell happens to be ahead is winner-selection, worth +1.4 to +2.9 vp by
this repo's own measurement.

⚠ **The refereed cell is the MIRROR**: the bar's row there is that script
playing itself from the disadvantaged player seat. A perfect imitation scores
the bar's number and no better, so **cloning cannot win that cell by
construction** -- only improving on a clone can.

## Decode headroom, and why it is printed beside the score

Measured 2026-09-04 (`reports/2026-09-04-ppo-spends-the-decodes-headroom.md`):
PPO collects rollouts at `decode_topk=1` with no charge decode, and is scored
here at `K=3` plus the charge decode. Those are different objectives. PPO from a
behaviour clone improved it by **+19.72 vp on 6 of 6 seeds** in the regime it
trains in while losing 14.92 in the regime it is scored in, because it spent the
decode's headroom: **+74.87 vp for the clone, +40.23 after**.

So `headroom` -- the same checkpoint at `K=3 cd=1` minus itself at `K=1 cd=0` --
is a first-class readout, not a diagnostic afterthought. A run that gains vp
while headroom collapses has not learned what it appears to have learned.

⚠ **Quote the revision with the table.** The bar below was re-measured at
`d5ec7d4` on 2026-09-04 and reproduced §38 on all four cells to the decimal; a
bar is void the moment a melee rule changes.

⚠ **The scripted rows reproduce exactly and so do the agent rows** -- four
repeat runs at n=45 returned every digit unchanged, and the six clone seeds
reproduced §46's published per-seed list. That contradicts the warning carried
by `measure_charges`, which recorded agent rows varying within one process on a
DIFFERENT config; it is not evidence that warning was wrong there.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.scenario_overrides import load_env_config, parse_overrides
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

CELLS = ("refereed", "vs_take", "vs_deny", "vs_shoot")

# `squad_march_take_charge`, n=45, seeds 700000+, no decode. Re-measured at
# d5ec7d4 on 2026-09-04, identical to melee-teaching-goal 38 on all four cells.
BAR = {"refereed": -5.3, "vs_take": 20.2, "vs_deny": 11.8, "vs_shoot": 56.6}

SEED_BASE = 700000


class CellResult:
    """One cell's episode-averaged readouts."""

    def __init__(
        self, vp: float, declared: float, stood: float, coherency: float
    ) -> None:
        self.vp = vp
        self.declared = declared
        self.stood = stood
        self.coherency = coherency


def score_cell(
    policy: str,
    config_path: str,
    n_episodes: int,
    decode_topk: int,
    charge_decode: bool,
    overrides: dict[str, str],
) -> CellResult:
    """Play `n_episodes` on one config and average the readouts.

    Seeds are `700000 + episode`, matching every other scoring recipe here, so
    an agent row and a baseline row taken at the same `n` are on identical
    layouts.
    """
    env = create_environment(load_env_config(config_path, **overrides))
    selector = build_action_selector(
        policy, env, decode_topk=decode_topk, charge_decode=charge_decode
    )
    total_vp = 0.0
    declared = 0
    stood = 0
    coherency_total = 0.0
    coherency_episodes = 0
    for episode in range(n_episodes):
        observation, _ = env.reset(seed=SEED_BASE + episode)
        done = False
        while not done:
            in_charge_phase = env.game_clock_state.phase is BattlePhase.charge
            if in_charge_phase:
                declared += _declared_units(env)
            action = selector.select(observation, env)
            observation, _reward, terminated, truncated, _info = env.step(action)
            if in_charge_phase:
                # Read AFTER the step: `charged_this_turn` is set by the
                # referee, so a charge counts as stood only if it was legal.
                stood += _units_that_charged(env)
            done = terminated or truncated
        rate = env.intended_coherency_rate
        if rate is not None:
            coherency_total += float(rate)
            coherency_episodes += 1
        total_vp += float(env.player_vp - env.opponent_vp)
    env.close()
    return CellResult(
        vp=total_vp / n_episodes,
        declared=declared / n_episodes,
        stood=stood / n_episodes,
        coherency=coherency_total / max(coherency_episodes, 1),
    )


def _declared_units(env: WargameEnv) -> int:
    return len(
        {
            int(model.group_id)
            for model in env.wargame_models
            if model.is_alive and getattr(model, "declared_charge", False)
        }
    )


def _units_that_charged(env: WargameEnv) -> int:
    return len(
        {
            int(model.group_id)
            for model in env.wargame_models
            if model.is_alive and getattr(model, "charged_this_turn", False)
        }
    )


def _config_for(family: str, cell: str) -> str:
    return f"configs/evaluation/25v25_maps_melee_{family}_{cell}.yaml"


def main(argv: list[str]) -> int:
    """Print the four cells, the gap to the bar, and the decode headroom."""
    positional, overrides = parse_overrides(argv)
    if not positional:
        print(__doc__)
        return 1
    policy = positional[0]
    family = positional[1] if len(positional) > 1 else "approach"
    n_episodes = int(positional[2]) if len(positional) > 2 else 45
    decode_topk = int(positional[3]) if len(positional) > 3 else 3
    charge_decode = bool(int(positional[4])) if len(positional) > 4 else True

    for cell in CELLS:
        if not Path(_config_for(family, cell)).exists():
            raise FileNotFoundError(
                f"no config for cell {cell!r} in family {family!r}: "
                f"{_config_for(family, cell)}"
            )

    print(
        f"melee ladder — {policy}\n"
        f"  family={family} n={n_episodes} decode_topk={decode_topk} "
        f"charge_decode={int(charge_decode)}"
        + (f" overrides={overrides}" if overrides else "")
    )
    print(
        f"\n{'cell':10s} {'vp':>8s} {'bar':>7s} {'gap':>8s} "
        f"{'decl/ep':>8s} {'stood/ep':>9s} {'coherent':>9s}"
    )
    cleared = 0
    for cell in CELLS:
        result = score_cell(
            policy,
            _config_for(family, cell),
            n_episodes,
            decode_topk,
            charge_decode,
            overrides,
        )
        gap = result.vp - BAR[cell]
        # Counted on the ROUNDED gap, so the count always agrees with the
        # column beside it. The bar constants carry one decimal, so a
        # full-precision comparison scores the bar as beating itself on
        # whichever cell rounds up -- which it did, before this line.
        cleared += round(gap, 1) > 0
        print(
            f"{cell:10s} {result.vp:+8.1f} {BAR[cell]:+7.1f} {gap:+8.1f} "
            f"{result.declared:8.2f} {result.stood:9.2f} {result.coherency:9.3f}"
        )
    print(
        f"\ncells ahead of the bar: {cleared}/4 (the goal is conjunctive) "
        f"-- ONE run, not a verdict: the goal asks for six seeds"
    )

    # The training regime, on the cell the goal turns on. Printed always: a
    # score without it cannot say whether a gain is the policy or the decode.
    refereed = _config_for(family, "refereed")
    undecoded = score_cell(policy, refereed, n_episodes, 1, False, overrides)
    decoded = score_cell(
        policy, refereed, n_episodes, decode_topk, charge_decode, overrides
    )
    print(
        f"\ndecode headroom (refereed): {undecoded.vp:+.2f} undecoded "
        f"→ {decoded.vp:+.2f} decoded = {decoded.vp - undecoded.vp:+.2f} vp"
        f"   [clone +74.87, after 300 epochs of PPO +40.23]"
    )
    print(
        f"  unaided coherency {undecoded.coherency:.3f} → decoded "
        f"{decoded.coherency:.3f}; unaided stood/ep {undecoded.stood:.2f} "
        f"→ {decoded.stood:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
