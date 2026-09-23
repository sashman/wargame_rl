"""The commitment layer's readouts (#384, B7): what a policy does with its
commitments, read live off the per-model env over seeded episodes.

Per policy (a scripted name, `random`, `set_network` or a `.pt`), one row:

- `persist`   share of unit-turns whose ground commitment is unchanged from
              the previous turn, over units alive on both turns
- `claim`     mean claimants on a claimed objective, and the max
- `complete`  share of ground commitments (unit, objective) that the unit's
              side held at the end of the episode
- `empty`     share of living unit-turns with no ground commitment while an
              objective was not ours
- `follow`    member FOLLOW-THROUGH: share of movement `act` steps by a member
              with a ground commitment that closed distance to it or ended
              inside it
- `leave`     the walk-off probe's number on the COMMITTED objective: share of
              movement decisions by a member inside its committed objective
              that ended outside it
- `hold`      share of committed unit-turns in HOLD mode (the unit has arrived
              at its objective and its plan weights are (0, 1)), and the share
              of movement openings by a unit in hold mode that declared the
              closing declaration (the whole unit stands): `hold <mode share>
              / decl <declaration share> (<openings>)`

    just measure-commitments <config.yaml> <n> <seed_base> <spec...>

The scripted bar writes its own assignment into the state (D8), so a script
has a row on every column and is the reference for each.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.per_model.commitment import NO_TARGET
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.types import BatchChooser, StepKind
from wargame_rl.wargame.envs.types import BattlePhase, WargameEnvConfig
from wargame_rl.wargame.selectors import build_per_model_chooser


def share(a: int, b: int) -> str:
    return f"{a / b:.2f}" if b else "n/a"


@dataclass(frozen=True)
class _Chooser:
    choose: BatchChooser


@dataclass
class Tally:
    persist_kept: int = 0
    persist_seen: int = 0
    claim_counts: list[float] = field(default_factory=list)
    # Distinct objectives claimed per turn over the objectives on the board
    # (#384: a plan that covers the board reads 1.0; a plan that stacks reads
    # less however many claimants it has).
    distinct_shares: list[float] = field(default_factory=list)
    # The same share on the episode's LAST turn only: an adaptive plan that
    # stacks early and covers by the end reads low on the mean and high here.
    distinct_end: list[float] = field(default_factory=list)
    complete_hits: int = 0
    complete_seen: int = 0
    empty_hits: int = 0
    empty_seen: int = 0
    follow_hits: int = 0
    follow_seen: int = 0
    leave_hits: int = 0
    leave_seen: int = 0
    hold_turns: int = 0
    committed_turns: int = 0
    hold_decl_hits: int = 0
    hold_decl_seen: int = 0

    def row_fields(self) -> str:
        """The readouts as table cells, in the plan-only readout's order."""
        claims = [c for c in self.claim_counts if c > 0]
        claim = f"{np.mean(claims):.2f}" if claims else "n/a"
        distinct = (
            f"{np.mean(self.distinct_shares):.2f}" if self.distinct_shares else "n/a"
        )
        end = f"{np.mean(self.distinct_end):.2f}" if self.distinct_end else "n/a"
        return (
            f"{share(self.persist_kept, self.persist_seen)} | {claim} | {distinct} / {end} | "
            f"{share(self.complete_hits, self.complete_seen)} | "
            f"{share(self.follow_hits, self.follow_seen)} | "
            f"{share(self.leave_hits, self.leave_seen)}"
        )

    def row(self, name: str) -> str:
        claims = [c for c in self.claim_counts if c > 0]
        claim = f"{np.mean(claims):.2f} / max {np.max(claims):.0f}" if claims else "n/a"
        distinct = (
            f"{np.mean(self.distinct_shares):.2f}" if self.distinct_shares else "n/a"
        )
        return (
            f"| {name} | persist {share(self.persist_kept, self.persist_seen)} "
            f"({self.persist_seen}) | claim {claim} | distinct {distinct} | complete "
            f"{share(self.complete_hits, self.complete_seen)} | empty "
            f"{share(self.empty_hits, self.empty_seen)} | follow "
            f"{share(self.follow_hits, self.follow_seen)} ({self.follow_seen}) | leave "
            f"{share(self.leave_hits, self.leave_seen)} ({self.leave_seen}) | hold "
            f"{share(self.hold_turns, self.committed_turns)} / decl "
            f"{share(self.hold_decl_hits, self.hold_decl_seen)} ({self.hold_decl_seen}) |"
        )


def _distance_to(env: PerModelEnv, location: object, objective: int) -> float:
    target = np.asarray(env.objectives[objective].location, dtype=float)
    return float(np.linalg.norm(np.asarray(location, dtype=float) - target))


def _ours(env: PerModelEnv) -> np.ndarray:
    from wargame_rl.wargame.envs.per_model.commitment import objective_counts_both_sides

    own, enemy, _norms = objective_counts_both_sides(
        env.wargame_models, env.opponent_models, env.objectives
    )
    return own > enemy


def run(spec: str, config: WargameEnvConfig, seeds: list[int]) -> Tally:
    env = PerModelEnv(config)
    chooser = build_per_model_chooser(spec, [env], seed=int(seeds[0]))
    return run_chooser(chooser.choose, env, config, seeds)


def run_chooser(
    choose: BatchChooser, env: PerModelEnv, config: WargameEnvConfig, seeds: list[int]
) -> Tally:
    """The tally of `choose` playing `env` over `seeds` (the plan-only readout
    passes a chooser whose head plans and whose members are scripted)."""
    chooser = _Chooser(choose)
    tally = Tally()
    radius = float(config.objective_radius_size)
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        state = env.player_commitments
        committed_pairs: set[tuple[int, int]] = set()
        while True:
            point = observation.decision
            action = chooser.choose([env], [observation])[0]
            if point.kind is StepKind.open and point.phase is BattlePhase.movement:
                opening = state.by_group.get(
                    int(env.wargame_models[action.model].group_id)
                )
                if (
                    opening is not None
                    and opening.ground != NO_TARGET
                    and opening.arrived
                ):
                    tally.hold_decl_seen += 1
                    if int(action.value) == 0:
                        tally.hold_decl_hits += 1
            is_move = point.kind is StepKind.act and point.phase is BattlePhase.movement
            before_ground = (
                state.ground_of(int(env.wargame_models[action.model].group_id))
                if is_move
                else NO_TARGET
            )
            before_location = (
                tuple(env.wargame_models[action.model].location) if is_move else None
            )
            observation, _r, terminated, _t, _info = env.step(action)
            if is_move and before_ground != NO_TARGET and before_location is not None:
                model = env.wargame_models[action.model]
                d0 = _distance_to(env, before_location, before_ground)
                d1 = _distance_to(env, model.location, before_ground)
                tally.follow_seen += 1
                if d1 < d0 - 1e-6 or d1 <= radius:
                    tally.follow_hits += 1
                if d0 <= radius:
                    tally.leave_seen += 1
                    if d1 > radius:
                        tally.leave_hits += 1
            for g, c in state.by_group.items():
                if c.ground != NO_TARGET:
                    committed_pairs.add((g, c.ground))
            if terminated:
                break
        history = state.history
        for previous, current in zip(history, history[1:]):
            for g, c in current.items():
                p = previous.get(g)
                if p is None or p.ground == NO_TARGET or c.ground == NO_TARGET:
                    continue
                tally.persist_seen += 1
                tally.persist_kept += int(p.ground == c.ground)
        ours_end = _ours(env)
        for turn in history:
            claims = np.zeros(len(env.objectives))
            for c in turn.values():
                if c.ground != NO_TARGET:
                    claims[c.ground] += 1
                    tally.committed_turns += 1
                    tally.hold_turns += int(c.arrived)
            tally.claim_counts.extend(claims.tolist())
            if len(claims):
                tally.distinct_shares.append(float(np.mean(claims > 0)))
        if history:
            last = np.zeros(len(env.objectives))
            for c in history[-1].values():
                if c.ground != NO_TARGET:
                    last[c.ground] += 1
            if len(last):
                tally.distinct_end.append(float(np.mean(last > 0)))
            unheld = bool(np.any(~ours_end))
            for g, c in turn.items():
                alive = any(
                    m.is_alive and int(m.group_id) == g for m in env.wargame_models
                )
                if not alive:
                    continue
                tally.empty_seen += 1
                if c.ground == NO_TARGET and unheld:
                    tally.empty_hits += 1
        for _g, objective in committed_pairs:
            tally.complete_seen += 1
            tally.complete_hits += int(bool(ours_end[objective]))
    return tally


def main(argv: list[str]) -> None:
    config = load_env_config(argv[1])
    n = int(argv[2])
    base = int(argv[3])
    seeds = [base + i for i in range(n)]
    print(
        f"commitment readouts on {argv[1]} n={n} seeds {base}+ | assignment="
        f"{config.commitments.assignment}"
    )
    print(
        "| policy | persist (unit-turns) | claimants per claimed objective | distinct | complete | empty | follow (member move steps) | leave (moves from inside) |"
    )
    for spec in argv[4:]:
        name = spec.split("/")[-2] if "/" in spec else spec
        print(run(spec, config, seeds).row(name), flush=True)


if __name__ == "__main__":
    main(sys.argv)
