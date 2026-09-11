"""Where a per-model decision's wall-clock goes, budgeted in ROUNDS.

The per-model counterpart of `measure_throughput.py`. Its unit of work is a
decision, not a phase, and decisions per round vary with the army and the
policy, so the budget is rounds and the report says how many decisions a
round took. Two rows, because the network is most of the cost and a row
without it is the env's own floor:

  random    the random legal seat -- tokens are never built, no forward runs
  network   the set network at fresh weights, SAMPLED (the training regime)

Per decision, split into: the token observation (`SetAgent.observe`), the
forward pass (`SetNetwork.forward` + `heads`), the rest of `act_batch` (the
draw and the legality check), `PerModelEnv.step`, the re-timed reward
(`PerStepReward.on_step`, per calculator), the decision observation the env
builds (`build_per_model_observation`), and the opponent's scripted plan.

Timed with `time.perf_counter` wrappers, not cProfile, for the reason the
whole-phase script gives: the reward loop's call shape is exactly what
cProfile inflates. Instrumented after the first reset, so construction and
the first layout are excluded.

Usage: just measure-throughput-per-model <env_config> [rounds]
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic_yaml import parse_yaml_raw_as

import wargame_rl.wargame.envs.per_model.env as per_model_env_module
from scripts.measure_throughput import _Timings
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.evaluate import random_chooser
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.types import (
    BatchChooser,
    PerModelAction,
    PerModelObservation,
    StepKind,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig

DEFAULT_ROUNDS = 20
SEED = 4242
COMBAT_SEED = 99


@dataclass(frozen=True)
class Row:
    """One policy's throughput over the round budget."""

    label: str
    decisions: int
    rounds: int
    seconds: float
    timings: _Timings

    @property
    def ms_per_decision(self) -> float:
        return 1000.0 * self.seconds / max(1, self.decisions)

    @property
    def ms_per_round(self) -> float:
        return 1000.0 * self.seconds / max(1, self.rounds)

    @property
    def decisions_per_round(self) -> float:
        return self.decisions / max(1, self.rounds)


def _instrument_env(
    env: PerModelEnv, retimer: PerStepReward, timings: _Timings
) -> None:
    timings.wrap(env, "step", "env.step")
    timings.wrap(
        per_model_env_module, "build_per_model_observation", "  env/observation"
    )
    opponent = env.opponent_seat.adapter
    if opponent is not None:
        timings.wrap(opponent, "plan", "  env/opponent plan")
    timings.wrap(retimer, "on_step", "reward (re-timed)")
    for phase in retimer.manager.phases:
        for name, calculator in [
            *phase.per_model_calculators,
            *phase.global_calculators,
        ]:
            timings.wrap(calculator, "calculate", f"  reward/{name}")


def _network_chooser(env: PerModelEnv, timings: _Timings) -> BatchChooser:
    """The set network at fresh weights, sampled, with its sections wrapped."""
    import torch

    from wargame_rl.wargame.model.per_model import SetAgent, SetNetwork

    torch.manual_seed(SEED)
    agent = SetAgent(SetNetwork.from_env(env))
    generator = torch.Generator().manual_seed(SEED)
    timings.wrap(agent, "observe", "  act/tokens")
    timings.wrap(agent.network, "forward", "  act/forward")
    timings.wrap(agent.network, "heads", "  act/heads")
    timings.wrap(agent, "act_batch", "act (tokens + forward + draw)")

    def choose(
        envs: Sequence[PerModelEnv], observations: Sequence[PerModelObservation]
    ) -> list[PerModelAction]:
        return [
            d.action for d in agent.act_batch(envs, observations, generator=generator)
        ]

    return choose


def _run(
    config: WargameEnvConfig,
    rounds: int,
    label: str,
    chooser_for: Callable[[PerModelEnv, _Timings], BatchChooser],
) -> Row:
    env = PerModelEnv(config)
    retimer = PerStepReward(env)
    timings = _Timings()
    observation, _ = env.reset(seed=SEED, options={"combat_seed": COMBAT_SEED})
    retimer.reset()
    _instrument_env(env, retimer, timings)
    choose = chooser_for(env, timings)

    decisions = 0
    closes = 0
    seconds = 0.0
    while closes < rounds:
        start = time.perf_counter()
        action = choose([env], [observation])[0]
        before = observation
        observation, _reward, terminated, _truncated, info = env.step(action)
        retimer.on_step(before, action, info["effect"], terminated)
        seconds += time.perf_counter() - start
        if before.decision.kind is StepKind.close_turn:
            closes += 1
        else:
            decisions += 1
        if terminated:
            observation, _ = env.reset(options={"combat_seed": COMBAT_SEED})
            retimer.reset()
    return Row(label, decisions, closes, seconds, timings)


def _print_row(row: Row) -> None:
    print(f"{row.label}")
    print(
        f"  per decision    {row.ms_per_decision:8.3f} ms   "
        f"{row.decisions_per_round:6.1f} decisions / round   "
        f"{row.ms_per_round:8.1f} ms / round"
    )
    for label in sorted(row.timings.total, key=lambda k: (k.startswith("  "), k)):
        seconds = row.timings.total[label]
        per_decision = 1000.0 * seconds / max(1, row.decisions)
        share = 100.0 * seconds / max(row.seconds, 1e-12)
        calls = row.timings.calls[label] / max(1, row.decisions)
        print(
            f"    {label:<34} {per_decision:8.3f} ms  {share:5.1f}%  {calls:6.2f} calls"
        )
    print()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    config_path = Path(argv[0])
    rounds = int(argv[1]) if len(argv) > 1 and argv[1] else DEFAULT_ROUNDS
    config = parse_yaml_raw_as(WargameEnvConfig, config_path.read_text())
    config.render_mode = None

    print(f"config          {config_path}")
    print(
        f"army            {config.number_of_wargame_models}v"
        f"{config.number_of_opponent_models}   board {config.board_width}x"
        f"{config.board_height}   rounds {rounds}   (per-model facade, batch 1)"
    )
    print()
    _print_row(
        _run(config, rounds, "random legal seat", lambda env, t: random_chooser(SEED))
    )
    _print_row(
        _run(config, rounds, "set network, fresh weights, sampled", _network_chooser)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
