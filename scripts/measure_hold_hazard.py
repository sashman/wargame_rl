"""What standing on an objective earns a model, and what it costs it.

The agent finishes with **52.9% of its army alive against the scripts'
27.4-30.9%** while holding 0.63 fewer objectives. Six explanations for that
offence deficit have been refuted -- the clock, weapon range, the travel-reward
gate, mixed profiles, squad structure and the VP cap -- which leaves one:
standing on an objective means standing in the open, and if the reward does not
pay for the casualties, hiding is *correct play* rather than a failure.

That is a claim about a per-model trade, so it needs a per-model differential.
`measure-income-share` cannot supply it: it reports each calculator's share of
*mean* income, and its own docstring warns that a share rules a term out as the
largest income stream but never as the driver of a behaviour. What moves a
policy gradient is a term's variation across the choice being made.

So this measures the choice directly. For every alive model on every step it
records whether the model is inside an objective radius (`norms_offset <=
radius`, the same test VP scoring uses), what `last_per_model_reward` paid it,
and whether it is dead on the next step. That yields:

- **income differential** -- mean per-model reward on an objective minus off it.
  What a model is paid for being there, including its share of the global terms.
- **death hazard** -- P(dead next step) on an objective against off it. What
  being there costs.
- **the break-even hazard** -- the excess hazard at which the differential stops
  covering the forgone future income. Above it, hiding maximises per-model
  return and the agent is playing the reward correctly.

⚠ This is a CONDITIONAL comparison, not a causal one. Models on objectives
differ from models off them in more than their footing -- they are further
forward, and later in the episode. Read it as "what the trade looks like from
where the policy actually stands", and treat a break-even margin inside the
noise as inconclusive rather than as either verdict.

Usage: just measure-hold-hazard <policy|ckpt> <env_config> [n_episodes] [decode_topk]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.measure_maps import config_for_map, load_maps
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

HELDOUT_SEED_BASE = 700_000
DEFAULT_MAPS_DIR = Path("configs/evaluation/maps_heldout")


@dataclass
class Tally:
    """Model-steps split by whether the model stood on an objective."""

    steps_on: int = 0
    steps_off: int = 0
    deaths_on: int = 0
    deaths_off: int = 0
    reward_on: float = 0.0
    reward_off: float = 0.0
    steps_remaining_on: float = 0.0
    episode_steps: int = 0
    episodes: int = 0

    def rate(self, deaths: int, steps: int) -> float:
        """Deaths per model-step. Zero when nothing was observed."""
        return deaths / steps if steps else 0.0

    def income(self, total: float, steps: int) -> float:
        """Mean per-model reward per model-step."""
        return total / steps if steps else 0.0


def on_objective(env: object) -> np.ndarray:
    """Boolean per player model: inside an objective radius, and alive.

    Uses `norms_offset <= radius` -- the same in-range test the VP scoring and
    the distance cache use, so "on an objective" here means what it means to the
    mission, not something adjacent to it.
    """
    alive = alive_mask_for(env.player_models)  # type: ignore[attr-defined]
    cache = compute_distances(
        env.player_models,  # type: ignore[attr-defined]
        env.objectives,  # type: ignore[attr-defined]
        alive_mask=alive,
    )
    inside = np.any(cache.model_obj_norms_offset <= cache.obj_radii, axis=1)
    standing: np.ndarray = np.asarray(inside) & np.asarray(alive)
    return standing


def collect(
    policy: str, config: WargameEnvConfig, seeds: list[int], decode_topk: int
) -> Tally:
    """Walk every map and seed, tallying the trade one model-step at a time."""
    tally = Tally()
    for terrain_map in load_maps(DEFAULT_MAPS_DIR):
        per_map = config_for_map(config, terrain_map)
        env = create_environment(env_config=per_map)
        select = build_action_selector(policy, env, decode_topk).select
        for seed in seeds:
            observation, _info = env.reset(seed=seed)
            done = False
            previous_on: np.ndarray | None = None
            previous_alive: np.ndarray | None = None
            steps = 0
            while not done:
                observation, _r, done, _t, _i = env.step(select(observation, env))
                steps += 1
                alive = np.asarray(alive_mask_for(env.player_models))
                standing = on_objective(env)
                rewards = np.asarray(env.last_per_model_reward, dtype=np.float64)

                # A death is scored against where the model stood on the step
                # BEFORE it died -- crediting it to the step it is already dead
                # on would attribute every casualty to wherever the corpse lies.
                if previous_on is not None and previous_alive is not None:
                    died = previous_alive & ~alive
                    tally.deaths_on += int(np.sum(died & previous_on))
                    tally.deaths_off += int(np.sum(died & ~previous_on))

                tally.steps_on += int(np.sum(standing))
                tally.steps_off += int(np.sum(alive & ~standing))
                tally.reward_on += float(np.sum(rewards[standing]))
                tally.reward_off += float(np.sum(rewards[alive & ~standing]))
                previous_on, previous_alive = standing, alive
            tally.episode_steps += steps
            tally.episodes += 1
        env.close()
    return tally


def report(label: str, tally: Tally) -> None:
    """Print the differential, the hazard, and the hazard that breaks even."""
    hazard_on = tally.rate(tally.deaths_on, tally.steps_on)
    hazard_off = tally.rate(tally.deaths_off, tally.steps_off)
    income_on = tally.income(tally.reward_on, tally.steps_on)
    income_off = tally.income(tally.reward_off, tally.steps_off)
    differential = income_on - income_off
    excess_hazard = hazard_on - hazard_off

    mean_episode = tally.episode_steps / tally.episodes if tally.episodes else 0.0
    # Half the episode is the average vantage point of a model choosing now.
    remaining = mean_episode / 2.0
    mean_income = tally.income(
        tally.reward_on + tally.reward_off, tally.steps_on + tally.steps_off
    )
    forgone = mean_income * remaining
    break_even = differential / forgone if forgone else float("nan")

    print(f"\n  {label}")
    print(f"    model-steps on an objective   {tally.steps_on:>10,}")
    print(f"    model-steps elsewhere         {tally.steps_off:>10,}")
    print(f"    reward per step, on           {income_on:>10.4f}")
    print(f"    reward per step, off          {income_off:>10.4f}")
    print(f"    INCOME DIFFERENTIAL           {differential:>10.4f}  <- what it pays")
    print(f"    death hazard, on              {100.0 * hazard_on:>9.2f}%")
    print(f"    death hazard, off             {100.0 * hazard_off:>9.2f}%")
    print(
        f"    EXCESS HAZARD                 {100.0 * excess_hazard:>9.2f}%  <- what it costs"
    )
    print(
        f"    forgone income if it dies     {forgone:>10.4f}  ({remaining:.1f} steps left)"
    )
    print(f"    BREAK-EVEN EXCESS HAZARD      {100.0 * break_even:>9.2f}%")
    verdict = (
        "holding does NOT pay -- hiding is correct play"
        if excess_hazard > break_even
        else "holding DOES pay -- the agent is leaving return on the table"
    )
    print(f"    -> {verdict}")


def main() -> None:
    """Price an objective-hold in casualties against what it earns."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 and argv[3] else 30
    decode_topk = int(argv[4]) if len(argv) > 4 and argv[4] else 1

    config = load_env_config(config_path, **overrides)
    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]
    tally = collect(policy, config, seeds, decode_topk)

    print(f"\n{policy}")
    print(
        f"{config_path}{describe(overrides)}  ({n_episodes} episodes per map, "
        f"seeds {seeds[0]}-{seeds[-1]}, decode_topk={decode_topk})"
    )
    report("the trade", tally)
    print()


if __name__ == "__main__":
    main()
