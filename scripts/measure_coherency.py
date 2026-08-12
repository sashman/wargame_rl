"""How often is a policy actually in unit coherency?

Coherency is [docs/rules/03-moving.md](../docs/rules/03-moving.md) § Coherency and
this environment does not enforce it -- the gap map rates the rule **divergent**.
Before choosing how to enforce it, the thing worth knowing is what enforcing it
would *cost*, and that is a question about the policies that already exist. If
the scripted bar spends most of the game out of coherency, then adopting the rule
reprices every baseline and the whole comparison ladder resets; if everything is
already coherent, the rule is nearly free and the choice of mechanism barely
matters.

Reports, per force and per step, the share of *units* in coherency and the mean
number of models outside the coherent body of their unit, split by which of the
three conditions failed. Both forces are measured, because coherency is symmetric
-- enforcement lands on the opponent too, and a rule that costs the opponent more
than the agent is a difficulty change wearing a rules change's clothes.

Measured at two distances: the rules' 2" chain / 9" spread, and whatever the
config's own `group_max_distance` is, so the cost of adopting the rules' numbers
is separated from the cost of enforcing the concept.

Accepts either a scripted baseline name or a checkpoint path, so the agent and
the bar are measured by one code path on identical layouts.

**Read the `deployment coherent` column for the player only.** It is sampled at
the first observation after `reset`, and `run_until_player_phase` auto-executes
the opponent's first turn before that returns -- so the player's figure is its
deployment, and the opponent's is already one move old.

Usage: just measure-coherency <policy|ckpt> <env_config> [n_episodes]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from pydantic_yaml import parse_yaml_raw_as

from scripts.measure_checkpoint import HELDOUT_SEED_BASE, build_selector
from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector, selector_for
from wargame_rl.wargame.envs.baseline.registry import (
    build_baseline_policy,
    get_registry,
)
from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.entities import WargameModel, alive_mask_for
from wargame_rl.wargame.envs.domain.rules_constants import (
    COHERENCY_FURTHEST_IN,
    COHERENCY_NEAREST_IN,
)
from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment


@dataclass
class CoherencyTally:
    """Running totals for one force under one pair of distances."""

    steps: int = 0
    units: int = 0
    units_coherent: int = 0
    fully_coherent_steps: int = 0
    models_out: int = 0
    chain_failures: int = 0
    spread_failures: int = 0
    split_units: int = 0
    deployment_coherent: int = 0
    deployment_episodes: int = 0

    def observe(
        self,
        models: list[WargameModel],
        nearest: float,
        furthest: float,
        at_deployment: bool = False,
    ) -> None:
        """Fold one instant of one force into the totals."""
        if not models:
            return
        report = evaluate_coherency(
            positions=np.array([m.location for m in models], dtype=float),
            group_ids=np.array([m.group_id for m in models], dtype=np.intp),
            alive_mask=alive_mask_for(models),
            base_radii=np.array([m.base_radius for m in models], dtype=float),
            nearest_distance=nearest,
            furthest_distance=furthest,
        )
        self.steps += 1
        self.units += report.n_units
        self.units_coherent += report.n_units_coherent
        self.fully_coherent_steps += int(report.all_coherent)
        self.models_out += report.n_models_out_of_coherency
        for unit in report.units:
            if unit.size <= 1:
                continue
            self.chain_failures += int(not unit.chain_ok.all())
            self.spread_failures += int(not unit.spread_ok.all())
            self.split_units += int(not unit.connected)
        if at_deployment:
            self.deployment_episodes += 1
            self.deployment_coherent += int(report.all_coherent)

    def row(self, label: str) -> str:
        """One formatted table row."""
        if self.steps == 0:
            return f"{label:<22}{'(no models)':>12}"
        unit_rate = self.units_coherent / max(self.units, 1)
        step_rate = self.fully_coherent_steps / self.steps
        deploy_rate = self.deployment_coherent / max(self.deployment_episodes, 1)
        return (
            f"{label:<22}{unit_rate:>10.3f}{step_rate:>10.3f}{deploy_rate:>12.3f}"
            f"{self.models_out / self.steps:>11.2f}"
            f"{self.chain_failures / max(self.units, 1):>9.3f}"
            f"{self.spread_failures / max(self.units, 1):>9.3f}"
            f"{self.split_units / max(self.units, 1):>8.3f}"
        )


def main() -> None:
    """Print the coherency profile for one policy on one config."""
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy_name = sys.argv[1]
    config_path = sys.argv[2]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    with open(config_path) as handle:
        env_config = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    env_config.render_mode = None
    env = create_environment(env_config=env_config)
    quantities = resolve_rules_quantities(env_config)
    scale = quantities.scale

    select: ActionSelector
    if policy_name in get_registry():
        select = selector_for(build_baseline_policy(policy_name))
        label = policy_name
    else:
        select, _net = build_selector(policy_name, env)
        label = policy_name.split("/")[-2] if "/" in policy_name else policy_name

    # Two rulers. The rules' own pair, and the single distance this scenario
    # currently calls coherency -- which has no chain/spread split at all, so it
    # is applied as both to show what the config would enforce if it enforced
    # anything.
    rulers = {
        'rules 2"/9"': (
            scale.to_units(COHERENCY_NEAREST_IN),
            scale.to_units(COHERENCY_FURTHEST_IN),
        ),
        f"config {env_config.group_max_distance:g}": (
            quantities.coherency_distance,
            quantities.coherency_distance,
        ),
    }
    tallies = {
        (ruler, force): CoherencyTally()
        for ruler in rulers
        for force in ("player", "opponent")
    }

    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        _observe_all(env, tallies, rulers, at_deployment=True)
        terminated = truncated = False
        while not (terminated or truncated):
            action = select(observation, env)
            observation, _r, terminated, truncated, _i = env.step(action)
            _observe_all(env, tallies, rulers)

    print(f"\n{label}   {config_path}")
    print(f"{n_episodes} episodes, seeds {seeds[0]}-{seeds[-1]}")
    print(
        f"{env_config.max_groups} units per force, base radius "
        f"{quantities.base_radius:.2f}u\n"
    )
    header = (
        f"{'':<22}{'units':>10}{'steps':>10}{'deployment':>12}"
        f"{'models':>11}{'chain':>9}{'spread':>9}{'split':>8}"
    )
    print(header)
    print(
        f"{'':<22}{'coherent':>10}{'coherent':>10}{'coherent':>12}"
        f"{'out':>11}{'fail':>9}{'fail':>9}{'unit':>8}"
    )
    print("-" * len(header))
    for ruler in rulers:
        for force in ("player", "opponent"):
            print(tallies[(ruler, force)].row(f"{ruler}  {force}"))
    print()


def _observe_all(
    env: WargameEnv,
    tallies: dict[tuple[str, str], CoherencyTally],
    rulers: dict[str, tuple[float, float]],
    at_deployment: bool = False,
) -> None:
    """Fold the current state of both forces into every tally."""
    for ruler, (nearest, furthest) in rulers.items():
        tallies[(ruler, "player")].observe(
            env.wargame_models, nearest, furthest, at_deployment
        )
        tallies[(ruler, "opponent")].observe(
            env.opponent_models, nearest, furthest, at_deployment
        )


if __name__ == "__main__":
    main()
