"""Play the four legs of a rated pairing.

This is the **only** module in `rating/` that touches a live `WargameEnv`;
everything else here is numpy and a config dump. That split -- env-free rather
than play-versus-fit -- is what keeps the rating mathematics testable on
synthetic arrays.

The scoring loop itself is `envs/baseline/evaluate.evaluate_selector`,
**unmodified**. It already resets per seed with a pinned combat seed, runs the
episode, and returns `vp_margin_per_episode` / `win_per_episode` /
`objectives_held_per_episode` in seed order -- exactly the per-layout rows the
Bradley-Terry fit and the layout bootstrap need.

What it lacks is opponent identity. But opponent identity is not a parameter of
the scoring loop: it is **state on the env**, installed once per leg. Threading
a field through `BaselineResult` would push it onto six other consumers for one
caller's benefit, and writing a second scoring loop is the exact defect
`measure_paired_policies` documents guarding against -- two implementations of
"score a policy over seeds" drifting apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from wargame_rl.wargame.envs.baseline.evaluate import evaluate_selector
from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.rating.entrant import Entrant
from wargame_rl.wargame.rating.schedule import (
    ELO_SEED_BASE,
    FOUR_LEGS,
    Leg,
    combat_seeds_for,
    config_for_leg,
    layout_seeds,
    with_opponent,
)


@dataclass(frozen=True, slots=True)
class LegResult:
    """One entrant against one opponent, on one leg, over the layout seeds.

    `margins` is A's `player_vp - opponent_vp`, one entry per layout **in seed
    order**, so two legs are index-aligned and a layout can be resampled whole.
    """

    entrant_a: str
    entrant_b: str
    leg: Leg
    layout_seeds: tuple[int, ...]
    combat_seeds: tuple[int, ...]
    margins: tuple[float, ...]
    wins: tuple[float, ...]
    objectives_held: tuple[float, ...]
    coherency_rate: float | None


class AsymmetricScenarioError(ValueError):
    """The config cannot host a rated match on equal terms."""


def opponent_config_for(entrant: Entrant) -> OpponentPolicyConfig:
    """How entrant B is seated on the opponent side.

    A scripted baseline goes through `scripted_baseline`, which already ships
    and runs any player-side baseline on the opponent seat. A checkpoint needs
    the `model` opponent policy, which does not exist yet -- so this refuses
    rather than producing a config the env would reject with a less useful
    message.
    """
    if entrant.kind == "baseline":
        return OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": entrant.name}
        )
    raise NotImplementedError(
        f"entrant {entrant.name!r} is a checkpoint, and seating one on the "
        "opponent side needs the 'model' opponent policy (phase 04 wave 3). "
        "Until then a checkpoint can be rated as entrant A only."
    )


def require_symmetric(config: WargameEnvConfig) -> None:
    """Refuse a scenario where the two seats are not the same game.

    Rating compares play strength, so the seats have to be comparable. Unequal
    armies are also what breaks the observation encoding: `_alive_feature_index`
    counts backwards assuming the trailing expected-damage block is exactly
    `n_opponents` wide, and `_alive_from_features` falls back to **all-alive**
    when the index lands out of range -- it degrades silently rather than
    raising. Removing this restriction is the size-agnostic policy work.
    """
    if config.number_of_wargame_models != config.number_of_opponent_models:
        raise AsymmetricScenarioError(
            "a rated scenario needs equal armies: "
            f"{config.number_of_wargame_models} player models against "
            f"{config.number_of_opponent_models} opponent models"
        )


def play_leg(
    entrant_a: Entrant,
    entrant_b: Entrant,
    base_config: WargameEnvConfig,
    leg: Leg,
    seeds: Sequence[int],
    combat_seeds: Sequence[int],
) -> LegResult:
    """Play one leg and return its per-layout rows.

    One env for the whole leg, which is what lets a network entrant load its
    checkpoint once rather than once per layout.
    """
    require_symmetric(base_config)
    config = with_opponent(
        config_for_leg(base_config, leg), opponent_config_for(entrant_b)
    )
    # Constructed directly rather than through `model.common.factory`, which is
    # a bare `WargameEnv(...)` wrapper that happens to live in the model layer.
    # Importing it here would point `rating -> model` and make the arrow
    # two-way. Registering an opponent policy that *does* need the model layer
    # is the entry point's job, not this module's.
    env = WargameEnv(config=config, renderer=None)
    try:
        result = evaluate_selector(
            entrant_a.build(env),
            env,
            list(seeds),
            name=entrant_a.name,
            combat_seeds=list(combat_seeds),
        )
    finally:
        env.close()

    return LegResult(
        entrant_a=entrant_a.name,
        entrant_b=entrant_b.name,
        leg=leg,
        layout_seeds=tuple(seeds),
        combat_seeds=tuple(combat_seeds),
        margins=tuple(float(margin) for margin in result.vp_margin_per_episode),
        wins=tuple(float(win) for win in result.win_per_episode),
        objectives_held=tuple(
            float(held) for held in result.objectives_held_per_episode
        ),
        # Carried because a `vp_margin` on its own is a result plus an unstated
        # claim that the moves earning it were legal, and only this column
        # carries the claim.
        coherency_rate=result.coherency_rate,
    )


def play_pairing(
    entrant_a: Entrant,
    entrant_b: Entrant,
    base_config: WargameEnvConfig,
    n_layouts: int,
    seed_base: int = ELO_SEED_BASE,
) -> list[LegResult]:
    """All four legs of one pairing, on identical layouts and dice."""
    seeds = layout_seeds(n_layouts, seed_base)
    combat = combat_seeds_for(seeds)
    return [
        play_leg(entrant_a, entrant_b, base_config, leg, seeds, combat)
        for leg in FOUR_LEGS
    ]
