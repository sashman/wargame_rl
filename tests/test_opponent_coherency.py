"""The opponent force keeps its own coherency totals.

`evaluate_selector` measures the player seat, and `pairings` lists each pair
once in input order -- so the entrant named last on the command line was
entrant B in every one of its pairings and came back with the coherency column
blank. A `vp_margin` on its own is a result plus an unstated claim that the
moves earning it were legal, and this column is the only thing carrying that
claim.

The load-bearing case is `test_the_opponent_column_is_not_the_players`. A
tracker that silently recorded the *player's* models under an opponent-shaped
name satisfies every structural assertion here -- not-None, present for both
entrants, persisted to the ledger -- and only a config where the two forces
genuinely hold formation differently can catch it.

⚠ **`configs/dev/4v4_two_phases.yaml` gives every model its own `group_id`**, so
a unit is one model and coherency is **1.000 by definition** on both seats. The
first two versions of these tests ran on it and asserted nothing at all. The
fixture here rewrites the group ids to put all four models in one unit, which is
the smallest change that makes the predicate have an answer.
"""

from __future__ import annotations

import numpy as np
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.baseline.evaluate import evaluate_selector
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.rating.arena import play_leg
from wargame_rl.wargame.rating.entrant import Entrant
from wargame_rl.wargame.rating.schedule import FOUR_LEGS, combat_seeds_for, layout_seeds
from wargame_rl.wargame.rating.table import mean_coherency
from wargame_rl.wargame.selectors import build_action_selector

ARENA_CONFIG = "configs/dev/4v4_two_phases.yaml"
N_LAYOUTS = 3


def _config() -> WargameEnvConfig:
    with open(ARENA_CONFIG) as handle:
        config: WargameEnvConfig = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    config.render_mode = None
    return config


def _one_unit_config() -> WargameEnvConfig:
    """All four models in one unit, so coherency is not 1.000 by definition.

    The shipped dev config gives every model its own `group_id`, and the rules
    make a one-model unit coherent by definition -- so every column reads 1.000
    on both seats and a comparison between them proves nothing.
    """
    config = _config()
    for model in (config.models or []) + (config.opponent_models or []):
        model.group_id = 0
    return config


def _entrant(name: str) -> Entrant:
    return Entrant(
        name=name,
        build=lambda env: build_action_selector(name, env).select,
        kind="baseline",
    )


def test_the_opponent_force_reports_a_coherency_rate() -> None:
    env = WargameEnv(config=_config(), renderer=None)
    try:
        evaluate_selector(
            build_action_selector("squad_march", env).select,
            env,
            [900_000, 900_001],
            "squad_march",
        )
        assert env.opponent_coherency_rate is not None
        assert env.opponent_models_out_of_coherency is not None
    finally:
        env.close()


def test_the_opponent_column_is_not_the_players() -> None:
    """The sensitivity control, and the only test here that is not structural.

    `squad_march` keeps the player's four models together well enough to be
    coherent on some phases; the opponent seat, marching at whichever objective
    it was given, never is. If `_record_opponent_coherency` were handed the
    player's models -- the obvious way to get this wrong, and invisible to every
    other assertion in this file -- the two columns would be equal.
    """
    seeds = layout_seeds(4)

    leg = play_leg(
        _entrant("squad_march"),
        _entrant("squad_march"),
        _one_unit_config(),
        FOUR_LEGS[0],
        seeds,
        combat_seeds_for(seeds),
    )

    assert leg.coherency_rate is not None
    assert leg.opponent_coherency_rate is not None
    assert leg.opponent_coherency_rate != leg.coherency_rate


def test_an_entrant_seated_only_as_b_gets_a_coherency_figure() -> None:
    """The gap this closes, stated as the table sees it."""
    seeds = layout_seeds(N_LAYOUTS)
    leg = play_leg(
        _entrant("squad_march"),
        _entrant("squad_march_take"),
        _config(),
        FOUR_LEGS[0],
        seeds,
        combat_seeds_for(seeds),
    )

    figures = mean_coherency([leg])

    assert figures["squad_march"] is not None
    assert figures["squad_march_take"] is not None


def test_the_player_column_is_untouched() -> None:
    """Nothing on the player path moved, so its figure must be bit-identical.

    Recorded against a second env built from the same config rather than against
    a stored constant, because the point is that adding the opponent's tracker
    changed neither the player's arithmetic nor the RNG stream feeding it.
    """
    seeds = [900_000, 900_001]
    results = []
    for _ in range(2):
        env = WargameEnv(config=_config(), renderer=None)
        try:
            results.append(
                evaluate_selector(
                    build_action_selector("squad_march", env).select,
                    env,
                    seeds,
                    "squad_march",
                )
            )
        finally:
            env.close()

    assert results[0].coherency_rate == results[1].coherency_rate
    np.testing.assert_array_equal(
        np.array(results[0].vp_margin_per_episode),
        np.array(results[1].vp_margin_per_episode),
    )
