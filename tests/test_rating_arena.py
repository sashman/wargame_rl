"""Four balanced legs, scored by the same loop `measure-baselines` uses."""

from __future__ import annotations

import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.baseline.evaluate import evaluate_selector
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.rating.arena import (
    AsymmetricScenarioError,
    LegResult,
    opponent_config_for,
    play_leg,
    play_pairing,
    require_symmetric,
)
from wargame_rl.wargame.rating.entrant import Entrant
from wargame_rl.wargame.rating.schedule import (
    FOUR_LEGS,
    Leg,
    Seat,
    Zone,
    combat_seeds_for,
    config_for_leg,
    layout_seeds,
    with_opponent,
)
from wargame_rl.wargame.selectors import build_action_selector

# 4v4 with explicit deployment zones and a real opponent army: the smallest
# shipped config a rated match can actually be played on.
ARENA_CONFIG = "configs/dev/4v4_two_phases.yaml"
N_LAYOUTS = 3


def _base_config() -> WargameEnvConfig:
    with open(ARENA_CONFIG) as handle:
        config: WargameEnvConfig = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    config.render_mode = None
    return config


def _entrant(name: str) -> Entrant:
    return Entrant(
        name=name,
        build=lambda env: build_action_selector(name, env).select,
        kind="baseline",
    )


def test_a_leg_returns_one_row_per_layout() -> None:
    seeds = layout_seeds(N_LAYOUTS)

    result = play_leg(
        _entrant("squad_march"),
        _entrant("random"),
        _base_config(),
        FOUR_LEGS[0],
        seeds,
        combat_seeds_for(seeds),
    )

    assert isinstance(result, LegResult)
    assert len(result.margins) == N_LAYOUTS
    assert len(result.wins) == N_LAYOUTS
    assert result.layout_seeds == tuple(seeds)


def test_a_leg_is_scored_by_evaluate_selector() -> None:
    """The arena wraps the scoring loop; it does not reimplement it.

    Two implementations of "score a policy over seeds" drifting apart is
    exactly the class of defect `measure_paired_policies` documents guarding
    against -- and it would silently break the cross-check between a rating
    and a `measure-paired` number, which is the only thing tying the Elo scale
    to numbers this repo already trusts.
    """
    base = _base_config()
    leg = FOUR_LEGS[0]
    seeds = layout_seeds(N_LAYOUTS)
    combat = combat_seeds_for(seeds)
    entrant_a, entrant_b = _entrant("squad_march"), _entrant("random")

    from_arena = play_leg(entrant_a, entrant_b, base, leg, seeds, combat)

    config = with_opponent(config_for_leg(base, leg), opponent_config_for(entrant_b))
    env = WargameEnv(config=config, renderer=None)
    direct = evaluate_selector(
        entrant_a.build(env), env, list(seeds), "squad_march", combat_seeds=list(combat)
    )
    env.close()

    assert from_arena.margins == tuple(
        float(margin) for margin in direct.vp_margin_per_episode
    )


def test_a_pairing_plays_exactly_the_four_legs() -> None:
    results = play_pairing(
        _entrant("squad_march"), _entrant("random"), _base_config(), N_LAYOUTS
    )

    assert len(results) == 4
    assert {(r.leg.sigma_zone, r.leg.sigma_turn) for r in results} == {
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    }


def test_every_leg_of_a_pairing_shares_its_seeds() -> None:
    """Identical layouts and dice across the four legs, so the only thing that
    differs is the axis under test."""
    results = play_pairing(
        _entrant("squad_march"), _entrant("random"), _base_config(), N_LAYOUTS
    )

    assert len({r.layout_seeds for r in results}) == 1
    assert len({r.combat_seeds for r in results}) == 1


def test_a_leg_carries_its_coherency() -> None:
    """A `vp_margin` on its own is a result plus an unstated claim that the
    moves earning it were legal, and only this column carries the claim."""
    seeds = layout_seeds(N_LAYOUTS)

    result = play_leg(
        _entrant("squad_march"),
        _entrant("random"),
        _base_config(),
        FOUR_LEGS[0],
        seeds,
        combat_seeds_for(seeds),
    )

    assert result.coherency_rate is not None


def test_swapping_the_zone_changes_the_result() -> None:
    """If the zone axis were inert, every rating's `h_zone` would be noise. The
    schedule refuses configs where the swap does nothing; this is the
    behavioural confirmation that it does something when it is allowed."""
    base = _base_config()
    seeds = layout_seeds(N_LAYOUTS)
    combat = combat_seeds_for(seeds)
    args = (_entrant("squad_march"), _entrant("random"), base)

    zone_one = play_leg(*args, Leg(Zone.zone_1, Seat.a), seeds, combat)
    zone_two = play_leg(*args, Leg(Zone.zone_2, Seat.a), seeds, combat)

    assert zone_one.margins != zone_two.margins


def test_an_asymmetric_scenario_is_refused() -> None:
    """Unequal armies also break the observation encoding: `_alive_feature_index`
    counts backwards assuming the trailing block is `n_opponents` wide, and the
    fallback is **all-alive** rather than an exception."""
    base = _base_config()
    base.number_of_opponent_models = base.number_of_wargame_models + 1
    base.opponent_models = None

    with pytest.raises(AsymmetricScenarioError, match="equal armies"):
        require_symmetric(base)


def test_a_checkpoint_is_seated_through_the_model_policy() -> None:
    """Named in the config rather than installed on a live env, so a rated match
    is reproducible from its provenance -- the recorded config says which
    checkpoint played and how it was decoded."""
    checkpoint = Entrant(
        name="run-armA",
        build=lambda env: (_ for _ in ()).throw(AssertionError("not called")),
        kind="checkpoint",
        source="checkpoints/run/last.ckpt",
        decode_topk=3,
    )

    config = opponent_config_for(checkpoint)

    assert config.type == "model"
    assert config.params["checkpoint"] == "checkpoints/run/last.ckpt"
    assert config.params["decode_topk"] == 3


def test_a_checkpoint_entrant_without_a_path_is_refused() -> None:
    nameless = Entrant(
        name="run-armA",
        build=lambda env: (_ for _ in ()).throw(AssertionError("not called")),
        kind="checkpoint",
    )

    with pytest.raises(ValueError, match="no source path"):
        opponent_config_for(nameless)
