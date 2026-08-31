"""A rated leg is a config transform, and it refuses to be inert.

Both imbalance axes are already config fields, so the four-leg schedule needs no
environment code at all: `turn_order` decides who moves first, and A's zone is a
swap of the two deployment-zone fields.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import (
    ModelConfig,
    OpponentPolicyConfig,
    TurnOrder,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config.terrain import MapPoolConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.rating.schedule import (
    ELO_SEED_BASE,
    FOUR_LEGS,
    InertLegError,
    Leg,
    Seat,
    Zone,
    combat_seeds_for,
    config_for_leg,
    layout_seeds,
    pairings,
    with_opponent,
)

TINY_CONFIG = "configs/dev/tiny.yaml"


def _base_config() -> WargameEnvConfig:
    """A rateable config: explicit zones, no fixed positions."""
    with open(TINY_CONFIG) as handle:
        config: WargameEnvConfig = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    config.render_mode = None
    config.deployment_zone = (0, 0, 3, config.board_height)
    config.opponent_deployment_zone = (
        config.board_width - 3,
        0,
        config.board_width,
        config.board_height,
    )
    config.models = None
    config.opponent_models = None
    return config


def test_the_schedule_is_balanced_in_both_axes() -> None:
    """Balanced by construction, so the ratings are unbiased even before the
    advantage terms are fitted."""
    assert len(FOUR_LEGS) == 4
    assert sum(leg.sigma_zone for leg in FOUR_LEGS) == 0.0
    assert sum(leg.sigma_turn for leg in FOUR_LEGS) == 0.0


def test_both_axes_vary_independently() -> None:
    """The property that makes `h_zone` and `h_turn` separately identifiable.

    If the two columns were proportional, only their sum would be measurable --
    which `fit_ratings` refuses outright rather than reporting a split.
    """
    zone = np.array([leg.sigma_zone for leg in FOUR_LEGS])
    turn = np.array([leg.sigma_turn for leg in FOUR_LEGS])

    assert float(zone @ turn) == 0.0


@pytest.mark.parametrize(
    ("first_mover", "expected"),
    [(Seat.a, TurnOrder.player), (Seat.b, TurnOrder.opponent)],
)
def test_the_first_mover_is_the_turn_order_field(
    first_mover: Seat, expected: TurnOrder
) -> None:
    """Entrant A sits on the player seat, so 'A moves first' is exactly
    `turn_order: player`. No new environment code is involved."""
    config = config_for_leg(_base_config(), Leg(Zone.zone_1, first_mover))

    assert config.turn_order == expected


def test_zone_two_swaps_the_deployment_zones() -> None:
    base = _base_config()

    swapped = config_for_leg(base, Leg(Zone.zone_2, Seat.a))

    assert swapped.deployment_zone == base.opponent_deployment_zone
    assert swapped.opponent_deployment_zone == base.deployment_zone


def test_a_leg_does_not_mutate_the_base_config() -> None:
    """One base config feeds four legs; mutating it would make leg four inherit
    leg three's zones."""
    base = _base_config()
    original = (base.deployment_zone, base.opponent_deployment_zone, base.turn_order)

    for leg in FOUR_LEGS:
        config_for_leg(base, leg)

    assert (base.deployment_zone, base.opponent_deployment_zone, base.turn_order) == (
        original
    )


def test_default_deployment_zones_are_refused() -> None:
    """`battle_factory.from_config` derives the zones when the fields are
    `None`, so swapping two `None`s is a **silent** no-op -- and `h_zone` would
    then be fitted from noise and reported as a measurement."""
    base = _base_config()
    base.deployment_zone = None

    with pytest.raises(InertLegError, match="silent no-op"):
        config_for_leg(base, Leg(Zone.zone_2, Seat.a))


def test_fixed_model_positions_are_refused() -> None:
    """With explicit coordinates the armies deploy where they are told and the
    zones are decorative."""
    base = _base_config()
    base.models = [
        ModelConfig(x=1, y=index + 1) for index in range(base.number_of_wargame_models)
    ]

    assert base.has_fixed_model_positions
    with pytest.raises(InertLegError, match="inert"):
        config_for_leg(base, Leg(Zone.zone_2, Seat.a))


def test_a_map_pool_is_refused() -> None:
    """A drawn table's own deployment outlines govern placement, and the config
    rectangles are ignored outright, so the swap is a **total no-op**.

    Measured on `25v25_maps_two_mode.yaml`: both armies' positions and the
    outline itself are bit-identical across the swap on 10 of 10 layouts. Same
    failure as the `None` case, reached by a different route -- `h_zone` would
    be fitted from noise on exactly the configs that train, and on this pool its
    true value is zero anyway, the two outlines being 180-degree rotations of
    each other on 45 of 45 tables.
    """
    base = _base_config()
    base.map_pool = MapPoolConfig(directory="configs/evaluation/maps")

    with pytest.raises(InertLegError, match="deployment outlines"):
        config_for_leg(base, Leg(Zone.zone_2, Seat.a))


def test_a_map_pool_is_refused_on_every_leg_including_the_unswapped_ones() -> None:
    """Zone 1 legs do not swap anything, so it would be tempting to let them
    through -- but a pairing needs all four legs to identify `h_zone`, and two
    legs that silently measure nothing are worse than a refusal."""
    base = _base_config()
    base.map_pool = MapPoolConfig(directory="configs/evaluation/maps")

    for leg in FOUR_LEGS:
        with pytest.raises(InertLegError):
            config_for_leg(base, leg)


def test_the_opponent_is_seated_separately_from_the_leg() -> None:
    """A leg is the two imbalance axes; the opponent is *who is being rated*.
    The ledger's scenario fingerprint drops both for that reason."""
    config = config_for_leg(_base_config(), FOUR_LEGS[0])

    seated = with_opponent(
        config, OpponentPolicyConfig(type="scripted_baseline", params={"baseline": "x"})
    )

    assert seated.opponent_policy is not None
    assert seated.opponent_policy.params["baseline"] == "x"
    assert config.opponent_policy != seated.opponent_policy


def test_the_four_legs_share_terrain_and_objectives() -> None:
    """Only deployment may differ between legs.

    This is also the guard for the `turn_order` layout-stream hazard:
    `_resolve_player_side` draws from `np_random` **only** under
    `turn_order: random`, and it runs before the map-pool draw and
    `place_for_episode`. Both fixed values draw nothing, so the four legs agree
    with each other -- which is what the fit needs. A config whose own
    `turn_order` is `random` is on a *different* layout stream from its own
    `measure-baselines` numbers, which is a caveat for reading a table, not a
    defect in the schedule.
    """
    base = _base_config()
    seed = ELO_SEED_BASE

    objectives: list[tuple[float, ...]] = []
    for leg in FOUR_LEGS:
        env = WargameEnv(config=config_for_leg(base, leg), renderer=None)
        env.reset(seed=seed)
        objectives.append(
            tuple(float(value) for obj in env.objectives for value in obj.location)
        )
        env.close()

    assert len(set(objectives)) == 1


def test_layout_seeds_claim_their_own_band() -> None:
    """Disjoint from rollout 0, baselines 10k, eval 500k, held-out 700k and
    clone 800k, so a rated match is played on layouts nothing else reports on."""
    seeds = layout_seeds(4)

    assert seeds == [900_000, 900_001, 900_002, 900_003]


def test_combat_seeds_are_pinned_apart_from_the_layout() -> None:
    """The dice contribute more outcome spread than the scenario does, so the
    combat stream is fixed across a pairing's four legs rather than left to
    follow from the layout seed."""
    seeds = layout_seeds(3)

    combat = combat_seeds_for(seeds)

    assert len(set(combat) & set(seeds)) == 0
    assert combat == combat_seeds_for(seeds)


def test_pairings_are_unordered_and_listed_once() -> None:
    """Ordered pairs would double the cost for nothing -- the four legs already
    balance both axes, so B-versus-A adds no information."""
    assert pairings(["a", "b", "c"]) == [("a", "b"), ("a", "c"), ("b", "c")]


def test_duplicate_entrants_are_refused() -> None:
    with pytest.raises(ValueError, match="distinct"):
        pairings(["a", "b", "a"])


def test_objectives_can_be_placed_with_the_player_on_the_right() -> None:
    """Regression: the objective band assumed the player deployed on the left.

    `objective_placement` read the free strip as
    `(deployment_zone.x_max, opponent_deployment_zone.x_min)`, which is the gap
    only while the player is the left-hand army. Every shipped config puts it
    there, so the assumption was invisible -- and giving the player the
    right-hand zone made those two numbers the *outer* edges of the board, so
    the range inverted and numpy raised `high - low < 0` at reset.

    A rated schedule plays both zones by construction, which is how this
    surfaced. It is a domain bug, not a scheduling one: any config deploying
    the player on the right hit it.
    """
    base = _base_config()
    base.number_of_opponent_models = base.number_of_wargame_models
    base.opponent_policy = OpponentPolicyConfig(
        type="scripted_advance_to_objective", params={}
    )
    # Objectives must be DRAWN, not authored: `objective_placement` is skipped
    # entirely for a config that fixes them, and every shipped dev config does
    # -- which is why a first version of this test passed with the bug back in.
    base.objectives = None
    base.number_of_objectives = 2

    for leg in FOUR_LEGS:
        env = WargameEnv(config=config_for_leg(base, leg), renderer=None)
        env.reset(seed=ELO_SEED_BASE)
        left, right = sorted(
            (
                float(env.deployment_zone[0]),
                float(env.opponent_deployment_zone[0]),
            )
        )
        assert left < right
        for objective in env.objectives:
            x = float(objective.location[0])
            assert 0.0 <= x <= base.board_width
        env.close()
