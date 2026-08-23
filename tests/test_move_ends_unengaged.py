"""A move must END unengaged, and passing through must stay legal.

`09-movement-phase.md`: "After moving: the unit must be unengaged."
`03-moving.md`: "Passing through an enemy unit's engagement range during a move
does **not** make the moving unit engaged. Only where it *ends* matters."

⚠ The first attempt at this inflated enemy blocker radii by the engagement
range, which turns an end-state rule into an impassable wall: review measured
**87% of opponent-held objectives with no legal spot at all**. These tests pin
the distinction that failure was made of.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.movement import back_off_to_unengaged


def _ring(x: float, y: float, reach: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([[x, y]], dtype=float), np.array([reach], dtype=float)


def test_an_endpoint_inside_the_ring_is_pulled_back_along_its_own_heading() -> None:
    """Backed off to just outside, and never sideways."""
    centres, reach = _ring(10.0, 0.0, 2.0)

    end = back_off_to_unengaged(
        np.array([0.0, 0.0]), np.array([9.0, 0.0]), centres, reach
    )

    assert end[1] == 0.0, "the model was displaced off its heading"
    assert np.linalg.norm(end - centres[0]) >= 2.0 - 1e-9, "still engaged"
    assert end[0] < 9.0, "it did not move back at all"


def test_passing_clean_through_is_legal() -> None:
    """The rule the reverted attempt broke: only the endpoint counts."""
    centres, reach = _ring(5.0, 0.0, 2.0)

    end = back_off_to_unengaged(
        np.array([0.0, 0.0]), np.array([10.0, 0.0]), centres, reach
    )

    assert np.allclose(end, [10.0, 0.0]), (
        "a move that ends clear of the ring was shortened -- this is the "
        "path-constraint bug that cost 87% of contested objectives"
    )


def test_an_endpoint_already_clear_is_untouched() -> None:
    """No ring in the way means bit-identical behaviour, so nothing else moves."""
    centres, reach = _ring(50.0, 50.0, 2.0)
    resolved = np.array([3.0, 4.0])

    end = back_off_to_unengaged(np.array([0.0, 0.0]), resolved, centres, reach)

    assert np.array_equal(end, resolved)


def test_with_no_enemies_the_move_is_returned_unchanged() -> None:
    """The overwhelmingly common path must allocate and decide nothing."""
    resolved = np.array([7.0, -2.0])

    end = back_off_to_unengaged(
        np.array([0.0, 0.0]),
        resolved,
        np.empty((0, 2), dtype=float),
        np.empty(0, dtype=float),
    )

    assert np.array_equal(end, resolved)


def test_a_move_with_no_legal_endpoint_is_not_made() -> None:
    """The rules' own remedy: return the model to where it started."""
    centres, reach = _ring(0.0, 0.0, 5.0)

    end = back_off_to_unengaged(
        np.array([0.0, 0.0]), np.array([3.0, 0.0]), centres, reach
    )

    assert np.array_equal(end, np.array([0.0, 0.0]))


def test_the_legal_set_is_not_an_interval_and_the_walk_handles_it() -> None:
    """Two rings with a gap: back off past the near one into the gap, not to zero.

    A bisection would fail here, which is the documented reason the movement
    solver's first rewrite was reverted.
    """
    centres = np.array([[4.0, 0.0], [9.0, 0.0]], dtype=float)
    reach = np.array([1.0, 1.0], dtype=float)

    end = back_off_to_unengaged(
        np.array([0.0, 0.0]), np.array([9.5, 0.0]), centres, reach
    )

    assert 5.0 <= end[0] <= 8.0, f"expected the gap between the rings, got {end[0]}"
    for centre, radius in zip(centres, reach):
        assert np.linalg.norm(end - centre) >= radius - 1e-9


def test_no_two_models_end_a_movement_phase_overlapping() -> None:
    """The composition test the unit tests above structurally cannot perform.

    ⚠ **This is the bug that shipped.** Backing off walks the endpoint into
    ground `resolve_move` had already cleared as passable-but-not-endable, so a
    model rescued from an engagement ring came to rest INSIDE a friendly base:
    **0.18% of friendly pairs, worst penetration 0.68"**, against 0.0000% with
    the rule off. `movement.py`'s own first line is that no two models may end a
    move overlapping.

    Every other test in this file calls the pure function. None of them could
    ever have seen this — the same defect this project already paid for on the
    joint decoder, where "seven tests covered the module and none called
    `env.step`, so every one asserted the decoder against its own relaxation".
    So this one drives the real env.
    """
    from pathlib import Path

    from scripts.measure_maps import config_for_map, load_maps
    from scripts.scenario_overrides import load_env_config
    from wargame_rl.wargame.envs.baseline.evaluate import selector_for
    from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
    from wargame_rl.wargame.envs.types.game_timing import BattlePhase
    from wargame_rl.wargame.model.common.factory import create_environment

    config = load_env_config("configs/evaluation/25v25_maps_advance_refereed.yaml")
    terrain_map = load_maps(Path("configs/evaluation/maps_heldout"))[0]
    env = create_environment(env_config=config_for_map(config, terrain_map))
    select = selector_for(build_baseline_policy("squad_march_take"))

    observation, _ = env.reset(seed=700000)
    done = False
    worst = 0.0
    while not done:
        was_movement = env.game_clock_state.phase is BattlePhase.movement
        observation, _r, done, _t, _i = env.step(select(observation, env))
        if not was_movement:
            continue
        alive = [m for m in env.player_models if m.is_alive]
        if len(alive) < 2:
            continue
        locations = np.array([m.location for m in alive], dtype=float)
        radii = np.array([m.base_radius for m in alive], dtype=float)
        gaps = np.linalg.norm(locations[:, None, :] - locations[None, :, :], axis=2)
        needed = radii[:, None] + radii[None, :]
        upper = np.triu_indices(len(alive), 1)
        worst = max(worst, float(np.max(needed[upper] - gaps[upper])))
    env.close()

    assert worst <= 1e-9, (
        f"two models ended a movement phase overlapping by {worst:.3f} in — the "
        "back-off placed an endpoint inside a base"
    )
