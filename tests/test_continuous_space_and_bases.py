"""The board is continuous and models have physical bases.

Two quantisations were removed here, and they were destroying information rather
than approximating it: `np.rint` on the displacement table collapsed the 96
movement actions of the 25v25 space to 80 distinct outcomes and made a "speed 1"
diagonal travel 41% further than a speed 1 orthogonal move, while `.astype(int)`
truncated the vector to the objective — the single most informative feature the
policy has.

Both failures are silent. An integer array assigned a float truncates with no
exception and no failing test, which is why these are pinned at the level of
*distance travelled* rather than at the level of a dtype.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

# 32mm across at one inch per unit, the rules' infantry base.
INFANTRY_RADIUS = 0.63


def _config(**overrides: object) -> WargameEnvConfig:
    base: dict[str, object] = {
        "board_width": 40,
        "board_height": 40,
        "number_of_wargame_models": 6,
        "number_of_objectives": 2,
        "number_of_battle_rounds": 3,
        "render_mode": None,
    }
    base.update(overrides)
    return WargameEnvConfig(**base)  # type: ignore[arg-type]


class TestExactDisplacement:
    """`np.rint` is gone from the displacement table."""

    def test_every_movement_action_has_a_distinct_outcome(self) -> None:
        """Rounding collapsed 96 actions to 80, 16 of them duplicates.

        A duplicated action is not merely wasteful — it is a pair the policy can
        never learn to tell apart, in the one head that steers.
        """
        handler = ActionHandler(_config(), n_shoot_targets=0)

        outcomes = {
            tuple(handler._decode_action(action))
            for action in range(1, handler.n_move_actions + 1)
        }

        assert len(outcomes) == handler.n_move_actions

    def test_a_speed_bin_travels_the_same_distance_in_every_direction(self) -> None:
        """The defect this replaces: a speed 1 diagonal travelled 1.414.

        Under rounding, "speed" meant a different distance depending on which way
        you faced, so the cheapest way to cover ground was to move diagonally.
        """
        config = _config(n_movement_angles=16, n_speed_bins=6)
        handler = ActionHandler(config, n_shoot_targets=0)
        max_speed = resolve_rules_quantities(config).max_move_speed

        for speed_index in range(config.n_speed_bins):
            expected = max_speed * (speed_index + 1) / config.n_speed_bins
            lengths = [
                float(
                    np.linalg.norm(
                        handler._decode_action(
                            handler.encode_action(angle_index, speed_index)
                        )
                    )
                )
                for angle_index in range(config.n_movement_angles)
            ]
            assert lengths == pytest.approx([expected] * len(lengths))

    def test_a_model_ends_up_off_the_grid(self) -> None:
        """The point of the whole change, asserted on the state and not the type.

        A position that is always a whole number would pass every dtype check
        while the board stayed a chessboard in everything but declaration.
        """
        env = WargameEnv(config=_config())
        env.reset(seed=7)
        env.action_space.seed(7)

        for _ in range(6):
            env.step(WargameEnvAction(actions=env.action_space.sample()))

        locations = np.array([m.location for m in env.wargame_models], dtype=float)
        assert not np.all(locations == np.round(locations))


class TestModelBases:
    """A model occupies ground, and two models cannot occupy the same ground."""

    def test_bases_do_not_overlap_at_placement(self) -> None:
        env = WargameEnv(config=_config(base_radius=INFANTRY_RADIUS))

        for seed in range(12):
            env.reset(seed=seed)
            locations = np.array([m.location for m in env.wargame_models], dtype=float)
            separations = np.linalg.norm(
                locations[:, None, :] - locations[None, :, :], axis=2
            )
            np.fill_diagonal(separations, np.inf)
            assert separations.min() >= 2.0 * INFANTRY_RADIUS

    def test_a_zone_too_small_for_the_army_fails_loudly(self) -> None:
        """Silently stacking models would be the worse failure.

        A 5x5 board's deployment zone is one unit wide and a 32mm base is 1.26
        across, so the small demo configs genuinely cannot hold an army — the
        error says so with the numbers rather than producing a legal-looking
        layout with every model on the same spot.
        """
        env = WargameEnv(
            config=_config(
                board_width=6,
                board_height=6,
                number_of_wargame_models=20,
                base_radius=INFANTRY_RADIUS,
                deployment_zone=(0, 0, 2, 6),
                opponent_deployment_zone=(4, 0, 6, 6),
            )
        )

        with pytest.raises(RuntimeError, match="too small for the army"):
            env.reset(seed=0)

    def test_no_base_leaves_placement_free(self) -> None:
        """Radius 0 is the historical behaviour and must stay unconstrained."""
        env = WargameEnv(config=_config(base_radius=0.0))
        env.reset(seed=0)

        assert all(m.base_radius == 0.0 for m in env.wargame_models)

    def test_a_base_cannot_hang_off_the_table(self) -> None:
        env = WargameEnv(config=_config(base_radius=INFANTRY_RADIUS))
        env.reset(seed=3)
        env.action_space.seed(3)

        for _ in range(12):
            env.step(WargameEnvAction(actions=env.action_space.sample()))

        locations = np.array([m.location for m in env.wargame_models], dtype=float)
        assert (locations >= INFANTRY_RADIUS).all()
        assert (locations[:, 0] <= env.board_width - INFANTRY_RADIUS).all()
        assert (locations[:, 1] <= env.board_height - INFANTRY_RADIUS).all()

    def test_objective_range_is_measured_from_the_base_edge(self) -> None:
        """The `norms_offset` seam: one subtraction, no branch downstream.

        Every consumer still asks `norms_offset <= obj_radii`, so the reward, VP
        and criteria layers need no knowledge that models have a size at all.
        """
        env = WargameEnv(config=_config(base_radius=INFANTRY_RADIUS))
        env.reset(seed=1)

        cache = compute_distances(env.wargame_models, env.objectives)

        assert cache.model_obj_norms_offset == pytest.approx(
            np.maximum(cache.model_obj_norms - INFANTRY_RADIUS, 0.0)
        )
