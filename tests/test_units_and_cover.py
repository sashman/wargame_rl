"""What blocks sight, and what puts a target in cover.

**Models do not block sight — only terrain does.** That is a deliberate
divergence from `docs/rules/06-visibility-and-damage.md`, recorded in the gap
map: no model here has a silhouette that is actually opaque, so a line that
clips a base can be drawn in practice. It takes the rules' *ignore your own
unit* exemption with it, since that exemption exists only to stop a squad
shielding itself from its own occlusion.

`base_radius` still shapes sight, in one way: it sets the width of the corridor
traced between two models. A target only partly visible along that corridor is
**in cover**, which worsens the attack by one — so cover is now a fact about
terrain, and at radius 0 the three rays coincide and cover cannot happen at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain import rules_constants
from wargame_rl.wargame.envs.domain.shooting import (
    DefenderStats,
    expected_damage,
    resolve_shooting,
)
from wargame_rl.wargame.envs.domain.sight import CLEAR, COVER, HIDDEN
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.env_components.shooting_masks import (
    compute_unit_shooting_masks,
)
from wargame_rl.wargame.envs.types import (
    NON_MOVEMENT_PHASES,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    TerrainPieceConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

RADIUS = 1.0


class _Weapon:
    """Minimal WeaponStats, so the roll can be pinned without config validation."""

    def __init__(self, ballistic_skill: int) -> None:
        self.attacks = 200
        self.ballistic_skill = ballistic_skill
        self.strength = 4
        self.ap = 0
        self.damage = 1


def _sight_env(
    player: list[tuple[float, float, int]],
    opponent: list[tuple[float, float, int]],
    terrain: list[TerrainPieceConfig] | None = None,
    skip_phases: list[BattlePhase] | None = None,
) -> WargameEnv:
    """Board with models at fixed spots, each carrying a declared group.

    `skip_phases` defaults to the config's own default (shooting skipped), so
    every sight test keeps stepping exactly as it did.
    """
    if skip_phases is None:
        skip_phases = list(NON_MOVEMENT_PHASES)
    return WargameEnv(
        config=WargameEnvConfig(
            board_width=60,
            board_height=40,
            number_of_wargame_models=len(player),
            number_of_opponent_models=len(opponent),
            number_of_objectives=1,
            number_of_battle_rounds=2,
            base_radius=RADIUS,
            terrain=terrain,
            skip_phases=skip_phases,
            models=[
                ModelConfig(
                    x=int(x), y=int(y), group_id=u, weapons=[WeaponProfile(range=50)]
                )
                for x, y, u in player
            ],
            opponent_models=[
                ModelConfig(x=int(x), y=int(y), group_id=u) for x, y, u in opponent
            ],
            opponent_policy=OpponentPolicyConfig(type="random"),
            render_mode=None,
        )
    )


def _visibility(env: WargameEnv) -> np.ndarray:
    return env.visibility_between(
        np.array([m.location for m in env.wargame_models], dtype=float),
        np.array([m.location for m in env.opponent_models], dtype=float),
        origin_models=env.wargame_models,
        target_models=env.opponent_models,
    )


class TestModelsDoNotOcclude:
    def test_a_solid_screen_of_enemies_does_not_hide_the_target(self) -> None:
        """Three enemy bases dead on the line, and the target behind is still
        fully visible — the divergence this file's docstring records."""
        env = _sight_env(
            [(5, 20, 0)],
            [(20, 19, 1), (20, 20, 1), (20, 21, 1), (40, 20, 2)],
        )
        env.reset(seed=0)

        assert _visibility(env)[0, 3] == CLEAR

    def test_a_squadmate_in_front_does_not_block_the_shot(self) -> None:
        """Through the mask the game actually builds, not through a helper.

        This was live until models stopped occluding. The unit exemption was
        threaded through a separate seam that only the exposure tracker called,
        while both real shooting masks passed `line_of_sight_matrix` — so a
        model *was* blocked by the squadmate standing in front of it, on both
        sides, for as long as bases occluded anything.
        """
        env = _sight_env(
            [(5, 20, 0), (20, 19, 0), (20, 20, 0), (20, 21, 0)], [(40, 20, 0)]
        )
        env.reset(seed=0)
        positions = np.array([m.location for m in env.wargame_models], dtype=float)
        enemies = np.array([m.location for m in env.opponent_models], dtype=float)

        mask = compute_unit_shooting_masks(
            positions,
            enemies,
            np.ones(4, dtype=bool),
            np.ones(1, dtype=bool),
            env.player_max_ranges,
            env.line_of_sight_matrix,
            np.array([m.group_id for m in env.opponent_models], dtype=int),
            1,
            player_advanced=np.zeros(4, dtype=bool),
            engagement_range=env.rules_quantities.engagement_range,
            base_diameter=2.0 * env.rules_quantities.base_radius,
        )

        assert bool(mask[0, 0]), "a squadmate must not block the shot"

    def test_terrain_still_blocks(self) -> None:
        """The other half of the claim: dropping model occlusion changes nothing
        about the ruin between them."""
        wall = TerrainPieceConfig(outline=[(19, 10), (22, 10), (22, 30), (19, 30)])
        env = _sight_env([(5, 20, 0)], [(40, 20, 1)], terrain=[wall])
        env.reset(seed=0)

        assert _visibility(env)[0, 0] == HIDDEN


class TestCover:
    def test_a_target_at_a_ruins_edge_is_in_cover(self) -> None:
        """Cover is now a fact about terrain: one edge of the corridor clipped
        by a ruin, the centre line and the other edge clear."""
        # The corridor is RADIUS wide either side of the line y = 20, so a ruin
        # occupying y in [20.5, 23] blocks only the upper ray.
        ledge = TerrainPieceConfig(outline=[(19, 20.5), (22, 20.5), (22, 23), (19, 23)])
        env = _sight_env([(5, 20, 0)], [(40, 20, 1)], terrain=[ledge])
        env.reset(seed=0)

        assert _visibility(env)[0, 0] == COVER

    def test_cover_worsens_the_hit_roll_by_one(self) -> None:
        """And the unmodified 6 still hits, so cover is never an absolute shield."""
        defender = DefenderStats(toughness=3, save=7)
        weapon = _Weapon(ballistic_skill=4)

        open_ground = resolve_shooting(
            weapon, defender, np.random.default_rng(1), in_cover=False
        )
        in_cover = resolve_shooting(
            weapon, defender, np.random.default_rng(1), in_cover=True
        )

        assert rules_constants.COVER_RANGED_SKILL_PENALTY == 1
        assert in_cover.hits < open_ground.hits
        assert in_cover.hits > 0

    def test_a_shot_into_cover_reports_the_expectation_it_was_rolled_under(
        self,
    ) -> None:
        """The analytical fields must be computed under the same rules as the dice.

        `expected_damage` predates cover and read the weapon's Ranged Skill
        straight off the profile, so every shot into terrain was recorded with
        the expectation for a target standing in the open — the one number a
        reader would use to decide whether the dice had been kind.
        """
        ledge = TerrainPieceConfig(outline=[(19, 20.5), (22, 20.5), (22, 23), (19, 23)])
        env = _sight_env(
            [(5, 20, 0)],
            [(40, 20, 1)],
            terrain=[ledge],
            skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        )
        env.reset(seed=0)
        env.step(WargameEnvAction(actions=[STAY_ACTION]))  # movement -> shooting

        shooting_slice = env.player_action_handler.shooting_slice
        assert shooting_slice is not None
        env.step(WargameEnvAction(actions=[shooting_slice.start + 1]))

        results = env.last_player_shooting_results
        assert results, "the covered unit must actually have been shot at"
        assert all(r.in_cover for r in results)

        recorded = env.to_snapshot().player_combat_results[0]
        models = env.config.models
        assert models is not None
        weapon = models[0].weapons[0]
        defender = DefenderStats(toughness=3, save=4)
        assert recorded.in_cover
        assert recorded.expected_damage == pytest.approx(
            expected_damage(weapon, defender, in_cover=True)
        )
        assert recorded.expected_damage < expected_damage(weapon, defender)

    def test_no_base_means_no_cover_ever(self) -> None:
        """The property that makes all of this a no-op for older configs.

        With no base the two edge rays coincide with the centre ray, so a pair
        is CLEAR or HIDDEN and never in between — which is why the golden gates
        still pass unchanged on every config that predates model bases.
        """
        ledge = TerrainPieceConfig(outline=[(19, 20.5), (22, 20.5), (22, 23), (19, 23)])
        env = _sight_env([(5, 20, 0)], [(40, 20, 1)], terrain=[ledge])
        env.config.base_radius = 0.0
        for model in (*env.wargame_models, *env.opponent_models):
            model.base_radius = 0.0
        env.reset(seed=3)

        assert not (_visibility(env) == COVER).any()


@pytest.mark.parametrize("seed", range(4))
def test_visibility_is_symmetric(seed: int) -> None:
    """A sees B exactly as well as B sees A.

    The corridor is as wide as the wider of the pair's two bases, which is
    symmetric in the pair, and the geometry is a segment. `firepower_ratio`
    reads an exposed model as one that can also fire, so an asymmetry here would
    make that metric count two different populations.
    """
    env = WargameEnv(
        config=WargameEnvConfig(
            board_width=60,
            board_height=40,
            number_of_wargame_models=6,
            number_of_opponent_models=6,
            number_of_objectives=2,
            number_of_battle_rounds=3,
            base_radius=RADIUS,
            opponent_policy=OpponentPolicyConfig(type="random"),
        )
    )
    env.reset(seed=seed)
    players = np.array([m.location for m in env.wargame_models], dtype=float)
    opponents = np.array([m.location for m in env.opponent_models], dtype=float)

    forward = env.visibility_between(
        players,
        opponents,
        origin_models=env.wargame_models,
        target_models=env.opponent_models,
    )
    backward = env.visibility_between(
        opponents,
        players,
        origin_models=env.opponent_models,
        target_models=env.wargame_models,
    )

    assert (forward == backward.T).all()
