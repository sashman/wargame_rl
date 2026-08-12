"""A scripted policy walks onto a terrain objective, not onto its centre point.

`WargameObjective.radius_size` is **0.0 for an area objective by design** — its
extent is the outline, and distance is reported to that edge through the
`norms_offset` seam. Every "have I arrived" test written against that field
therefore waits for the model to reach the *centroid* exactly.

On a marker objective that is merely the middle of a small disc and nothing goes
wrong. On a terrain objective the size of a real ruin it means marching a whole
squad onto a single point — and once bases are real they collide there, so the
models behind stop dead in the open.

Nothing raised, and no test caught it: the policies were doing exactly what they
were told, against a field that means something different for the two kinds of
objective.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.baseline.policy import objective_extent
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types.config import ObjectiveConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

# A 12 x 12 ruin in the middle of the board — big enough that "inside it" and
# "on its centre" are obviously different places.
BIG_RUIN = [(24.0, 16.0), (36.0, 16.0), (36.0, 28.0), (24.0, 28.0)]


def _env(policy_models: int = 10) -> WargameEnv:
    return WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            board_width=60,
            board_height=44,
            number_of_wargame_models=policy_models,
            number_of_objectives=1,
            max_groups=2,
            number_of_battle_rounds=20,
            deployment_zone=(0, 0, 20, 44),
            objectives=[ObjectiveConfig(area=BIG_RUIN)],
        )
    )


def _run(policy_name: str, seed: int = 0) -> WargameEnv:
    env = _env()
    observation, _info = env.reset(seed=seed)
    policy = build_baseline_policy(policy_name)
    done = False
    while not done:
        action = policy.select_action(
            env.player_models, env, action_mask=observation.action_mask
        )
        observation, _reward, terminated, truncated, _step = env.step(action)
        done = terminated or truncated
    return env


class TestObjectiveExtent:
    def test_an_area_reports_its_equivalent_radius_not_zero(self) -> None:
        """0.0 is what `radius_size` says, and it is what caused the bug."""
        env = _env()
        env.reset(seed=0)
        objective = env.objectives[0]

        assert objective.radius_size == 0.0
        # 12 x 12 = 144; a disc of that area has radius sqrt(144/pi) = 6.77.
        assert objective_extent(objective) == pytest.approx(6.77, abs=0.01)

    def test_a_disc_still_reports_its_own_radius(self) -> None:
        env = WargameEnv(
            config=WargameEnvConfig(
                render_mode=None,
                number_of_objectives=1,
                objectives=[ObjectiveConfig(x=25, y=25, radius_size=3)],
            )
        )
        env.reset(seed=0)

        assert objective_extent(env.objectives[0]) == pytest.approx(3.0)


class TestArrivalIsARegionTest:
    def test_the_squad_marcher_arrives_and_spreads_over_the_ruin(self) -> None:
        """The bar's movement, and the one that had to be right.

        `squad_march_shoot` inherits this, so an understated squad marcher
        understates the reference every learned policy is quoted against. It
        was: on the 25v25 real-geometry scenario, fixing arrival took the bar
        from +27.8 to **+38.0** vp_margin, win 0.67 to 0.75, and occupancy 0.61
        to 0.91 (n=100, seeds 700000+).
        """
        env = _run("squad_march")
        objective = env.objectives[0]
        area = objective.area
        assert area is not None
        centre = np.asarray(objective.location, dtype=float)

        positions = np.array(
            [
                np.asarray(model.location, dtype=float)
                for model in env.player_models
                if model.is_alive
                and area.contains(float(model.location[0]), float(model.location[1]))
            ]
        )
        assert len(positions) >= 8, "the squad should have arrived on the ruin"

        # The ruin is 12 across, equivalent radius 6.77. Anything below ~2 is
        # a heap on the centroid; a region test leaves the squad spread over
        # the ground it walked onto.
        spread = float(np.mean(np.linalg.norm(positions - centre, axis=1)))
        assert spread > 2.5, f"mean distance from centre only {spread:.2f}"

    @pytest.mark.parametrize("policy", ("greedy_nearest", "split_evenly"))
    def test_the_centroid_seekers_still_funnel(self, policy: str) -> None:
        """A narrower defect than before, pinned rather than fixed.

        Both send every model at the objective's *centre*, so a squad
        approaching from one bearing converges on a single point of the near
        face. The first arrivals stop there, the rest collide with them and halt
        outside: three of ten get in.

        Steering at the nearest point of the footprint instead was tried and is
        worse -- models spawned above and left of a ruin all clamp to the same
        *corner*, and one of ten gets in. Fixing it properly needs per-model
        target slots or collision-aware steering, and neither is the bar:
        `squad_march` is unaffected because it moves under a cohesion limit,
        and these two sit far below it either way.
        """
        env = _run(policy)
        area = env.objectives[0].area
        assert area is not None

        inside = sum(
            1
            for model in env.player_models
            if model.is_alive
            and area.contains(float(model.location[0]), float(model.location[1]))
        )
        assert inside < 8, (
            f"{policy} now gets {inside} models in — if this is a real fix, "
            "update the numbers in the docstring and in tests/test_baselines.py"
        )
