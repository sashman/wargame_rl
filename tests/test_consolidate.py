"""A unit that fought shuffles up to 3" afterwards — in Objective mode only.

`docs/rules/12-fight-phase.md` § Consolidate step. The three modes are assessed
in order and the first match is **compulsory**, so most of these tests are about
what does NOT happen: a unit still engaged is in Ongoing mode and a unit with an
enemy within 3" is in Engaging mode, and neither may consolidate onto an
objective instead. Both are `DEFERRED`, so for those units the step is a no-op.

⚠ Contact is set BY HAND, as in `tests/test_fight_phase.py`. Every mover on both
seats is walked back out of engagement, so no sequence of legal non-charge moves
reaches the state these tests are about.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.consolidate import consolidate_objective
from wargame_rl.wargame.envs.domain.entities import WargameObjective
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.envs.wargame_model import WargameModel
from wargame_rl.wargame.model.common.factory import create_environment

LETHAL = MeleeWeaponProfile(attacks=12, melee_skill=2, strength=10, ap=6, damage=4)


def _model(x: float, y: float = 10.0, group: int = 0) -> WargameModel:
    return WargameModel(
        location=np.array([x, y]),
        stats={"toughness": 3, "save": 4, "max_wounds": 1, "current_wounds": 1},
        distances_to_objectives=np.zeros(1),
        group_id=group,
        base_radius=0.0,
    )


def _consolidate(
    models: list[WargameModel],
    enemies: list[WargameModel],
    objectives: list[WargameObjective],
) -> list[int]:
    return consolidate_objective(
        models,
        enemies,
        objectives,
        eligible_units={0},
        objective_offsets=lambda: (
            compute_distances(models, objectives).model_obj_norms_offset
        ),
        max_distance=3.0,
        engagement_range=1.0,
        base_radius=0.0,
        board=(60.0, 44.0),
        coherency_nearest=2.0,
        coherency_furthest=9.0,
    )


def _objective(x: float, radius: float = 1.0) -> WargameObjective:
    return WargameObjective(location=np.array([x, 10.0]), radius_size=radius)


def test_a_unit_that_fought_and_is_now_clear_walks_onto_the_objective() -> None:
    models = [_model(10.0), _model(10.5)]
    enemies = [_model(40.0, group=1)]
    objectives = [_objective(12.5)]

    assert _consolidate(models, enemies, objectives) == [0]

    offsets = compute_distances(models, objectives).model_obj_norms_offset
    assert (offsets[:, 0] <= 1.0).any(), "nobody ended in range of the objective"


def test_ongoing_mode_pre_empts_objective_mode() -> None:
    """Still engaged, so the mode is Ongoing — and Ongoing is deferred."""
    models = [_model(10.0), _model(10.5)]
    enemies = [_model(11.2, group=1)]  # inside the 1" engagement range
    objectives = [_objective(12.5)]
    before = [m.location.copy() for m in models]

    assert _consolidate(models, enemies, objectives) == []
    assert all(np.array_equal(b, m.location) for b, m in zip(before, models))


def test_engaging_mode_pre_empts_objective_mode() -> None:
    """Unengaged but within 3" of an enemy, so the mode is Engaging."""
    models = [_model(10.0), _model(10.5)]
    enemies = [_model(12.5, group=1)]  # 2.0" from the nearest model
    objectives = [_objective(12.0)]
    before = [m.location.copy() for m in models]

    assert _consolidate(models, enemies, objectives) == []
    assert all(np.array_equal(b, m.location) for b, m in zip(before, models))


def test_an_objective_further_than_three_inches_cannot_be_consolidated_onto() -> None:
    models = [_model(10.0), _model(10.5)]
    enemies = [_model(40.0, group=1)]
    objectives = [_objective(20.0)]
    before = [m.location.copy() for m in models]

    assert _consolidate(models, enemies, objectives) == []
    assert all(np.array_equal(b, m.location) for b, m in zip(before, models))


def test_a_unit_that_did_not_fight_does_not_consolidate() -> None:
    """Eligibility is "was eligible to fight this phase", not "is near a point"."""
    models = [_model(10.0), _model(10.5)]
    enemies = [_model(40.0, group=1)]
    objectives = [_objective(12.5)]
    before = [m.location.copy() for m in models]

    moved = consolidate_objective(
        models,
        enemies,
        objectives,
        eligible_units=set(),
        objective_offsets=lambda: (
            compute_distances(models, objectives).model_obj_norms_offset
        ),
        max_distance=3.0,
        engagement_range=1.0,
        base_radius=0.0,
        board=(60.0, 44.0),
        coherency_nearest=2.0,
        coherency_furthest=9.0,
    )
    assert moved == []
    assert all(np.array_equal(b, m.location) for b, m in zip(before, models))


def test_a_unit_that_came_out_of_the_fight_strung_out_cannot_consolidate() -> None:
    """All-or-nothing at the UNIT, per `03-moving.md`.

    The trailing model is 6" behind — past the 2" chain — and a 3" walk does not
    close that, so the end state is still incoherent and the whole move is
    reverted rather than the offending model.

    ⚠ This, not a move that *breaks* the chain, is what the coherency guard
    catches. Every model walks toward the same point, so an Objective-mode
    consolidation is **contractive**: it can only bring a unit closer together.
    The guard is still the rule, and still worth having — melee casualties are
    what leave a unit strung out in the first place.
    """
    models = [_model(10.0), _model(4.0)]
    enemies = [_model(40.0, group=1)]
    objectives = [_objective(12.5)]
    before = [m.location.copy() for m in models]

    assert _consolidate(models, enemies, objectives) == []
    assert all(np.array_equal(b, m.location) for b, m in zip(before, models))


def test_a_model_already_in_range_does_not_shuffle() -> None:
    """The rules only bind a model that *moves*, and moving buys it nothing."""
    models = [_model(12.0), _model(10.6)]
    enemies = [_model(40.0, group=1)]
    objectives = [_objective(12.5)]
    arrived = models[0].location.copy()

    assert _consolidate(models, enemies, objectives) == [0]
    assert np.array_equal(arrived, models[0].location)


def _env() -> WargameEnv:
    """One player model that kills in melee, against one that cannot fight back."""
    config = WargameEnvConfig(
        number_of_wargame_models=1,
        number_of_opponent_models=1,
        number_of_objectives=1,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[ModelConfig(melee_weapons=[LETHAL])],
        opponent_models=[ModelConfig(melee_weapons=[])],
        melee=MeleeConfig(enabled=True),
        skip_phases=[BattlePhase.command, BattlePhase.shooting],
    )
    env = create_environment(config)
    env.reset(seed=7)
    return env


def test_consolidation_happens_through_env_step() -> None:
    """The whole path: locked in melee, kill the target, walk onto the point."""
    env = _env()
    player = env.wargame_models[0]
    opponent = env.opponent_models[0]
    player.location = np.array([10.0, 10.0], dtype=player.location.dtype)
    opponent.location = np.array([11.0, 10.0], dtype=opponent.location.dtype)
    objective = env.objectives[0]
    objective.location = np.array([7.5, 10.0], dtype=objective.location.dtype)
    radius = objective.radius_size

    for _ in range(3):
        env.step(WargameEnvAction(actions=[STAY_ACTION]))
        if not opponent.is_alive:
            break

    assert not opponent.is_alive, "the melee did not kill the target"
    offset = compute_distances([player], [objective]).model_obj_norms_offset[0, 0]
    assert offset <= radius, "the survivor did not consolidate onto the objective"
