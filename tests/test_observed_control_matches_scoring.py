"""The count the agent SEES must be the count the mission SCORES.

There were three implementations of "on an objective". VP, `objective_hold` and
every control read use `norms_offset <= obj_radii`, which measures from the
model's **base edge**. The observation builder had its own: a point-in-polygon
test on the model *centre*. A model whose base overlapped a ruin while its
centre sat outside scored for the mission and was invisible to the network.

Measured on the held-out nine before the fix: **206 of 2,700 (objective, step)
slots disagreed -- 7.6%**, 215 models miscounted. `player_count` is the feature
every objective-keyed reward term reads, so the standing rule "check the agent
can observe what the lever keys on" was quietly false for all of them.

These tests fail on the pre-fix builder.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.baseline.evaluate import selector_for
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_counts_from_norms_offset,
)
from wargame_rl.wargame.model.common.factory import create_environment

# Area objectives, base radius 0.63, and `observe_objective_control: true` --
# all three are needed for the two definitions to be able to disagree at all.
CONFIG = "configs/golden/25v25_maps_two_mode.yaml"


def _scored_counts(env: object, models: list) -> np.ndarray:
    """The count the mission would score for one side, right now."""
    cache = compute_distances(
        models,
        env.objectives,  # type: ignore[attr-defined]
        alive_mask=alive_mask_for(models),
    )
    return objective_counts_from_norms_offset(
        cache.model_obj_norms_offset, cache.obj_radii
    )


@pytest.mark.parametrize("seed", [700000, 700001])
def test_observed_control_equals_scored_control_every_step(seed: int) -> None:
    """Integration: over a whole episode, not one hand-built frame.

    A hand-built frame can be constructed to agree by accident; an episode on a
    real table cannot. This is the check that measured 7.6% before the fix.
    """
    config = load_env_config(CONFIG)
    establishment = max(
        1, config.number_of_wargame_models, config.number_of_opponent_models
    )
    env = create_environment(env_config=config)
    # A marching policy, so models actually cross objective boundaries -- the
    # only place the two definitions can disagree.
    select = selector_for(build_baseline_policy("squad_march_take"))
    observation, _info = env.reset(seed=seed)

    disagreements = 0
    slots = 0
    done = False
    while not done:
        observation, _r, done, _t, _i = env.step(select(observation, env))
        player = _scored_counts(env, env.player_models)
        opponent = _scored_counts(env, env.opponent_models)
        for index in range(len(env.objectives)):
            # `player_count` is optional on the observation because it is only
            # populated under `observe_objective_control`; this config sets it.
            objective = observation.objectives[index]
            assert objective.player_count is not None
            assert objective.opponent_count is not None
            seen_player = round(float(objective.player_count) * establishment)
            seen_opponent = round(float(objective.opponent_count) * establishment)
            slots += 1
            disagreements += int(seen_player != int(player[index]))
            disagreements += int(seen_opponent != int(opponent[index]))
    env.close()

    assert slots > 0, "the episode produced no objective slots to compare"
    assert disagreements == 0


def test_a_model_whose_base_overlaps_but_centre_does_not_is_counted() -> None:
    """The exact case the old centre test missed, isolated.

    Place one model just outside an area objective's boundary by less than its
    base radius. Scoring counts it; the old centre test did not.
    """
    config = load_env_config(CONFIG)
    env = create_environment(env_config=config)
    env.reset(seed=700000)

    objective = next((o for o in env.objectives if o.area is not None), None)
    assert objective is not None, "this config should draw area objectives"
    area = objective.area
    assert area is not None

    model = env.player_models[0]
    minx, miny, maxx, maxy = area.bounds
    # Just outside the right edge, closer than one base radius.
    model.location = np.array(
        [maxx + model.base_radius * 0.5, (miny + maxy) / 2.0],
        dtype=model.location.dtype,
    )

    inside_by_centre = bool(area.contains_points(np.array([model.location]))[0])
    scored = int(_scored_counts(env, [model])[env.objectives.index(objective)])
    env.close()

    assert not inside_by_centre, (
        "the centre must be outside, or the case is not the one"
    )
    assert scored == 1, "scoring measures from the base edge, so this model counts"
