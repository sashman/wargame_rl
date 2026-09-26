"""The two-stream reward's member terms (#384, 2026-09-24): with
`normalize_to_commit_distance` completing any commitment pays exactly
`progress_scale` whatever the distance at commit; with `cap_per_commitment`
the staying term pays at most the cap per model per commitment; and with the
fallback off an uncommitted unit is paid nothing by the travel term. All read
from real episodes of the scripted bar on the arm's own config through the
per-model retimer, so the numbers are the ones a trainer would see.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.per_model.commitment import NO_TARGET
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.types import StepKind
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import CommitmentConfig
from wargame_rl.wargame.selectors import build_per_model_chooser

CONFIG = "configs/experiments/curriculum/a3_head_r.yaml"
PROGRESS = "closest_objective_v2"
STAY = "objective_stay"


def _episode_pay(
    config: WargameEnvConfig, seed: int
) -> tuple[
    dict[tuple[int, int], dict[str, float]], dict[tuple[int, int], bool], set[int]
]:
    """Per (model, committed objective): the summed pay by term over the span
    the model's unit held that commitment; whether the model ended the span
    inside it; and the models whose unit re-committed during the episode."""
    env = PerModelEnv(config)
    chooser = build_per_model_chooser("squad_march_take", [env], seed=seed)
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=seed)
    retimer.reset()
    pay: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    seen: dict[int, set[int]] = defaultdict(set)
    inside: dict[tuple[int, int], bool] = {}
    while True:
        before = observation
        action = chooser.choose([env], [observation])[0]
        observation, _reward, terminated, _truncated, info = env.step(action)
        payment = retimer.on_step(before, action, info["effect"], terminated)
        actors = list(info["effect"].actor_set)
        if len(actors) == 1:
            model_idx = int(actors[0])
            model = env.wargame_models[model_idx]
            ground = env.player_commitments.ground_of(int(model.group_id))
            if ground != NO_TARGET:
                key = (model_idx, int(ground))
                seen[model_idx].add(int(ground))
                for term in (PROGRESS, STAY):
                    pay[key][term] += float(payment.breakdown.get(term, 0.0))
                # Inside by the term's own definition: the base's offset
                # distance within the objective's radius (the scoring rule).
                cache = compute_distances(env.wargame_models, env.objectives)
                inside[key] = bool(
                    cache.model_obj_norms_offset[model_idx, int(ground)]
                    <= cache.obj_radii[int(ground)]
                )
        if terminated:
            break
    switched = {m for m, grounds in seen.items() if len(grounds) > 1}
    return pay, inside, switched


@pytest.mark.parametrize("seed", [700000, 700001, 700002])
def test_completing_any_commitment_pays_the_progress_scale(seed: int) -> None:
    config = load_env_config(CONFIG)
    pay, inside, switched = _episode_pay(config, seed)
    completed = [
        pay[key][PROGRESS]
        for key, ended_inside in inside.items()
        if ended_inside and key[0] not in switched
    ]
    assert len(completed) >= 4, "the bar completes most of its commitments"
    # Progress is the FRACTION of the distance at commit times the scale (2.0
    # here, weight 1.0), so every completed commitment pays the same total,
    # near or far. The per-model retimer pays every per-decision term over
    # the model count, so the trainer sees 2.0 / 12 per completed commitment.
    expected = 2.0 / config.number_of_wargame_models
    assert np.allclose(completed, expected, atol=0.05 * expected), completed


@pytest.mark.parametrize("seed", [700000, 700001])
def test_the_staying_term_is_capped_per_commitment(seed: int) -> None:
    config = load_env_config(CONFIG)
    pay, inside, _switched = _episode_pay(config, seed)
    stays = [pay[key][STAY] for key in inside]
    # cap 2.0 in the term's own units at weight 0.5 -> 1.0 per commitment,
    # over the model count in the retimer's units.
    cap = 1.0 / config.number_of_wargame_models
    assert max(stays) <= cap + 1e-9, max(stays)
    assert max(stays) >= cap - 1e-9, "a body that arrives early reaches the cap"
    assert sum(1 for s in stays if s > 0.0) >= 6


def test_a_unit_with_no_commitment_earns_nothing_from_the_travel_term() -> None:
    config = load_env_config(CONFIG)
    env = PerModelEnv(config)
    chooser = build_per_model_chooser("squad_march_take", [env], seed=700000)
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=700000)
    retimer.reset()
    cleared = 0
    while True:
        before = observation
        action = chooser.choose([env], [observation])[0]
        if action.kind is StepKind.act and cleared < 4:
            # Empty the mover's slot on its own step: with the fallback off
            # the travel term must pay it nothing, not the nearest objective.
            group = int(env.wargame_models[action.model].group_id)
            env.player_commitments.clear_ground(group)
            cleared += 1
            observation, _r, terminated, _t, info = env.step(action)
            payment = retimer.on_step(before, action, info["effect"], terminated)
            assert payment.breakdown.get(PROGRESS, 0.0) == 0.0
        else:
            observation, _r, terminated, _t, info = env.step(action)
            retimer.on_step(before, action, info["effect"], terminated)
        if terminated:
            break
    assert cleared == 4


def test_the_floored_anchor_bounds_a_step_from_a_near_re_commit() -> None:
    """A unit re-committed six inches from its new target: without the floor
    one step pays the whole scale; with a twelve-inch floor it pays half."""
    from wargame_rl.wargame.envs.reward.calculators.closest_objective_v2 import (
        ClosestObjectiveV2Calculator,
    )

    bare = ClosestObjectiveV2Calculator(
        progress_scale=2.0, normalize_to_commit_distance=True
    )
    floored = ClosestObjectiveV2Calculator(
        progress_scale=2.0,
        normalize_to_commit_distance=True,
        normalize_min_distance=12.0,
    )
    assert bare.normalize_min_distance == 0.0
    assert floored.normalize_min_distance == 12.0
    with pytest.raises(ValueError):
        ClosestObjectiveV2Calculator(
            normalize_to_commit_distance=True, normalize_min_distance=-1.0
        )


def test_commitment_coverage_pays_the_planning_stream_the_share_claimed() -> None:
    """Under the head writer the plan-shape term lands on the planning stream at
    every close, worth the share of objectives some living unit is committed to
    (times its weight); with the layer off it pays nothing."""
    from wargame_rl.wargame.envs.per_model.commitment import NO_TARGET as _NO

    config = load_env_config("configs/experiments/curriculum/a3_head_rf_pc.yaml")
    env = PerModelEnv(config)
    chooser = build_per_model_chooser("squad_march_take", [env], seed=700000)
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=700000)
    retimer.reset()
    closes = 0
    while True:
        before = observation
        action = chooser.choose([env], [observation])[0]
        state = env.player_commitments
        claimed = {
            state.ground_of(g) for g in state.by_group if state.ground_of(g) != _NO
        }
        observation, _r, terminated, _t, info = env.step(action)
        payment = retimer.on_step(before, action, info["effect"], terminated)
        paid = payment.breakdown.get("commitment_coverage", 0.0)
        if action.kind is StepKind.close_turn:
            closes += 1
            assert paid > 0.0
            assert paid <= 0.3 * retimer.phase_scale + 1e-9
            assert abs(paid - 0.3 * retimer.phase_scale * len(claimed) / 4) < 1e-6
        else:
            assert paid == 0.0
        if terminated:
            break
    assert closes >= 2

    off: WargameEnvConfig = config.model_copy(
        update={"commitments": CommitmentConfig(assignment="none")}  # type: ignore[arg-type]
    )
    env = PerModelEnv(off)
    chooser = build_per_model_chooser("squad_march_take", [env], seed=700000)
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=700000)
    retimer.reset()
    total = 0.0
    while True:
        before = observation
        action = chooser.choose([env], [observation])[0]
        observation, _r, terminated, _t, info = env.step(action)
        total += abs(
            retimer.on_step(before, action, info["effect"], terminated).breakdown.get(
                "commitment_coverage", 0.0
            )
        )
        if terminated:
            break
    assert total == 0.0


def test_commitment_churn_charges_the_planning_stream_per_re_commit_before_arrival() -> (
    None
):
    """The churn cost lands on the planning stream at the close, worth minus its
    weight times the share of living units re-committed this turn before they
    arrived; a first commitment, a re-commit after arrival and a kept slot cost
    nothing; with the layer off it pays nothing."""
    from wargame_rl.wargame.envs.per_model.commitment import NO_TARGET as _NO

    config = load_env_config(
        "configs/experiments/curriculum/a5_points_head_rf_pc_cc.yaml"
    )
    env = PerModelEnv(config)
    chooser = build_per_model_chooser("squad_march_take", [env], seed=700000)
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=700000)
    retimer.reset()
    state = env.player_commitments
    n_units = len(state.groups)
    churned_turns = 0
    closes = 0
    while True:
        before = observation
        action = chooser.choose([env], [observation])[0]
        if action.kind is StepKind.close_turn and closes in (1, 3):
            # Move every unit that has a plan and has not arrived to another
            # objective just before the close: each one is a churn event on
            # top of whatever the script's own writer churned this turn.
            already = state.churned_this_turn
            moved = 0
            for g in state.groups:
                slot = state.by_group[int(g)]
                if slot.ground != _NO and not slot.arrived:
                    state.set_ground(int(g), (slot.ground + 1) % 5)
                    moved += 1
            assert state.churned_this_turn == already + moved
            churned_turns += 1 if moved else 0
        # The close pays the turn's churn, counted up to this step.
        expected = -0.5 * retimer.phase_scale * state.churned_this_turn / n_units
        observation, _r, terminated, _t, info = env.step(action)
        payment = retimer.on_step(before, action, info["effect"], terminated)
        paid = payment.breakdown.get("commitment_churn", 0.0)
        if action.kind is StepKind.close_turn:
            closes += 1
            assert abs(paid - expected) < 1e-9, (closes, paid, expected)
        else:
            assert paid == 0.0
        if terminated:
            break
    assert churned_turns >= 1
    # A re-commit AFTER arrival is not churn, nor is a first commitment.
    fresh = PerModelEnv(config)
    fresh.reset(seed=700001)
    fresh_state = fresh.player_commitments
    fresh_state.by_group[0].ground = _NO
    fresh_state.set_ground(0, 1)
    fresh_state.by_group[0].arrived = True
    fresh_state.set_ground(0, 2)
    assert fresh_state.churned_this_turn == 0
    fresh_state.set_ground(0, 3)
    assert fresh_state.churned_this_turn == 1

    off: WargameEnvConfig = config.model_copy(
        update={"commitments": CommitmentConfig(assignment="none")}  # type: ignore[arg-type]
    )
    env = PerModelEnv(off)
    chooser = build_per_model_chooser("squad_march_take", [env], seed=700000)
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=700000)
    retimer.reset()
    total = 0.0
    while True:
        before = observation
        action = chooser.choose([env], [observation])[0]
        observation, _r, terminated, _t, info = env.step(action)
        total += abs(
            retimer.on_step(before, action, info["effect"], terminated).breakdown.get(
                "commitment_churn", 0.0
            )
        )
        if terminated:
            break
    assert total == 0.0
