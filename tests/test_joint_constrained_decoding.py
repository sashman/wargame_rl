"""Joint constrained decoding: the most probable action combination the rules allow.

The policy emits one categorical per model and samples them independently, but
coherency is a property of the joint configuration. These pin the decoder on
hand-built preferences, so the behaviour is checked against geometry rather
than against whatever a checkpoint happened to prefer.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.decoding import decode_joint_coherent
from wargame_rl.wargame.model.common.factory import create_environment


@pytest.fixture
def env() -> WargameEnv:
    """A four-model env whose first two models share a unit."""
    config = parse_yaml_raw_as(WargameEnvConfig, open("configs/dev/tiny.yaml").read())
    environment = create_environment(config)
    environment.reset(seed=1)
    return environment


def unit_is_coherent(env: WargameEnv, indices: list[int]) -> bool:
    models = env.player_models
    return evaluate_coherency(
        positions=np.array([models[i].location for i in indices], dtype=float),
        group_ids=np.zeros(len(indices), dtype=np.intp),
        alive_mask=np.array([models[i].is_alive for i in indices], dtype=bool),
        base_radii=np.array([models[i].base_radius for i in indices], dtype=float),
        nearest_distance=env.config.coherency.nearest_distance,
        furthest_distance=env.config.coherency.furthest_distance,
    ).all_coherent


def test_top_k_of_one_is_exactly_the_independent_decode(env: WargameEnv) -> None:
    """K=1 must be a no-op, so every number measured before this existed holds."""
    handler = env.player_action_handler
    rng = np.random.default_rng(0)
    log_probs = rng.normal(size=(len(env.player_models), handler.n_actions))
    actions = [int(a) for a in log_probs.argmax(axis=1)]

    assert decode_joint_coherent(log_probs, actions, env, top_k=1) == actions


def test_it_replaces_a_joint_action_that_would_break_the_unit(env: WargameEnv) -> None:
    """The whole point: independently-best actions that are jointly illegal.

    Model 0 is told to sprint east and model 1 to sprint west, which is the
    canonical way an independently factorised policy tears a unit apart. A legal
    combination exists in the candidate set — both moving the same way — and the
    decoder has to find it.
    """
    models = env.player_models
    unit = [i for i, m in enumerate(models) if m.group_id == models[0].group_id]
    assert len(unit) >= 2, "fixture must give the first unit at least two models"
    # Put them adjacent so only the *move* can break the chain.
    models[unit[1]].location = models[unit[0]].location + np.array(
        [1.0, 0.0], dtype=models[unit[0]].location.dtype
    )
    assert unit_is_coherent(env, unit)

    handler = env.player_action_handler
    east = handler.best_action_toward(1.0, 0.0)
    west = handler.best_action_toward(-1.0, 0.0)

    n_actions = handler.n_actions
    log_probs = np.full((len(models), n_actions), -20.0)
    # Both models rank a full-speed opposed sprint first and agreeing second.
    log_probs[unit[0]][east] = 0.0
    log_probs[unit[0]][west] = -1.0
    log_probs[unit[1]][west] = 0.0
    log_probs[unit[1]][east] = -1.0
    for index in range(len(models)):
        if index not in unit:
            log_probs[index][0] = 0.0

    actions = [int(a) for a in log_probs.argmax(axis=1)]
    assert actions[unit[0]] == east and actions[unit[1]] == west

    decoded = decode_joint_coherent(log_probs, actions, env, top_k=2)

    assert decoded != actions, "the decoder should have rejected the opposed sprint"
    # Whatever it chose must be one of the candidates it was offered.
    assert decoded[unit[0]] in (east, west)
    assert decoded[unit[1]] in (east, west)
    # And the two must now agree, which is the only legal pair here.
    assert decoded[unit[0]] == decoded[unit[1]]


def test_it_keeps_the_original_actions_when_nothing_legal_is_available(
    env: WargameEnv,
) -> None:
    """No legal combination in the candidate set leaves the caller's enforcement.

    The decoder must never invent an action outside the top-K it was given; when
    every candidate combination is illegal it declines, and `enforce_move` then
    applies exactly as it did before.
    """
    models = env.player_models
    unit = [i for i, m in enumerate(models) if m.group_id == models[0].group_id]
    # Park them far enough apart that no single move can close the chain.
    models[unit[1]].location = models[unit[0]].location + np.array(
        [40.0, 0.0], dtype=models[unit[0]].location.dtype
    )
    handler = env.player_action_handler
    n_actions = handler.n_actions
    log_probs = np.full((len(models), n_actions), -20.0)
    # One candidate each, and it cannot possibly bridge 40 inches.
    log_probs[unit[0]][0] = 0.0
    log_probs[unit[1]][0] = 0.0
    for index in range(len(models)):
        if index not in unit:
            log_probs[index][0] = 0.0
    actions = [int(a) for a in log_probs.argmax(axis=1)]

    assert decode_joint_coherent(log_probs, actions, env, top_k=1) == actions
    assert decode_joint_coherent(log_probs, actions, env, top_k=2) == actions


def test_a_unit_too_large_for_the_candidate_budget_is_left_alone(
    env: WargameEnv,
) -> None:
    """`top_k ** size` explodes, so an oversized unit keeps its independent decode."""
    handler = env.player_action_handler
    rng = np.random.default_rng(1)
    log_probs = rng.normal(size=(len(env.player_models), handler.n_actions))
    actions = [int(a) for a in log_probs.argmax(axis=1)]

    decoded = decode_joint_coherent(log_probs, actions, env, top_k=8, max_candidates=4)

    assert decoded == actions
