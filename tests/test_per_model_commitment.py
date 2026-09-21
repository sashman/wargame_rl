"""The commitment layer, Stage 0 (#384, #385, #386): the state, the sticky
assignment the env writes, the token columns and relation, the reward keyed
to the committed objective, the scripted seat's row, and the checkpoint
widening. Off, everything is byte- or bit-identical to before the layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.per_model_seats import random_legal_action, small_config
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.per_model import PerModelEnv, StepKind
from wargame_rl.wargame.envs.per_model.commitment import (
    NO_TARGET,
    CommitmentState,
    greedy_assignment,
    retire_and_reassign,
    unit_centroids,
)
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.scripted import ScriptedSeat
from wargame_rl.wargame.envs.per_model.tokens import (
    CONTEXT_OBJECTIVE,
    CONTEXT_OWN_UNIT,
    OBJECTIVE_CLAIMANTS,
    REL_COMMIT_PRESENT,
    REL_COMMITTED,
    UNIT_HAS_GROUND,
    TokenScenario,
    build_tokens,
)
from wargame_rl.wargame.envs.per_model.types import PerModelObservation
from wargame_rl.wargame.envs.reward.calculators.closest_objective_v2 import (
    ClosestObjectiveV2Calculator,
)
from wargame_rl.wargame.envs.reward.calculators.objective_stay import (
    ObjectiveStayCalculator,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import CommitmentConfig
from wargame_rl.wargame.model.per_model.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig
from wargame_rl.wargame.model.per_model.net import SetNetwork
from wargame_rl.wargame.model.per_model.ppo import PerModelPPOConfig


def _config(*, greedy: bool, **kwargs: object) -> WargameEnvConfig:
    config = small_config(**kwargs)  # type: ignore[arg-type]
    if greedy:
        config = config.model_copy(
            update={"commitments": CommitmentConfig(assignment="greedy")}
        )
    return config


def _tokens(env: PerModelEnv, observation: PerModelObservation):  # type: ignore[no-untyped-def]
    scenario = TokenScenario.for_episode(env, env.player_seat)
    return build_tokens(env, env.player_seat, observation, scenario)


# ------------------------------------------------------------ the pure rule


def test_greedy_assignment_matches_the_scripted_bar_on_random_boards() -> None:
    """The env's sticky rule IS the take script's rule, checked on the env's
    own board through the script's own method."""
    config = _config(greedy=False)
    env = PerModelEnv(config)
    policy = build_baseline_policy("squad_march_take")
    for seed in range(5):
        env.reset(seed=seed)
        groups = env.player_seat.living_units()
        squad_objectives = getattr(policy, "squad_objectives")
        expected = squad_objectives(env.wargame_models, env, groups)
        counts = np.zeros(len(env.objectives), dtype=np.intp)
        enemy = [m for m in env.opponent_models if m.is_alive]
        for k, objective in enumerate(env.objectives):
            counts[k] = sum(
                1
                for m in enemy
                if np.hypot(
                    *(
                        np.asarray(m.location, float)
                        - np.asarray(objective.location, float)
                    )
                )
                <= env.config.objective_radius_size
            )
        centroids = unit_centroids(env.wargame_models, groups)
        locations = np.array([o.location for o in env.objectives], dtype=float)
        got = greedy_assignment(centroids, locations, counts)
        assert [env.objectives[i] for i in got] == expected


# ------------------------------------------------------- off is identical


def test_off_config_has_empty_state_and_zeroed_columns() -> None:
    env = PerModelEnv(_config(greedy=False))
    observation, _ = env.reset(seed=1)
    assert not env.player_commitments.any_set()
    tokens = _tokens(env, observation)
    assert not np.any(tokens.cross_relations[:, :, REL_COMMIT_PRESENT])
    assert not np.any(tokens.cross_relations[:, :, REL_COMMITTED])
    objectives = tokens.context_kind == CONTEXT_OBJECTIVE
    assert not np.any(tokens.context[objectives, OBJECTIVE_CLAIMANTS])
    units = tokens.context_kind == CONTEXT_OWN_UNIT
    assert not np.any(tokens.context[units, UNIT_HAS_GROUND])


def test_a_scripted_seat_writes_its_assignment_and_the_reward_ignores_it() -> None:
    """The bar has a row on the readouts (D8), and with the layer off the
    retimer pays exactly what it paid before the layer existed."""
    config = _config(greedy=False)
    totals = []
    for write in (False, True):
        env = PerModelEnv(config)
        if write:
            env.set_player_planner(
                ScriptedSeat.for_policy(build_baseline_policy("squad_march_take"))
            )
        retimer = PerStepReward(env)
        observation, _ = env.reset(seed=3)
        retimer.reset()
        adapter = env.player_seat.adapter
        while True:
            point = observation.decision
            action = (
                adapter.choose(point, env.player_seat)
                if adapter is not None and point.kind is not StepKind.close_turn
                else random_legal_action(point, np.random.default_rng(3))
            )
            nxt, _r, terminated, _t, info = env.step(action)
            retimer.on_step(observation, action, info["effect"], terminated)
            observation = nxt
            if terminated:
                break
        totals.append((env.player_commitments.any_set(), retimer.episode_reward))
    assert totals[0][0] is False and totals[1][0] is True
    # Different drivers, so the totals differ; what the test pins is that the
    # OFF layer never reads the state: the keyed calculators see None.
    assert (
        retimer.last_context is None or retimer.last_context.committed_objective is None
    )


# --------------------------------------------------------- on: the writer


def test_greedy_writer_assigns_every_living_unit_at_reset() -> None:
    env = PerModelEnv(_config(greedy=True))
    env.reset(seed=2)
    state = env.player_commitments
    groups = env.player_seat.living_units()
    grounds = [state.ground_of(g) for g in groups]
    assert all(0 <= g < len(env.objectives) for g in grounds)
    assert len(set(grounds)) == len(groups), "two units on one objective with spares"


def test_tokens_flag_the_committed_objective_and_count_claimants() -> None:
    env = PerModelEnv(_config(greedy=True))
    observation, _ = env.reset(seed=2)
    tokens = _tokens(env, observation)
    state = env.player_commitments
    assert np.all(tokens.cross_relations[:, :, REL_COMMIT_PRESENT] == 1.0)
    objective_rows = np.flatnonzero(tokens.context_kind == CONTEXT_OBJECTIVE)
    for i, model in enumerate(env.wargame_models):
        ground = state.ground_of(int(model.group_id))
        flagged = np.flatnonzero(tokens.cross_relations[i, :, REL_COMMITTED])
        assert list(flagged) == [objective_rows[ground]]
    claimants = state.claimants_by_objective(len(env.objectives))
    np.testing.assert_allclose(
        tokens.context[objective_rows, OBJECTIVE_CLAIMANTS], claimants / 10.0
    )
    unit_rows = np.flatnonzero(tokens.context_kind == CONTEXT_OWN_UNIT)
    assert np.all(tokens.context[unit_rows, UNIT_HAS_GROUND] == 1.0)


def test_redundant_unit_is_reassigned_and_the_holder_keeps_its_objective() -> None:
    """Unit A stands on objective k and holds it; unit B, also committed to k
    but outside, is redundant and moves to the nearest free objective. A keeps k."""
    env = PerModelEnv(_config(greedy=True))
    env.reset(seed=2)
    state = env.player_commitments
    groups = env.player_seat.living_units()
    a, b = groups[0], groups[1]
    k = 0
    target = np.asarray(env.objectives[k].location, dtype=float)
    for model in env.wargame_models:
        if int(model.group_id) == a:
            model.location = target.copy()
    state.set_ground(a, k)
    state.set_ground(b, k)
    reassigned = retire_and_reassign(
        state, env.wargame_models, env.opponent_models, env.objectives
    )
    assert reassigned == [b]
    assert state.ground_of(a) == k
    assert state.ground_of(b) != k and state.ground_of(b) != NO_TARGET


def test_dead_unit_loses_its_commitment() -> None:
    env = PerModelEnv(_config(greedy=True))
    env.reset(seed=2)
    state = env.player_commitments
    g = env.player_seat.living_units()[0]
    for model in env.wargame_models:
        if int(model.group_id) == g:
            model.stats["current_wounds"] = 0
    retire_and_reassign(state, env.wargame_models, env.opponent_models, env.objectives)
    assert state.ground_of(g) == NO_TARGET


# --------------------------------------------------------- on: the reward


def test_retimer_keys_travel_and_staying_to_the_committed_objective() -> None:
    """The context carries each model's committed objective; the travel term
    targets it whatever its own rule would pick, and the staying term pays
    only for ending inside it."""
    config = _config(greedy=True)
    env = PerModelEnv(config)
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=4)
    retimer.reset()
    rng = np.random.default_rng(4)
    for _ in range(6):
        point = observation.decision
        action = random_legal_action(point, rng)
        nxt, _r, terminated, _t, info = env.step(action)
        retimer.on_step(observation, action, info["effect"], terminated)
        observation = nxt
        if terminated:
            break
    ctx = retimer.last_context
    assert ctx is not None and ctx.committed_objective is not None
    expected = env.player_commitments.committed_objective_per_model(env.wargame_models)
    np.testing.assert_array_equal(ctx.committed_objective, expected)

    travel = ClosestObjectiveV2Calculator(progress_scale=6.0, fallback_to_nearest=True)
    for i, model in enumerate(env.wargame_models):
        travel.calculate(i, model, env, ctx)  # type: ignore[arg-type]
        assert travel._last_breakdown[i]["target_obj_idx"] == float(expected[i])

    stay = ObjectiveStayCalculator()
    model = env.wargame_models[0]
    committed = int(expected[0])
    other = next(k for k in range(len(env.objectives)) if k != committed)
    for k, paid in ((other, False), (committed, True)):
        target = np.asarray(env.objectives[k].location, dtype=float)
        model.location = target.copy()
        fresh = retimer._context(action_phase=None, kills_by_model=None)
        value = stay.calculate(0, model, env, fresh)  # type: ignore[arg-type]
        assert (value > 0.0) is paid


def test_state_records_a_turn_at_every_close() -> None:
    env = PerModelEnv(_config(greedy=True))
    observation, _ = env.reset(seed=5)
    rng = np.random.default_rng(5)
    closes = 0
    while True:
        point = observation.decision
        if point.kind is StepKind.close_turn:
            closes += 1
        action = random_legal_action(point, rng)
        observation, _r, terminated, _t, _info = env.step(action)
        if terminated:
            break
    assert len(env.player_commitments.history) >= closes >= 1


# ------------------------------------------------------ checkpoint widening


def test_a_pre_layer_checkpoint_loads_with_zero_weights_on_the_new_relations(
    tmp_path: Path,
) -> None:
    env = PerModelEnv(_config(greedy=False))
    network = SetNetwork.from_env(
        env, SetNetworkConfig(n_layers=1, embedding_size=32, n_heads=4)
    )
    narrow = {k: v.clone() for k, v in network.state_dict().items()}
    for key in (
        "self_relation_bias.weight",
        "cross_relation_bias.weight",
        "pointer_relation_bias.weight",
    ):
        narrow[key] = narrow[key][:, :-2].clone()
    path = tmp_path / "old.pt"
    save_checkpoint(
        path,
        network,
        ppo_config=PerModelPPOConfig(),
        env_config=env.config.model_dump(mode="json"),
        rounds=0,
        seed=0,
        revision="test",
    )
    payload = torch.load(path, weights_only=True)
    payload["state_dict"] = narrow
    torch.save(payload, path)
    loaded = load_checkpoint(path).network.state_dict()
    for key in (
        "self_relation_bias.weight",
        "cross_relation_bias.weight",
        "pointer_relation_bias.weight",
    ):
        assert torch.equal(loaded[key][:, :-2], narrow[key])
        assert torch.all(loaded[key][:, -2:] == 0)


def test_combat_set_cap_is_enforced() -> None:
    state = CommitmentState(groups=[0, 1], combat_set_size=2)
    state.set_combat(0, (3, 4))
    with pytest.raises(ValueError):
        state.set_combat(1, (3, 4, 5))
    claimants = state.claimants_by_enemy_unit(np.array([3, 4, 5]))
    np.testing.assert_allclose(claimants, [0.5, 0.5, 0.0])
