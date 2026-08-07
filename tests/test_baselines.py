"""Tests for the scripted baseline policies used as measurement references."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from wargame_rl.wargame.envs.baseline.evaluate import (
    evaluate_baseline,
    evaluate_selector,
    record_episode,
    selector_for,
)
from wargame_rl.wargame.envs.baseline.registry import (
    build_baseline_policy,
    get_registry,
)
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types import (
    WargameEnvAction,
    WargameEnvConfig,
    WargameEnvObservation,
)
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.model.net import TransformerNetwork

BASELINE_NAMES = ("random", "greedy_nearest", "split_evenly", "squad_march")


def _make_env(
    number_of_wargame_models: int = 8,
    number_of_objectives: int = 2,
    max_groups: int = 2,
) -> WargameEnv:
    """Small opponent-free env. Squads are contiguous by model index."""
    return WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=number_of_wargame_models,
            number_of_objectives=number_of_objectives,
            max_groups=max_groups,
            number_of_battle_rounds=12,
        )
    )


def test_all_baselines_are_registered() -> None:
    """Every baseline the harness reports on can be built by name."""
    registry = get_registry()
    assert set(BASELINE_NAMES) <= set(registry)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_emits_one_legal_action_per_model(name: str) -> None:
    """Baselines produce a full, in-bounds action vector in every phase."""
    env = _make_env()
    policy = build_baseline_policy(name)
    observation, _ = env.reset(seed=0)

    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.select_action(env.wargame_models, env)
        assert len(action.actions) == len(env.wargame_models)
        assert all(0 <= a < env.n_actions for a in action.actions)
        observation, _reward, terminated, truncated, _info = env.step(action)


@pytest.mark.parametrize(
    ("name", "floor"),
    (("greedy_nearest", 0.65), ("split_evenly", 0.8), ("squad_march", 1.0)),
)
def test_objective_seeking_baselines_reach_objectives(name: str, floor: float) -> None:
    """The scripted baselines end the episode with nearly every model on an objective.

    This is the property that makes them a usable bar: they come close to saturating
    the occupancy criteria, so a learned policy well below them is unambiguously
    worse.

    They no longer saturate it exactly. Models have bases that cannot overlap, so a
    squad has to flow into an objective rather than stacking on its centre, and a
    policy that sends every model at the same point leaves a few orbiting the cluster
    — `greedy_nearest` plateaus around 0.7 however long the episode runs. The disc
    has room for roughly thirty of these bases, so this is a pathfinding limit, not a
    capacity one. `squad_march`, which spreads a squad across a frontage, still
    reaches every model in.
    """
    env = _make_env()
    result = evaluate_baseline(build_baseline_policy(name), env, seeds=[0, 1, 2])
    assert result.final_fraction_at_objectives >= floor


def test_random_baseline_is_the_floor() -> None:
    """The random baseline occupies objectives at chance level."""
    env = _make_env()
    result = evaluate_baseline(build_baseline_policy("random", seed=0), env, [0, 1, 2])
    assert result.final_fraction_at_objectives < 0.5


def test_squad_march_keeps_squads_together() -> None:
    """Squad marching holds coherency; per-model assignment breaks it.

    Sized so the contrast is real: 3 squads of 3 over 3 objectives means
    `split_evenly` (model *i* to objective *i mod 3*) sends every member of a
    squad to a different objective, leaving each one isolated. Marching on the
    squad centroid keeps them on one point together.
    """
    env = _make_env(number_of_wargame_models=9, number_of_objectives=3, max_groups=3)
    seeds = [0, 1, 2]
    marching = evaluate_baseline(build_baseline_policy("squad_march"), env, seeds)
    splitting = evaluate_baseline(build_baseline_policy("split_evenly"), env, seeds)
    assert marching.worst_cohesion_gap < splitting.worst_cohesion_gap


def test_squad_march_sends_whole_squads_to_one_objective() -> None:
    """Every member of a squad ends on the same objective."""
    env = _make_env()
    policy = build_baseline_policy("squad_march")
    _observation, _ = env.reset(seed=0)

    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.select_action(env.wargame_models, env)
        _observation, _reward, terminated, truncated, _info = env.step(action)

    models = env.wargame_models
    cache = compute_distances(models, env.objectives, alive_mask=alive_mask_for(models))
    nearest = np.argmin(cache.model_obj_norms_offset, axis=1)
    for group_id in {m.group_id for m in models}:
        members = [i for i, m in enumerate(models) if m.group_id == group_id]
        assert len(set(nearest[members].tolist())) == 1


def test_evaluate_baseline_is_deterministic_for_a_seed_set() -> None:
    """Two runs over the same seeds give the same numbers.

    Baselines are only useful as a reference if they are reproducible; this
    pins that they do not depend on global RNG state.
    """
    env = _make_env()
    seeds = [7, 8]
    first = evaluate_baseline(build_baseline_policy("squad_march"), env, seeds)
    second = evaluate_baseline(build_baseline_policy("squad_march"), env, seeds)
    assert first == second


def test_a_learned_policy_scores_through_the_same_path_as_a_baseline() -> None:
    """A network-driven selector is scored by the identical code as a baseline.

    This is what makes "agent 0.95 vs squad_march_shoot 1.00" a comparison
    rather than two numbers from two loops that can drift apart. Uses an
    untrained network — the point is the plumbing, not the score.
    """
    env = _make_env()
    policy_net = TransformerNetwork.policy_from_env(env)
    policy_net.eval()

    def select(
        observation: WargameEnvObservation, _env: WargameEnv
    ) -> WargameEnvAction:
        with torch.no_grad():
            logits = policy_net(observation_to_tensor(observation, policy_net.device))
        actions = logits.argmax(dim=-1).flatten().tolist()
        return WargameEnvAction(actions=[int(a) for a in actions])

    result = evaluate_selector(select, env, [0, 1], name="untrained")

    assert result.name == "untrained"
    assert result.n_episodes == 2
    assert 0.0 <= result.final_fraction_at_objectives <= 1.0
    assert result.vp_margin == result.player_vp - result.opponent_vp


def test_recorded_episode_is_written_for_a_selector(tmp_path: Path) -> None:
    """`record_episode` accepts any selector, so agent traces are comparable.

    The reference traces `just analyze-compare` reads are only meaningful if the
    agent's trace is produced by the same recorder as the baseline's.
    """
    env_config = WargameEnvConfig(
        render_mode=None,
        number_of_wargame_models=4,
        number_of_objectives=2,
        number_of_battle_rounds=4,
    )
    policy = build_baseline_policy("squad_march")
    output = tmp_path / "trace.jsonl"

    written = record_episode(
        selector_for(policy), env_config, seed=3, output_path=output
    )

    assert written.exists()
    assert written.stat().st_size > 0
