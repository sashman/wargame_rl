"""The per-model facade's coherency readout is the phase facade's tracker, on
the phase facade's grid: sampled at the movement boundary, with the policy's
intent judged before each unit's own referee."""

from __future__ import annotations

import numpy as np

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.domain.movement.coherency import coherency_counts
from wargame_rl.wargame.envs.per_model import PerModelEnv, scripted_chooser
from wargame_rl.wargame.envs.per_model.random_seat import random_legal_action
from wargame_rl.wargame.envs.types import BattlePhase


def test_the_intent_at_a_movement_close_is_the_force_wide_count() -> None:
    """Per-unit counts taken at each unit's close, summed, equal one count over
    the whole force -- coherency is intra-unit -- and with no referee the
    intent equals the realised board."""
    env = PerModelEnv(small_config(opponent_x=22))
    rng = np.random.default_rng(5)
    observation, _ = env.reset(seed=500020)
    while env.game_clock_state.phase is BattlePhase.movement:
        observation, _, done, _, info = env.step(
            random_legal_action(observation.decision, rng)
        )
        if done or info["reward_settled"]:
            break
    quantities = env.rules_quantities
    expected = coherency_counts(
        env.wargame_models, quantities.coherency_nearest, quantities.coherency_furthest
    )
    units, coherent, models_out = expected
    assert env.intended_coherency_rate == coherent / units
    assert env.intended_models_out_of_coherency == float(models_out)
    assert env.coherency_rate == env.intended_coherency_rate
    assert env.models_out_of_coherency == env.intended_models_out_of_coherency


def test_the_trackers_reset_per_episode_and_the_opponents_is_opt_in() -> None:
    config = small_config(opponent_x=22)
    env = PerModelEnv(config)
    assert env.coherency_rate is None and env.opponent_coherency_rate is None
    choose = scripted_chooser(build_baseline_policy("squad_march_take"), [env])
    observation, _ = env.reset(seed=500021)
    done = False
    while not done:
        observation, _, done, _, _ = env.step(choose([env], [observation])[0])
    assert env.coherency_rate is not None
    assert env.opponent_coherency_rate is None
    env.reset(seed=500022)
    assert env.coherency_rate is None

    tracked = PerModelEnv(config.model_copy(update={"track_opponent_coherency": True}))
    choose = scripted_chooser(build_baseline_policy("squad_march_take"), [tracked])
    observation, _ = tracked.reset(seed=500021)
    done = False
    while not done:
        observation, _, done, _, _ = tracked.step(choose([tracked], [observation])[0])
    assert tracked.opponent_coherency_rate is not None
    assert tracked.opponent_intended_coherency_rate is not None
