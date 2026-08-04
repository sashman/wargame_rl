"""Tests for lockstep batched evaluation.

Evaluation used to run every episode sequentially through a single env. Because
`terminate_on_success: false` makes every episode exactly `max_turns` steps,
episodes can run lockstep across several envs with one batched forward pass —
measured 2.5x faster on the 25v25 config, ~21% of an epoch.

The property that matters is that batching changes only the speed.
"""

from __future__ import annotations

import pytest
import torch

from wargame_rl.wargame.envs.state import EventLogExporter
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.lightning_base import EVAL_SEED_BASE
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer

N_EPISODES = 4


def _make_module() -> PPOLightning:
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=3,
            number_of_objectives=2,
            number_of_battle_rounds=4,
        )
    )
    return PPOLightning(
        env=env,
        ppo_model=PPO_Transformer.from_env(env),
        log=False,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        num_rollout_envs=2,
    )


def test_batched_evaluation_matches_sequential() -> None:
    """Batching the forward pass changes throughput, not results.

    Runs the same greedy policy over the same seeds one env at a time, and
    asserts the batched path produces identical VP and step counts.
    """
    module = _make_module()
    module._set_policy_mode(True)
    seeds = [EVAL_SEED_BASE + i for i in range(N_EPISODES)]

    with torch.no_grad():
        batched = module._run_episodes_batched(N_EPISODES, seeds)
    assert batched is not None
    steps, player_vps, opponent_vps = (
        batched.steps,
        batched.player_vps,
        batched.opponent_vps,
    )

    env = module.env
    sequential_vps: list[float] = []
    sequential_steps: list[int] = []
    with torch.no_grad():
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            count = 0
            while not (terminated or truncated):
                actions = module._batch_greedy_actions(
                    observation_to_tensor(observation, module._policy_model_device())
                )
                action = WargameEnvAction(actions=[int(a) for a in actions[0].tolist()])
                observation, _r, terminated, truncated, _info = env.step(action)
                count += 1
            sequential_vps.append(float(env.player_vp))
            sequential_steps.append(count)

    assert player_vps == pytest.approx(sequential_vps)
    assert steps == sequential_steps
    assert len(opponent_vps) == N_EPISODES


def test_evaluation_uses_fixed_seeds_across_calls() -> None:
    """Two evaluations of an unchanged policy give identical numbers.

    Objective placement dominates episode variance, so resampling layouts each
    epoch would make a training curve mostly a record of which maps were drawn.
    Fixed seeds make epoch-to-epoch movement attributable to the policy.
    """
    module = _make_module()
    module._set_policy_mode(True)
    seeds = [EVAL_SEED_BASE + i for i in range(N_EPISODES)]

    with torch.no_grad():
        first = module._run_episodes_batched(N_EPISODES, seeds)
        second = module._run_episodes_batched(N_EPISODES, seeds)

    assert first is not None and second is not None
    assert first == second


def test_eval_envs_share_the_curriculum_position() -> None:
    """Evaluation scores the phase the curriculum has actually reached."""
    module = _make_module()
    envs = module._ensure_eval_envs(2)

    assert envs
    for env in envs:
        assert env.phase_manager.position is module.env.phase_manager.position


def test_eval_envs_are_built_once_and_reused() -> None:
    """Eval envs persist across evaluations rather than being rebuilt."""
    module = _make_module()

    first = module._ensure_eval_envs(2)
    second = module._ensure_eval_envs(2)

    assert first[0] is second[0]


def test_evaluation_still_records_event_logs() -> None:
    """`--record-events` keeps working once evaluation runs on its own envs.

    The exporter is attached to the training env, which batched evaluation no
    longer steps. Without forwarding it, a run set to record silently produces
    an empty log — the same failure that left five earlier runs with
    `--record-events` set and nothing on disk.

    Only env 0 records; recording every env would interleave concurrent
    episodes into a single log.
    """
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=3,
            number_of_objectives=2,
            number_of_battle_rounds=4,
        ),
        state_exporters=[EventLogExporter(anchor_interval=10)],
    )
    module = PPOLightning(
        env=env,
        ppo_model=PPO_Transformer.from_env(env),
        log=False,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        num_rollout_envs=2,
    )
    module._set_policy_mode(True)

    with torch.no_grad():
        module._run_episodes_batched(2, [EVAL_SEED_BASE, EVAL_SEED_BASE + 1])

    exporter = env.state_exporters[0]
    assert isinstance(exporter, EventLogExporter)
    assert len(exporter.log) > 1, "evaluation recorded no steps"

    eval_envs = module._eval_envs or []
    assert eval_envs[0].state_exporters
    assert not eval_envs[1].state_exporters
