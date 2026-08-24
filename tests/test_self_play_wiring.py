"""Self-play is opt-in, and off costs nothing.

The three properties here are the ones that make a self-play arm comparable to
its own control. Everything else about the pool and the sampler is tested in
`test_rating_pool.py` and `test_pfsp.py`, which need no env at all.

⚠ **The load-bearing one is
`test_a_training_module_builds_no_scheduler_when_self_play_is_off`.** A version
that constructed the scheduler unconditionally and simply did not use it would
pass every other assertion in this file, and would leave an object whose stream
a later change could quietly seed from a shared source -- which is how
`augment_start` nearly shifted every layout in the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.opponent.registry import build_opponent_policy
from wargame_rl.wargame.envs.opponent.scripted_baseline_policy import (
    ScriptedBaselineOpponentPolicy,
)
from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.self_play import (
    SELF_PLAY_SEED_BASE,
    OpponentScheduler,
    SelfPlayConfig,
)
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer
from wargame_rl.wargame.selectors import build_action_selector

ARENA_CONFIG = "configs/dev/4v4_two_phases.yaml"


def _config() -> WargameEnvConfig:
    with open(ARENA_CONFIG) as handle:
        config: WargameEnvConfig = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    config.render_mode = None
    return config


def _scheduler(tmp_path: Path, **overrides: object) -> OpponentScheduler:
    settings = SelfPlayConfig(enabled=True, **overrides)  # type: ignore[arg-type]
    return OpponentScheduler(settings, tmp_path, seed=0)


def test_the_default_is_off() -> None:
    """A config carrying this block trains the control."""
    assert SelfPlayConfig().enabled is False


def test_off_refuses_to_build_a_scheduler(tmp_path: Path) -> None:
    """Nothing should construct one when self-play is off, so that no stream is
    drawn from -- the property the whole opt-in design rests on."""
    with pytest.raises(ValueError, match="enabled self-play only"):
        OpponentScheduler(SelfPlayConfig(), tmp_path)


@pytest.mark.parametrize("settings", [None, SelfPlayConfig()])
def test_a_training_module_builds_no_scheduler_when_self_play_is_off(
    settings: SelfPlayConfig | None,
) -> None:
    """The sensitivity control for the whole feature.

    A version that constructed the scheduler unconditionally and merely declined
    to use it would pass every other assertion in this file, and would leave an
    object whose stream a later change could quietly seed from a shared source
    -- which is how `augment_start` nearly shifted every layout in the repo. So
    the assertion is that the object **does not exist**, and that a rollout
    epoch starts without so much as building the envs.
    """
    env = WargameEnv(config=_config())
    module = PPOLightning(
        env=env,
        ppo_model=PPO_Transformer.from_env(env),
        log=False,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        num_rollout_envs=2,
        self_play=settings,
    )

    assert module._opponent_scheduler is None

    module.on_train_epoch_start()
    module.on_train_epoch_end()
    assert module._rollout_envs is None


def test_the_opponent_stream_has_its_own_seed_band() -> None:
    """Disjoint from rollout (0), baselines (10k), eval (500k), held-out (700k),
    cloning (800k) and ratings (900k), so turning self-play on changes who the
    learner plays and not which boards it plays on."""
    assert SELF_PLAY_SEED_BASE not in (0, 10_000, 500_000, 700_000, 800_000, 900_000)
    assert SELF_PLAY_SEED_BASE > 1_000_000


def test_enabling_it_needs_somewhere_to_put_the_snapshots() -> None:
    env = WargameEnv(config=_config())
    with pytest.raises(ValueError, match="snapshot_dir"):
        PPOLightning(
            env=env,
            ppo_model=PPO_Transformer.from_env(env),
            log=False,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            num_rollout_envs=2,
            self_play=SelfPlayConfig(enabled=True),
        )


def test_the_anchor_is_seated_as_a_scripted_baseline(tmp_path: Path) -> None:
    """Entry zero has no checkpoint, so it goes through `scripted_baseline`.

    This is also the case a fresh run spends its first epochs in: the pool holds
    only the anchor until the first snapshot epoch.
    """
    scheduler = _scheduler(tmp_path, anchor="squad_march_take")
    env = WargameEnv(config=_config(), renderer=None)
    try:
        drawn = scheduler.seat([env])
        assert drawn[0].is_anchor
        assert isinstance(env._opponent_policy, ScriptedBaselineOpponentPolicy)
        assert env._opponent_policy.baseline_name == "squad_march_take"
    finally:
        env.close()


def test_seating_survives_a_reset(tmp_path: Path) -> None:
    """`reset()` never touches `_opponent_policy`, which is why an opponent is
    installed once per epoch rather than once per episode."""
    scheduler = _scheduler(tmp_path)
    env = WargameEnv(config=_config(), renderer=None)
    try:
        scheduler.seat([env])
        installed = env._opponent_policy
        env.reset(seed=900_000)
        assert env._opponent_policy is installed
    finally:
        env.close()


def test_a_snapshot_joins_the_pool_and_lands_on_disk(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, snapshot_every_n_epochs=5)
    weights = {"layer.weight": torch.zeros(2, 2)}

    assert not scheduler.should_snapshot(4)
    assert scheduler.should_snapshot(5)
    entry = scheduler.snapshot(5, weights)

    assert Path(entry.checkpoint).exists()
    assert scheduler.pool.entries[-1].name == "epoch_5"
    assert scheduler.pool.anchor.is_anchor


def test_epoch_zero_is_not_a_snapshot(tmp_path: Path) -> None:
    """An untrained network in the pool is the anchor's job, and the anchor is
    already there -- a duplicate would just crowd out a real one."""
    assert not _scheduler(tmp_path, snapshot_every_n_epochs=5).should_snapshot(0)


def test_the_opponent_schedule_is_reproducible(tmp_path: Path) -> None:
    """Training is deterministic given seed, config and code. An opponent
    schedule that varied run to run would end that."""
    weights = {"layer.weight": torch.zeros(2, 2)}
    schedules = []
    for run in range(2):
        scheduler = _scheduler(tmp_path / f"run{run}", snapshot_every_n_epochs=1)
        for epoch in (1, 2, 3):
            scheduler.snapshot(epoch, weights)
        scheduler.rate({"epoch_1": -100.0, "epoch_2": 0.0, "epoch_3": 100.0}, 0.0)
        schedules.append(
            [
                scheduler.pool.sample(0.0, scheduler._rng, mode="hard").name
                for _ in range(20)
            ]
        )

    assert schedules[0] == schedules[1]


def test_a_rated_pool_prioritises_the_harder_opponents(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, snapshot_every_n_epochs=1)
    weights = {"layer.weight": torch.zeros(2, 2)}
    for epoch in (1, 2):
        scheduler.snapshot(epoch, weights)
    scheduler.rate({"epoch_1": -400.0, "epoch_2": 400.0}, learner_rating=0.0)

    rng = np.random.default_rng(0)
    drawn = [scheduler.pool.sample(0.0, rng, mode="hard").name for _ in range(400)]

    assert drawn.count("epoch_2") > drawn.count("epoch_1")


def test_a_snapshot_loads_back_as_an_opponent(tmp_path: Path) -> None:
    """The regression for the bug this wiring shipped with.

    A snapshot is read back through `convert_state_dict`, which looks for
    `policy_net.` or `ppo_model.policy_network.` -- the prefixes a real Lightning
    checkpoint carries. Saving `self.ppo_model.state_dict()` gives bare
    `policy_network.` keys and raises on load, which is what a three-epoch smoke
    run found.

    It raised loudly only because this path loads **strict**. The very same
    mistake through `_apply_warm_start_weights`, which uses `strict=False`,
    loads *nothing* and reports a warm start -- so this test asserts the
    opponent actually plays rather than merely that it constructs.
    """
    env = WargameEnv(config=_config())
    module = PPOLightning(
        env=env,
        ppo_model=PPO_Transformer.from_env(env),
        log=False,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        num_rollout_envs=2,
        self_play=SelfPlayConfig(enabled=True, snapshot_every_n_epochs=1),
        snapshot_dir=tmp_path,
    )
    assert module._opponent_scheduler is not None
    entry = module._opponent_scheduler.snapshot(1, module.state_dict())

    play_env = create_environment(env_config=_config())
    try:
        policy = build_opponent_policy(
            OpponentPolicyConfig(type="model", params={"checkpoint": entry.checkpoint}),
            play_env,
        )
        play_env.set_opponent_policy(policy)
        select = build_action_selector("squad_march", play_env).select
        observation, _ = play_env.reset(seed=900_000)
        for _ in range(3):
            observation, _r, done, _t, _i = play_env.step(select(observation, play_env))
            if done:
                break
    finally:
        play_env.close()
