from collections.abc import Iterator
from functools import lru_cache
from typing import cast

import pytest
import torch

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.config import TransformerConfig
from wargame_rl.wargame.model.net import RL_Network, TransformerNetwork
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer
from wargame_rl.wargame.types import Experience


@pytest.fixture(autouse=True)
def full_float32_precision() -> Iterator[None]:
    """Run every test at full float32 precision, whatever ran before it.

    `torch.set_float32_matmul_precision` is *process-wide* and sticky, so one
    test enabling TF32 silently changes the arithmetic of every test that
    follows it in the same worker. That is not hypothetical: with TF32 on,
    `test_transformer_shooting_policy.py::test_transformer_policy_batched_matches_single_obs`
    fails **25 times out of 25** — it compares a batched forward against single
    forwards at `atol=1e-5`, and TF32's 11 mantissa bits cannot hold that. It
    presented as a once-per-16-runs flake, which reads like numerical noise and
    is not.

    Pinned per test rather than per session so that a test which changes the
    setting deliberately -- `test_precision.py` does, and restores it -- cannot
    leave the rest of the suite running under different arithmetic if its own
    restore ever regresses.

    Tests that need another mode set it themselves; this only decides the
    starting point. TF32 is off in training too, and for the same reason it
    matters here: it costs ~8.5 vp_margin. See
    reports/2026-08-09-tf32-costs-eight-vp.md.
    """
    torch.set_float32_matmul_precision("highest")
    yield


@pytest.fixture
def n_steps() -> int:
    return 256


@pytest.fixture
def env() -> WargameEnv:
    return WargameEnv(
        config=WargameEnvConfig(render_mode=None, number_of_battle_rounds=100)
    )


@pytest.fixture(scope="session")
def _session_env() -> WargameEnv:
    """Shared env instance used only by the session-scoped experience fixtures."""
    return WargameEnv(
        config=WargameEnvConfig(render_mode=None, number_of_battle_rounds=100)
    )


@pytest.fixture(scope="session")
def experiences(_session_env: WargameEnv) -> list[Experience]:
    previous_state, _ = _session_env.reset()
    output: list[Experience] = []

    for _ in range(256):
        action = WargameEnvAction(actions=_session_env.action_space.sample())
        state, reward, terminated, _, _ = _session_env.step(action)
        output.append(
            Experience(previous_state, action, reward, terminated, state, None)
        )
        previous_state = state

    return output


# The trunk every network fixture here is built with, and it is NOT the
# production one. `TransformerConfig()` defaults to 8 layers of 256 at 8 heads
# -- ~12.7M parameters -- and the suite was paying that on a 2-model 20x20
# board, where it dominated both runtime and the 153 MB a checkpoint costs.
# Measured: the whole suite went from 155s to ~100s wall clock under `-n auto`,
# and `test_z_e2e_training` from 66.0s to 1.2s run on its own. ⚠ Read `--durations`
# under `-n auto` as wall time including worker starvation, not as a test's own
# cost -- the e2e test still reports ~40s there while taking 1.2s alone.
#
# ⚠ Any test asserting on a NUMBER a trained network produces must build its own
# production-sized trunk, because this one is a different network. Nothing here
# does: these tests assert on shapes, dtypes, wiring and invariances, none of
# which the trunk's depth changes. `test_network_size` pins that the production
# default is still 8/8/256, so shrinking the *fixtures* can never quietly shrink
# what ships.
#
# `n_heads` must divide `embedding_size`; 32 / 2 = 16 per head.
TEST_TRANSFORMER_CONFIG = TransformerConfig(n_layers=2, n_heads=2, embedding_size=32)


@pytest.fixture
@lru_cache(maxsize=1)
def transformer_net(env: WargameEnv) -> TransformerNetwork:
    return TransformerNetwork.policy_from_env(
        env=env, transformer_config=TEST_TRANSFORMER_CONFIG
    )


@pytest.fixture
def policy_net(env: WargameEnv) -> RL_Network:
    """A bare policy network, for tests that take any `RL_Network`."""
    return TransformerNetwork.policy_from_env(
        env=env, transformer_config=TEST_TRANSFORMER_CONFIG
    )


@pytest.fixture
def ppo_transformer_net(env: WargameEnv) -> PPO_Transformer:
    net = PPO_Transformer.from_env(env=env, transformer_config=TEST_TRANSFORMER_CONFIG)
    return cast(PPO_Transformer, net.to("cpu"))


@pytest.fixture(
    params=[
        pytest.param("ppo_transformer_net", id="ppo_transformer"),
    ]
)
def ppo_net(request: pytest.FixtureRequest, env: WargameEnv) -> PPO_Transformer:
    """Parametrized fixture for PPO_Transformer."""
    assert request.param == "ppo_transformer_net"
    return cast(
        PPO_Transformer,
        PPO_Transformer.from_env(
            env=env, transformer_config=TEST_TRANSFORMER_CONFIG
        ).to("cpu"),
    )
