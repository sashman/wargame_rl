from functools import lru_cache
from typing import cast

import pytest

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.net import RL_Network, TransformerNetwork
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer
from wargame_rl.wargame.types import Experience


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


@pytest.fixture
@lru_cache(maxsize=1)
def transformer_net(env: WargameEnv) -> TransformerNetwork:
    return TransformerNetwork.policy_from_env(env=env)


@pytest.fixture
def policy_net(env: WargameEnv) -> RL_Network:
    """A bare policy network, for tests that take any `RL_Network`."""
    return TransformerNetwork.policy_from_env(env=env)


@pytest.fixture
def ppo_transformer_net(env: WargameEnv) -> PPO_Transformer:
    net = PPO_Transformer.from_env(env=env)
    return cast(PPO_Transformer, net.to("cpu"))


@pytest.fixture(
    params=[
        pytest.param("ppo_transformer_net", id="ppo_transformer"),
    ]
)
def ppo_net(request: pytest.FixtureRequest, env: WargameEnv) -> PPO_Transformer:
    """Parametrized fixture for PPO_Transformer."""
    assert request.param == "ppo_transformer_net"
    return cast(PPO_Transformer, PPO_Transformer.from_env(env=env).to("cpu"))
