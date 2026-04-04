from train import AlgorithmType, _build_default_run_base_name
from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.model.dqn.config import NetworkType


def test_build_default_run_base_name_without_opponent_policy() -> None:
    env_config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=0,
        number_of_objectives=2,
        board_width=60,
        board_height=44,
    )

    run_base_name = _build_default_run_base_name(
        AlgorithmType.PPO, NetworkType.TRANSFORMER, env_config
    )

    assert run_base_name == "ppo-transformer-m4-opp0-obj2-b60x44-ph1"


def test_build_default_run_base_name_with_opponent_policy() -> None:
    env_config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=2,
        board_width=60,
        board_height=44,
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
    )

    run_base_name = _build_default_run_base_name(
        AlgorithmType.PPO, NetworkType.TRANSFORMER, env_config
    )

    assert (
        run_base_name
        == "ppo-transformer-m4-opp4-obj2-b60x44-ph1-vs-scripted_advance_to_objective"
    )
