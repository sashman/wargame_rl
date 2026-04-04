from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.model.ppo.config import SelfPlayConfig
from wargame_rl.wargame.model.ppo.self_play_callback import PPOSelfPlayCallback


def test_mode_for_epoch_inactive_defaults_to_update() -> None:
    callback = PPOSelfPlayCallback(
        run_name="test-run",
        env_config=WargameEnvConfig(
            number_of_wargame_models=1,
            number_of_opponent_models=1,
            number_of_objectives=1,
            objective_radius_size=1,
            opponent_policy=OpponentPolicyConfig(type="random"),
        ),
        self_play_config=SelfPlayConfig(update_epochs=3, self_play_epochs=1),
    )
    assert callback.mode_for_epoch(0) == "update"
    assert callback.mode_for_epoch(10) == "update"


def test_mode_for_epoch_after_activation_uses_3_to_1_cycle() -> None:
    callback = PPOSelfPlayCallback(
        run_name="test-run",
        env_config=WargameEnvConfig(
            number_of_wargame_models=1,
            number_of_opponent_models=1,
            number_of_objectives=1,
            objective_radius_size=1,
            opponent_policy=OpponentPolicyConfig(type="random"),
        ),
        self_play_config=SelfPlayConfig(update_epochs=3, self_play_epochs=1),
    )
    callback._activation_epoch = 5
    assert callback.mode_for_epoch(5) == "update"
    assert callback.mode_for_epoch(6) == "update"
    assert callback.mode_for_epoch(7) == "update"
    assert callback.mode_for_epoch(8) == "self_play"
    assert callback.mode_for_epoch(9) == "update"
