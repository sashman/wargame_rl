from wargame_rl.wargame.envs.renders.human import HumanRender
from wargame_rl.wargame.envs.types import (
    OpponentPolicyConfig,
    WargameEnvAction,
    WargameEnvConfig,
)


class EnvTest:
    def __init__(
        self,
        board_width: int = 50,
        board_height: int | None = None,
        render_mode: str | None = "human",
    ) -> None:
        from wargame_rl.wargame.envs.wargame import WargameEnv

        if board_height is None:
            board_height = board_width

        # In CI / headless runs we want to avoid constructing `HumanRender()`
        # since `WargameEnv.reset()` will immediately call `renderer.setup()`.
        renderer = HumanRender() if render_mode is not None else None
        self.env = WargameEnv(
            WargameEnvConfig(
                board_width=board_width,
                board_height=board_height,
                render_mode=render_mode,
                objective_radius_size=2,
                number_of_wargame_models=3,
                number_of_objectives=2,
                number_of_opponent_models=3,
                opponent_policy=OpponentPolicyConfig(
                    type="scripted_advance_to_objective"
                ),
            ),
            renderer=renderer,
        )
        self.env.reset()

    def run_actions(self, num_actions: int = 5) -> None:
        for _ in range(num_actions):
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(
                WargameEnvAction(actions=action)
            )
            self.env.render()

            print(
                f"Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Info: {info}"
            )

            if terminated or truncated:
                self.env.reset()

    def reset(self) -> None:
        self.env.reset()
