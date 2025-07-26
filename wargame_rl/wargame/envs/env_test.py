from wargame_rl.wargame.envs.env_types import WargameEnvAction, WargameEnvConfig


class EnvTest:
    # Set up the environment and and run N numbe of acitons

    def __init__(self, size, render_mode="human"):
        from wargame_rl.wargame.envs.wargame import WargameEnv

        self.env = WargameEnv(
            WargameEnvConfig(
                size=size, render_mode=render_mode, number_of_wargame_models=3
            )
        )
        self.env.reset()

    def run_actions(self, num_actions=5):
        for _ in range(num_actions):
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(
                WargameEnvAction(actions=action)
            )

            print(
                f"Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Info: {info}"
            )

            if terminated or truncated:
                self.env.reset()
