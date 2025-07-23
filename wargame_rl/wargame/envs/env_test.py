class EnvTest:
    # Set up the environment and and run N numbe of acitons
    
    def __init__(self, size, render_mode="human"):
        from wargame_rl.wargame.envs.wargame import WargameEnv
        self.env = WargameEnv(size=size, render_mode=render_mode)
        self.env.reset()

    def run_actions(self, num_actions=5):
        for _ in range(num_actions):
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)

            print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Info: {info}")

            if terminated or truncated:
                self.env.reset()