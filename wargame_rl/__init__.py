from gymnasium.envs.registration import register
register(
    id="gymnasium_env/WargameEnv-v0",
    entry_point="wargame_rl.rl.gym.wargame.envs.wargame:WargameEnv",
)