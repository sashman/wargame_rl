from gymnasium.envs.registration import register
register(
    id="gymnasium_env/WargameEnv-v0",
    entry_point="wargame_rl.wargame.envs.wargame:WargameEnv",
)