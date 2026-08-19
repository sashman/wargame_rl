from __future__ import annotations

# Registers the `model` opponent key. Imported here rather than in `envs`
# because the policy needs a network, and `envs` importing `model` would be
# a real import cycle. Every training, evaluation and scoring path builds
# its env through this factory, so naming a checkpoint as the opponent works
# wherever it matters; a direct `WargameEnv(config)` gets a message saying so.
import wargame_rl.wargame.model.opponent  # noqa: F401  (registration side effect)
from wargame_rl.wargame.envs.renders import renderer
from wargame_rl.wargame.envs.state.exporter import StateExporter
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def create_environment(
    env_config: WargameEnvConfig,
    renderer: renderer.Renderer | None = None,
    state_exporters: list[StateExporter] | None = None,
) -> WargameEnv:
    """Create the Wargame environment.

    Returns:
        Configured gymnasium environment
    """
    env = WargameEnv(env_config, renderer, state_exporters=state_exporters)

    return env
