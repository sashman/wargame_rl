"""Environment configuration models.

Split into modules by what they configure, but re-exported here in full: this
was one 667-line module and roughly 30 call sites import from
`...envs.types.config` directly, so the package boundary is deliberately
invisible to them.

- `battle`    — turn order, opponent policy, mission
- `entities`  — weapon profiles, models, objectives
- `terrain`   — fixed pieces, named maps, the random generator
- `env`       — `WargameEnvConfig`, which composes all of the above
- `_validation` — the checks the models share

The dependency runs one way: `env` imports the rest, and nothing imports `env`.
"""

from wargame_rl.wargame.envs.types.config.battle import (
    MissionConfig,
    OpponentPolicyConfig,
    TurnOrder,
)
from wargame_rl.wargame.envs.types.config.entities import (
    ModelConfig,
    ObjectiveConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.config.terrain import (
    MapPoolConfig,
    RandomTerrainConfig,
    TerrainMapConfig,
    TerrainPieceConfig,
)

__all__ = [
    "MissionConfig",
    "ModelConfig",
    "ObjectiveConfig",
    "OpponentPolicyConfig",
    "MapPoolConfig",
    "RandomTerrainConfig",
    "TerrainMapConfig",
    "TerrainPieceConfig",
    "TurnOrder",
    "WargameEnvConfig",
    "WeaponProfile",
]
