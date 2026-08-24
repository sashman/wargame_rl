"""Environment configuration models.

Split into modules by what they configure, but re-exported here in full: this
was one 667-line module and roughly 30 call sites import from
`...envs.types.config` directly, so the package boundary is deliberately
invisible to them.

- `battle`    — turn order, opponent policy, mission
- `coherency` — the unit coherency rule's distances and enforcement switches
- `melee`     — whether the charge and fight phases are played at all
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
from wargame_rl.wargame.envs.types.config.coherency import CoherencyConfig
from wargame_rl.wargame.envs.types.config.entities import (
    MeleeWeaponProfile,
    ModelConfig,
    ObjectiveConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.config.melee import MeleeConfig
from wargame_rl.wargame.envs.types.config.terrain import (
    DeploymentConfig,
    MapPoolConfig,
    RandomTerrainConfig,
    TerrainMapConfig,
    TerrainPieceConfig,
)

__all__ = [
    "CoherencyConfig",
    "MeleeConfig",
    "MeleeWeaponProfile",
    "MissionConfig",
    "ModelConfig",
    "ObjectiveConfig",
    "OpponentPolicyConfig",
    "DeploymentConfig",
    "MapPoolConfig",
    "RandomTerrainConfig",
    "TerrainMapConfig",
    "TerrainPieceConfig",
    "TurnOrder",
    "WargameEnvConfig",
    "WeaponProfile",
]
