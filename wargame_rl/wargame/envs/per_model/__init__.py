"""The per-model facade: one env step is one decision, over the unchanged domain.

A second application context beside `envs/wargame.py`. See `docs/ddd-envs.md`
and GitHub issue #283 for why it exists and what it shares with the phase
facade (the domain, the reward calculators, `BattleView`) and what it does not
(the step, the observation, every checkpoint).
"""

from wargame_rl.wargame.envs.per_model.dice_seeded import SeededDice
from wargame_rl.wargame.envs.per_model.env import (
    EpisodeOver,
    PerModelEnv,
    SettledWindow,
)
from wargame_rl.wargame.envs.per_model.evaluate import (
    evaluate_per_model_chooser,
    random_chooser,
    scripted_chooser,
)
from wargame_rl.wargame.envs.per_model.recording import ChooserFactory, record_episode
from wargame_rl.wargame.envs.per_model.scripted import ScriptedSeat
from wargame_rl.wargame.envs.per_model.types import (
    FACADE_TAG,
    BatchChooser,
    Chooser,
    DecisionPoint,
    FacadeDivergence,
    PerModelAction,
    PerModelObservation,
    PerModelProvenance,
    StepEffect,
    StepKind,
    facade_of,
    require_per_model,
)

__all__ = [
    "FACADE_TAG",
    "BatchChooser",
    "Chooser",
    "ChooserFactory",
    "DecisionPoint",
    "EpisodeOver",
    "PerModelAction",
    "PerModelEnv",
    "PerModelObservation",
    "PerModelProvenance",
    "FacadeDivergence",
    "ScriptedSeat",
    "SeededDice",
    "SettledWindow",
    "StepEffect",
    "StepKind",
    "evaluate_per_model_chooser",
    "facade_of",
    "random_chooser",
    "record_episode",
    "require_per_model",
    "scripted_chooser",
]
