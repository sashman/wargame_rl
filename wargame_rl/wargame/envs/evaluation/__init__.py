"""The evaluation kernel: what both facades' evaluators share.

Two application contexts play the same game (`docs/ddd-envs.md` § Two
application contexts over one domain), and a score has to mean the same
thing whichever one produced it. This package holds the result value object
and the end-of-episode readouts (`held`, `on_obj`, `alive`, the cohesion gap)
so the two runners cannot answer "how many objectives were held" differently.

It depends on `domain/`, `env_components/` and `types/` only -- never on a
facade, `model/`, `rating/` or torch -- which is what lets `per_model/` import
it without importing the phase facade through `baseline/evaluate.py`.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.evaluation.constants import EVAL_WAVE_SIZE
from wargame_rl.wargame.envs.evaluation.readouts import (
    EndOfEpisode,
    read_end_of_episode,
)
from wargame_rl.wargame.envs.evaluation.result import (
    EvalResult,
    format_optional_metric,
    mean_of_measured,
    paired_difference,
    standard_error,
)

__all__ = [
    "EVAL_WAVE_SIZE",
    "EndOfEpisode",
    "EvalResult",
    "format_optional_metric",
    "mean_of_measured",
    "paired_difference",
    "read_end_of_episode",
    "standard_error",
]
