"""A seat that draws a uniformly random LEGAL decision at every point.

Not a way of playing the game -- a way of exercising the facade: the fuzz
test drives it to prove no legal decision can break the env, and the play and
record recipes use it to watch decisions no script would take.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.sequencing.activation import CHARGE_TARGET_DECLINE
from wargame_rl.wargame.envs.per_model.types import (
    DecisionPoint,
    PerModelAction,
    StepKind,
)


def random_legal_action(
    point: DecisionPoint, rng: np.random.Generator
) -> PerModelAction:
    """A uniformly random legal decision at `point`."""
    if point.kind is StepKind.close_turn:
        return PerModelAction.close_turn()
    candidates = np.flatnonzero(point.selector_mask)
    if candidates.size == 0:
        raise AssertionError(f"{point.kind.value} point offers no selectable model")
    model = int(rng.choice(candidates))
    if point.kind is StepKind.open:
        values = np.flatnonzero(point.declaration_mask[model])
        return PerModelAction.open(model, int(rng.choice(values)))
    if point.kind is StepKind.act:
        values = np.flatnonzero(point.action_mask[model])
        return PerModelAction.act(model, int(rng.choice(values)))
    choices = [int(v) for v in np.flatnonzero(point.target_mask[model])] + [
        CHARGE_TARGET_DECLINE
    ]
    return PerModelAction.target(model, int(rng.choice(choices)))
