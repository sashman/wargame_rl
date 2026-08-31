"""The surplus-reallocation decode: a play-time policy-improvement operator.

⚠ **This is a DECODE, not a policy and not a reward.** After the network has
chosen its moves, one surplus squad per movement phase is redirected onto the
objective `choose_surplus_reallocation` nominates — the opponent's weakest-held
point, falling back to the nearest empty one — by overwriting that squad's
movement actions with the ONE shared grid cell that carries its centroid
closest to the target. Rigid, so a coherent squad stays coherent, and the env's
own referee still judges the executed move.

**Why it exists.** Measured on six trained checkpoints, n=45, K=3, frozen
weights: the contest form is worth **+8.3 ± 4.25 vp** (`docs/melee-teaching-goal.md`
§40c) — the largest lever on file — and the empty-ground form is worth
**+1.6 ± 5.00** and fails its own kill. ⚠ The gain is **not** ground taken: our
own objectives held move **+0.002 ± 0.039** per step. It is denial (**−0.053**
of theirs) plus attrition (**−4.3 pp** of their army) (§40d). Do not describe it
as allocation.

⚠ **PLAY-TIME ONLY, like every other decode here.** Folding a decode into PPO
means the executed action is not the sampled one, which measured **−51.8 vp**
from scratch (`reports/2026-08-20-decoding-does-not-belong-in-training.md`).
The supported route into the weights is **distillation** — clone the decoded
policy, then train from that basin — which is what this module exists to make
possible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wargame_rl.wargame.envs.baseline.reallocation import choose_surplus_reallocation
from wargame_rl.wargame.envs.types.game_timing import BattlePhase


def apply_reallocation(
    actions: list[int],
    env: Any,
    min_stack: int = 4,
) -> list[int]:
    """Redirect one surplus squad's movement actions, or return `actions` as-is.

    A no-op outside the movement phase, on boards offering no surplus move, and
    whenever the handler exposes no movement slice — so a caller may apply it
    unconditionally.
    """
    # ⚠ MOVEMENT PHASE ONLY, and this guard is load-bearing. Without it the
    # redirect overwrites the shooting slice and the charge ladder with
    # movement indices — measured on three checkpoints at −16 to −24 vp
    # against the same weights undecoded, while the docstring claimed the
    # no-op. The phase gate is the whole difference between the +8.3 rule and
    # a policy that shoots at movement bins.
    if env.game_clock_state.phase is not BattlePhase.movement:
        return actions
    handler = env.player_action_handler
    movement = getattr(handler, "movement_slice", None)
    if movement is None:
        return actions
    branch = choose_surplus_reallocation(env.player_models, env, min_stack)
    if branch is None:
        return actions
    donor, target = branch
    members = [
        index
        for index, model in enumerate(env.player_models)
        if int(model.group_id) == donor and model.is_alive
    ]
    if not members:
        return actions
    centre = np.asarray(env.objectives[target].location, dtype=float)
    positions = np.array([env.player_models[i].location for i in members], dtype=float)
    centroid = positions.mean(axis=0)
    grid = handler.movement_displacements()
    best = int(
        np.argmin(
            np.linalg.norm(
                (centroid[np.newaxis, :] + grid) - centre[np.newaxis, :], axis=1
            )
        )
    )
    redirected = list(actions)
    for index in members:
        redirected[index] = movement.start + best
    return redirected
