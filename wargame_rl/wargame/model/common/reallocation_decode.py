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

⚠ **ITERATING IT IS MEASURED AND REJECTED.** `max_redirects` > 1 redirects
further squads in the same phase; at n=180 on six seeds it is **−1.28 ± 0.76**
overall and **−4.33 (t=−3.17, 1 of 6 seeds)** on `vs_deny`. The default of 1 is
the shipped rule and the parameter is kept only as the measured-rejected
control, the way `squad_march_take_charge_realloc` is. See
`reports/2026-09-06-three-decode-knobs-and-none-of-them-pays.md`.

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
    max_redirects: int = 1,
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
    redirected = list(actions)
    grid = handler.movement_displacements()
    moved_groups: set[int] = set()
    used_targets: set[int] = set()
    changed = False
    # Iterated: each pass re-asks the rule with the groups already committed
    # treated as gone, so a board with several over-stacked points can empty
    # more than one of them in a turn. `max_redirects=1` is the original rule.
    for _ in range(max(1, max_redirects)):
        branch = choose_surplus_reallocation(
            env.player_models,
            env,
            min_stack,
            frozenset(moved_groups),
            frozenset(used_targets),
        )
        if branch is None:
            break
        donor, target = branch
        members = [
            index
            for index, model in enumerate(env.player_models)
            if int(model.group_id) == donor and model.is_alive
        ]
        if not members:
            break
        centre = np.asarray(env.objectives[target].location, dtype=float)
        positions = np.array(
            [env.player_models[i].location for i in members], dtype=float
        )
        centroid = positions.mean(axis=0)
        best = int(
            np.argmin(
                np.linalg.norm(
                    (centroid[np.newaxis, :] + grid) - centre[np.newaxis, :], axis=1
                )
            )
        )
        for index in members:
            redirected[index] = movement.start + best
        moved_groups.add(donor)
        used_targets.add(target)
        changed = True
    return redirected if changed else actions
