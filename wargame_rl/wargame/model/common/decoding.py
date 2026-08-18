"""Decode per-model action preferences into a joint action the rules allow.

The policy emits one categorical per model and samples them independently, but
unit coherency is a property of the *joint* configuration the whole unit lands
in. The two do not match, and the mismatch is expensive: measured, an
independently decoded policy has **33.1%** of its unit-moves cancelled by the
referee, which destroys **48.9%** of all intended movement.

This module changes only the decode. For each unit it solves

    a* = argmax over (a_1..a_k) of  sum_i log pi_i(a_i | s)
         subject to coherency(x')

over the top-K actions per model, which is `K^k` candidates -- 243 at K=3 on a
five-model unit. That is a tiny constraint-satisfaction problem, not a search,
and it needs no retraining: the policy already says which actions it likes and
this picks the most probable *combination* the rules permit.

**Why this rather than repairing positions afterwards.** A position repair moves
models to places the policy never chose, so it can silently change tactical
intent. Decoding only ever reranks combinations the policy already ranked
highly, so the distortion is bounded by the top-K set.

Measured on three seeds, 30 episodes, `enforce_move: revert_unit`, attrition on:

    decode    vp_margin (mean)   intended coherency
    argmax          -26.0              0.662
    top-3            -3.5              0.852
    top-5            -5.5              0.871

Coherency rises monotonically in K; vp does not, because a wider candidate set
buys legality by accepting a less-preferred action. It changes 14-33% of
unit-moves and finds no legal combination in 0.3-1.4%.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv

# Above this many candidates the enumeration stops being free. A unit is left to
# its independent decode rather than silently costing a second a step -- K^k
# grows fast, and this project has been burned by per-model python loops before.
DEFAULT_MAX_CANDIDATES = 4096


def _displacement_table(env: WargameEnv) -> np.ndarray:
    """Every action's (dx, dy), read from the handler rather than rebuilt.

    Recomputing the polar encoding here would be a second source of truth for
    the action space, and a drift between them would move models to places the
    decoder did not evaluate.

    Only the stay and movement slices are decodable — `decode_action` indexes the
    displacement grid directly and raises on a shooting-slice action — so the
    table is built over those and zero-padded to the full width. A shooting
    action displaces nothing, which is what the zeros say.
    """
    handler = env.player_action_handler
    table = np.zeros((handler.n_actions, 2), dtype=float)
    for action in range(1 + handler.n_move_actions):
        table[action] = handler.decode_action(action)
    return table


def _coherent_mask(
    candidates: np.ndarray, radii: np.ndarray, nearest: float, furthest: float
) -> np.ndarray:
    """Which of `(C, k, 2)` joint configurations satisfy the whole rule.

    All three clauses, vectorised over candidates. Connectivity is the one that
    needs care: it is a transitive closure, not a pairwise test, so it is taken
    as `(I + A)^(k-1) > 0` everywhere -- true exactly when every model reaches
    every other along chain edges.
    """
    deltas = candidates[:, :, None, :] - candidates[:, None, :, :]
    centres = np.sqrt((deltas**2).sum(-1))
    gaps = np.maximum(centres - (radii[None, :, None] + radii[None, None, :]), 0.0)
    size = candidates.shape[1]
    eye = np.eye(size, dtype=bool)
    chain_edges = (gaps <= nearest) & ~eye[None]
    chain_ok = chain_edges.any(axis=2).all(axis=1)
    spread_ok = ((gaps <= furthest) | eye[None]).all(axis=(1, 2))
    reach = (chain_edges | eye[None]).astype(np.float32)
    power = reach.copy()
    for _ in range(size - 1):
        power = np.einsum("cij,cjk->cik", power, reach)
    connected = (power > 0).all(axis=(1, 2))
    result: np.ndarray = chain_ok & spread_ok & connected
    return result


def decode_joint_coherent(
    log_probs: np.ndarray,
    actions: list[int],
    env: WargameEnv,
    top_k: int,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[int]:
    """Rerank each unit's actions into the most probable coherency-legal joint one.

    Args:
        log_probs: ``(n_models, n_actions)`` per-model log probabilities, already
            masked by the network.
        actions: The independently decoded actions, returned unchanged for any
            unit this declines to handle.
        env: Read for positions, unit membership, base radii and the coherency
            distances. Nothing is mutated.
        top_k: Candidates per model. 1 disables the decoder entirely.
        max_candidates: Skip a unit whose ``top_k ** size`` exceeds this.

    Returns:
        A new action list. Units of fewer than two live models, and units with
        no legal combination in the candidate set, keep their original actions —
        the caller's enforcement then applies as it always did.
    """
    if top_k <= 1:
        return list(actions)
    # Coherency is judged on where a *move* leaves the unit, so there is nothing
    # to decode in any other phase — and a shooting action displaces nobody.
    if env.game_clock_state.phase is not BattlePhase.movement:
        return list(actions)

    decoded = list(actions)
    models = env.player_models
    quantities = env.rules_quantities
    nearest = quantities.scale.to_units(env.config.coherency.nearest_distance)
    furthest = quantities.scale.to_units(env.config.coherency.furthest_distance)
    displacements = _displacement_table(env)

    by_unit: dict[int, list[int]] = {}
    for index, model in enumerate(models):
        if model.is_alive and index < len(decoded):
            by_unit.setdefault(int(model.group_id), []).append(index)

    for member_indices in by_unit.values():
        size = len(member_indices)
        if size < 2 or top_k**size > max_candidates:
            continue
        positions = np.array([models[i].location for i in member_indices], dtype=float)
        radii = np.array([models[i].base_radius for i in member_indices], dtype=float)
        # Restrict to actions the network's own mask left possible: a masked
        # action has -inf log-probability and must never be decoded into.
        per_model = []
        for index in member_indices:
            ranked = np.argsort(-log_probs[index])[:top_k]
            per_model.append(ranked[np.isfinite(log_probs[index][ranked])])
        if any(candidates.size == 0 for candidates in per_model):
            continue
        combos = np.array(list(itertools.product(*per_model)))
        ends = positions[None, :, :] + displacements[combos]
        legal = _coherent_mask(ends, radii, nearest, furthest)
        if not legal.any():
            continue
        score = np.stack(
            [log_probs[i][combos[:, j]] for j, i in enumerate(member_indices)], axis=1
        ).sum(axis=1)
        best = combos[int(np.argmax(np.where(legal, score, -np.inf)))]
        for j, index in enumerate(member_indices):
            decoded[index] = int(best[j])
    return decoded
