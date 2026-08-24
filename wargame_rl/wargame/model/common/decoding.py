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

from wargame_rl.wargame.envs.domain.movement import resolve_move
from wargame_rl.wargame.envs.domain.value_objects import position
from wargame_rl.wargame.envs.env_components.actions import _base_arrays
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv

# Above this many candidates the enumeration stops being free. A unit is left to
# its independent decode rather than silently costing a second a step -- K^k
# grows fast, and this project has been burned by per-model python loops before.
DEFAULT_MAX_CANDIDATES = 4096

# Action 0 is the no-op the handler decodes to a zero displacement. Standing
# still cannot *break* coherency -- positions do not change -- which is why the
# scripted policies collect 38-57% of their moves legal for free.
STAY_ACTION = 0


def _displacement_table(env: WargameEnv) -> np.ndarray:
    """Every model's every action as a (dx, dy) — ``(n_models, n_actions, 2)``.

    Read from the handler rather than rebuilt. Recomputing the polar encoding
    here would be a second source of truth for the action space, and a drift
    between them would move models to places the decoder did not evaluate.

    Per model, not per action alone, because a model's speed bins span its own
    Move characteristic: with differing M the same action index is a different
    displacement for different models, and one shared row would certify
    combinations nobody can make. Every row is identical when the army is
    uniformly fast, which is every config shipped today.

    Every slice that DISPLACES is built; the rest stay zero because they really
    do displace nobody — a shooting action names a target and a `move_type`
    action declares one.

    ⚠ The advance slice is built here too, and for a long time it was not. The
    loop ran over `1 + n_move_actions`, which is stay plus the movement slice
    only, and the advance slice is registered *after* shooting — so every
    advance action was modelled as a **zero displacement** while `env.step`
    applied a real 8-12". That did not merely mis-score advances, it reversed
    the decoder's opinion of them: `ends == positions` makes `_coherent_mask`
    return True whenever the unit is *already* coherent, so advance combinations
    were certified legal at ~90% against a true rate near 68%, and among legal
    candidates the highest log-prob wins. Measured on the shipped table, the
    executed advance share was inflated 1.32-1.43x. `verify_moves` could not
    catch it because `_resolve_endpoints` is handed this same array.

    `advance_roll` must be passed per model: the rung is absolute but
    `decode_action` clamps it to `M + roll`, so a table built with the default
    roll of 0 would model every advance as a normal move instead.
    """
    handler = env.player_action_handler
    models = env.player_models
    table = np.zeros((len(models), handler.n_actions, 2), dtype=float)
    advance = handler.advance_slice
    for model_idx, model in enumerate(models):
        roll = float(getattr(model, "advance_roll", 0.0))
        for action in range(1 + handler.n_move_actions):
            table[model_idx, action] = handler.decode_action(
                action, model_idx=model_idx
            )
        if advance is None:
            continue
        for action in range(advance.start, advance.end):
            table[model_idx, action] = handler.decode_action(
                action, model_idx=model_idx, advance_roll=roll
            )
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


def _resolve_endpoints(
    env: WargameEnv,
    member_indices: list[int],
    combo: np.ndarray,
    displacements: np.ndarray,
) -> np.ndarray:
    """Where the environment would actually put this unit, for one combination.

    Mirrors `ActionHandler.apply`: clamp into the board first (the edge is
    clamped into the *displacement*, before collisions), then `resolve_move`
    against enemy bases as blockers and friendly bases as passable-but-not-
    endable, in model index order so earlier members displace later ones.

    Two documented divergences from `apply`, both unavoidable here and both
    conservative:

    * models of *other* units move in the same loop, so their post-move
      positions are unknowable at decode time. They are taken where they stand.
    * the moving unit's members are resolved against that snapshot rather than
      against a board where every unit has already moved.

    The alternative -- the free translation this replaced -- was wrong on 49.8%
    of models with a median error of 1.75in against a 2in decision band, and
    certified 9.3% of unit-moves legal that landed incoherent.
    """
    models = env.player_models
    opponents = env.opponent_models
    radius = float(models[member_indices[0]].base_radius)
    lower = position(radius, radius)
    upper = position(env.config.board_width - radius, env.config.board_height - radius)
    blocker_centres, blocker_radii = _base_arrays(opponents if radius > 0.0 else None)
    live = [m for m in models if m.is_alive]
    friendly_centres = np.array([m.location for m in live], dtype=float)
    friendly_radii = np.array([m.base_radius for m in live], dtype=float)
    live_ids = {id(m): j for j, m in enumerate(live)}

    ends = np.zeros((len(member_indices), 2), dtype=float)
    for j, index in enumerate(member_indices):
        model = models[index]
        start = np.asarray(model.location, dtype=float)
        keep = np.ones(len(live), dtype=bool)
        keep[live_ids[id(model)]] = False
        in_bounds = np.clip(start + displacements[index, combo[j]], lower, upper)
        ends[j] = resolve_move(
            start,
            in_bounds - start,
            model.base_radius,
            blocker_centres,
            blocker_radii,
            friendly_centres[keep],
            friendly_radii[keep],
        )
        # Earlier members hold the ground they just took, exactly as `apply`
        # rebuilds its friendly array from live locations on every iteration.
        friendly_centres[live_ids[id(model)]] = ends[j]
    return ends


def decode_joint_coherent(
    log_probs: np.ndarray,
    actions: list[int],
    env: WargameEnv,
    top_k: int,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    include_stay: bool = False,
    safety_margin: float = 0.0,
    verify_moves: bool = True,
    max_verifications: int = 24,
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
        safety_margin: Inches to tighten the chain distance by when judging a
            candidate. **The decoder validates `position + displacement`; the
            environment does not put models there.** `resolve_move` backs a
            model off any base it would end inside, stops it dead on an enemy
            base, and clamps to the board edge, and models resolve sequentially
            so earlier movers displace later ones. Measured on the control
            checkpoint over five held-out episodes with enforcement off:
            **49.8% of models do not land where this function predicted**, mean
            offset 0.820in and p90 2.005in -- the same size as the whole 2in
            chain band -- and **9.3% of the combinations certified legal here
            were illegal by the time the referee looked**. Simulating the
            resolution exactly is not possible per unit (models of other units
            move in the same loop) and 243 combos x k models of python collision
            resolution would cost far more than the decode itself. A margin buys
            the same protection generically, at the price of a smaller legal set.
        verify_moves: Re-check shortlisted candidates against the endpoints the
            environment would actually produce (`_resolve_endpoints`) and take
            the best that survives, instead of trusting the free-translation
            relaxation. **On by default, because the relaxation was a defect.**
            Paired on three control seeds plus a distilled clone, nine held-out
            tables at n=30, only the decode differing: **+6.4 vp (sd 1.4, t=7.7)
            and +0.096 unit coherency (sd 0.004, t=40)**, every policy improving
            on both axes. Pass False to reproduce a number measured before this
            existed. This is the principled version of `safety_margin`: the
            relaxation is wrong on 49.8% of models by a median 1.75in against a
            2in band, and ~83% of the decoder's residual illegality is that
            error rather than a poor candidate set.
        max_verifications: How many shortlisted candidates to resolve exactly
            before giving up and taking the relaxation's best. Bounded because
            243 x k collision resolutions would cost far more than the decode;
            only ~5% of actions change at all, so the first candidate usually
            passes.
        include_stay: Stand a unit still when its top-K set contains no legal
            combination at all, instead of handing it back for the referee to
            revert. It applies to the 0.3-1.4% of unit-moves the decoder cannot
            solve, and the two outcomes put the unit in the *same place* -- but
            a revert additionally runs the overlap cascade, which drags
            *neighbouring* units back and accounts for 9.2-15.3% of all freezes,
            while a deliberate stay triggers nothing. Declined when the unit is
            already incoherent, because standing still cannot close a casualty
            split.

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
        # `(members, 1)` against `(C, k)` picks each member's own row, so a
        # unit of models with different Move characteristics is enumerated at
        # the distances each of them can actually travel.
        member_rows = np.asarray(member_indices, dtype=int)[None, :]
        ends = positions[None, :, :] + displacements[member_rows, combos]
        legal = _coherent_mask(ends, radii, max(nearest - safety_margin, 0.0), furthest)
        if not legal.any():
            # A strict fallback, never a competitor: standing still is only ever
            # taken when the ranked set offers nothing legal, so it cannot make
            # the policy passive. It is declined when the unit is *already*
            # incoherent -- a casualty split cannot be closed by not moving,
            # which is exactly what attrition is for.
            # Judged at the TRUE distance: standing still moves nobody, so
            # nothing can back off and the margin has no work to do here.
            if (
                include_stay
                and _coherent_mask(positions[None], radii, nearest, furthest)[0]
            ):
                for index in member_indices:
                    decoded[index] = STAY_ACTION
            continue
        score = np.stack(
            [log_probs[i][combos[:, j]] for j, i in enumerate(member_indices)], axis=1
        ).sum(axis=1)
        ranked_legal = np.argsort(-np.where(legal, score, -np.inf))[: int(legal.sum())]
        best = combos[ranked_legal[0]]
        if verify_moves:
            # The relaxation is a cheap SHORTLIST, not the answer: walk it in
            # score order and take the first candidate that is still coherent
            # once the env's own resolution has been applied. Bounded, because
            # only ~5% of actions change at all -- the first candidate usually
            # passes -- and an unbounded walk would put 243 x k collision
            # resolutions on the hot path.
            for candidate in ranked_legal[:max_verifications]:
                resolved = _resolve_endpoints(
                    env, member_indices, combos[candidate], displacements
                )
                if _coherent_mask(resolved[None], radii, nearest, furthest)[0]:
                    best = combos[candidate]
                    break
        for j, index in enumerate(member_indices):
            decoded[index] = int(best[j])
    return decoded
