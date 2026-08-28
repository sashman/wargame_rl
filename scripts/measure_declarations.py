"""The declaration census — §31's S1 farm screen and S2 ablation data source.

    just measure-declarations <ckpt> <config.yaml> [n_episodes] [decode_topk] [ablate_onehot]

Per policy it reports, per episode: how many units hold a declaration, how often
units REdeclare, whether declarations track the nearest objective, the cosine
between unit displacement and the declared direction, and — the farm screen's
statistic — per-unit `declared_objective_progress` income against the 5.0/model
cap, with the top unit printed separately (the army mean is provably blind to a
one-unit farm: it reads 39% of cap).

`ablate_onehot=1` zeroes the declaration one-hot at PLAY (frozen weights) — §31
S2: if vp and held do not move, the plan feeds nothing through the observation
channel.

Income is recomputed here with the calculator's own formula and the scoring
distance definition (`compute_distances`), not read from the reward pipeline,
so the census works on eval configs that do not carry the term.
"""

from __future__ import annotations

import sys

import numpy as np

from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.env_components import observation_builder
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

VALUE = 0.25  # the trained term's params (declare config: value 0.25, span 6.0)
SPAN = 6.0
CAP = 5.0  # value * ~20 movement steps: the farm ceiling §31 S1 prices against


def measure(
    selector_spec: str,
    config_path: str,
    n_episodes: int,
    decode_topk: int,
    ablate_onehot: bool,
    overrides: dict[str, str],
) -> None:
    """Print the census for one policy on one config."""
    if ablate_onehot:
        budget_sized_zeros = observation_builder._declared_onehot

        def _zeros(declared: int, budget: int) -> np.ndarray:
            return budget_sized_zeros(-1, budget)

        observation_builder._declared_onehot = _zeros  # type: ignore[assignment]

    env = create_environment(load_env_config(config_path, **overrides))
    selector = build_action_selector(selector_spec, env, decode_topk=decode_topk)

    declared_unit_phases = 0  # unit-command-phases holding a declaration
    unit_command_phases = 0
    redeclarations = 0
    declarations = 0
    declared_nearest = 0
    cosines: list[float] = []
    incomes: list[dict[int, float]] = []  # per episode: unit -> summed income
    unit_alive_counts: list[dict[int, list[float]]] = []
    redecl_by_episode: list[dict[int, int]] = []
    vp_total = 0.0
    held_total = 0.0

    for episode in range(n_episodes):
        observation, _ = env.reset(seed=700000 + episode)
        prev_declared: dict[int, int] = {}
        income: dict[int, float] = {}
        alive_steps: dict[int, list[float]] = {}
        episode_redecl: dict[int, int] = {}
        done = False
        while not done:
            phase = env.game_clock_state.phase
            models = env.wargame_models
            pre_cache = compute_distances(models, env.objectives)
            pre_centroids: dict[int, np.ndarray] = {}
            unit_declared: dict[int, int] = {}
            for index, model in enumerate(models):
                if not model.is_alive:
                    continue
                group = int(model.group_id)
                unit_declared.setdefault(group, int(model.declared_objective))
            if phase is BattlePhase.movement:
                for group in unit_declared:
                    member_locs = [
                        m.location
                        for m in models
                        if m.is_alive and int(m.group_id) == group
                    ]
                    pre_centroids[group] = np.mean(np.array(member_locs), axis=0)

            action = selector.select(observation, env)
            observation, _r, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            post_models = env.wargame_models
            if phase is BattlePhase.command:
                post_declared: dict[int, int] = {}
                leader_row: dict[int, int] = {}
                for index, model in enumerate(post_models):
                    if not model.is_alive:
                        continue
                    group = int(model.group_id)
                    if group not in post_declared:
                        post_declared[group] = int(model.declared_objective)
                        leader_row[group] = index
                unit_command_phases += len(post_declared)
                cache = compute_distances(post_models, env.objectives)
                for group, declared in post_declared.items():
                    if declared >= 0:
                        declared_unit_phases += 1
                    before = prev_declared.get(group, -1)
                    if declared >= 0 and declared != before:
                        declarations += 1
                        if before >= 0:
                            redeclarations += 1
                            episode_redecl[group] = episode_redecl.get(group, 0) + 1
                        gaps = cache.model_obj_norms_offset[leader_row[group]]
                        if int(np.argmin(gaps)) == declared:
                            declared_nearest += 1
                    prev_declared[group] = declared
            elif phase is BattlePhase.movement:
                post_cache = compute_distances(post_models, env.objectives)
                for index, model in enumerate(post_models):
                    declared = int(model.declared_objective)
                    if not model.is_alive or declared < 0:
                        continue
                    if declared >= pre_cache.model_obj_norms_offset.shape[1]:
                        continue
                    closed = float(
                        pre_cache.model_obj_norms_offset[index, declared]
                    ) - float(post_cache.model_obj_norms_offset[index, declared])
                    pay = VALUE * float(np.clip(closed / SPAN, 0.0, 1.0))
                    group = int(model.group_id)
                    income[group] = income.get(group, 0.0) + pay
                for group, declared in unit_declared.items():
                    if declared < 0 or group not in pre_centroids:
                        continue
                    member_locs = [
                        m.location
                        for m in post_models
                        if m.is_alive and int(m.group_id) == group
                    ]
                    if not member_locs:
                        continue
                    displacement = (
                        np.mean(np.array(member_locs), axis=0) - pre_centroids[group]
                    )
                    target_dir = (
                        np.array(env.objectives[declared].location)
                        - pre_centroids[group]
                    )
                    norm = np.linalg.norm(displacement) * np.linalg.norm(target_dir)
                    if norm > 1e-9:
                        cosines.append(float(np.dot(displacement, target_dir) / norm))
            if phase is BattlePhase.movement:
                for group in unit_declared:
                    members = sum(
                        1
                        for m in post_models
                        if m.is_alive and int(m.group_id) == group
                    )
                    if members:
                        totals = alive_steps.setdefault(group, [0.0, 0.0])
                        totals[0] += float(members)
                        totals[1] += 1.0
        vp_total += float(env.player_vp) - float(env.opponent_vp)
        cache = compute_distances(env.wargame_models, env.objectives)
        opp_cache = compute_distances(env.opponent_models, env.objectives)
        alive = np.array([m.is_alive for m in env.wargame_models])
        opp_alive = np.array([m.is_alive for m in env.opponent_models])
        ours = ((cache.model_obj_norms_offset <= cache.obj_radii) & alive[:, None]).sum(
            axis=0
        )
        theirs = (
            (opp_cache.model_obj_norms_offset <= opp_cache.obj_radii)
            & opp_alive[:, None]
        ).sum(axis=0)
        held_total += float((ours > theirs).sum())
        incomes.append(income)
        unit_alive_counts.append(alive_steps)
        redecl_by_episode.append(episode_redecl)

    per_model_ep: list[float] = []
    top_unit_ep: list[float] = []
    for income, alive_steps in zip(incomes, unit_alive_counts):
        # per-model income = unit income / mean alive members over its
        # declared movement phases (counted directly, never inferred from a
        # phase count -- skip_phases varies by config).
        per_unit = {}
        for group, total in income.items():
            members_sum, phases = alive_steps.get(group, [1.0, 1.0])
            per_unit[group] = total / max(members_sum / max(phases, 1.0), 1e-9)
        if per_unit:
            top_unit_ep.append(max(per_unit.values()))
            per_model_ep.append(float(np.mean(list(per_unit.values()))))
        else:
            top_unit_ep.append(0.0)
            per_model_ep.append(0.0)

    label = "ablated" if ablate_onehot else "live"
    print(
        f"  onehot={label}  decl_frac={declared_unit_phases / max(unit_command_phases, 1):.3f}  "
        f"declarations/ep={declarations / n_episodes:.2f}  "
        f"redecl/ep={redeclarations / n_episodes:.2f}  "
        f"P(nearest)={declared_nearest / max(declarations, 1):.3f}"
    )
    print(
        f"  cosine={np.mean(cosines) if cosines else 0.0:+.3f}  "
        f"income/model/ep={np.mean(per_model_ep):.3f}  "
        f"top_unit/model/ep={np.mean(top_unit_ep):.3f}  cap={CAP}  "
        f"top_unit_frac_of_cap={np.mean(top_unit_ep) / CAP:.3f}"
    )
    print(
        f"  mean_unit_redecl/ep={np.mean([v for d in redecl_by_episode for v in d.values()]) if any(redecl_by_episode) else 0.0:.2f}  "
        f"max_unit_redecl/ep={max((max(d.values()) for d in redecl_by_episode if d), default=0)}  "
        f"vp={vp_total / n_episodes:+.1f}  held={held_total / n_episodes:.2f}"
    )


def main() -> None:
    """CLI entry point."""
    positionals, overrides = parse_overrides(sys.argv[1:])
    if len(positionals) < 2:
        print(__doc__)
        raise SystemExit(1)
    selector_spec = positionals[0]
    config_path = positionals[1]
    n_episodes = int(positionals[2]) if len(positionals) > 2 else 20
    decode_topk = int(positionals[3]) if len(positionals) > 3 else 3
    ablate_onehot = bool(int(positionals[4])) if len(positionals) > 4 else False
    print(
        f"{config_path}{describe(overrides)}  (n={n_episodes}, seeds 700000+, "
        f"K={decode_topk}, onehot={'ABLATED' if ablate_onehot else 'live'})"
    )
    measure(
        selector_spec, config_path, n_episodes, decode_topk, ablate_onehot, overrides
    )


if __name__ == "__main__":
    main()
