"""Does a policy use the charge phase, and does it use it competently?

The primary readouts of [the melee teaching goal](../docs/melee-teaching-goal.md),
which exists because melee is a **core rule**: the question is how well a policy
charges, never whether charging is worth having.

    just measure-charges <policy|ckpt> <config.yaml> [n_episodes] [decode_topk]

⚠ **Read `stood` per episode, not the standing fraction.** The fraction's
denominator is the policy's own declaration count, so it *rises* when a policy
declares less -- it is not monotone in competence, and a gate written on it
rejects policies that land more charges. `stood/ep` is the numerator alone, with
a hard floor at zero.

⚠ **Quote the K.** At `decode_topk` 3 the joint decoder picks legal combinations
FOR the network, so these counts measure the decoder: a randomly-initialised
network stands 1.17-3.67 charges an episode at K=3 and 0.00-1.67 at K=1. Training
decodes at K=1, so **K=1 is the column that decides**.

⚠ **A charge STOOD iff the REFEREE SAYS SO.** `charged_this_turn` is set at
`actions.py:1160` only when `_charge_preconditions_hold` and `_charge_is_legal`
both pass; the reverting branch is its `else`. Read it *inside* the step, right
after `_apply_player_action` returns.

⚠ **DO NOT count a charge as stood because its models MOVED.** That was this
script's rule until 2026-08-26 and it is wrong in a policy-dependent direction.
One `env.step` also runs, after the charge referee has already reverted a failed
charge: pile-in for both forces, the fight step, consolidate for both, and **the
entire opponent turn** (`turn_execution.py:55-60`), whose own pile-in moves the
player's models again. A unit whose charge was reverted and which the opponent
then charged is displaced by pile-in and was scored as having STOOD.

The false positives are drawn from the *failed-attempt* pool, so the error grows
with incompetence: measured inflation was **+2.9% on the scripted teacher, +300%
on an untrained network and +500% on the arm**. The old rule was therefore
ANTI-monotone in competence -- the exact disease this docstring already warned
about for the standing *fraction* while asserting the numerator was free of it.
⚠ **A docstring that lists its author's past mistakes reads as audited and is
not.** Six panellists trusted this one without validating it.

⚠ **AN AGENT ROW DOES NOT REPRODUCE AND A SCRIPTED ROW DOES.** Measured
2026-08-26: `squad_march_take_charge` scored 4.44/4.22/3.89/+13.3 on three
consecutive runs, identical to every digit, while the SAME checkpoint on the
same config, seeds and `decode_topk` scored `stood/ep` 0.22, 0.22, 0.33 -- and
declarations, coherency and vp moved with it, so whole episodes diverged. It
varies **within one process** as well as across processes.

Not the env: the dice, map draws and opponent are all seeded in `reset`
(`wargame.py:1247-1248`), which is why the scripted rows are exact. Not
sampling: `_resolve_checkpoint` is a per-model `argmax` under `no_grad` with
`eval()`. The logits themselves differ between calls, and at K=1 one flipped
near-tie diverges the episode. ⚠ **UNRESOLVED** --
`torch.use_deterministic_algorithms(True, warn_only=False)`, `cudnn.deterministic`
and `CUBLAS_WORKSPACE_CONFIG` set before the first matmul do **not** fix it, so
the cause is not yet established and must not be asserted.

**Until it is: quote an agent row from repeats, never from one run**, and prefer
a larger `n`. At n=9 with `stood/ep` under 1.0 a single flipped charge is a
30-50% relative move -- larger than most differences this goal cares about. The
noise floor is ASYMMETRIC, so an agent-versus-script comparison inherits it on
one side only.

⚠ **A probe needs a KNOWN-ANSWER row.** `squad_march_take_charge` must read
close to its published `stood/ep`; if it does not, the instrument changed and
not the policy.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.domain.engagement import engagement_matrix
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

STAY_ACTION = 0


def measure(
    selector_spec: str,
    config_path: str,
    n_episodes: int,
    decode_topk: int,
    overrides: dict[str, str],
) -> dict[str, float]:
    """Per-episode charge counts for one policy, judged by the referee's verdict."""
    env = create_environment(load_env_config(config_path, **overrides))
    declared = attempted = stood = 0
    # §34's charge census rider: per declared unit, the PRE-move edge gap to
    # its nearest alive enemy, the 2D6 it rolled, and the referee's verdict.
    # This is the calibration dataset the branch-true ChargeField is specified
    # against — its P_2D6 must beat the empirical P(stood | gap) printed here.
    charge_rows: list[tuple[float, float, bool]] = []
    # The fold pre-registration's LABEL (i), pinned by panel B's B-4: stood
    # charges split by route (grant-attributable = the unit held a declared
    # enemy target; manual = it did not), and — within the grant route — by
    # whether the unit ended engaged with its DECLARED group or a bystander.
    # The mismatch denominator is the grant route ONLY.
    stood_grant = stood_manual = stood_on_declared = 0
    # Panel B-6's PIN-SKEW label: stood charges split by whether the charging
    # unit began the charge step with a member on an objective its side
    # CONTROLS (the hold-plus-pin composite). >70% reads an A-pass as
    # pin-enablement; <50% kills the pin-skew account. Membership uses THE
    # control rule (`norms_offset <= obj_radii`, base edge, alive only).
    stood_from_own_objective = 0

    # The referee's verdict, captured before the fight boundary clears it.
    # `_apply_player_action` runs the charge referee; `run_after_player_action`
    # (the very next statement in `step`) resolves the fight and clears the
    # flag, so this is the only window in which it can be read.
    stood_units: set[int] = set()
    apply_player_action = env._apply_player_action

    def capture_referee_verdict(chosen: Any) -> None:
        apply_player_action(chosen)
        stood_units.clear()
        stood_units.update(
            int(model.group_id)
            for model in env.wargame_models
            if model.is_alive and getattr(model, "charged_this_turn", False)
        )

    env._apply_player_action = capture_referee_verdict  # type: ignore[assignment]
    vp = 0.0
    coherent_steps = 0.0
    coherent_total = 0
    try:
        selector = build_action_selector(selector_spec, env, decode_topk=decode_topk)
        for episode in range(n_episodes):
            observation, _ = env.reset(seed=700000 + episode)
            done = False
            while not done:
                phase = env.game_clock_state.phase
                action = selector.select(observation, env)
                units: dict[int, list[int]] = {}
                unit_gaps: dict[int, tuple[float, float]] = {}
                pinner_units: set[int] = set()
                if phase is BattlePhase.charge:
                    for index, model in enumerate(env.wargame_models):
                        if model.is_alive and getattr(model, "declared_charge", False):
                            units.setdefault(int(model.group_id), []).append(index)
                    declared += len(units)
                    enemies = [m for m in env.opponent_models if m.is_alive]
                    for group, members in units.items():
                        gaps = [
                            float(
                                np.linalg.norm(
                                    np.asarray(enemy.location, dtype=float)
                                    - np.asarray(
                                        env.wargame_models[i].location, dtype=float
                                    )
                                )
                                - enemy.base_radius
                                - env.wargame_models[i].base_radius
                            )
                            for i in members
                            for enemy in enemies
                        ]
                        roll = float(
                            getattr(env.wargame_models[members[0]], "charge_roll", 0.0)
                        )
                        unit_gaps[group] = (min(gaps) if gaps else float("inf"), roll)
                    if units and env.objectives:
                        alive = alive_mask_for(env.wargame_models)
                        cache = compute_distances(
                            env.wargame_models, env.objectives, alive_mask=alive
                        )
                        on_objective = cache.model_obj_norms_offset <= cache.obj_radii
                        player_counts = on_objective[alive].sum(axis=0)
                        opp_alive = alive_mask_for(env.opponent_models)
                        opponent_counts = (
                            compute_distances(
                                env.opponent_models,
                                env.objectives,
                                alive_mask=opp_alive,
                            ).model_obj_norms_offset
                            <= cache.obj_radii
                        ).sum(axis=0)
                        controlled = player_counts > opponent_counts
                        pinner_units = {
                            group
                            for group, members in units.items()
                            if bool(on_objective[members][:, controlled].any())
                        }
                moving = {
                    group
                    for group, members in units.items()
                    if any(action.actions[i] != STAY_ACTION for i in members)
                }
                attempted += len(moving)

                observation, _r, terminated, truncated, info = env.step(action)

                # `charged_this_turn` implies the unit moved, so this is a
                # subset of `moving` -- but it is the REFEREE's subset, not
                # every unit something displaced during the step.
                #
                # ⚠ **GATED ON THE CHARGE PHASE, and it has to be.** The flag is
                # set in the charge step and cleared in `_resolve_fight_phase`,
                # which runs on the boundary leaving the FIGHT phase. While
                # `fight` was skipped both happened inside one step; once it is
                # stepped they are two, the flag survives into the fight step
                # and an ungated count reports every charge TWICE. Caught by an
                # impossible value -- a standing fraction of 1.636, stood 8.00
                # against tried 4.89 -- which is the argument for printing a
                # ratio whose bound is known.
                if phase is BattlePhase.charge:
                    stood += len(stood_units)
                    stood_from_own_objective += len(stood_units & pinner_units)
                    for group in moving:
                        gap, roll = unit_gaps.get(group, (float("inf"), 0.0))
                        charge_rows.append((gap, roll, group in stood_units))
                    for group in stood_units:
                        declared_tgt = next(
                            (
                                int(getattr(m, "declared_target", -1))
                                for m in env.wargame_models
                                if m.is_alive and int(m.group_id) == group
                            ),
                            -1,
                        )
                        if declared_tgt < 0:
                            stood_manual += 1
                            continue
                        stood_grant += 1
                        member_locs = np.array(
                            [
                                m.location
                                for m in env.wargame_models
                                if m.is_alive and int(m.group_id) == group
                            ],
                            dtype=float,
                        )
                        alive_enemies = [m for m in env.opponent_models if m.is_alive]
                        if not len(member_locs) or not alive_enemies:
                            continue
                        contacts = np.asarray(
                            engagement_matrix(
                                member_locs,
                                np.array(
                                    [m.location for m in alive_enemies],
                                    dtype=float,
                                ),
                                np.ones(len(alive_enemies), dtype=bool),
                                np.ones(len(member_locs), dtype=bool),
                                engagement_range=env.config.engagement_range,
                                base_diameter=2.0 * env.config.base_radius,
                            )
                        )
                        engaged_groups = {
                            int(alive_enemies[j].group_id)
                            for j in np.nonzero(contacts.any(axis=0))[0]
                        }
                        if declared_tgt in engaged_groups:
                            stood_on_declared += 1
                done = terminated or truncated
            # ⚠ The POLICY'S OWN figure, not the realised one. This config
            # referees with `enforce_move: revert_unit`, under which the
            # realised rate is 1.000 whatever the policy does -- a metric
            # sampled after a corrective wrapper measures the wrapper, and
            # reading it that way once published a policy intending 0.630 as
            # 1.000. `evaluate.py` prefers the same field for the same reason.
            intended = env.intended_coherency_rate
            if intended is not None:
                coherent_steps += float(intended)
                coherent_total += 1
            vp += float(env.player_vp - env.opponent_vp)
    finally:
        env.close()
    _print_charge_census(charge_rows)
    if stood_grant or stood_manual:
        mismatch = 1.0 - stood_on_declared / max(stood_grant, 1)
        print(
            f"  stood by route: grant={stood_grant} manual={stood_manual}  "
            f"on_declared={stood_on_declared}  "
            f"mismatch(grant)={mismatch:.3f}"
        )
        pin_share = stood_from_own_objective / max(stood_grant + stood_manual, 1)
        print(
            f"  pin-skew: from_own_objective={stood_from_own_objective}  "
            f"share={pin_share:.3f}"
        )
    return {
        "declared": declared / n_episodes,
        "attempted": attempted / n_episodes,
        "stood": stood / n_episodes,
        "fraction": stood / attempted if attempted else float("nan"),
        "vp": vp / n_episodes,
        "coherent": coherent_steps / coherent_total if coherent_total else float("nan"),
    }


def _print_charge_census(rows: list[tuple[float, float, bool]]) -> None:
    """The gap-binned calibration table: n, P(stood), mean roll per gap bin.

    Gaps are in board units, which is inches on every shipped melee config
    (`inches_per_unit` 1.0). Nothing prints when nothing charged.
    """
    if not rows:
        return
    edges = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, float("inf")]
    print("  charge census (attempted, by pre-move edge gap):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        binned = [r for r in rows if lo <= r[0] < hi]
        if not binned:
            continue
        stood_rate = sum(1 for r in binned if r[2]) / len(binned)
        mean_roll = float(np.mean([r[1] for r in binned]))
        label = f'{lo:g}-{hi:g}"' if np.isfinite(hi) else f'{lo:g}"+'
        print(
            f"    gap {label:>7s}  n={len(binned):4d}  "
            f"P(stood)={stood_rate:.3f}  mean_roll={mean_roll:.2f}"
        )


def main() -> None:
    """Print one row of charge counts for the named policy or checkpoint."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    selector_spec = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 else 20
    decode_topk = int(argv[4]) if len(argv) > 4 else 1

    print(
        f"{config_path}{describe(overrides)}  ({n_episodes} episodes, "
        f"seeds 700000+, decode_topk={decode_topk})\n"
    )
    result = measure(selector_spec, config_path, n_episodes, decode_topk, overrides)
    print(
        f"  {'policy':38s} {'decl/ep':>8s} {'tried/ep':>9s} {'stood/ep':>9s} "
        f"{'frac':>7s} {'coherent':>9s} {'vp':>8s}"
    )
    print(
        f"  {selector_spec[:38]:38s} {result['declared']:8.2f} "
        f"{result['attempted']:9.2f} {result['stood']:9.2f} {result['fraction']:7.3f} "
        f"{result['coherent']:9.3f} {result['vp']:+8.1f}"
    )


if __name__ == "__main__":
    main()
