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

from scripts.scenario_overrides import describe, load_env_config, parse_overrides
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
                if phase is BattlePhase.charge:
                    for index, model in enumerate(env.wargame_models):
                        if model.is_alive and getattr(model, "declared_charge", False):
                            units.setdefault(int(model.group_id), []).append(index)
                    declared += len(units)
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
    return {
        "declared": declared / n_episodes,
        "attempted": attempted / n_episodes,
        "stood": stood / n_episodes,
        "fraction": stood / attempted if attempted else float("nan"),
        "vp": vp / n_episodes,
        "coherent": coherent_steps / coherent_total if coherent_total else float("nan"),
    }


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
