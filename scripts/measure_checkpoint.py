"""Score a trained checkpoint on held-out seeds, next to the scripted baselines.

A `success_rate` from a training log is not comparable across configs — it
changes definition with the reward phase, and it is a per-epoch binomial over
`n_eval_episodes`. This runs a checkpoint through the *same* code path as
`measure-baselines`, on the same kind of fixed seed set, so a learned policy and
`squad_march_shoot` land in one table on identical layouts.

Pass `record` as a fourth argument to also write an event log to `recordings/`,
which `just analyze-compare <agent> <baseline>` reads.

Usage: just measure-checkpoint <checkpoint> <env_config> [n_episodes] [record]
       [decode_topk]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.baseline.evaluate import (
    ActionSelector,
    BaselineResult,
    evaluate_selector,
    format_optional_metric,
    record_episode,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.decoding import decode_joint_coherent
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.model.net import TransformerNetwork

# Disjoint from ROLLOUT_SEED_BASE (0), the training eval base (500_000) and the
# baseline base (10_000), so a checkpoint is scored on layouts it never trained
# or model-selected on.
HELDOUT_SEED_BASE = 700_000


def build_selector(
    checkpoint_path: str,
    env: WargameEnv,
    decode_topk: int = 1,
    decode_stay: bool = False,
) -> tuple[ActionSelector, TransformerNetwork]:
    """Load a policy network and wrap it as an `ActionSelector`.

    Greedy (argmax) rather than sampled: this measures the policy the agent
    would play, not the exploration distribution around it. The network applies
    the action mask internally, so illegal actions cannot be selected.

    `decode_topk` > 1 replaces the independent per-model argmax with **joint
    constrained decoding** — the most probable action *combination* each unit
    can legally make (`model/common/decoding.py`). It changes only the decode,
    never the weights, and is worth a great deal: measured over three seeds at
    `enforce_move: revert_unit`, on nine held-out tables, `vp_margin` goes
    −38.5 → +1.1 at K=3 and intended coherency 0.639 → 0.936. Across all 45
    tables the per-map gain is +40.5 vp, positive on 45 of 45. Left at 1 by
    default so every existing number stays reproducible.

    `decode_stay` stands a unit still when the top-K set yields no legal
    combination at all, rather than letting the referee revert it to the same
    positions *plus* an overlap cascade onto its neighbours.
    """
    policy_net = TransformerNetwork.from_checkpoint(env, checkpoint_path)
    policy_net.eval()

    def select(
        observation: WargameEnvObservation, env_: WargameEnv
    ) -> WargameEnvAction:
        with torch.no_grad():
            state = observation_to_tensor(observation, policy_net.device)
            logits = policy_net(state)
            actions = [int(a) for a in logits.argmax(dim=-1).flatten().tolist()]
            if decode_topk > 1:
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                actions = decode_joint_coherent(
                    log_probs,
                    actions,
                    env_,
                    decode_topk,
                    include_stay=decode_stay,
                )
        return WargameEnvAction(actions=actions)

    return select, policy_net


def label_for(checkpoint_path: str) -> str:
    """Name a run by its `--run-suffix`, falling back to the directory name.

    Checkpoint directories are `<scenario>-<YYYY-MM-DD-HH-MM-SS>[-<suffix>]`, and
    the scenario part is identical across the arms of a screen — so the suffix
    is the only part that identifies which arm a row belongs to.
    """
    directory = Path(checkpoint_path).parent.name
    match = re.search(r"\d{4}(?:-\d{2}){5}-(.+)$", directory)
    return match.group(1) if match else directory


def format_result(result: BaselineResult) -> str:
    """One aligned row: the fields that rank policies, in one line."""
    return (
        f"{result.name:<28}{result.final_fraction_at_objectives:>10.3f}"
        f"{result.win_rate:>9.2f}{result.player_vp:>12.1f}"
        f"{result.opponent_vp:>10.1f}{result.vp_margin:>11.1f}"
        f"{format_optional_metric(result.vp_margin_se, 1):>8}"
        f"{result.objectives_held:>7.2f}{result.final_fraction_alive:>8.3f}"
        f"{format_optional_metric(result.coherency_rate):>10}"
        f"{format_optional_metric(result.models_out_of_coherency, 2):>8}"
        f"{format_optional_metric(result.exposure_rate):>10}"
        f"{format_optional_metric(result.terrain_proximity, 1):>11}"
        f"{format_optional_metric(result.firepower_ratio, 2):>12}"
    )


def main() -> None:
    """Score one checkpoint and print its row of the baseline table."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    checkpoint_path = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 else 30
    record = len(argv) > 4 and argv[4].lower() in {"record", "true", "1"}
    # Joint constrained decoding, off by default so every historical row here
    # reproduces. It matters most for `record`: a recording made at K=1 shows
    # the referee cancelling a third of the unit-moves, which is the play the
    # decoder exists to replace, and is not what the checkpoint would do now.
    decode_topk = int(argv[5]) if len(argv) > 5 else 1

    env_config = load_env_config(config_path, **overrides)
    env_config.render_mode = None

    env = create_environment(env_config=env_config)
    select, _policy_net = build_selector(checkpoint_path, env, decode_topk)
    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]

    name = label_for(checkpoint_path)
    print(f"\n{checkpoint_path}")
    print(
        f"{config_path}{describe(overrides)}  ({n_episodes} episodes, "
        f"seeds {seeds[0]}-{seeds[-1]})\n"
    )
    header = (
        f"{'policy':<28}{'on obj':>10}{'win':>9}{'player VP':>12}"
        f"{'opp VP':>10}{'VP margin':>11}{'±SE':>8}{'held':>7}{'alive':>8}"
        f"{'coherent':>10}{'adrift':>8}"
        f"{'exposure':>10}{'terrain_d':>11}{'firepower':>12}"
    )
    print(header)
    print("-" * len(header))

    result = evaluate_selector(select, env, seeds, name)
    print(format_result(result))

    if record:
        output = Path("recordings") / f"agent_{name}-k{decode_topk}.jsonl"
        written = record_episode(select, env_config, seeds[0], output)
        print(f"\nwrote {written}")

    print()


if __name__ == "__main__":
    main()
