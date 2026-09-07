"""Record a per-model checkpoint's greedy episode to a match event log.

The per-model counterpart of `just record`: plays one episode of a driver
checkpoint (`train_per_model.py`) through the per-model facade with an
`EventLogExporter` attached, writes `recordings/<name>_events.jsonl`, and
prints a behaviour summary. The log replays through every existing tool
(`just replay`, `just replay-summary`, `just analyze`, `just replay-render`)
— snapshots are taken at phase boundaries, today's schema semantics.

Usage:
    uv run python scripts/record_per_model.py <checkpoint.pt> <config.yaml> [seed]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.envs.state import EventLogExporter, JsonMatchCodec
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.per_model import SetAgent


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    checkpoint_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 500000

    from pydantic_yaml import parse_yaml_raw_as

    from train_per_model import load_per_model_checkpoint

    torch.set_num_threads(1)
    config = parse_yaml_raw_as(WargameEnvConfig, config_path.read_text())
    config.render_mode = None
    exporter = EventLogExporter()
    env = PerModelEnv(config, state_exporters=[exporter], build_info=False)
    network, metadata = load_per_model_checkpoint(checkpoint_path, env)
    network.eval()
    agent = SetAgent(network)

    observation, _ = env.reset(seed=seed)
    terminated = False
    while not terminated:
        decision = agent.act(env, observation, greedy=True)
        observation, _r, terminated, _t, _ = env.step(decision.action)

    out = Path("recordings")
    out.mkdir(exist_ok=True)
    name = f"per_model_{checkpoint_path.parent.name}_r{metadata['rounds']}_s{seed}"
    path = out / f"{name}_events.jsonl"
    path.write_bytes(JsonMatchCodec().encode(exporter.log))

    counts = agent.reset_declaration_counts()
    movement = sum(v for k, v in counts.items() if k.startswith("movement:"))
    shooting = sum(v for k, v in counts.items() if k.startswith("shooting:"))
    print(f"recording: {path}")
    print(
        f"checkpoint: {metadata['rounds']} rounds, revision {metadata['revision']}, "
        f"map {env.map_name}"
    )
    print(
        f"outcome: vp {env.player_vp}-{env.opponent_vp} "
        f"(margin {env.player_vp - env.opponent_vp}), "
        f"alive {np.mean([m.is_alive for m in env.wargame_models]):.2f}"
    )
    if movement:
        print(
            f"declarations: stationary {counts.get('movement:1', 0)}/{movement} "
            f"movement, hold-fire {counts.get('shooting:1', 0)}/{max(shooting, 1)} "
            f"shooting"
        )


if __name__ == "__main__":
    main()
