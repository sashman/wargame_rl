"""Record one episode of the per-model facade to a match event log.

An EVENT LOG, not the MP4 `just record-per-model` writes. At `phase` cadence
the log is schema-identical to the phase facade's, so `just replay`,
`replay-summary`, `replay-render` and `analyze` read it unchanged; at
`decision` cadence there is one snapshot per decision, which replay and
render accept and `analyze` refuses by name.

`policy` is a scripted baseline name, `random`, `set_network` or a per-model
checkpoint (`*.pt`); every one of them is played on the PER-MODEL facade
(a baseline name through its scripted seat). To record the phase facade's
own game, use `just measure-checkpoint ... record` or `just record`.

Usage: just record-per-model-events <env_config> [policy] [cadence] [seed] [out]
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.state.provenance import CADENCES
from wargame_rl.wargame.scoring import record_per_model

DEFAULT_SEED = 700_000


def main() -> None:
    """Record one episode and print where it went."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    config_path = argv[1]
    policy = argv[2] if len(argv) > 2 and argv[2] else "squad_march_take"
    cadence = argv[3] if len(argv) > 3 and argv[3] else "phase"
    if cadence not in CADENCES:
        raise SystemExit(f"cadence must be one of {CADENCES}, not {cadence!r}")
    seed = int(argv[4]) if len(argv) > 4 and argv[4] else DEFAULT_SEED
    label = Path(policy).parent.name if policy.endswith(".pt") else policy
    out = (
        Path(argv[5])
        if len(argv) > 5 and argv[5]
        else Path("recordings") / f"per_model_{label}-s{seed}-{cadence}_events.jsonl"
    )
    env_config = load_env_config(config_path, **overrides)
    env_config.render_mode = None
    written = record_per_model(
        policy,
        env_config,
        seed,
        out,
        cadence=cadence,  # type: ignore[arg-type]
    )
    print(
        f"{config_path}{describe(overrides)}  {policy}  seed {seed}  {cadence} cadence"
    )
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
