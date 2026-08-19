"""Play the rated schedule and record the result, so policies share one scale.

Every score in this repo is quoted against one hand-picked opponent, so "did
this get better" has no answer that survives changing the opponent. A rating
puts the scripted baselines and any checkpoints on **one** scale, and reports
the deployment-zone and first-turn advantages as fitted numbers rather than
assuming them away.

Each pairing plays four legs per layout -- every combination of (A's zone) x
(who moves first) -- so the schedule is balanced in both axes by construction
and both advantage terms are separately identifiable. Legs are appended to
`ratings/<scenario>.json`; `just elo-table` fits and prints them.

⚠ **On a config whose own `turn_order` is `random`, a rated leg is played on a
different layout stream from that config's `measure-baselines` numbers.**
`_resolve_player_side` draws from the layout RNG only under `random`, and it
draws before the terrain and objectives are placed. The four legs agree with
each other, which is what the fit needs -- but do not expect a rating and a
`measure-baselines` row on such a config to be describing the same maps.

⚠ **Ratings are comparable only at a fixed decode.** `decode_topk` is recorded
per entrant; do not mix K in one table.

Usage: just measure-elo <env_config> [n_layouts] <entrant> <entrant> [...]
"""

from __future__ import annotations

import sys

from pydantic_yaml import parse_yaml_raw_as

# Registers the `model` opponent key, so a checkpoint can be entrant B.
# The arena deliberately does not import this: `rating` never imports
# `model`, and pulling a registration into the library would invert that.
import wargame_rl.wargame.model.opponent  # noqa: F401,E402
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.rating.arena import play_pairing, require_symmetric
from wargame_rl.wargame.rating.entrant import Entrant
from wargame_rl.wargame.rating.ledger import append, fingerprint, path_for
from wargame_rl.wargame.rating.schedule import pairings
from wargame_rl.wargame.selectors import build_action_selector

DEFAULT_ENTRANTS = ("random", "squad_march", "squad_march_shoot")


def _entrant_for(spec: str, config: WargameEnvConfig, decode_topk: int) -> Entrant:
    """Resolve a baseline name or a checkpoint path into a rated entrant.

    The kind is settled by building the selector once against a throwaway env,
    which also fails early and loudly on a bad path or an unknown name rather
    than in the middle of a long schedule.
    """
    probe = WargameEnv(config=config.model_copy(deep=True), renderer=None)
    try:
        resolved = build_action_selector(spec, probe, decode_topk)
    finally:
        probe.close()

    return Entrant(
        name=resolved.label,
        build=lambda env: build_action_selector(spec, env, decode_topk).select,
        kind=resolved.kind,
        source=resolved.source,
        decode_topk=decode_topk,
    )


def main() -> None:
    """Play every pairing and append the legs to this scenario's ledger."""
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)

    config_path = sys.argv[1]
    n_layouts = int(sys.argv[2]) if sys.argv[2] else 100
    specs = [spec for spec in sys.argv[3:] if spec]
    if len(specs) < 2:
        print("at least two entrants are needed to rate anything")
        raise SystemExit(1)

    with open(config_path) as handle:
        base_config = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    base_config.render_mode = None
    require_symmetric(base_config)

    entrants = [_entrant_for(spec, base_config, 1) for spec in specs]
    by_name = {entrant.name: entrant for entrant in entrants}
    schedule = pairings(list(by_name))

    digest = fingerprint(base_config)
    print(f"\n{config_path}")
    print(f"scenario {digest}  ({n_layouts} layouts x 4 legs per pairing)")
    print(f"entrants: {', '.join(by_name)}\n")

    played = []
    for name_a, name_b in schedule:
        print(f"  {name_a} vs {name_b} ...", flush=True)
        played.extend(
            play_pairing(by_name[name_a], by_name[name_b], base_config, n_layouts)
        )

    append(played, base_config, entrants)
    print(f"\nwrote {len(played)} legs to {path_for(digest)}")
    print(f"fit and print with: just elo-table {config_path}")


if __name__ == "__main__":
    main()
