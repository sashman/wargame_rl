"""Fit and print the rating table for a scenario, from legs already played.

Ratings are **fitted, not accumulated**, so this reads the raw per-layout legs
out of `ratings/<scenario>.json` and solves for them fresh every time. Nothing
is replayed, which is what lets an entrant be added, or the margin scale
recalibrated, without re-running a single match.

Read the table beside `objectives_held` and `vp_margin`, which say *how* a
margin was produced -- **Elo ranks, it does not explain.** And quote the
interval, not the point: a rating without its bootstrap interval is the same
failure as a `success_rate` with no floor and no bar.

Usage: just elo-table <env_config>
"""

from __future__ import annotations

import sys

from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.rating.ledger import fingerprint, leg_results, load, path_for
from wargame_rl.wargame.rating.score import DEFAULT_MARGIN_SCALE
from wargame_rl.wargame.rating.table import format_table, rate


def main() -> None:
    """Print the fitted table for the scenario this config describes."""
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    config_path = sys.argv[1]
    with open(config_path) as handle:
        config = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    config.render_mode = None

    digest = fingerprint(config)
    ledger = load(digest)
    if ledger is None:
        print(f"no ratings for scenario {digest} ({config_path})")
        print(f"expected {path_for(digest)}; play some with `just measure-elo`")
        raise SystemExit(1)

    legs = leg_results(ledger)
    table = rate(legs, margin_scale=DEFAULT_MARGIN_SCALE)

    print(f"\n{config_path}")
    print(f"scenario {digest}   {len(legs)} legs, {len(ledger.entrants)} entrants")
    decodes = {entrant.decode_topk for entrant in ledger.entrants}
    if len(decodes) > 1:
        print(
            f"⚠ this table mixes decodes {sorted(decodes)} -- the rows are not "
            "comparable to each other"
        )
    print()
    print(format_table(table, legs, DEFAULT_MARGIN_SCALE))


if __name__ == "__main__":
    main()
