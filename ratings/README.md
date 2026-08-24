# Rating ledgers

One file per scenario, named by its fingerprint: `<16 hex>.json`. Each holds the
**raw per-layout legs** every rated pairing has ever played on that scenario,
not the fitted ratings — `just elo-table <env_config>` fits on read.

These are committed, in the same class as [`reports/`](../reports/README.md):
a durable record of what was measured, under a named code revision, on a named
set of layouts. They are superseded rather than edited, and like `reports/` they
are exempt from the docs-drift hook.

## Reading one

```
just measure-elo <env_config> [n_layouts] <entrant> <entrant> ...   # play and append
just elo-table <env_config>                                        # fit and print
```

An entrant is a baseline name or a path to a `.ckpt`. Legs accumulate: adding an
entrant replays only its own pairings, and recalibrating the margin scale
replays nothing at all.

## Why a fingerprint, and why it is not the config

A rating means something only *within* one scenario, so a ledger refuses to hold
two — mixing them raises rather than warns. The fingerprint deliberately
**excludes** `turn_order` and `opponent_policy`, and **sorts** the deployment-zone
and army pairs: those are the leg axes and the entrants, and fingerprinting them
would scatter the four legs of a single pairing across four ledgers. See
[docs/elo.md](../docs/elo.md) § The fingerprint.

## Before you add a scenario

⚠ **Run `just measure-seat-parity <env_config>` first.** A rating assumes the two
seats are the same game, and on `configs/golden/25v25_shooting_opponent.yaml`
they are not — one policy played from both seats loses from the player seat by
**−24.6 ± 9.4 vp**
([report](../reports/2026-08-19-the-two-seats-are-not-the-same-game.md)). Nothing
in the code enforces this; it is a precondition you are expected to check. See
[docs/elo.md](../docs/elo.md) § Open gaps.

⚠ **`code_revision` is on every leg for a reason.** Three open bugs
(`polygons_contain_points`' padded outlines, the LOS asymmetry, `_cover_mask`'s
hidden models) all touch the board every rating is measured on. Fixing any of
them shifts every number here, and the revision field is how a stale ledger is
recognised rather than re-fitted.
