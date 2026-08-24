# Pre-registration — is the offence deficit a twenty-round artefact?

**Written 2026-08-24, before any number in it exists.** This is M4 of the
five-round plan, the one step of it never run. It costs **no GPU**: scoring
never touches the reward, so evaluating at a different horizon needs none of
the recalibration that *training* there would.

## The question

The agent holds **1.9–2.1** distinct objectives against the scripts' **2.9–3.9**,
and that shortfall is the whole of its offence deficit — flat at −42 to −71 vp
against every opponent, and −75.9 against `advance_and_shoot`.

Reaching a third or fourth objective means walking the table. At Move 6 with
objectives 20–40" away, **only a twenty-round game affords it**. So the
shortfall has two possible readings and nobody has separated them:

- **CLOCK** — the scripts' extra objectives are bought with time the agent
  spends elsewhere. A shorter game compresses the difference.
- **GENUINE** — the agent allocates badly at any horizon, and the clock is not
  what is between it and a third point.

This matters because it decides whether the next spend goes on allocation at
all. If the deficit is largely the clock, "allocation" is partly a scenario
property; if it survives a 4x shorter game, it is a search failure and the
critic-probe evidence stands unqualified.

## Design

| | |
|---|---|
| layouts | the held-out nine, `configs/evaluation/maps_heldout` |
| n | 30 per table, seeds 700000+ |
| decode | K=3, verified |
| configs | **refereed** — `25v25_maps_take_opponent_refereed.yaml` (the primary documented row) and `25v25_maps_vs_advance_and_shoot.yaml` (the worst, −75.9 on 0 of 9) |
| horizons | `rounds=20` (the configs' own value) and `rounds=5` |
| policies | the six `-newmaps` seeds at `last.ckpt`, plus `squad_march_take`, `squad_march_deny`, `squad_march_shoot`, `contest_and_spread` |

## Primary readout, and why it is `held` and not vp

**`held_agent − held_best_script`, per horizon.** `held` is a count of
objectives and means the same thing at either horizon.

⚠ **Absolute vp is NOT comparable across horizons and must not be quoted as
though it were.** Five rounds is 4 scoring events against twenty's 19, the VP
cap is per turn, and the five-round outcome sd is ~12 against twenty's ~91. The
vp gap is therefore reported **normalised by the per-episode sd**, never raw.

## Criteria

- **CLOCK** — the `held` shortfall shrinks by **≥ 50%** at five rounds, on
  **both** opponents.
- **GENUINE** — the shortfall is unchanged within noise, or widens, on both.
- **MIXED** — the two opponents disagree. Report both and claim nothing; a
  single opponent has inverted a reading here before.

## The asymmetry, stated in advance

⚠ **The agent was trained at twenty rounds; the scripts do not plan.** So the
agent's five-round score is a *transfer* result and is handicapped in a way the
scripts' is not. This biases **against** CLOCK — it makes a narrowing harder to
find, not easier. That is the safe direction, and any CLOCK verdict is therefore
conservative while a GENUINE verdict is not.

## The instrument check that voids everything

Training is deterministic given seed + config + code, and these checkpoints and
this scoring path are unchanged. **The `rounds=20` rows must reproduce the
published table.** If they do not, the instrument moved since 2026-08-21 and
neither horizon can be read.

## What is deliberately not asked

Whether to *train* at five rounds. That needs reward recalibration — gamma, the
per-turn cap, `objective_coverage` — and is downstream of this verdict.
