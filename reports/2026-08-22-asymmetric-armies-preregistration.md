# Asymmetric armies — criteria, written 2026-08-22 before any numbers

ELITE 15 models / 5 squads of 3 / range 24 / 2 attacks / S5 — 30 shots.
HORDE 30 models / 6 squads of 5 / range 12 / 1 attack / S4 — 30 shots.
Matched on firepower, differing in bodies and reach. Control is a HEADCOUNT, so
30 bodies beat 15 wherever both arrive; the elite has to thin the horde at 24"
before it closes to 12".

Screen: scripted policies only, held-out nine, n=30, seeds 700000+, K=1, on
BOTH configs (each side gets a turn as the player). No GPU.

## Reject as DEGENERATE if either holds

- **One-sided.** The best script on one side wins **> 0.85** of episodes while
  the best on the other wins **< 0.15**. A scenario one side cannot lose teaches
  nothing, and an agent trained on it would be scoring the match-up, not itself.
- **No contest.** `hold_deployment` lands within **1 SE** of the best marching
  script on either side — the same floor-meets-ceiling test that would have
  caught a dead scenario earlier.

## Reject as NO NEW QUESTION if

- The script **ranking is the same on both sides** and matches the mirror's
  (`24v24_maps_spare_squads`: take > deny > shoot). If one policy is best
  whichever army it commands, asymmetry changed the numbers and not the game.

## Accept as HEALTHY if

- **The best script differs by side**, or the rankings differ between sides.
  That is the whole claim: two armies that need to be played differently. Only
  then is it worth GPU, and only then does "the mirror was hiding equilibrium
  play" have evidence behind it.

## Then, and only then

Train the side whose ranking differs most from the mirror, three seeds, 300
epochs, `ent_coef` 0.003, recording on, scored against scripts re-measured on
that config — a new scenario inherits no bar.

⚠ Both configs are UNREFEREED, like every training config here. Score a trained
agent on a refereed twin, never on the config it trained on.
