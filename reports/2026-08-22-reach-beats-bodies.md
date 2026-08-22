# Reach beats bodies. The first asymmetric match-up is lopsided, and the ranking change is noise.

**2026-08-22.** Script-only screen, no GPU. Held-out nine, n=30, seeds 700000+,
K=1, both configs unrefereed. Criteria pre-registered in
[the pre-registration](2026-08-22-asymmetric-armies-preregistration.md) before
any number existed.

## The design

Every config in this repo had been a **mirror** — both armies identical — which
leaves the game with no reason for the two sides to play differently. This is the
first that is not.

| | ELITE | HORDE |
|---|---|---|
| models / squads | 15 / 5 of 3 | **30** / 6 of 5 |
| weapon | range **24**, 2 attacks, S5, AP1 | range 12, 1 attack, S4, AP1 |
| shots per round | **30** | **30** |

Matched on firepower, differing only in distribution — the mixed-role arms were
a null precisely because they changed total lethality while claiming to change
roles. The tension is that **control is a headcount**, so 30 bodies beat 15
wherever both arrive; the elite has to thin the horde at 24" before it closes to
12". Both orientations were screened, so each army got a turn as the player.

## The result

| policy | ELITE vp / win / held / alive | HORDE vp / win / held / alive |
|---|---|---|---|
| `hold_deployment` | −253.4 / 0.00 / 0.01 / 0.296 | −238.0 / 0.00 / 0.07 / 0.117 |
| `random` | −253.3 / 0.00 / 0.00 / 0.048 | −236.8 / 0.00 / 0.01 / 0.004 |
| `squad_march_shoot` | +12.5 / 0.63 / 2.44 / 0.407 | −66.2 / 0.29 / 0.86 / 0.113 |
| `squad_march_deny` | **+22.2** / **0.72** / 2.39 / 0.408 | −43.5 / 0.37 / 0.91 / 0.101 |
| `squad_march_take` | +13.4 / 0.66 / 2.47 / 0.417 | **−41.1** / 0.33 / 0.87 / 0.097 |
| `contest_and_spread` | +18.1 / 0.67 / **2.59** / **0.519** | −89.4 / 0.20 / 0.67 / 0.088 |

### It is not degenerate, and it is badly imbalanced

Neither pre-registered rejection fires. The best script wins **0.72** on one side
and **0.33** on the other — inside the 0.85 / 0.15 band — and the floors sit
**197 to 276 vp** below the marchers, so there is a real game on both sides.

But the elite wins the match-up decisively, in both orientations, with the same
policies. **The horde has twice the bodies, control is a headcount, and it holds
a third of the ground**: `held` 0.67–0.91 against 2.39–2.59. It finishes with
**9–11% of its army alive** against the elite's 41–52%.

**Matching firepower was not enough.** At 24" against 12" the elite gets two free
rounds of shooting, and 30 shots at S5 into an army that cannot reply settles the
game before the bodies matter. **Reach dominates bodies at matched firepower.**

### ⚠ The accept criterion fired on noise, and is not treated as met

The pre-registration accepts if the best script differs by side. It does —
`deny` on the elite, `take` on the horde. Every such difference is inside its
own error bar:

| comparison | diff | se | t |
|---|---|---|---|
| ELITE `deny` − `take` | +8.8 | 16.2 | **0.54** |
| HORDE `take` − `deny` | +2.4 | 20.4 | **0.12** |
| ELITE `contest` − `shoot` | +5.6 | 9.5 | **0.59** |
| HORDE `shoot` − `contest` | +23.2 | 22.7 | **1.02** |

A ranking that reorders within noise is the same error as reading one layout-set
cell as a result. **The question "does asymmetry make the two sides need
different play" is not answered by this screen** — only "the elite army is
stronger" is, and that is a balance fact, not a behavioural one.

## What was done about it

The horde gets **Move 12**, double the default, using the per-model `move` field.
It is the minimal compensation the measurement asks for: the free-fire window is
`(24 − 12) / move` rounds, so doubling Move halves it. Firepower stays matched at
30 shots; only closing time changes. `15v30_elite_vs_fast_horde.yaml` and its
swapped twin are the re-screen.

## Do not re-run

- **Matched firepower alone does not balance an asymmetric pair.** Reach is worth
  more than bodies here by a wide margin, and `alive` is the diagnostic that says
  so — 9–11% against 41–52%.
- **A ranking change of 2.4 vp against an se of 20.4 is not a ranking change.**
