# Melee self-play runs end to end, and the control refused to charge

**One seed. No verdict.** This was a demonstration that the full feature stack —
melee, the command-phase charge declaration, self-play, the in-run ratings —
works together, not a measurement of whether any of it is good. The sharpest
thing in it is a hypothesis, flagged as one.

Run 2026-09-01 at `20cede8`. `configs/experiments/25v25_maps_melee_approach.yaml`,
300 epochs, seed 1, arm and control with byte-identical flags apart from
`--self-play`, `--pfsp-mode uniform`, `--pool-anchor squad_march_take_charge`.
3h02m, `rc=0`.

## Preconditions, both cheap and both run first

- **Seat parity PASSES on the melee config.** `squad_march_take_charge` on both
  seats, 120 layouts: aggregate **−3.6 ± 5.6 vp**, inside the 2 se threshold.
  Map-pool config, so it plays the turn-order pair only and appends nothing to a
  ledger — the zone axis is dead there by construction.
- **The mechanism fires for a script.** `just measure-charges
  squad_march_take_charge`, refereed, n=10: **8.00 declared / 7.10 attempted /
  5.50 stood** per episode (0.775), coherency 0.947, `bind_violations=0`.

## The anchor had to change, and that is a general trap

`just train-self-play-screen` hardcoded `--pool-anchor squad_march_take`, whose
`select_charge` returns STAY. The anchor is entry zero of the pool and is
**never evicted**, so it defines what "no better than where we started" means.
On a melee config that floor is a policy that **never charges** — the learner
could top the ladder without ever meeting a charge. Fixed in #263; the general
form is in `docs/self-play.md`: **a floor that cannot use a rule is not a floor
for a game that has it**, and every future move type has to be asked it.

## Mechanism: PASS

Pool filled to capacity 8 and ended **spanning epochs 0-275**, still anchored at
0 after five evictions — not collapsed onto its newest member. Zero exceptions.
Snapshots landed in the run's **own** directory, which is #262's fix running
outside a test for the first time.

## The two ratings disagreed, which is the point of logging both

| | arm (self-play) | control |
|---|---|---|
| `eval/vp_margin` | −71.17 | −63.33 |
| `eval/elo` | −185.8 | −154.5 |
| `self_play/learner_elo` | **+146.3** | — |
| `eval/coherency_rate` | 0.783 | 0.815 |

The arm's **ladder reads +146 while its rating against the script reads −186**.
It climbed steadily against its own history while staying far below the scripted
bar. This is the ladder-versus-rating caveat demonstrated rather than asserted:
anyone reading `self_play/learner_elo` as "the agent's Elo" would have concluded
the opposite of the truth. ⚠ The two are not on one scale and never will be.

## The observation worth testing: the control refused to charge

Final charge census, **refereed** eval config, n=10, K=3:

| | decl/ep | tried/ep | stood/ep | frac | coherent | vp |
|---|---|---|---|---|---|---|
| arm | 14.50 | 10.00 | 7.20 | 0.720 | 0.956 | −57.0 |
| **control** | **0.20** | **0.10** | **0.10** | — | 0.969 | **−29.5** |

**Both faced a charging opponent** — the config's own is
`squad_march_take_charge` — so the difference is the *pool*, not the presence of
charges. The arm learned to use the feature; the control learned to ignore it.
And **the control scored better while ignoring it**, which is consistent with the
standing finding that closing is priced only by what it captures.

⚠ **This is one seed at n=10 and resolves nothing.** The pre-registration's own
power table says three seeds cannot resolve a difference under 28 vp; the
arm−control gap here is 27.5 vp at one seed. The control's `pin-skew 1.000`
comes from a **single** charge and means nothing. Two readings are available and
this data cannot separate them:

1. A pool of charging past selves teaches the agent to charge.
2. Charging does not pay at this horizon, the control found that out, and the
   arm is being dragged into a bad habit by an anchor that charges.

Reading 2 is the one the repo's existing evidence favours, and it is the reason
not to celebrate the arm's higher charge count.

## ⚠ A trajectory that does not exist

Two mid-run censuses were taken on the **training** config
(`configs/experiments/...`, unrefereed, n=5, off a moving `last.ckpt`): ~epoch 50
read `12.20 decl / 0.604 frac / vp −127.0`, ~epoch 175 read
`20.60 / 0.702 / −5.0`. The final table above is the **refereed evaluation**
config. **Those are different measurements and −127 → −5.0 → −57.0 is not a
trajectory.** The within-config pair (−127 → −5.0) is a real improvement; the
third number belongs to another scale entirely. Recorded because the table was
one paste away from implying otherwise.

## What is NOT claimed

- That self-play helps, or hurts, on a melee config. One seed.
- That melee is worth training on at all. Untested — a shooting army has no
  reason to close except to stand on an objective, so any move whose value is
  "arrive sooner" is being measured in a game that may not reward arriving.
- Anything about `hard` or `even` scheduling. `uniform` was the arm.

## The cheap next step

Three seeds of the same pair, and read `decl/ep` beside `vp_margin`. If the
charge count separates the arms while vp does not, reading 1 is right and the
pool is teaching a habit that costs nothing; if vp tracks the charge count
downward, reading 2 is right and the charging anchor is the problem.
