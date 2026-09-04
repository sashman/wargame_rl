# A tenth of the step — interpolation keeps PPO's gain and the decode's headroom

**2026-09-04**, code `d5ec7d4`, **no GPU training**. Six seeds, n=45, seeds
700000+, `K=3` + charge decode, the `approach` eval family. Comparator fixed by
name before any number: `squad_march_take_charge`, re-measured the same day and
identical to §38 on all four cells.

Follows directly from
[PPO spends the decode's headroom](2026-09-04-ppo-spends-the-decodes-headroom.md).
If PPO's problem is that it travels too far from a decodable policy, the
cheapest possible intervention is to **travel a tenth as far**: interpolate the
weights between the §46 clone and its own §47 PPO endpoint. Both already exist,
so this costs arithmetic and inference, not a training run.

## The dose-response, refereed cell (the goal's hard cell)

| α | mean | paired v clone | SE | t | signs |
|---|---|---|---|---|---|
| 0.0 — the clone | −10.67 | — | — | — | — |
| **0.1** | **−2.02** | **+8.65** | 2.75 | **3.15** | **6/6** |
| 0.25 | −2.85 | +7.82 | 4.96 | 1.58 | 5/6 |
| 0.5 | −20.70 | −10.03 | 5.06 | −1.98 | 1/6 |
| 1.0 — 300 epochs of PPO | −25.58 | — | — | — | — |

The grid `{0.1, 0.25, 0.5}` was **fixed before any six-seed number existed** and
is reported whole. It is a curve, not a pick: small helps, large hurts, and the
endpoint is worst.

## All four cells at α=0.1, against the same reading rule

"WON" means the gap to the bar exceeds 2 SE, "ahead" 1 SE, "tie" within 1 SE.

| cell | α=0.1 | SE | bar | v bar | read | clone | paired v clone | t |
|---|---|---|---|---|---|---|---|---|
| refereed | −2.02 | 3.66 | −5.3 | +3.28 | tie | −10.67 | **+8.65** | **3.15** |
| `vs_take` | +22.80 | 4.67 | +20.2 | +2.60 | tie | +28.82 | −6.02 | −1.24 |
| `vs_deny` | **+28.02** | 4.88 | +11.8 | **+16.22** | **WON** | +19.07 | +8.95 | 1.19 |
| `vs_shoot` | +56.50 | 3.42 | +56.6 | −0.10 | tie | +58.45 | −1.95 | −0.43 |

**The same rule applied to the clone itself**: refereed **LOST** (−1.80 SE),
`vs_take` **WON** (+3.22), `vs_deny` ahead (+1.80), `vs_shoot` tie (+0.98).

## VERDICT: the goal is NOT MET

It is conjunctive — beat the bar on all four — and three of four cells are
ties. What changed is narrower and worth stating exactly:

**The clone reads 1 won / 1 ahead / 1 tie / 1 LOST. α=0.1 reads 1 won / 3 ties
/ 0 lost.** It is the first policy in this line with **no losing cell**, and the
cell it repaired is the head-to-head — the one §48 named as the hard one, and
the only one a clone cannot win by construction. It bought that by giving back
`vs_take`, which fell from a win to a tie.

⚠ **α=0.25 is worse**, not better: it **loses** `vs_shoot` (−4.53, −1.68 SE).
Both doses cleared the refereed screen, so both were carried to all four cells
rather than the winner being reported alone.

## ⚠ The falsifier fires, partly

If any small displacement of the clone helped, this would be regularisation and
not anything PPO learned. Control: displace the clone by a **random** direction,
per-tensor norm-matched to the α=0.1 step, three draws per seed on the refereed
cell.

| | paired v clone | SE | t | signs |
|---|---|---|---|---|
| interpolation α=0.1 | **+8.65** | 2.75 | 3.15 | 6/6 |
| random, same size | +2.91 | 2.40 | 1.21 | 4/6 |
| **the difference** | **+5.74** | 2.73 | **2.10** | 5/6 |

So **roughly two-thirds of the gain is specifically PPO's direction** and about
a third is displacement of any kind — and that third is **not distinguishable
from zero**. ⚠ On the first six-draw pass the difference was t=0.86 and the two
were **not** separable; it took 18 draws to resolve. Do not quote the direction
effect without the control beside it.

## The mechanism, measured rather than assumed

| | mean headroom | drift (nats/model) |
|---|---|---|
| clone | **+74.87** | 0 |
| **α=0.1** | **+65.40** | **0.0155** |
| 300 epochs of PPO | +40.23 | ~2.35 |

α=0.1 keeps **87%** of the clone's decode headroom while capturing PPO's gain.
That is the pre-registered mechanism claim, and it holds: the intervention works
by *not leaving the decodable region*, and the region is very small — the
control run has already drifted to 0.255 by epoch 25 and lost two thirds of its
headroom there.

## What this is NOT

- ⚠ **Not self-play.** These are §47's fixed-opponent PPO endpoints. The
  self-play arms are a separate, pre-registered experiment still running.
- ⚠ **Not "a learned policy beats the bar unaided".** §46's caveat governs
  verbatim: the charge move is the **script's geometry** supplied by the charge
  decode, the bar's rows are scripts at no decode, and six clone seeds share one
  teacher.
- ⚠ **Not a new training method.** It is arithmetic on two checkpoints. Whether
  a run *trained* under an equivalent constraint lands in the same place is
  exactly what the KL-anchor arm tests, and that is unresolved.
- **Not a resolution of the refereed cell.** +3.28 at 0.90 SE is a tie. Settling
  it needs more seeds — the SE here is across six, and only six clones exist.
