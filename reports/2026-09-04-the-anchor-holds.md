# The anchor holds — self-play from a clone, with a leash

**2026-09-04**, code `d5ec7d4` + the KL anchor. Pre-registration committed to git
at `71011a0` **while seeds 2–6 were still training**.
`configs/experiments/25v25_maps_melee_approach.yaml`, 300 epochs, `ent_coef`
0.003, `--no-tf32`, `--self-play --pfsp-mode uniform --pool-anchor
squad_march_take_charge`, warm-started 1:1 from the §46 clones. Scored n=45,
seeds 700000+, `K=3` + charge decode, all at a **verified epoch 300**.

Follows [PPO spends the decode's headroom](2026-09-04-ppo-spends-the-decodes-headroom.md):
if PPO destroys a clone by travelling too far from a decodable policy, hold it
near one.

## ⚠ Self-play alone is NOT the fix — that is the control

| | refereed |
|---|---|
| the §46 clones (s1–s3) | −9.07 |
| **unanchored self-play from those clones** | **−27.17** |
| fixed-opponent PPO from them (§47) | −18.47 |

Changing the opponent distribution does not rescue the clone; it destroys it
slightly *faster* than a fixed opponent does. **Anything that follows here is a
property of the anchor, not of self-play.**

## The three-seed screen — PASSES its pre-registered bound decisively

Against its own unanchored control, refereed cell, paired by seed:

**+29.63 ± 1.74, t = 17.0, 3/3 seeds.** The bound was ≥ +8.0 with 3/3.

| cell | mean | SE | bar | gap | SEs | read | per-seed |
|---|---|---|---|---|---|---|---|
| refereed | +2.47 | 12.28 | −5.3 | +7.77 | 0.63 | tie | +0.4 / −17.7 / +24.7 |
| `vs_take` | +37.33 | 9.93 | +20.2 | +17.13 | 1.73 | ahead | +35.9 / +20.9 / +55.2 |
| `vs_deny` | **+32.30** | 3.20 | +11.8 | **+20.50** | 6.42 | **WON** | +30.9 / +27.6 / +38.4 |
| `vs_shoot` | +61.10 | 4.85 | +56.6 | +4.50 | 0.93 | tie | +69.4 / +61.3 / +52.6 |

⚠ **Ahead of the bar on all four point estimates — a first here — and that is
NOT the goal.** The committed rule requires every cell to clear **2 SE**, and
only `vs_deny` does. Three seeds decide almost nothing on the refereed cell,
whose seed spread is ±21.

## The mechanism was pre-registered and it CONFIRMS

| | decode headroom |
|---|---|
| the §46 clone | +74.87 |
| **KL arm (3 seeds)** | **+73.47** |
| unanchored control | +49.9 |
| 300 epochs of plain PPO | +40.23 |

Drift, measured against the clone in nats per model: **arm 0.039 at epoch 125
against the control's 1.770**. The anchor does exactly and only what it was
built to do — it keeps the policy inside the region where the decode still
works — and the vp follow.

## ⚠ Arm 2 is REJECTED, and it NARROWS the corollary I published this morning

`require_coherent: false` in training (`..._freecoh.yaml`, verified to differ
from its control in exactly two lines). One seed, epoch 300:

| | refereed | `vs_take` | `vs_deny` | `vs_shoot` | decoded coherency | decl/ep |
|---|---|---|---|---|---|---|
| freecoh | −19.6 | −14.0 | +0.9 | +7.7 | **0.906–0.921** | 5.6–6.6 |
| control | −27.8 | — | — | +43.4 | 0.938–0.945 | 6.0–8.9 |
| KL arm | +0.4 | +35.9 | +30.9 | +69.4 | **0.960–0.976** | 9.3–12.4 |

The registration said in advance that a reject here would be evidence the gate
does work the decode does not substitute for. **It is.**

**And the mechanism corrects me.** I wrote this morning that *"a play-time decode
makes the corresponding training-time skill worthless."* Too strong. freecoh's
**decoded** coherency is only 0.906–0.921: the decode **could not repair** a
policy that never learned formation. The joint decode chooses the most probable
*legal* combination from each model's top-K — if the policy was never pushed
toward coherency, a legal combination is often not in the top-K to choose. So:

> **A decode substitutes for the EXECUTION of a skill, not for the training
> pressure that makes the skill representable.** Training-time coherency
> pressure is what stocks the candidate set the decode selects from.

That also explains why the anchor works: it holds the policy where the decode
still has good candidates.

## Where the goal stands, on one reading rule applied to every route

| route | refereed | `vs_take` | `vs_deny` | `vs_shoot` | seeds |
|---|---|---|---|---|---|
| §46 clone | **LOST** | WON | ahead | tie | 6 |
| interpolation α=0.1 | tie | tie | **WON** | tie | 6 |
| **KL-anchored self-play** | tie | ahead | **WON** | tie | **3** |

**NOT MET.** The goal is conjunctive and needs 4/4 WON at six seeds. Seeds 4–6
are running.

⚠ **Projection committed before they land**: if per-seed variance holds, six
seeds give refereed ~0.89 SE (still a tie), `vs_take` ~2.4 SE (could become
WON), `vs_shoot` ~1.3 SE (ahead, not won). Most likely **2 WON / 1 ahead /
1 tie** — a real advance and still not the goal.

⚠ §46's caveat governs every row: the charge move is the **script's geometry**
supplied by the charge decode, the bar's rows are scripts at no decode, and six
seeds warm-started from six clones share one teacher.
