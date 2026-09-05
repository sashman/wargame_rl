# Pre-registration — ARM 4: arm 3 extended to 1000 epochs

Written 2026-09-05, **before any 1000-epoch checkpoint exists**.

## Why

Arm 3 (two-anchor pool) at six seeds wins `vs_take`, `vs_deny` and `vs_shoot`
and **ties the refereed mirror at −0.12 SE**. That single cell is the goal.

Pool composition cannot fix it: the mirror is played against
`squad_march_take_charge`, and the arm that anchors on it **alone** (arm 1)
still only reaches +1.45, short of the ~+5 a win needs. So the remaining gap is
not the opponent distribution.

This project's own rule is *"screen at ~300 epochs, quote effect sizes at
1000+"*, and the historical 300→1000 gain is **~+8 vp** — the size required.
The anchored arm is the first lineage here that does **not** degrade with
training (headroom held at +67 to +78 against plain PPO's +40), so it is the
first for which extending is a reasonable bet rather than a known loss.

## The arm

`--resume-ckpt-path` from each arm 3 seed's epoch-300 `last.ckpt` to
`--max-epochs 1000`, everything else identical. **Three seeds** (1–3) as a
screen. ⚠ Resume was broken for anchored runs until today and is now verified by
resuming one to epoch 302; without that fix this arm was impossible.

## Bounds, fixed now

Primary readout: **refereed**, because that is the only cell not won.

- **PASS** (extend to six seeds): refereed six-... three-seed mean ≥ **+6.0**
  AND `vs_shoot` stays above **+60** (it must not give back the cell arm 3 won).
- **FAIL**: refereed ≤ arm 3's −5.95, or `vs_shoot` falls below +56.6.
- **INDETERMINATE** between.

⚠ **Power, stated before the numbers**: arm 3's refereed per-seed sd is ~12.8,
so three seeds give SE ≈ 7.4. A +12 improvement is 1.6 SE. **This screen cannot
establish a win on that cell**; it can only say whether six seeds are worth
spending. Written down because this project has read an underpowered screen as a
result twice.

⚠ **The record's own warning applies with full force**: the advance lever read
"+2.2, free" at 300 epochs and **−16.3 at 1000**, with two of three seeds
flipping sign. **A 1000-epoch result can reverse a 300-epoch one rather than
sharpen it.** If this arm comes back worse, that is a real finding about the
anchor and not a fluke to be re-run.

## Committed in advance
- Scored identically to every other row: n=45, seeds 700000+, K=3 + charge
  decode, all four cells, at a **verified** epoch (checked inside the file, not
  inferred from `last.ckpt` existing — that mistake cost a mislabelled table
  yesterday).
- Decode headroom reported whatever it says. If the cells improve while headroom
  collapses, the anchor stopped being the mechanism and that must be said.


---

# RESULT — the mirror is WON, and `vs_shoot` is given back. Still 3 of 4.

⚠ **FIRST ATTEMPT RETRACTED, and it is the important part of this entry.** The
launch script dropped `--kl-ref-target`/`--kl-ref-coef` on the resume. The
reference loaded from the checkpoint, nothing errored, and the term was
multiplied by **zero**: three 1000-epoch runs completed looking healthy, lost
all four cells, and drifted to **2.69 nats against a 0.03 target** — worse than
the never-anchored control's 1.77. That collapse was one step from being written
up as *"training longer destroys the policy"*. **It measured nothing of the
sort.** What caught it was the mechanism check, not the score: a drift of 2.69
against a 2.61-nat *total action entropy* is not a bad number, it is an
impossible one. Two guard bugs were fixed as a result (a resume at coefficient
zero is now refused; a resume is now an accepted source of the anchor).

Rerun with the flags verified present and the leash verified holding
(0.045 → 0.055 → 0.043 across epochs 300→675). Three seeds, epoch 1000.

| cell | 300 ep | 1000 ep | change | bar | gap | read |
|---|---|---|---|---|---|---|
| **refereed** | −3.77 | **+8.77** | **+12.53** | −5.3 | **+14.07** | **WON** (4.05 SE) |
| `vs_take` | +30.37 | +32.63 | +2.27 | +20.2 | +12.43 | WON |
| `vs_deny` | +32.47 | +40.87 | +8.40 | +11.8 | +29.07 | WON |
| `vs_shoot` | +67.87 | **+55.70** | **−12.17** | +56.6 | −0.90 | tie |

**VERDICT: FAIL against the committed bound** (refereed ≥ +6.0 **and**
`vs_shoot` above +60). Refereed passed handsomely — **the mirror is won for the
first time in this line** — and `vs_shoot` fell below both +60 and its own bar.
Still 3 of 4, a *different* 3.

**Headroom +80.67**, the highest recorded here (clone +74.87, arm 3 +67.27,
the unanchored 1000-epoch run +34.90). The anchor held across 700 extra epochs.

## The finding: refereed and `vs_shoot` are a measured TRADE-OFF

Paired, same seeds (the extension is a resume, so fully paired):

| cell | change | t | signs |
|---|---|---|---|
| refereed | +12.53 | +1.68 | **3/3** |
| `vs_shoot` | **−12.17** | **−2.05** | **0/3** |

`vs_take` and `vs_deny` did not move resolvably. **Two independent levers —
pool composition and training length — move these two cells in OPPOSITE
directions**, at roughly 1.5 vp of refereed per 1 vp of `vs_shoot`:

| arm | refereed | `vs_shoot` |
|---|---|---|
| 100% charge floor, 300ep | +1.45 | +56.03 |
| 50/50 floor, 300ep | −5.95 | **+65.55** |
| 50/50 floor, 1000ep | **+8.77** | +55.70 |

⚠ **The frontier does not reach the winning region.** Winning both needs
refereed ≥ +5.1 and `vs_shoot` ≥ +59.8. Sliding along the measured frontier
from arm 3, buying refereed's 11.1 vp costs 7.4 of `vs_shoot` and lands at
**+58.1 — short by ~1.7 vp**. No point on the observed trade-off wins both, and
the shortfall is small enough that a modest overall improvement would close it.

**Next**: seeds 4–6 of this arm, launched. `vs_shoot`'s three-seed spread is
wide (+65 / +59 / +43) and its mean sits 0.9 below the bar, so six seeds may
land either side.


---

# Addendum — the mirror's 44% IS par, and margin hides it

Prompted by a plain question ("does the agent now reliably win games?"), which
the margins invite you to answer yes to and which is **wrong**. Best checkpoint,
45 games per opponent, K=3 + charge decode:

| opponent | mean margin | won | drew | lost | win rate |
|---|---|---|---|---|---|
| refereed (the mirror) | **+2.33** | 20 | 2 | 23 | **44%** |
| `vs_take` | +34.22 | 30 | 1 | 14 | 67% |
| `vs_deny` | +50.56 | 35 | 0 | 10 | 78% |
| `vs_shoot` | +65.22 | 33 | 1 | 11 | 73% |

**In the mirror the agent finishes ahead on points while losing more games than
it wins.** It wins its wins by more than it loses its losses. A positive
`vp_margin` and a losing record are not in tension here, and every headline
figure in this record is the former.

**And 44% is PAR for that seat**, measured rather than assumed:

| in the mirror seat | won | drew | lost | win rate | mean |
|---|---|---|---|---|---|
| **the bar** (`squad_march_take_charge` vs itself) | 20 | 1 | 24 | **44%** | −5.33 |
| **the agent** | 20 | 2 | 23 | **44%** | **+2.33** |
| `squad_march_take` (never charges) | 13 | 1 | 31 | 29% | −33.56 |

The agent **matches the script's record exactly — 20 wins each — while scoring
+7.7 more per game.** 44% is the going-second handicap, not weakness; the 29%
row proves the seat is not doing all the work, since a worse policy really does
fall away.

⚠ **The rule this earns.** `vp_margin` is the right instrument for this project
(win rate cannot resolve differences under ~7pp here) but it is **not** an
answer to "does it win". Quote a win/draw/loss record whenever the question is
about games rather than points — and quote the BAR's record beside it, because
in a near-draw matchup the absolute rate is mostly the seat.
