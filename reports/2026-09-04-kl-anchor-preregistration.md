# Pre-registration — the KL anchor arm, and the goal's verdict rule

⚠ **Committed to git while seeds 2-6 were still training**, so its timestamp is
checkable rather than asserted. It is reproduced verbatim from the working copy,
amendments and disclosures included — including a dose recalibration made on
control-only data, a prediction that was later falsified, and a criterion later
found to be underpowered. Nothing has been tidied.

# Pre-registration — self-play from the clone, with and without a KL anchor

Written 2026-09-04 at `d5ec7d4`, before any run in this experiment is launched
and before any score in it exists.

## What the diagnosis says (D1/D2, same day, six paired seeds, n=45)

| regime | clone | ppo-from-clone | paired Δ | t | signs |
|---|---|---|---|---|---|
| K=1 cd=0 — **what PPO trains in** | −85.53 | −65.82 | **+19.72** | 8.22 | 6/6 |
| K=1 cd=1 | −69.05 | −50.83 | +18.22 | 3.50 | 6/6 |
| K=3 cd=0 | −32.18 | −28.72 | +3.47 | 0.52 | 2/6 |
| K=3 cd=1 — **what it is scored in** | −10.67 | −25.58 | **−14.92** | −2.60 | 1/6 |

Harness reproduces §46 (−10.67) and §47 (−25.58) to the decimal.
**Decode headroom: clone +74.87 vp, after PPO +40.23.** PPO bought 19.7 vp of
unaided skill and spent 34.6 vp of headroom doing it. §47 is therefore not an
optimisation failure — PPO succeeded at the objective it was given.

## The intervention, and why this one

The misalignment is **not** the charge decode (D2 falsified: Δ is +18.2 at
K=1 cd=1, barely moved from +19.7). It is the **joint coherent decode**, and
that cannot be moved into training — constrained sampling is refuted at −51.8
from scratch and −43.7 warm-started, scored decoded both times
(reports/2026-08-20). So training regime cannot be made to equal play regime.

What is left is to stop the drift. The §47 intermediate checkpoints show the
damage is **progressive**, not immediate (epochs ≤100 mean −7.3 over 5 points,
>100 mean −24.7 over 13) — ⚠ those epochs are top-k-by-training-reward and so
are not a clean curve, and the early window is NOT established as beating the
clone. It is enough to motivate a trust-region fix and no more.

**KL anchor**: add `beta * KL(pi_theta || pi_ref)` to the PPO loss with
`pi_ref` the frozen clone. Standard, targets the measured mechanism directly,
and probes the learning-dynamics question §48 records as the whole remaining
question.

## The arms

Both warm-started **1:1** from `barclone-s{1,2,3}.ckpt`, on
`configs/experiments/25v25_maps_melee_approach.yaml`, 300 epochs, `ent_coef`
0.003, `--no-tf32`, `--self-play --pfsp-mode uniform --pool-anchor
squad_march_take_charge --pool-capacity 8 --snapshot-every-n-epochs 25`.

- **CONTROL**: as above, no anchor (`--kl-ref-coef 0`, which must be a verified
  no-op — no reference network constructed).
- **ARM**: as above, `--kl-ref-coef <beta>` against the seed's own clone.

Three seeds each. Paired by seed (same `seed_everything`, same warm start, and
the anchor changes no parameter shape, so per-seed differences are paired).

## Readouts, fixed now

Primary: `vp_margin` on `25v25_maps_melee_approach_refereed.yaml`, n=45, seeds
700000+, **K=3 + charge decode** — identical to §46/§47, so all three tables are
comparable. Comparator fixed **by name before any score**: the **§46 clone at
−10.67**, and the **bar at −5.3**.

Secondary, reported always: `decl` / `stood` per episode, `coherent`, and
**decode headroom** (the same checkpoint at K=1 cd=0 subtracted from K=3 cd=1) —
the quantity the diagnosis says is being spent.

## Bounds, power-checked before being written

Per-seed paired sd on this cell is **~14** (the §47 six-seed spread). At three
seeds SE is ~8. So:

- A **per-seed** bound is refused outright: at n=3 it cannot separate anything
  under ~20 vp, and this project has twice recorded a screen as a result and
  reversed it (advance lever, §"300 epochs said FREE; at 1000 it says −16.3").
- **SCREEN PASS** (arm proceeds to six seeds at 1000 epochs): arm − control
  ≥ **+8.0** paired, with **3/3 seeds positive**. That is ~1 SE and is
  deliberately a *screening* threshold, not a claim.
- **SCREEN FAIL**: arm − control ≤ 0, or ≤ 2/3 seeds positive.
- **INDETERMINATE** otherwise — report as such; do not spin, do not pick a beta
  post hoc and call it a result.
- ⚠ **No verdict on the GOAL comes from three seeds.** The goal's cells need six
  seeds; nothing here may be quoted against the bar as met or unmet.

## Committed in advance
- The control is launched on `d5ec7d4` and the arm on the anchor commit. The
  anchor code MUST be verified bit-identical at `--kl-ref-coef 0` before the
  two are compared, by a seeded digest; if it is not, the control is re-run.
- If the anchor helps, the mechanism claim is that **decode headroom is
  preserved**. That is measured, not inferred: headroom must not fall as far in
  the arm as in the control. If the arm wins on vp while headroom falls just as
  far, the stated mechanism is REFUTED even though the arm passed.
- `last.ckpt` is only valid if the run reaches `Epoch 299` and exits cleanly;
  a SIGKILLed run is scored from its highest `ppo-NNN`.

---

# ARM 2 — `require_coherent: false` in training (registered before launch)

`configs/experiments/25v25_maps_melee_approach_freecoh.yaml`, verified to differ
from the control config in exactly two lines (`config_name` and the gate).

**Why, from the same finding.** The gate withholds all `objective_hold` income
from a unit that is incoherent. The §46 clone is coherent **0.754 unaided** and
**0.96 decoded** — so on the first epoch the gate punishes it for a defect that
does not exist in the regime it is scored in, and pays it to acquire a skill the
decode already supplies. That is a candidate for the largest single force
pushing a warm start out of its basin.

Same warm starts, same self-play settings, same three seeds, **same controls**
(the control is unchanged and serves both arms). Training reward only — the
referee, `vp_margin` and every scored quantity are identical, so arm and control
are scored by the same instrument.

**Bounds** — as ARM 1, and for the same power reason:
- **SCREEN PASS**: arm − control ≥ **+8.0** paired with **3/3** seeds positive.
- **SCREEN FAIL**: ≤ 0, or ≤ 2/3 positive. **INDETERMINATE** between.
- Mechanism, measured not assumed: **decode headroom must fall less than the
  control's**. If vp improves while headroom falls as far, the stated mechanism
  is refuted even though the arm passed.

⚠ **A REJECT here is expected to be informative in the other direction too**:
`require_coherent` is the measured-good lever (0.55 → 0.78 for free) and this
turns it off. If the arm loses badly, that is evidence the gate is doing work
the decode does *not* substitute for, which would narrow the corollary.

⚠ Committed now: with two arms against one shared control, **the comparison is
one control per arm, not a best-of-two**. Reporting only whichever arm wins
would be winner-selection on a 2-way choice, worth +1.4 to +2.9 vp by this
repo's own measurement. Both arms are reported whatever they say.

---

# AMENDMENT 2026-09-04, ~11:40 — the anchor target was recalibrated BEFORE any arm score existed

⚠ **What I had seen when I made this change, and what I had not.** I had seen
**control-only** data: the τ=0 control's own trajectory (its epoch 25/50/75/100
snapshots) and interpolations between the clone and its epoch-25 snapshot. I had
**not** scored any arm checkpoint, and no arm-versus-control comparison existed.
Recalibrating a dose on control data before any treatment effect is observed is
legitimate; moving it afterwards would not be. Recorded here so the distinction
is checkable rather than asserted.

**Why it had to move.** Seed 1's control, refereed cell, n=45:

| point | drift (mean KL) | undecoded | decoded | headroom |
|---|---|---|---|---|
| clone | 0.000 | −75.0 | +2.2 | **+77.2** |
| interp α=0.15 | 0.006 | −58.4 | **+10.7** | +69.1 |
| interp α=0.35 | 0.024 | −68.3 | −10.6 | +57.7 |
| interp α=0.65 | 0.096 | −57.3 | −2.6 | +54.7 |
| control epoch 25 | 0.255 | −64.9 | **−37.2** | **+27.7** |
| control epoch 100 | 1.695 | −66.3 | −46.3 | +20.0 |

**Headroom is two-thirds destroyed by a drift of 0.255**, which the run reaches
by epoch 25. A target of 0.3 therefore permits the entire collapse and the
anchor would never have engaged — measured directly: at epoch 25 the arm's drift
is **0.267** and the control's **0.255**, indistinguishable.

**New target: 0.03**, between the α=0.35 point (drift 0.024, headroom 57.7) and
the α=0.15 point (drift 0.006, headroom 69.1). Arm 1 seed 1 is killed at ~epoch
30 and relaunched; nothing from the 0.3 dose enters any table.

⚠ **A consequence I am NOT free to ignore**: the same evidence says the anchor
may be unable to work at all, because the drift that preserves headroom may be
too small to learn anything. That is a real possible outcome and the arm is
still worth running, because it is the only way to tell "too tight to learn"
from "drift is not the mechanism".

---

# NEW EXPERIMENT — weight interpolation, registered before any six-seed number

The pilot above suggests something cheaper than training: a **small step along
the line from the clone to a PPO endpoint** may keep the gain and the
decodability. `+10.7` beats the clone's `+2.2` **and the bar's −5.3** on the
goal's hardest cell — but it is **ONE SEED, ONE CELL, and α was read off a grid
of three**, which is exactly the winner-selection this repo prices at +1.4 to
+2.9 vp. It is a hypothesis, not a result.

**The test.** Interpolate `barclone-s{1..6}` toward
`*-s{1..6}-clone-ppo/last.ckpt` (the §47 endpoints — all six exist, no GPU
training needed), at a **grid fixed now: α ∈ {0.1, 0.25, 0.5}**, six seeds,
refereed cell, n=45, seeds 700000+, scored `K=3` + charge decode.

- Report the **whole grid**. Quoting the best α is selection on three doses.
- Seed 1's α=0.15 pilot is on a **different line** (toward the self-play control's
  epoch 25, not toward the §47 endpoint) and is **not** part of this table.
- **PASS** requires a dose whose six-seed mean beats the clone's −10.67 by
  ≥ +5.0 with ≥ 5/6 seeds positive; only then are the other three cells scored.
- ⚠ Even a pass is **not the goal met**: it would need the other three cells,
  and the interpolated policy still executes charges with the SCRIPT's geometry
  via the charge decode, which §46's caveat governs verbatim.

---

# REPLICATION — does interpolation work on the SELF-PLAY line too?

Registered 2026-09-04 while the controls are at epoch 250/300, **before any
control endpoint exists or is scored**.

α=0.1 was measured on the clone → §47 (fixed-opponent PPO) line. The three
self-play controls give a **different endpoint lineage from the same three
clones**. If the effect is a property of "a small step toward wherever PPO
went", it should replicate; if it was specific to §47's endpoints, it should
not.

**Test.** Interpolate `barclone-s{1,2,3}` → `*-s{1,2,3}-spctl/last.ckpt` at
**α=0.1** (the dose already fixed, not re-tuned), refereed cell, n=45, seeds
700000+, K=3 + charge decode. Report the paired difference against the same
three clones.

- **REPLICATES** if the paired mean is ≥ +4.0 with 3/3 seeds positive.
- **FAILS TO REPLICATE** if ≤ 0 or ≤ 1/3 positive.
- Three seeds cannot resolve anything finer, and no verdict about the goal
  comes from it.

⚠ Committed now: the self-play control endpoints are **worse** than §47's on the
evidence so far (their epoch-25 decoded score is −37.2 against §47's −9.1 at
epoch 24 on the same seed). A replication that fails may therefore say the
endpoint was too poor to be worth stepping toward, not that the method is
wrong — and that reading is only admissible because it is written **before** the
numbers.

---

# THE GOAL'S VERDICT RULE, fixed 2026-09-04 before seeds 2-6 exist

⚠ **Disclosure: seed 1 of the KL arm is already scored** (all four cells ahead
of the bar). So this rule is written with one of six seeds visible. What
protects it is that **the same rule was already applied, earlier today and
unchanged, to two prior tables** — the §46 clone and the α=0.1 interpolation —
before any arm number existed. It is not fitted to the arm.

**The rule.** Per cell, over six seeds, n=45, seeds 700000+, K=3 + charge
decode, against `squad_march_take_charge` re-measured at this revision
(−5.3 / +20.2 / +11.8 / +56.6):

- **WON** — the six-seed mean exceeds the bar by more than **2 SE**.
- **ahead** — by more than 1 SE.
- **tie** — within 1 SE either way.
- **LOST** — below by more than 1 SE.

**The goal is MET only if all four cells read WON.** It is conjunctive; three
wins and a tie is not the goal, and must not be reported as one. "Ahead on the
point estimate" is not a win and never gets quoted as one.

**Also required, and reported whatever it says:**
- **Decode headroom** per seed. The pre-registered mechanism is that the anchor
  works by preserving it. If the cells are won while headroom collapses, the
  mechanism claim is REFUTED even though the goal is met.
- The **paired difference against the unanchored control** at a **matched
  epoch**, since that is what isolates the anchor from self-play itself.
- The per-seed spread. ⚠ A single checkpoint at n=45 carries **~±10 vp** here
  (measured: arm s1's `vs_shoot` read +54.0 / +60.4 / +69.4 at epochs
  199 / 275 / 300 of ONE run), so a cell decided by less than that on one seed
  is not decided.

**Standing caveat that travels with any result**: the charge move is the
SCRIPT'S geometry supplied by the charge decode (§46), the bar's rows are
scripts at no decode, and the six seeds are warm-started from six clones that
share one teacher. A win here is "a learned policy, executing charges with the
script's geometry, beats the script" — never "unaided".
