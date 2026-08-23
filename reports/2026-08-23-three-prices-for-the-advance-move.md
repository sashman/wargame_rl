# Three prices for the Advance move, and none of them is low enough

**2026-08-23.** No GPU. `just measure-advance-use` (new), `just measure-paired`,
`just measure-maps`, all on `configs/experiments/25v25_maps_advance.yaml`.

The goal this serves: *make the movement action space one a policy can actually
learn.* Advance is the first move type past "normal", it went in as a 48-action
slice bolted on after shooting, and the arm that used it was rejected at **−26.7
unpaired** against a non-advance control. Three encoding defects were nominated
as the cause. This priced them, and then priced the move itself.

⚠ **Everything below is negative.** Four candidate defects eliminated, one
proposed mechanism refuted, three scripted rules rejected. That is the result.

---

## 1. The census: all four candidate defects fail to bind

`just measure-advance-use <policy|ckpt> <config> [n] [decode_topk]` walks the
held-out nine and separates what a policy buys with the advance from what it
pays. Three seeds of the rejected arm, `last.ckpt` (epoch 299), n=10, at **both**
`decode_topk` 1 and 3 — the joint decoder enumerates unit-level combinations and
could manufacture unit agreement by itself, so it has to be ruled out.

| | s1 K=1 | s1 K=3 | s2 K=1 | s2 K=3 | s3 K=1 | s3 K=3 |
|---|---|---|---|---|---|---|
| advance share of model-steps | 9.5% | 9.4% | 15.4% | 15.0% | 10.9% | 10.2% |
| **dominated** (≤ Move) | 2.0% | 0.5% | 4.6% | 5.9% | 0.8% | 0.4% |
| waste (start *and* end inside one objective) | 4.0% | 3.0% | 3.2% | 3.9% | 2.3% | 1.8% |
| reallocation (started inside, left) | 18.7% | 20.2% | 18.0% | 17.3% | 17.2% | 17.5% |
| **unanimous 5-of-5 trigger** | 75% | 81% | 78% | 81% | 64% | 75% |
| one model dragged four | 9% | 8% | 9% | 9% | 11% | 7% |
| shooting slots forfeited | 10.8% | 10.4% | 17.3% | 16.5% | 13.2% | 11.6% |

**K=1 ≈ K=3 in every cell**, so nothing here is the decoder's doing.

- ⚠ **Dominated actions do not bind.** Half the advance slice is strictly
  dominated in expectation — bin 1 is always within a normal move's reach and bin
  2 is whenever the roll is ≤ 3 — and trained policies pick one on **0.4–5.9%**
  of their advances. They learned to avoid them.
- ⚠ **The accidental unit trigger does not bind.** A move type is a unit decision
  in the rules and a per-model action here, resolved upward: one model choosing an
  advance forfeits all five models' shooting. At initialisation that fires on
  `1 − (1 − 48/150)^5` = **85.5%** of five-model unit-turns. At convergence the
  policy chooses unanimously on **64–81%** and one model drags four on **7–11%**.
  It learned the unit-level structure without being given it.
- ⚠ **"Advancing from inside an objective" was a misleading statistic and was
  corrected before publication.** Counted raw it is 21–31% and reads as waste.
  Split by where the model *ends*, only **1.8–4.0%** start and end inside the same
  objective; **17–20%** start inside one and leave, which is reallocation — the
  behaviour the record says is missing. **An advance that goes somewhere is not
  waste.**
- ⚠ **Within-unit distance spread has a control, and the control kills it.** It is
  p90 **4–6"** on advancing unit-turns against a 2" coherency chain — and **the
  same on the walking unit-turns of the same policy** (mean 1.28" walking v 0.88"
  advancing on s1). It is how the policy moves, not what the advance does.

**The move type was not the problem the census was asked to find.** What the
census cannot see is the *path*: 48 actions the policy must learn to ignore is a
sample-efficiency cost that shows up as a worse policy at a 300-epoch screen, not
as a worse action distribution at convergence.

## 2. Three scripted rules, three rejections

The record's standing requirement is that a claim is priced as a scripted policy —
one inference run, no GPU — before it becomes a training run. Paired against
`squad_march_take` on the same config, n=100, **three seed bases**, because the
bar is a distribution over layout sets.

| rule | what it prices | advancing unit-turns | paired difference | seed bases positive |
|---|---|---|---|---|
| `advance_when_out_of_reach` | nothing | — | **≈ −78** (2×2) | — |
| `advance_when_no_shot` | the forfeited shooting | 11.2% | **−18.4** (−35.0 / −6.3 / −13.8) | **0 of 3** |
| `advance_to_arrive` + no-shot | shooting *and* the gain | 2.2% | **−11.9** (−20.1 / −8.9 / −6.7) | **0 of 3** |

- `advance_when_no_shot` advances only when a normal move would have left no enemy
  within weapon range of any member — so the fire it gives up is fire it never had.
  Range only, deliberately: sight can only *remove* shots, so this errs toward
  declining a free advance rather than spending a real shot.
- `advance_to_arrive` adds the clause that the run must convert a two-turn approach
  into a one-turn arrival. Control is a headcount at the scoring moment, so being
  nearer scores exactly what being much nearer scores: nothing.

⚠ **The loss shrinks with usage and never becomes a gain.** 11.2% of unit-turns
costs 18.4 vp; 2.2% costs 11.9. The family converges on `squad_march_take` — which
advances 0% — **from below**. That is the signature of a move whose value is
negative wherever it is spent, not of a rule that is aimed badly.

## 3. The mechanism I proposed, and the measurement that refuted it

**Proposed:** D-14, threat is move plus range. An advance crosses twice the ground,
so it ends inside the enemy's reach a turn earlier, having already spent the shot
that would answer.

**Refuted, by the statistic built to test it.** Model-moves ending inside an alive
enemy's weapon reach:

| | advancing | walking |
|---|---|---|
| `squad_march_take_advance` | **4.1%** | 22.4% |
| `squad_march_take_arrive` | **4.9%** | 20.1% |
| the trained arm (s1) | **8.6%** | 44.7% |

Advancing moves end **five times safer** than walking ones. Obvious in hindsight —
the rule only advances when nothing is in range, and nothing is in range when you
are far away. The proposal had the sign backwards.

## 4. The mechanism the numbers do support

All 45 tables, n=10, `squad_march_take` against `squad_march_take_advance`:

| | `take` | `take_advance` | Δ |
|---|---|---|---|
| VP margin | −2.8 | **−23.7** | −20.9 |
| own VP | 216.6 | 208.9 | −7.7 |
| **opponent VP** | 219.3 | **232.5** | **+13.2** |
| held | 2.573 | 2.276 | −0.30 |
| alive | 0.396 | 0.349 | −0.047 |
| **exposure** | 0.2156 | **0.2388** | **+10.8%** |
| **firepower** | 1.091 | **1.004** | **−8.0%** |
| coherent | 0.845 | 0.859 | +0.013 |
| adrift | 0.794 | 0.660 | −0.134 |

Coherency *improves*, so this is not a formation failure. Nearly two thirds of the
loss is what the opponent gains. The move itself lands safely; the **episode** does
not — arriving early buys turns standing forward under fire, and the firepower
ratio falls below parity because a volley was forfeited on the way in.

⚠ **This is a whole-episode cost that no end-of-move statistic can see.** Read
`exposure` and `firepower` beside any movement feature, not just the geometry of
the move.

## 5. The clock — pre-registered, and confirmed

If arriving a turn early is what an advance buys, its value is a **fraction of the
game**: one turn of nineteen scoring events at 20 rounds, one of four at 5. The
prediction was written before the numbers — *the paired difference rises
monotonically as the round count falls, and turns in the advance's favour at five
rounds* — and `rounds=` is already a scenario override on every measure recipe, so
it cost one command.

`squad_march_take_arrive` against `squad_march_take`, n=100, three seed bases.
**Positive means plain walking wins.**

| rounds | 700000 | 800000 | 900000 | mean | outcome sd | normalised |
|---|---|---|---|---|---|---|
| **5** | **−1.4** | **−1.8** | **−1.9** | **−1.7** | 12–13 | **+0.14 sd to advancing, 3 of 3** |
| 10 | +5.2 | −1.0 | −0.2 | +1.3 | 31–35 | −0.04 sd |
| 20 | +20.1 | +8.9 | +6.7 | +11.9 | 83–100 | −0.13 sd, **0 of 3** |

Monotone, and the five-round column is 3 of 3 with t = −1.78, −2.25, −2.73.

⚠ **Absolute vp are NOT comparable across horizons** — a five-round game's whole
outcome sd is 12 against twenty rounds' 91, so 1.7 there is a larger share of the
game than 11.9 is here. The normalised column is the one to read.

**And the five-round game is not degenerate**, which is the artefact this result
would otherwise be. All 45 tables, n=10, `rounds=5`:

| policy | VP margin | held | alive |
|---|---|---|---|
| `hold_deployment` (the floor) | **−33.1** | 0.79 | 0.916 |
| `squad_march_take` | −0.7 | 2.50 | 0.632 |
| `squad_march_take_arrive` | **+0.1** | 2.50 | 0.623 |

The floor sits 32.4 vp below the marcher on a game worth ~45 vp a side, and both
marchers reach `held` 2.50 — five rounds at Move 6 covers 30", and the objectives
are 20–40" out. Policies separate.

## 6. What this means

**Advance is a five-round move being played in a twenty-round game.** The config
that trains runs 20 battle rounds, and there the move is worth ≤ 0 however
carefully it is spent — which is why every rule aimed at it lands short of plain
walking, and why the loss shrinks toward zero as the rule fires less.

That reframes the goal it was measured for, in two directions at once, and both
are worth stating because they pull opposite ways:

- **It lowers the value of re-encoding.** The census says a trained policy already
  uses the slice sanely — it avoids dominated actions, it agrees at unit level, its
  advances go somewhere. Making a worth-≤0 move easier to choose is not obviously
  progress.
- **It raises the value of shrinking the slice.** If the move should almost never
  be taken at this horizon, then **32% of the action space is an option the policy
  must spend samples learning to decline**, half of it strictly dominated. That is
  an argument for criterion 1 of the goal — no dominated actions — on entirely
  different grounds from the one it was written on: not "the policy picks them" but
  "they cost exploration to rule out".

⚠ **What is still not measured is the path, and it is the only live explanation
left for the −26.7.** Every statistic here is taken at convergence. A 300-epoch
screen prices sample efficiency, and nothing above prices sample efficiency.

## 7. Cautions earned

- ⚠ **Split a statistic by where the model ENDS, not where it starts.** "Advances
  from inside an objective" reads 21–31% and looks like waste; it is 1.8–4.0% waste
  and 17–20% reallocation.
- ⚠ **Every behavioural statistic needs its within-policy control.** Within-unit
  distance spread looked like an advance defect at p90 4–6" against a 2" chain,
  until the same policy's *walking* unit-turns came out the same.
- ⚠ **An end-of-move statistic cannot see a whole-episode cost.** Advancing moves
  end five times safer than walking ones and still raise episode exposure 10.8%.
- ⚠ **Check the decoder is not the author of a behavioural finding.** K=1 and K=3
  agreed in every cell here; had they not, the unit-agreement result was the
  decoder's.
- ⚠ **`random` is not a control for action-slice usage.** `RandomBaselinePolicy`
  samples `0..n_move_actions` and can never choose an advance, so it reports 0%
  regardless.

## 8. Reproducing

```
just measure-advance-use <policy|ckpt> configs/experiments/25v25_maps_advance.yaml 10 1
just measure-paired squad_march_take_arrive squad_march_take \
    configs/experiments/25v25_maps_advance.yaml 100 700000 rounds=5
just measure-maps squad_march_take_arrive configs/experiments/25v25_maps_advance.yaml 10 "" 1 rounds=5
```

---

## 9. Addendum — the slice, re-encoded

Two of the goal's four criteria are unambiguous and do not depend on the horizon
question, so they were built and verified the same day.

**Absolute rungs.** An advance bin is now `M + (bin + 1) × (6 / bins)` — at `M = 6`
with three bins, **8" / 10" / 12"** — and the unit's D6 gates which rungs are
**legal** (`ActionHandler.advance_legality`, masked on both seats) rather than
deciding what an action means.

| | old: `fraction × (M + roll)` | new: absolute rungs |
|---|---|---|
| bin 1 at roll 1 / roll 6 | 2.33" / 4.00" | 8", legal only at roll ≥ 2 |
| bin 3 at roll 1 / roll 6 | 7.00" / 12.00" | 12", legal only at roll 6 |
| dominated share of the slice | ~50% in expectation | **0 by construction** |
| dominated advances *measured* | 3.5–13.8% (scripts), 0.4–5.9% (agents) | **0.0%** |

- ⚠ **The cross-config bridge holds.** `squad_march_take` never advances and scores
  **−2.8 / `held` 2.57 / `alive` 0.396 / `coherent` 0.845** on all 45 tables either
  side of the change — identical to every printed digit. That is what makes any
  comparison across the re-encoding legitimate.
- `n_advance_speed_bins` defaults to 0, so **no golden config is touched** and every
  reward and observation golden stays bit-identical.
- ⚠ **It voids the advance arm's checkpoints behaviourally.** Tensor width is
  unchanged so they load; their indices now mean different distances.
- ⚠ **At three bins a roll of 1 leaves no legal rung.** The rules permit a 7"
  advance and the ladder cannot express it. Deliberate — a 1" gain never repays a
  turn of fire.

**The exploration burden, measured.** 120 movement phases on the advance config:
**25.1 of 48** advance actions are legal per model on average (52% of the slice — the
roll masks the rest), and **0.00** of them are dominated. Under the old ladder roughly
**24 of 150 actions — 16% of the whole space — were strictly dominated** and always
legal. That is the entire mechanism the re-encoding buys, and it can only be cashed by
training; every other statistic in this report is taken at convergence.

**And the other two criteria are blocked together, on a mistake worth recording.**
"Move type is a unit declaration" has an obvious implementation — let the unit's
leader declare and mask the advance slice for everyone else. It is a genuine
unit-level declaration and it cuts the initialisation trigger rate from 85.5% to
32%. **It would also shatter formation.** Move type and displacement are the same
action here, so a leader-only advance caps every other model at `M`: the scripts
advance **5-of-5 with a within-unit distance spread of 0.00"**, and leader-binds
forces that spread to ~6" against a 2" chain.

A declaration therefore has to be *separable* from the displacement, which needs an
**action-bearing command phase** — and `command` is in `skip_phases` on every
config, so making it act changes steps-per-round everywhere. That is the same
machinery the "additive cost" criterion needs, so the two are one change, and it is
not worth spending before the horizon question is settled.

---

## 10. Addendum — the declaration

The remaining two criteria, built the same day once the blocker turned out to be
one change rather than two.

**A `move_type` slice of two actions** (`normal`, `advance`), valid in the
**command phase**, registered last so no existing index moves. The unit's
lowest-indexed alive model declares and the whole unit is bound.

| | before | after |
|---|---|---|
| how the move type is decided | an **OR over five per-model movement actions** | one declaration by the unit's leader |
| P(a 5-model unit advances) at init | 0.855 | one binary choice per unit |
| action space, advance on | 150 | 152 |
| action space, `n_advance_speed_bins: 0` | 102 | **102** |
| cost of adding fall back or charge | ~48 actions + a unit-resolution hack | **one value in `move_type`** |

- **STAY declares `normal`**, so every policy written before the declaration
  behaves as it did. Verified: a non-advancing script scores **bit-identically on
  10 of 10 seeds** across the change.
- **Declaring spends the unit's shooting immediately**, whether or not a member
  then uses a long rung — the rules' cost attaches to the move type, not the
  distance.
- ⚠ **The advance roll moved to the start of the side's turn.** A declaration made
  in the command phase would otherwise be blind, and since rung legality is gated
  on `M + roll`, no rung would ever be legal. Idempotent and keyed on
  `(battle_round, active_player)` rather than hung on a phase transition, because
  command is the *first* phase of a turn and the first turn of an episode never
  advances into it.
- **`n_advance_speed_bins > 0` now requires the command phase**, rejected at
  construction otherwise — the rungs would exist and no declaration would ever be
  legal, and a run would measure a feature it never had.

⚠ **The cheaper version is actively wrong, and this is the finding.** Masking the
advance rungs to the leader alone makes the declaration unit-level inside the
existing structure — and **shatters formation**, because a move type and a
displacement were the same action, so a leader-only advance caps every squadmate at
`M`. The scripts advance **5-of-5 with a within-unit spread of 0.00"**; leader-binds
forces ~6" against a 2" chain. Separating the declaration from the distance is the
whole reason the unit decision is safe.

### What it voids, and what it does not

The command phase is now a real agent step on advance configs. Verified neutral on
the game — the golden config scores **bit-identically on 8 of 8 seeds** with command
skipped or active — **except in episodes that end early by elimination**:

| seed 700003, table_02 | steps | final round | scoring events | own VP |
|---|---|---|---|---|
| command skipped | 20 | 11 | **10** | 120 |
| command active | 30 | 11 | **9** | 105 |

10 of 45 tables moved by exactly **−1.5** at n=10 — 15 VP in one episode each. The
skipped command phase used to be traversed *inside* the terminating step and scored
there; it is now a phase the agent never gets to leave. **Arguably more correct**: a
scoring event that requires reaching your next turn should not fire in a game that
has already ended. Either way it is a change, so re-measure rather than carry a
figure across it.

**Throughput: ~15% more wall-clock per battle round, not the 50% the step count
suggests.** Per-step cost *falls* from 4.338 ms to 3.334 ms because a command step
does almost nothing, so 1.5× the steps nets 8.68 → 10.00 ms per round. A 2048-step
epoch runs 9.5 s → 7.5 s and covers a third fewer rounds.
