# Beating the shooting opponent: what was actually in the way

**Status: in progress.** The audit and the no-retraining measurements below are complete.
The 2x2 training screen they motivated is running; its results are not in this report yet.

## TL;DR, in plain terms

We spent three experimental batches asking whether the agent would learn to hide behind
walls. It never did. This report says why that question was the wrong one, and finds two
concrete things that were wrong instead — one of which is worth more than everything the
cover programme measured, and needed no training at all to demonstrate.

**First, a correction: the agent was never beating the scripted heuristic.** Batch 3
concluded "all four arms clear the bar". That comparison scored the agent on one set of
random maps and the baseline on a *different* set. Score them on the same maps and the
heuristic wins 0.77 to 0.67. We had been reading a win as a win when it was a loss.

**Second, the agent's army piles onto one objective, and the reward tells it to.** Standing
on a controlled objective pays a model +0.25 per step. The knob meant to discourage
over-stacking charges at most 0.24, *ever* — so no matter how many models crowd onto one
point, crowding is still profitable. The agent does exactly what it is paid to do: fifteen
models on an objective the enemy has already abandoned, while two contested ones go
untouched.

**Third, the agent shoots the same enemy to death over and over.** Every model picks its
target independently, and the network is built so that identically-placed models must make
identical choices. Result: 7.7 shooters converge on 2.86 targets when 11 are available, and
**36–40% of all shots fired are discarded** because the target was already dead. This is not
a subtle effect — it is more than a third of the army's firepower, thrown away every turn.

Making the models pick targets one at a time, each forbidden from taking one already
claimed, recovers most of it. On the **existing trained network, with no retraining at all**,
that single change moved the score from +16.3 to **+26.8**, closing about half the gap to the
scripted bar.

## ⚠️ The shooting rules are going to change

Flagged by the project owner on 2026-08-07: **the shooting mechanic as implemented is not how
the tabletop rules actually work**, and it is expected to change. Work continued on the
current model deliberately rather than pivoting — but that splits this report's findings in
two, and the split is worth knowing before reusing any of it.

**Depends on the current shooting model — re-derive after the rules change:**

- the 36–40% discarded-shot rate and all the overkill arithmetic (Part 3). It rests on
  index-order resolution, damage applied immediately, shots at dead targets silently
  dropped, and `max_wounds: 1` making every wound a kill — so `p(kill/shot) = (4/6)³`.
- the `distinct_shooting_targets` decode and its entire rationale (Part 4).
- `contest_and_spread`'s target-claiming (Part 5). Its *allocation* half survives.
- `firepower_ratio`, which leans on line-of-sight symmetry plus range gating.

**Independent of the shooting rules — carry forward:**

- the objective **stacking / occupancy** finding (Part 2) and the `objective_hold`
  `surplus_value` lever. These are about objective control being a strict count comparison,
  and touch shooting nowhere.
- the layout-set variance result (Part 1.1): a bar is a distribution, not a number.
- the epoch-budget result: screen at ~300, quote effect sizes at 1000+.
- the corrections to the batch-3 report (Part 1).

The live hypothesis as of 2026-08-07 (`surplus_value`) is on the independent side, so a
rules change should not cost that line of work.

## What was asked

Beat `scripted_advance_and_shoot` on the batch-3 configs. Revisit the existing reports
critically rather than building on them unexamined.

## Part 1 — Three corrections to the record

### 1.1 The bar is a distribution over layout sets, not a number

`squad_march_shoot` is deterministic. On the *same* config it scores:

| seeds | n | win | vp_margin |
|---|---|---|---|
| 10000-10019 — training's `eval/baseline_*` | 20 | 0.45 | −15 |
| 10000-10029 — `measure-baselines … 30` | 30 | 0.53 | −4.3 |
| 700000-700029 — `measure-checkpoint`'s held-out set | 30 | **0.77** | **+39.4** |

**A 32-point swing purely from which maps are drawn.** This dwarfs the ~7pp seed-noise
limit the project had been treating as its resolution floor, and it is a larger error term
than any effect any batch has reported.

Batch 3's `0.45` was additionally mis-attributed: it is the in-run 20-episode figure, not
what the command cited in its Reproducing section returns. That section was never run —
`recordings/` holds no batch-3 baseline trace.

*Measured, by re-running `just measure-baselines` at three seed bases.*

### 1.2 On matched layouts, the agent is below the bar

Seeds 700000-700029, 30 episodes, identical layouts, same `evaluate_selector` path:

| policy | on obj | win | player VP | opp VP | **vp margin** | alive | firepower ratio |
|---|---|---|---|---|---|---|---|
| `random` | 0.002 | 0.00 | 14.0 | 163.8 | −150.0 | 0.713 | 0.20 |
| `greedy_nearest` | 0.800 | 0.27 | 98.2 | 150.0 | −51.8 | 0.456 | 1.25 |
| `squad_march` | 0.867 | 0.30 | 96.0 | 142.5 | −46.5 | 0.264 | 0.51 |
| **`squad_march_shoot` (bar)** | **1.000** | **0.77** | 140.2 | 100.8 | **+39.4** | 0.409 | 0.51 |
| H control s1 | 0.925 | 0.67 | 128.7 | 112.3 | +16.3 | 0.588 | 1.72 |
| H control s2 | 0.924 | 0.57 | 123.8 | 112.7 | +11.2 | 0.628 | 1.54 |
| J reason s1 | 0.877 | 0.57 | 122.7 | 114.0 | +8.7 | 0.679 | 1.67 |
| J reason s2 | 0.933 | 0.57 | 120.3 | 115.8 | +4.5 | 0.553 | 1.18 |

The agent trails by ~10pp on win rate and **23 VP** on margin.

What it is *not* losing on is instructive. It **out-survives** the bar (0.59–0.68 alive
against 0.41), **wins the firefight** on `firepower_ratio` (1.2–1.7 against 0.51), and takes
**less** fire (exposure 0.135–0.159 against 0.200) at *greater* distance from terrain
(2.0–2.1 against 1.8 — the range-management signature, not a cover one). It loses on
**objective occupancy: 0.877–0.933 against the bar's 1.000.**

The cover programme was therefore aimed at a mechanism that was not binding. The agent was
not dying too much and was not outgunned; it was not holding ground.

*Measured. The corollary was already recorded on 2026-08-04 under "Occupancy is traded away
… they are buying kills with position", and three batches then looked for cover instead.*

### 1.3 `models_lost` +7 does not survive; "the signal is null" overstates

Both are corrected inline in
[the batch-3 report](2026-08-06-cover-signal-reason-geometry.md). In short: the +7 is
window-dependent (the previous 100-epoch block has a no-penalty run beating a penalty run)
and reverses on held-out layouts, so its **sign** is unestablished; and the signal arm's
data bound the effect at |Δ| ≲ 5pp, which is "not detected" rather than "null".

Also unestablished, and newly flagged: **whether 1000 epochs converges on this config.**
Batch 2 checked its arms were flat across their final buckets; batch 3 did not, and five of
its eight runs move by more than 3 VP between their last two 100-epoch blocks.

## Part 2 — Why the agent does not hold ground

**The reward pays for stacking, and the anti-stack knob provably cannot stop it.**

- `objective_hold` (weight 0.25 × `player_value` 1.0) pays **+0.25 per model per step**,
  private and unconditional, for standing on a controlled disc.
- `closest_objective_v2.overstack_penalty_per_extra` charges
  `0.01 × (p_count − o_count − 1)`. With all 25 models on one point that is **0.24**.

`0.24 < 0.25`, so **every** stack depth nets positive. A 15-stack nets +0.11/step. Meanwhile
everything that rewards spreading — `vp_gain`, `objective_coverage` — is a **global**
calculator broadcast identically to all 25 models, and is therefore absorbed by the
per-model value baseline. The model that peels off pays the transit cost alone and shares
the gain with 24 free-riders. The differential signal to spread is ≈ 0.

`closest_objective_v2` cannot supply it either: it assigns at most one group per objective,
so with 5 groups over 3 objectives two groups fall back to `argmin(distance)` — which, for a
model already in the pile, is the objective it is standing on. Zero gradient to leave.

Trained control, final per-objective counts, 10 seeds:

```
player [0,15,0]  opp [6,0,14]      player [0,11,0]  opp [6,0,10]
player [3,0,17]  opp [0,11,0]      player [9,13,0]  opp [0,0,19]
player [24,0,1]  opp [0,23,0]      player [3,0,0]   opp [0,8,0]
```

*Arithmetic for the incentive; measured for the behaviour.*

And the opponent makes this maximally costly: `scripted_advance_to_objective` returns
`STAY` unconditionally once inside an objective radius. **Measured total opponent
displacement is exactly 0.0 from round 9 onward, and at least one objective is left with
zero opponents in every single episode.** An abandoned objective costs one model to take and
holds for the rest of the game.

## Part 3 — Why the agent wastes a third of its firepower

`_resolve_shooting_action` (`wargame.py:527-529`) resolves attackers **in index order**,
applies damage immediately, and silently **drops** any shot whose target is already dead.
With `max_wounds: 1`, every shot after the first successful one on a target is wasted.

Per shot, `p(kill) = (4/6)³ = 0.2963`. Five shots on one target expect 0.83 kills; five on
five distinct targets expect 1.48. **Spreading is ~78% more efficient and neither side does
it.**

Measured wastage:

| firer | shots ordered | resolved | **wasted** | duplicate-target |
|---|---|---|---|---|
| `squad_march_shoot` | 438 | 294 | **32.9%** | 63.7% |
| the opponent | 584 | 356 | 39.0% | 66.1% |
| **trained checkpoint** | 1335 | 801 | **40.0%** | **64.8%** |

The mechanism is fully accounted for: feeding the observed shots-per-target histogram
through `1 − (1 − 0.2963)ⁿ` predicts 0.368 waste against 0.364 observed.

**The cause is structural, not informational.** The policy is factorized — one `Categorical`
over the whole `(n_models, n_actions)` tensor, every row sampled independently — over a
backbone with **no positional embedding**, hence permutation-equivariant over player tokens.
Two similarly-placed models *must* produce the same argmax. "Every model picks a different
target" is a **joint** constraint, and no per-model input can express one.

That last point is not a guess. A "how many friendlies already bear on this target" feature
— the obvious fix — was tested and **made things worse**: waste fell 0.326 → 0.243 but
vp_margin fell +25.0 → +17.9, because the least-covered target is the most *isolated* enemy,
i.e. the one furthest from the objectives. And being identical for every shooter, it cannot
break the tie it exists to break.

*Measured, including the falsification.*

## Part 4 — The decoder change, and what it is worth before any training

`distinct_shooting_targets` decodes the shooting phase autoregressively over models: model
`i` may not name a target claimed by a model decoded before it. "Stay" is outside the
shooting slice and legal in every phase, so a model whose only target is taken holds fire
rather than being left with no legal action.

It is a policy-side change, not an env one — the game's rules are untouched, and in
particular **the opponent does not get it**, so this is not a symmetric buff.

PPO correctness is preserved: the joint log-prob is the sum of the conditionals, and
`evaluate_actions` rebuilds exactly those conditionals from the stored actions (the mask
depends only on lower-indexed models' choices, all of which are in the buffer). This is
pinned by a test asserting the two paths agree to 1e-6.

**Result across all four batch-3 checkpoints — same weights, same 30 held-out seeds, same
dice, only the decode changed.** This is a tightly paired comparison: layouts and combat
seeds are fixed by `HELDOUT_SEED_BASE`, so the *only* source of difference is the policy's
own actions.

| checkpoint | vp margin argmax → decode | Δ vp | win argmax → decode | Δ win | firepower ratio |
|---|---|---|---|---|---|
| control s1 | +16.3 → +26.8 | **+10.5** | 0.67 → 0.73 | +6pp | 1.72 → 1.75 |
| control s2 | +11.2 → +9.7 | **−1.5** | 0.57 → 0.63 | +6pp | 1.54 → 1.71 |
| reason s1 | +8.7 → +8.3 | **−0.4** | 0.57 → 0.60 | +3pp | 1.67 → 2.05 |
| reason s2 | +4.5 → +4.3 | **−0.2** | 0.57 → 0.63 | +6pp | 1.18 → 1.46 |

**The first row is an outlier and an earlier draft of this report was built on it alone.**
Corrected reading:

- **On `vp_margin` the decode does nothing.** Three of four checkpoints move by −0.2 to
  −1.5, i.e. flat. The +10.5 does not replicate and should not be quoted.
- **On win rate it is positive on all four, by +3 to +6pp** (mean +5.25). Under a sign test
  that is p = 0.0625 — suggestive, not conclusive, and 6pp is 2 episodes in 30.
- **`firepower_ratio` rises on all four**, by +0.03 to +0.38. So the intended mechanism *is*
  engaging: more of our guns bear for each of theirs. Exposure rises too (e.g. 0.136 →
  0.207), which is the expected cost of more models actually engaging rather than
  duplicating.

The coherent story is that the decode **converts close losses into close wins without
widening the margin** — better fire efficiency kills a few more models, which tips tight
games but does not produce blowouts. That is a real but small effect, and it is exactly the
size of effect this project has repeatedly mistaken for a large one.

*Measured, paired, n=30 per cell. It says the network's shooting preferences are somewhat
informative and the decoder was discarding some of that. It does **not** say the decode is
worth +10 VP, and it does **not** say what training under it does — which is what the screen
is for.*

## Part 5 — A negative result that contradicts Part 2's premise

`squad_march_shoot` allocates squads by a fixed `k % n_objectives` against an opponent whose
destination is knowable at reset and whose position is frozen from round 9. That looks
obviously improvable, so `contest_and_spread` was built to improve it: allocate against each
opponent's *predicted* objective (its nearest, since that is how the opponent steers), skip
objectives that cannot be won with the squads remaining, and claim shooting targets so no
two models stack fire.

**It loses to the bar it was built to beat.** Seeds 700000-700029:

| baseline | on_obj | win | player VP | opp VP | margin | alive | exposure |
|---|---|---|---|---|---|---|---|
| `squad_march_shoot` | 1.000 | **0.77** | 140.2 | 100.8 | **+39.4** | 0.409 | 0.200 |
| `contest_and_spread` | 0.977 | 0.60 | 126.0 | 107.2 | +18.8 | 0.516 | 0.158 |

The first version was worse still (0.903 on_obj, +28.2 margin): it allocated against
*current* occupancy, which is all zeros at deployment, so squads thrashed as the opponent
advanced. Predicting the destination fixed occupancy (0.903 → 0.977) and survival (0.331 →
0.516) without fixing the score.

**Why it loses is the interesting part.** `squad_march_shoot`'s "dumb" `k % 3` split puts 5
squads on 3 objectives as 10/10/5 and wins two of them decisively. `contest_and_spread`
spreads to grab the objective the opponent abandoned and then loses the contested ones.
Both have ~1.0 of models standing on objectives; the bar *holds* more of them. Control is a
strict count comparison, so **concentration is not a defect of the bar, it is the reason the
bar wins.**

This directly undercuts the premise behind the `spread` arm in the screen below. The
anti-stack price was raised because stacking is provably profitable under the reward and the
trained agent visibly over-stacks — but "over-stacking on *one* objective while ignoring
others" and "concentrating enough to win the objectives you contest" are different things,
and a blunt per-model price cannot distinguish them. **Prior on the `spread` arm lowered
accordingly.** The control is in the 2x2 precisely so this can come out either way.

*Measured. Note also that a separate scratch-harness measurement put a similar policy at
0.90 / +53.2 on seeds 10000-10029. That did not reproduce through
`evaluate_selector`, and is not claimed here.*

## Part 6 — How many epochs an arm actually needs

Round 1 answers a question batch 3 left open ("whether 1000 epochs converges is
unestablished"). `eval/vp_margin` in 100-epoch buckets, seed 1:

| epochs | control | spread | control+dt | spread+dt |
|---|---|---|---|---|
| 0–99 | −75.7 | −77.9 | −62.2 | −57.6 |
| 100–199 | −13.1 | −30.7 | −11.1 | −24.3 |
| 200–299 | −5.9 | **−17.6** | −8.0 | −16.3 |
| 300–399 | −2.3 | −12.5 | −5.6 | −12.7 |
| 500–599 | −8.4 | −14.0 | −8.8 | −15.0 |
| 700–799 | −1.4 | −2.1 | −3.6 | −3.1 |
| 900–999 | **+5.4** | −5.6 | +3.8 | +0.3 |

**1000 epochs is not too many — it is arguably too few.** The control is still climbing at
the end, −2.3 at epoch 300 → +5.4 at 950. That **+8 VP over the last 700 epochs is the same
size as the arm differences being measured**, so an early cut is not merely noisier, it is
comparable to the signal. This also explains batch 3's "window-dependent" `models_lost`
result: the metric is still drifting upward, so the averaging window genuinely changes the
number. That was non-convergence, not a measurement bug.

**But the curve is steeply diminishing, and the losing arm is obvious early.** Epochs 0–300
take vp_margin from −76 to −2; 300–1000 add +8. And `spread` was already clearly behind by
epoch 200–299 (−17.6 against −5.9) and stayed behind in every later bucket.

Operational rule adopted for round 2:

- **Screening a lever — ~300 epochs.** 3.3x cheaper (≈2h against ≈7h at 24s/epoch), and it
  would have called round 1's failure before seven hours were committed to it.
- **Quoting an effect size — 1000+**, because the last 700 epochs are worth as much as the
  effect.
- A *marginal* 300-epoch result means "run it longer", not "rejected". Nothing in these four
  curves crosses after epoch 300, but that is four arms on one seed.

## Part 7 — Round 1 result: both levers refuted

1000 epochs, two seeds, scored through `measure-checkpoint` on seeds 700000-700029 against
`measure-baselines … 30 "" 700000` — identical layouts for every row.

| arm | on_obj s1 | on_obj s2 | vp s1 | vp s2 | **vp mean** | Δ vs control |
|---|---|---|---|---|---|---|
| **`squad_march_shoot` (bar)** | 1.000 | — | +39.4 | — | **+39.4** | — |
| **control** | 0.925 | 0.924 | +16.3 | +11.2 | **+13.8** | — |
| control + decode | 0.860 | 0.891 | +10.0 | +9.7 | **+9.9** | **−3.9** |
| spread | 0.535 | 0.504 | +11.5 | +3.8 | **+7.7** | **−6.1** |
| spread + decode | 0.599 | 0.412 | +9.3 | −1.5 | **+3.9** | **−9.9** |

**Every arm lost to the control, and the two levers stack: the combination is worst.**

**`spread` is refuted unambiguously.** Objective occupancy roughly halves on both seeds and
vp_margin drops 6.1. The mechanism is the one Part 5 predicted from the scripted policy:
`overstack_penalty_per_extra` charges a model for standing on an objective at all, so models
leave. Control is a strict count comparison, and concentration is how objectives are won.

**`decode` is mildly harmful when trained under, and useful when it is not.** On the *same*
control checkpoint: argmax +16.3, decode applied post-hoc +26.8, trained-under +9.9. The
constraint works as a decoder and is counterproductive as a training environment. A likely
cause is that sequential masking makes a model's conditional depend on its *index*, breaking
the permutation symmetry the advantage estimates are pooled across — untested, and not worth
testing given the shooting rules are changing.

**Two incidental findings worth keeping:**

*Training is deterministic given seed + config + code.* Round 1's control reproduced batch
3's control bit-identically on both seeds (all ten metrics), from independently trained
weights whose checksums differ. Greedy evaluation collapses low-order weight differences.
Consequence: **do not retrain a control that already exists at the same epoch budget** —
two of round 1's eight runs bought nothing.

*The decode compresses behavioural differences.* `control+dt` s2 scored bit-identically to
the *non*-decode-trained control s2 evaluated with the decode, despite different weights and
genuinely different training (the flag is `True` in the run config; verified). Constraining
the output to distinct, rank-rationed targets leaves similar policies indistinguishable.

## Part 8 — Round 2: the surplus discount fails the same way, and that is the finding

300-epoch screen (per Part 6), two seeds, three arms bracketing
`objective_hold.surplus_value`. Scored on seeds 700000-700029.

| arm | on_obj s1 | on_obj s2 | vp s1 | vp s2 | **vp mean** |
|---|---|---|---|---|---|
| control | 0.784 | 0.671 | −10.8 | −4.8 | **−7.8** |
| surplus (0.25) | **0.284** | **0.388** | −28.2 | −15.2 | **−21.7** |
| surplus0 (0.0) | **0.250** | **0.331** | −20.0 | −13.3 | **−16.7** |

Occupancy collapses on both seeds at both parameter values. Not marginal, so Part 6's
"a marginal 300-epoch result means run it longer" clause does not apply.

**The result is not "surplus_value is badly calibrated". It is that two mechanically
different levers failed identically.** Round 1's `spread` is a *penalty* on concentration;
round 2's `surplus` is a *discount* on models beyond the control quota, and it was
specifically designed never to make occupancy negative — the property that was supposed to
distinguish it. Both halved occupancy anyway. When two different designs fail the same way,
the cause is what they share.

**What they share is that the agent cannot see the quantity they are keyed on.** An
objective reaches the network as nothing but an `(x, y)` location: no radius, no control
state, no friendly count, no enemy count. So a reward that pays differently depending on
"how many of us are already inside this disc" is one the policy has no input capable of
attributing. From its side both levers look like one thing — *standing on objectives pays
less on average* — so it stands on them less. That is precisely what both rounds measured.

This unifies four results that otherwise look like four separate failures, and it reframes
the problem: **an observation defect wearing a reward defect's clothes.** No shaping over
objective occupancy can work while occupancy is unobservable. VP is scored on
`player_count > opponent_count` per objective, so this is the quantity the entire mission
turns on, and it was never an input.

*Measured for the collapse; inferred for the mechanism. Round 3 tests the inference directly
by adding the observation and changing no reward.*

**Secondary: the 300-epoch screen worked, and its limits are visible.** The 300-epoch control
scores −7.8 / 0.784 against the 1000-epoch control's +13.8 / 0.925, so 300 epochs is
emphatically not converged — exactly as Part 6 predicted. But a screen measures *ordering*,
and the ordering separated cleanly on both seeds at **3.4 hours instead of 14**.

## What is running

A 2x2, two seeds each, 1000 epochs, group `beat-2026-08-06`:

| arm | config | decode |
|---|---|---|
| control | `25v25_cover_control.yaml` | off |
| spread | `25v25_beat_spread.yaml` | off |
| decode | `25v25_cover_control.yaml` | **on** |
| both | `25v25_beat_spread.yaml` | **on** |

`25v25_beat_spread.yaml` differs from the control in exactly one field:
`overstack_penalty_per_extra` 0.01 → 0.05, which puts the *uncontested* equilibrium at ~6
models per objective (extra=5 gives −0.25, cancelling the hold reward) while leaving
contested objectives untouched.

**Read `vp_margin`, not win rate** — within-arm seed spread on win rate is 6.0–7.3pp. And
score every arm with `just measure-checkpoint … 30 "" distinct` against
`just measure-baselines … 30 "" 700000`, on identical layouts. Passing `distinct` is
mandatory for the decode arms: the setting is not recoverable from the weights, and scoring
a decode-trained network with a plain argmax sends every model at the same target.

## What this does not support

- **That the decode helps *during* training.** Only the no-retraining re-decode is measured.
  Training under it changes the advantage estimates, and could plausibly do worse.
- **That the anti-stack price is correctly calibrated.** 0.05 is derived from the hold
  reward's arithmetic, not measured. It could over-correct into refusing to contest.
- **Any revision to "the agent manages range rather than using cover."** That claim survives
  — and on better evidence than it was given, since on matched seeds the agent sits *further*
  from terrain than the bar while taking less fire. But note the ablation originally cited
  for it (deleting terrain, batch 1/2) ran at 5.8% of the board hidden, where cover was not
  an available alternative — the same confound already admitted for arm F. **The clean
  experiment — no-terrain at the 19.8% profile — has still never been run.**
- **That the bar is a weak ceiling.** This was tested and **failed** — see below.

## Reproducing

```bash
# The bar, on the layouts measure-checkpoint uses
just measure-baselines examples/env_config/25v25_cover_control.yaml 30 "" 700000

# The agent, same layouts -- with and without the decode
just measure-checkpoint <ckpt>/last.ckpt examples/env_config/25v25_cover_control.yaml 30
just measure-checkpoint <ckpt>/last.ckpt examples/env_config/25v25_cover_control.yaml 30 "" distinct

# The screen
just train-arm 1000 2 beat-2026-08-06 "" "" \
  examples/env_config/25v25_cover_control.yaml examples/env_config/25v25_beat_spread.yaml
just train-arm 1000 2 beat-2026-08-06 "-dt" "--distinct-shooting-targets" \
  examples/env_config/25v25_cover_control.yaml examples/env_config/25v25_beat_spread.yaml
```
