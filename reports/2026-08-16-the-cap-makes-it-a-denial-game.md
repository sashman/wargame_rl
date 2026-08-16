# The VP cap makes this a denial game — and copying beats the bar where the reward could not

**2026-08-16.** Everything here is measured on current physics (post-#193), nine
held-out tables and nine *matched* training tables, **n=100 per map**, seeds
700000+, with error bars taken across maps.

Two method notes. The **error bar on a mean is taken across maps, not across
episodes**, because a map is the unit this generalises over: nine maps at n=100
is nine samples of "an unseen table", not nine hundred, and quoting the
episode-level error understates it by ~3x. And the harness is **deterministic
given seeds** — `squad_march_shoot` on the held-out set was measured twice,
before and after the VP columns were added, and returned 111.8 +/- 5.3 both
times.

The two map sets are matched on objective and terrain count (5.67 / 15.67 each).
They are **not** matched on the axis that turns out to predict difficulty; see
"the held-out split is confounded" below.

---

## In short, with no jargon

Each round you score 5 points per objective you hold, but **no more than 15 a
round** — so holding three objectives scores exactly as much as holding six. The
maps have five or six.

That means once you hold three, the only way left to increase your *lead* is to
stop the other side scoring. The trained agent worked this out and holds almost
exactly three. What it never learned is the second half: taking ground off the
opponent. It sat on its three, kept 94% of its army alive, and let the enemy
score freely — conceding 21 points a game more than the scripted benchmark.

**Why it could not learn better.** Two separate reasons, both measured. Sending
one squad forward alone gets that squad killed, so the learning rule is right to
refuse — even though the whole army committing together scores better. And the
training setup pays a bonus for keeping decisions *uncertain*, which held the
agent's play permanently vague; that is why changing what the reward pays for
never moved anything.

**What fixed it.** Not a better reward. A better *hand-written* policy — send
spare squads to the weakest ground rather than attacking the strongest, which is
one line — and then teaching the network to copy it. Copied faithfully enough
(98% of decisions matched), the network scores **113.6 on average** against the
benchmark's **111.8**.

The best network policy on this scenario went from **82.3 to 115.8** in a day,
and a typical one to 113.6. Worth being clear about what did the work: the
copying, not the learning. The reinforcement-learning step contributed nothing
measurable on top.

**⚠ This section first read "the network scores 115.8, ahead on 8 of the 9
unseen maps", and that was the best of five identical copies.** Repeating the
copying five times — same demonstrations, same settings, only the random seed
differing — gives 115.8 / 114.6 / 114.3 / 112.1 / 111.1: **mean 113.6, sd 1.9**.
Four of the five beat the benchmark and one falls 0.7 short, so the win is
**+1.8 +/- 0.9**, less than half the +4.04 originally published. See
[the correction](#correction-the-clone-is-a-distribution-not-a-number).

---

## The mission

`DefaultVPCalculator` scores `min(cap_per_turn, controlled * vp_per_objective)`
= **`min(15, held * 5)`**, from round 2, over 20 rounds — a ceiling of
19 x 15 = **285**. The cap is the rule, not an accident:
`docs/rules/constants.yaml` sets `primary_cap_per_round: 15`.

**Own VP is saturated for anything competent**, which is the fact that
reorganises everything else:

| policy, held-out | player VP | of 285 |
|---|---|---|
| `squad_march_shoot` (the bar) | 272.2 | 95.5% |
| `squad_march_deny` | 277.3 | 97.3% |
| agent, best seed | 266.2 | 93.4% |

So `vp_margin` is decided almost entirely by the **opponent's** score. This is
why `measure-maps` now prints `plr VP` and `opp VP` separately: the margin alone
cannot say which half moved, and the two halves move for different reasons.

## The floor, the bar and the agent

Nine **held-out** tables, n=100:

| policy | vp_margin | +/- | player VP | opp VP | held | alive |
|---|---|---|---|---|---|---|
| `hold_deployment` (the FLOOR) | -88.9 | 8.1 | 121.1 | 210.0 | 1.27 | 0.998 |
| `random` | -26.4 | 6.4 | 182.2 | 208.6 | 1.96 | 0.823 |
| `squad_march` | 79.4 | 5.7 | 270.3 | 190.9 | 3.50 | 0.673 |
| `squad_march_shoot` (the BAR) | 111.8 | 5.3 | 272.2 | **160.4** | 4.00 | 0.762 |
| `squad_march_deny` (new) | 112.3 | 4.8 | 277.3 | 165.1 | **3.00** | 0.590 |
| agent s1-xB @300 | 84.7 | 6.6 | 266.2 | 181.5 | 3.29 | 0.940 |
| agent s4-xB @300 | 84.5 | 7.4 | 266.0 | 181.5 | 3.31 | 0.947 |
| agent s3-xA @300 | 84.2 | 9.4 | 265.3 | 181.1 | 3.26 | 0.967 |
| agent s2-xA @300 | 75.9 | 9.8 | 258.3 | 182.4 | 3.08 | 0.962 |

**Opponent VP across four independently trained seeds is 181.5 / 181.5 / 181.1 /
182.4 — a standard deviation of 0.6.** Whatever the agent has learned about
denial, it has learned exactly the same thing every time, and it is 21 points
short of the bar. Measured against the do-nothing floor, the agent achieves
28.5 of the bar's 49.6 points of denial: **57%**.

`squad_march` and `squad_march_shoot` differ **only** in whether they fire, so
the 30.5 points between them (190.9 -> 160.4) is what shooting is worth as
denial. The agent is not that pair's third member — it moves differently too —
but it reaches a *higher* firepower ratio than the bar on most maps and still
lands at 181.5, closer to the policy that never fires than to the one that does.
It wins the firefight where it happens to be standing, and never brings it to
the ground that is scoring.

Nine **matched training** tables, n=100:

| policy | vp_margin | +/- | player VP | opp VP | held | alive |
|---|---|---|---|---|---|---|
| `squad_march_shoot` | 98.7 | 2.7 | 272.1 | 173.4 | 3.84 | 0.733 |
| `squad_march_deny` | **102.6** | 2.7 | 278.8 | 176.1 | 3.00 | 0.597 |
| agent s1-xB | 92.3 | 1.1 | — | — | 3.65 | 0.945 |
| agent s4-xB | 91.5 | 1.4 | **278.1** | 186.6 | 3.63 | 0.940 |
| agent s2-xA | 86.5 | 2.6 | 274.4 | 187.9 | 3.36 | 0.959 |

**On ground it knows, the agent out-scores the bar on its own VP** (278.1 v
272.1) and still loses by 7, because it concedes 13 more. Opponent VP across
three held-out seeds is 181.5 / 182.4 / 181.5 against the bar's 160.4 — the
denial deficit is not seed noise, it is a property of the policy.

## Holding a fourth objective is better denial than raiding a fifth

`squad_march_deny` was written to price the denial hypothesis before any reward
that teaches it was trained — this project's standing discipline for a cheap
proxy. It banks the cap with `cap // vp_per_objective` squads and sends the rest
at whatever the opponent holds.

It **ties the bar** on held-out (112.3 v 111.8) and beats it on training tables
(102.6 v 98.7) — **while holding 3.00 objectives against the bar's 4.00**. That
is the mechanism proof: `held` ranks policies only up to the cap, and past it
the metric is blind to what actually wins.

But the two reach the same margin by opposite routes, and the bar's is better:
it denies *more* (160.4 v 165.1) by holding a fourth objective **outright**. The
raider ends on **exactly 3.00** objectives — which is the evidence, not an
inference: its denial squads never finish controlling what they were sent at, or
`held` would exceed the three it commits to banking. They contest without
flipping, and **a raid that does not flip an objective denies nothing**.

So the lesson for the agent is not to raid the enemy zone. It is to *hold*
contested ground it currently walks away from.

### Taking the weakest ground beats both, and beats the bar

That reading is testable, and it holds. `squad_march_take` is
`squad_march_deny` with **one line inverted**: after the cap is banked, surplus
squads go to the objectives with the **fewest** opponents rather than the most.
Nine held-out tables, n=100:

| policy | vp_margin | +/- | player VP | opp VP | held | alive |
|---|---|---|---|---|---|---|
| `squad_march_shoot` (the old BAR) | 111.8 | 5.3 | 272.2 | **160.4** | 4.00 | 0.762 |
| `squad_march_deny` | 112.3 | 4.8 | **277.3** | 165.1 | 3.00 | 0.590 |
| **`squad_march_take`** | **116.7** | 4.7 | **277.5** | **160.8** | **4.02** | 0.756 |

On the nine matched training tables the same ordering holds and widens:
`squad_march_take` **108.8 +/- 2.2** against the bar's **98.7 +/- 2.7** — +10.1
there against +4.9 held out.

It is exactly the hybrid the decomposition predicted: the denier's scoring
efficiency (277.5) *and* the bar's denial (160.8), for **+4.9 over the bar**.

**Why inverting one comparison is worth five points.** Denial is still where the
whole margin lives — own VP is capped at three objectives. What was wrong was
the *means*: a raid on defended ground has to cross a strict count threshold to
change anything, and `squad_march_deny`'s `held` of exactly 3.00 says its raids
never did. Weakly-held ground flips on arrival and then denies for the remaining
rounds. **Holding an objective is a far more reliable way to deny it than
contesting one**, and a contest that does not cross the threshold denies
nothing at all.

### What `squad_march_take` actually does

Written out because it is now the strongest policy on the scenario, the target
every clone imitates, and the thing a stronger *opponent* would be built from —
and because "one line inverted" is the effect, not the mechanism.

**Inherited unchanged**, from `squad_march` and `squad_march_shoot`:

- **Squads move as one body.** Every model in a unit follows the *same* vector,
  computed from the squad centroid to its target, capped at
  `min(max_move_speed, distance)`. Relative positions are preserved, which is
  what keeps the unit in coherency. Only once the centroid is within the
  objective's extent does each model settle onto the objective individually, via
  `step_toward_objective` — which steers at the objective's *boundary*, not its
  centroid, so arriving models stop at different points on the perimeter and the
  squad spreads across it rather than piling on one spot.
- **Shooting** is `squad_march_shoot`'s, untouched.
- The only seam a subclass overrides is `squad_objectives(models, env, group_ids)`
  → one objective per squad.

**The assignment, recomputed every movement phase** (control changes, and a
squad en route to a point the opponent has abandoned should be re-tasked):

1. Count living **opponent** models on each objective, under VP's own membership
   rule — `occupants()` uses `area.contains_points` for an area objective,
   because `radius_size` is 0.0 by design there and a distance-to-centre test
   would count only models standing exactly on the centroid.
2. Take the squad centroids, one per `group_id`, over living members.
3. Sort objectives by **ascending opponent count**, ties broken on index so the
   result is deterministic.
4. Walk that order; for each objective, assign the **nearest unassigned squad**.
   Stop when squads run out.
5. Any leftover squads (more squads than objectives) reinforce down the same
   order, `order[squad % len(order)]`.

That is the whole policy: **one squad per objective, cheapest ground first,
nearest squad to each.**

**How it really differs from `squad_march_deny`** — worth stating precisely,
because the report's "one comparison inverted" describes the outcome and would
mislead anyone reimplementing it:

| | `squad_march_deny` | `squad_march_take` |
|---|---|---|
| objective order | ascending opponent count | **same** |
| first `needed` (= `cap // vp_per_objective`) | banked as hold targets | — no such step |
| the remainder | **re-sorted**: contested first, then cheapest | left in ascending order |
| leftover squads | reinforce the *held* subset | reinforce down the *whole* order |
| reads `mission.params` | yes (`cap_per_turn`, `vp_per_objective`) | **no** |

So `take` does not bank the cap and then spend a surplus — **it has no cap logic
at all.** It drops `deny`'s re-sort of the remainder, and the two-tier structure
collapses into one flat ascending list. Banking still happens, implicitly: the
first few entries of that list *are* the cheapest objectives. The class docstring
and the inline comment both describe the effect in cap-banking terms, which reads
as though `needed` were computed; it is not.

**Why it holds 4.02 objectives rather than 3.00.** With five squads and five or
six objectives, step 4 gives nearly every squad its own objective, so the policy
spreads onto the cheapest five. That overshoots the three-objective cap on
purpose: objectives four and five pay nothing in own VP but deny 5 each per round
off the opponent's score.

**The two side-specific reads**, flagged for task #125 (making this playable as
an opponent). Neither mirrors itself, and both fail *quietly*:

- `env.opponent_models` in step 1 — from the other side this must count the
  **player's** models. Unmirrored, the policy targets the ground its own team
  holds. It runs, it looks plausible, and it scores badly for a reason nobody
  would guess from the output.
- `env.player_action_handler` inside `step_toward_objective` and
  `select_movement` — wrong handler for opponent models.

## Two failures, not one

| failure | training maps | held-out maps | nature |
|---|---|---|---|
| denial deficit (~13-21 opp VP) | **yes** | yes | systematic — a **reward** problem |
| occupation collapse (`on_obj` < 0.75) | 0-1 of 9 | 2-3 of 9 | unfamiliar maps only — **generalisation** |

The second is worth about half the held-out gap and is a different bug:

| map | policy | on_obj | held | player VP | alive | exposure |
|---|---|---|---|---|---|---|
| table_35 | agent s1-xB | 0.957 | 3.95 | **279.5** | 0.958 | 0.024 |
| table_35 | bar | 0.967 | 4.10 | 276.1 | 0.794 | 0.040 |
| table_05 | agent s1-xB | **0.389** | 2.37 | 224.6 | 0.964 | **0.002** |
| table_05 | agent s2-xA | 0.393 | 2.03 | **198.7** | 0.990 | **0.001** |
| table_05 | bar | 0.968 | 4.20 | 271.4 | 0.804 | 0.032 |

On table_35 the agent is genuinely competitive and loses by 1.2. On table_05 it
ends with **two thirds of a 96%-alive army standing on no objective**, exposure
near zero. It is not trading badly; it has found somewhere to hide and is
sitting out the game.

## The held-out split is confounded, and should be stratified

Grouping the held-out maps by how much contested ground they force:

| held-out map type | agent deficit v bar (both seeds) |
|---|---|
| **1 own-zone objective** (20, 25, 45) | **-51.5** |
| 2 own-zone objectives (10, 15, 30, 35, 40) | -10.6 |

| map set | share with only 1 own-zone objective |
|---|---|
| held-out (9) | **33%** |
| training pool (36) | 14% |
| training subset scored (9) | 11% |

**The held-out set is enriched ~3x in the map type the agent is worst on.**
"Every table divisible by 5" was a convenient split, not a stratified one, so
the train-versus-held-out difference this project has been calling a
generalisation gap is partly a difference in mission mix. Stratify by own-zone
objective count before quoting a transfer number again.

Two caveats kept rather than smoothed over: the hard group is **3 maps**, and
**table_05 is an unexplained outlier** — it has the easy 2/2/2 shape, and its
geometry (objective spacing, terrain count) does not distinguish it from
table_15 / 30 / 35, where the agent is fine.

## What this predicts, and what was changed

`objective_hold` pays `player_value` 1.0, `contested_value` 0.5,
`opponent_value` 0.25. With `crowding_exponent: 1.0` an objective's pot is split
among its occupants, and `require_coherent: true` pays a detached model nothing
— so the unit, not the model, is the decision-maker, and a squad moving onto
enemy ground takes a **4x pay cut** for the highest-margin action on the board.

Arms in flight at the time of writing (300 epochs, warm starts matched to the
`coherency-crossover` controls, because outcomes here are bimodal by lineage):

- **`deny_high`** — `contested_value` and `opponent_value` both to 1.0, removing
  the pay cut. Only raises income and lowers nothing, so it cannot repeat the
  income-destruction failure that killed `overstack_penalty_per_extra` and
  `surplus_value`.
- **`gamma 0.99`** — the episode is **40 steps** and the default `gamma: 0.9`
  gives a GAE horizon of **6.9 steps**, so a committed move that repays over the
  remaining twelve rounds is discounted to ~4% of face value. Hiding with an
  intact army is what that horizon rewards. `CLAUDE.md` records that the case
  for 0.9 was measured on a different scenario and its refutation retracted.

Two other arms were launched and killed before they answered anything, and are
recorded here so the compute is accounted for: a `deny_mid` (0.75 / 0.5) killed
at epoch ~40 because halving a 4x pay cut still leaves a 2x one, and a combined
`deny_high` + `gamma 0.99` cell killed at epoch ~21 once the teleport audit
above showed *why* the denial arm is null — the committing squad dies, and no
discount factor pays a corpse. Only
`configs/experiments/25v25_maps_deny_high.yaml` is kept, because it backs the
measured null; `git log -- configs/` restores the others.

## The denial arm is a null, and the reason was already in the repo

`contested_value` and `opponent_value` both raised to 1.0, two warm-start
lineages, two matched controls each, measured at the plateau (epochs 140-180 of
runs that reached 149 and 124):

| run | opp VP 90-130 | opp VP 140-180 |
|---|---|---|
| s1-high (**arm**) | 188.5 | 188.0 |
| s1-xB / s4-xB (controls) | 188.5 / 188.5 | 188.6 / 188.6 |
| s2-high (**arm**) | 189.1 | 189.4 |
| s2-xA / s3-xA (controls) | 189.4 / 188.8 | 188.4 / 188.9 |

Nothing, on either lineage, in either direction. The params were verified
present in the config the running process loaded, so this is a real null.

**And it was predictable from work already done here.** The audit behind
`248ac36` / `996614d` teleported a squad onto contested ground and paired 39
episodes properly:

| | control -> teleported |
|---|---|
| episode reward (scalar) | 20.94 -> 24.65 (+3.71) |
| **the moved squad's own income** | 83.03 -> **53.62** (**-29.41**) |
| **the moved squad's survivors** | 3.03 -> **1.33** |

The squad dies on a point holding **4.91 opponents**. **PPO trains on the
per-model reward vector, not the scalar**, so in the units the optimiser sees
the deviating squad *loses* 29.4 income. That is why the earlier local-optimum
verdict and its 12:1 reward-to-cost ratio were withdrawn.

**This is why raising the rate cannot work.** `contested_value` sets what a
*living* model earns for standing on contested ground, and
`phase_manager` iterates *alive* models — so the income is not lost to a low
rate, it is lost to casualties. No setting of that parameter pays a corpse.

**Nor is it a steering problem.** `closest_objective_v2._candidate_mask` already
filters to objectives where a model's arrival would improve control
(`_is_positive_transition` admits opponent -> contested and contested ->
player), so the potential term already pulls squads toward contested and
enemy-held ground. Only the group-to-objective assignment among those candidates
is by distance.

So all three of the obvious config-level levers are closed: the steering already
points the right way, the pay is already collected on arrival, and raising that
pay is a null because the arriving squad is dead.

**The obstacle is coordination, not pricing.** A *unilateral* squad commitment is
correctly punished, and the per-model gradient is right to refuse it. The bar
survives the same commitment because it commits everything at once and trades in
aggregate — 24% casualties for 4.00 objectives held. That is a large, discrete,
coordinated change of strategy, and it is not reachable by local perturbation
from a defensive optimum. **The agent is not mis-trained; it is in a good local
optimum that a gradient cannot leave.**

## Behaviour cloning crosses the gap, and proves the network was never the limit

Two facts measured here are only consistent if the agent sits in a *local*
optimum, and together they say exactly what to do:

- a **unilateral** commitment loses the deviating squad 29.4 income -> every
  gradient step toward committing is **downhill**;
- the **joint** committing policy earns **30.29** training reward an episode
  against the agent's **24.77**, ahead on *every* calculator -> the destination
  is **uphill**.

A gradient cannot cross that, because the path down and the destination up are
separated by a discrete distance. So don't walk there — *start* there.
`scripts/behaviour_clone.py` (`just behaviour-clone`) plays a scripted baseline,
records what it saw and what it did, and fits the policy network to it by masked
cross-entropy, writing a checkpoint `--warm-start-ckpt-path` accepts.

400 episodes, 12 epochs, **90.6% action match**. Scored on the nine held-out
tables, n=100:

| policy | vp_margin | +/- | player VP | opp VP | held | alive |
|---|---|---|---|---|---|---|
| `squad_march_shoot` (the BAR) | 111.8 | 5.3 | 272.2 | 160.4 | 4.00 | 0.762 |
| **the clone** | **109.7** | 4.2 | 271.4 | **161.6** | 3.84 | 0.735 |
| trained agent, 4 seeds | 82.3 | — | 263.9 | 181.6 | 3.24 | 0.954 |

**Within error of the bar on every column, and +27 vp above the RL agent.** On
table_05 — where the trained agent collapses to 40.5 with two thirds of its army
hidden — the clone scores **110.9** at `on_obj` 0.801. The collapse simply does
not happen.

On the nine matched **training** tables it is level too, and nominally ahead:

| policy | vp_margin | +/- | player VP | opp VP | held | alive |
|---|---|---|---|---|---|---|
| `squad_march_shoot` | 98.7 | 2.7 | 272.1 | 173.4 | 3.84 | 0.733 |
| **the clone** | **100.3** | 2.5 | 272.2 | **171.9** | 3.85 | 0.734 |

So the clone is the scripted bar, reproduced inside the network, on both map
sets and on every column — 90.6% action match is enough to carry the whole
policy, not merely most of it.

**So the network class, the observation and the action space were never the
limitation.** The same architecture, on the same inputs, plays at bar level when
it is *put* there. Everything measured above is a statement about the
optimisation, not the model.

### Clone fidelity is the bottleneck, and it is buyable

Cloning `squad_march_take` — the stronger but far more *reactive* target, which
recomputes its assignment every step from live opponent counts — showed what
sets a clone's ceiling:

| clone of `squad_march_take` | episodes | epochs | action match | held-out vp_margin |
|---|---|---|---|---|
| first attempt | 400 | 14 | 86.8% | **101.4** (-15.3 v its target) |
| larger | 700 | 35 | **94.7%** | **112.2** (-4.5 v its target) |

The target scores 116.7, so the gap is **compounding error**, and it closes as
action match rises. A fixed-assignment target (the bar, `k mod n`) loses only
2.1 vp at 90.6% match; a reactive one loses 15.3 at 86.8% and 4.5 at 94.7%.
**Reactive policies are harder to clone, and the fix is more data and more
epochs rather than anything clever.**

### Where this lands against the bar: a tie, not a win

| held-out, n=100 | vp_margin | +/- | player VP | opp VP | held |
|---|---|---|---|---|---|
| `squad_march_shoot` (the bar) | 111.8 | 5.3 | 272.2 | 160.4 | 4.00 |
| **clone of `squad_march_take`** | **112.2** | 4.9 | 272.5 | 160.3 | 3.73 |
| `squad_march_take` (the target) | **116.7** | 4.7 | 277.5 | 160.8 | 4.02 |

Paired per map against the bar, which is the powerful test: **+0.36 +/- 1.93,
t = 0.18, ahead on 5 of 9 maps.** That is a dead tie. The best *network* policy
now matches the bar and does not beat it — while the best *scripted* policy
beats the old bar by 4.9.

### Pushing fidelity further clears the bar

The two-point trend was worth one more test, so: 1200 episodes, 60 epochs.

| clone of `squad_march_take` | episodes | epochs | action match | held-out |
|---|---|---|---|---|
| first | 400 | 14 | 86.8% | 101.4 |
| larger | 700 | 35 | 94.7% | 112.2 |
| **XL** | **1200** | **60** | **98.3%** | **115.8** |

At 98.3% match the clone reproduces its target to within **0.9 vp** (115.8
against 116.7), and the compounding-error gap has closed from 15.3 to under one
point. **Fidelity was the whole ceiling, and it is buyable with data.**

Paired per map against the bar, n=100 on each side:

| | table_05 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| XL clone - bar | +0.8 | +5.8 | +11.2 | +1.3 | +6.2 | +2.6 | +13.3 | **-5.4** | +0.6 |

**+4.04 +/- 1.92, ahead on 8 of 9 maps.** Be precise about how strong that is:
t = 2.10 on 8 degrees of freedom is p ~ 0.07 two-tailed, which is marginal
alone. What makes it convincing is three independent lines agreeing — the sign
test on 8-of-9 gives p ~ 0.04, the clone's *target* beats the bar by +4.9 with
clean separation, and the clone tracks that target to within 0.9.

**So a network policy now beats the bar on the real tables**, and the best
network on this scenario has moved **82.3 -> 115.8** in a day. The route was not
a better reward: it was a better *scripted* policy (one inverted comparison,
`squad_march_take`) plus enough imitation data to reproduce it faithfully.

### Correction: the clone is a distribution, not a number

Everything above this line treats one checkpoint as *the* clone. It is not. The
cloning procedure was run **five times at identical settings** — same cached
1200-episode demonstration set, same 60 epochs, differing only in the torch seed
— and scored n=100 on the same nine held-out tables:

| clone | held-out vp_margin |
|---|---|
| A (the one quoted above) | **115.8** |
| B | 114.6 |
| C | 114.3 |
| D | 112.1 |
| E | 111.1 |
| | **mean 113.6, sd 1.9, SE 0.9** |

**115.8 was the maximum of five draws**, quoted as the result before the spread
was known. Three things above are weaker than stated:

- **The headline.** The procedure beats the bar by **+1.8 +/- 0.9**, not +4.04.
  Four of five clones clear 111.8; one lands 0.7 below it. "A cloned network
  beats the bar" survives as a statement about the mean, but it is a modest win,
  not a clean one.
- **The +4.04 +/- 1.92 pairing is still correct for clone A** and is not
  retracted — what was wrong was reading a single checkpoint's paired margin as
  the procedure's effect. The per-map rows were not kept for clones B-E, so the
  spread cannot be re-paired without re-scoring.
- **The fidelity ladder's last rung does not survive.** With sd 1.9 per clone,
  the 400 -> 700 episode step (101.4 -> 112.2, **+10.8**) is far outside the
  noise and stands. The 700 -> 1200 step is 112.2 -> **113.6** once the mean
  replaces the max: **+1.4 against a sd of 1.9**, i.e. not established. So
  fidelity bought the first jump; it is unproven that it bought the second. The
  measured *match rates* (86.8% / 94.7% / 98.3%) are unaffected — only what the
  last one is worth.
- **Target reproduction.** At 98.3% match the clone reproduces `squad_march_take`
  to within **3.1 vp** on average (113.6 against the target's 116.7), not the
  0.9 that clone A alone suggested.

**The lesson is the one this repo keeps relearning in new clothes.** A stochastic
procedure was run once, its output quoted as a property of the procedure, and the
figure that got published was the lucky one. The repo already says this about
*training* seeds and about layout seeds; behaviour cloning is a third instance,
and the tell was available cheaply — two early clones had already scored 115.8
and 111.1 at identical settings, which was noted as a caveat and then not
propagated into the headline.

**What is untouched by this.** `squad_march_take` itself (116.7) is a
deterministic scripted policy and carries no clone variance. And the
PPO-degradation result below compares against the *whole* range: every refined
arm scored 97.6-108.3, beneath even the worst clone (111.1), so its direction
does not depend on which clone it is measured against.

**What this does not claim.** This is not PPO beating the bar. The learning
algorithm's contribution to the final number is zero, and it is worse than that:

### PPO actively destroys the clone, and the rate scales with learning rate

Refining the 115.8 clone at `ent_coef 0.0`, measured at epoch 124, n=100
held-out:

| | the clone | lr 1e-5 | lr 5e-5 |
|---|---|---|---|
| vp_margin | **115.8** | 106.3 | 103.2 |
| fraction alive | 0.673 | 0.857 | **0.887** |
| opp VP | **158.0** | 170.2 | 174.1 |

A clean dose-response: the gentler learning rate degrades less, and both slide
**monotonically toward conservatism** — straight back to the original agent's
signature (alive 0.954, opp VP 181.6). PPO is not failing to improve the clone.
It is dragging it home.

### The discounting, not the reward, is what disagrees

This looks like it contradicts the income measurement above — the bar earns
**30.29** training reward an episode against the agent's 24.77, so how can PPO
prefer the agent's behaviour?

Because those are different quantities. **30.29 is the undiscounted episode
sum. PPO optimises the discounted per-model return**, and at `gamma: 0.9` with
`gae_lambda: 0.95` the effective window is **6.9 steps of a 40-step episode**.
Committing costs income now and repays over the remaining rounds; discounted at
0.9 a step, a payoff 30 steps out is worth 4% of face value. **Under PPO's
actual objective the conservative policy really is better**, even though it is
worse over the episode and worse on `vp_margin`.

That suggested the discount factor, not the reward, was the wedge — that
`gamma` decides which policy basin PPO regards as optimal. **It was tested
directly and it is WRONG.** Same clone, same two learning rates, only `gamma`
changed, all measured at epoch 124:

| gamma | lr | vp_margin | alive | opp VP |
|---|---|---|---|---|
| — (the clone) | — | **115.8** | 0.673 | 158.0 |
| 0.9 | 1e-5 | 106.3 | 0.857 | 170.2 |
| **0.99** | 1e-5 | **103.5** | 0.823 | 172.1 |
| 0.9 | 5e-5 | 103.2 | 0.887 | 174.1 |
| **0.99** | 5e-5 | **98.2** | 0.913 | 177.9 |

**Raising gamma made it worse at both learning rates**, and the damage scales
with **learning rate** rather than with gamma.

**⚠ But that test was under-specified, and the sentence that followed it here
was arithmetically wrong.** It read "`gamma 0.99` gives a ~100-step horizon on a
40-step episode, i.e. effectively undiscounted". It does not. **`gae_lambda`
caps the advantage window whatever gamma does** — the window is
`1/(1 - gamma*lambda)`:

| gamma | lambda | window |
|---|---|---|
| 0.9 | 0.95 | **6.9 steps** |
| 0.99 | 0.95 | **16.8 steps** |
| 0.99 | 0.99 | 50.3 steps |

Episodes are 40 steps. So that arm moved the horizon from 7 steps to 17 — still
under half an episode — and never came near undiscounted. **The horizon
hypothesis is therefore NOT refuted; it was tested with the wrong knob.**
`--gae-lambda` is now exposed for that reason.

### The horizon, properly specified: it is a null. Only the learning rate moves it

`gamma 0.99` **with** `gae_lambda 0.99` — a 50.3-step window, longer than the
40-step episode, so a payoff at the far end of the game is finally inside what
the advantage estimate can see. Four arms, two seeds per learning rate, from the
critic-fitted clone at `ent_coef 0.0`. Scored at epoch 125 and again at a matched
epoch 150, n=100 held-out:

| arm | window | lr | e125 | e150 | alive @150 | opp VP @150 |
|---|---|---|---|---|---|---|
| the clone | — | — | 111.1-115.8 | — | 0.673 | 158.0 |
| s1-h5 | 50 | 5e-5 | 100.3 | 97.6 | 0.917 | 178.0 |
| s2-h5 | 50 | 5e-5 | — | 99.9 | 0.908 | 176.4 |
| s3-h1 | 50 | 1e-5 | 108.3 | 107.4 | 0.828 | 168.5 |
| s4-h1 | 50 | 1e-5 | — | 105.7 | 0.842 | 170.1 |
| *reference:* 7-step | 7 | 1e-5 | 106.3 | — | 0.857 | 170.2 |
| *reference:* 7-step | 7 | 5e-5 | 103.2 | — | 0.887 | 174.1 |

**The horizon is a null. The learning rate is the whole effect.**

- **lr 1e-5**: 50-step window gives 107.4 / 105.7, mean **106.6**. The 7-step
  window at the same learning rate gave **106.3**. No difference.
- **lr 5e-5**: 50-step gives 97.6 / 99.9 against the 7-step's 103.2 — if
  anything slightly worse.
- The learning rate separates cleanly and reproducibly: **~106.6 against ~98.8**,
  with both seeds tight inside each pair.

**A window longer than the entire game does not stop the decay.** Every arm sits
below the *worst* clone (111.1), and each is still sliding at epoch 150.

**⚠ This section first read "at the slow learning rate the horizon genuinely
helps, and helps along the mechanism", built on s3-h1's 108.3 at epoch 125 —
the best refinement then measured, with an `alive`/opp-VP signature sitting
neatly between the clone and the basin.** The second seed at that learning rate
came back at **105.7**, making the pair 107.4 / 105.7 with a spread of 1.7 — wide
enough to swallow the +2.0 over the 7-step arm that the whole reading rested on.
One seed, one epoch, and a mechanism story that fit: the same shape of mistake as
the +4.04 clone and the λ arithmetic, caught this time only because the
replication was already armed before the first number arrived. **Arm the second
seed before reading the first.**

### Where this was stopped, and the one candidate that was never tested

This line of work was closed here deliberately. The goal — beat the bar on the
real tables — was already met by `squad_march_take` and its clone; everything in
this section is upside on top of a met goal, and it had reached the point of
turning knobs against a stable negative.

**The negative is worth stating cleanly, because it is well-supported.** Across
**cold critic, a critic fitted to 0.976 explained variance, gamma 0.9, gamma
0.99, a 7-step window, a 17-step window and a 50-step window, at two learning
rates and eight training runs**, PPO refinement of a good clone landed at
**97.6-108.3** — always below the clone, always by the same route (`alive` up,
opponent VP up), always converging on the independently-trained agent's own
signature. That basin is robust to everything tried.

**Only one variable moved the number at all, and it was the learning rate** —
which is a statement about how *fast* the policy leaves the clone, not about
what it is being pulled toward. Every knob that reached the *objective* (gamma,
lambda) or the *gradient's quality* (critic) measured null. That is the shape of
a result where the destination is set by something none of these knobs touch.

**The untested candidate, named so the next person starts there.** No arm ever
made *drifting itself* costly. Every knob touched the speed of the drift
(learning rate, horizon) or the quality of the gradient (critic); none added a
penalty for moving away from the clone's decisions — the standard trust-region /
KL-anchor fix for exactly this failure. That is the first thing to try.

**And a prior question that should be answered before any of it**, because it
may dissolve the problem: **no reward term pays for denial.** Every point of
`vp_margin` on this scenario above the saturated own-score comes from taking VP
off the opponent, and the reward pays for holding ground, not for denying it.
It is entirely possible PPO is optimising correctly and the objective it is
given genuinely ranks the cautious basin first — in which case no amount of
optimiser tuning is the fix, and the repo's own rule applies: *check the agent
is paid for what you are measuring.* Scoring the clone and a degraded checkpoint
under the training reward itself is a desk-check, costs no GPU, and would settle
it.

### The critic was the next suspect, and it is not that either

The clone supplies the *policy* and nothing else, so the first updates carry a
randomly initialised value function and the advantages driving them are noise —
which predicts damage proportional to step size. `behaviour_clone` was extended
to fit the critic too (per-model discounted returns, MSE, reported as explained
variance) and reached **0.976**, better than the 0.86-0.90 PPO reaches on its
own. Refined at the identical epoch 124:

| | clone -> after PPO | alive | opp VP |
|---|---|---|---|
| cold critic, lr 5e-5 | 115.8 -> **103.2** | 0.887 | 174.1 |
| **warm critic (EV 0.976)**, lr 5e-5 | 111.1 -> **104.0** | 0.882 | 172.9 |

**The same destination**, with near-identical `alive` and opponent VP. A good
critic does not prevent it.

**So the attractor is robust**: across cold and warm critics and both gammas
tried, PPO converges on `alive` 0.88-0.91 and opponent VP 173-178 — the trained
agent's own signature. Whatever selects that basin survives fixing the critic,
and the one hypothesis not yet properly tested is the horizon, for the lambda
reason above.

Warm-starting PPO from the clone is running at the time of writing (two seeds at
the default learning rate, two at 1e-4 — the clone supplies the policy but not
the critic, so the first updates carry a random value function and could undo
it). The prediction that makes this worth running: because the committing policy
scores *higher* on the training reward, PPO has no incentive to drift back.

**One hazard closed on the way.** `_apply_warm_start_weights` loads with
`strict=False`, so a wrong key prefix loads **nothing** and trains a random
network while reporting a warm start. The cloner writes
`ppo_model.policy_network.*`, verifies the overlap before exiting, and
`tests/test_behaviour_clone_checkpoint.py` pins both the prefix and that loading
actually changes the weights.

## What is worth doing next, and what is not

**Worth doing.** Stratify the map split by own-zone objective count and
re-measure transfer. Investigate table_05 directly with `just debug-recording`
— a map where two thirds of the army ends hidden with exposure 0.001 should be
watchable. Consider whether `objective_hold` should price *marginal* control
rather than presence: the fourth model on a point we already hold contributes
nothing, and `crowding_exponent` only partly expresses that.

**Not worth doing.** `map_pool.mirror` **was measured and is a null.** Four
checkpoints trained on 2026-08-13 — two with `mirror: true`, two without,
matched pairwise on warm start and epoch (1000/1000) — had never been scored
against each other. Paired per map on the nine held-out tables, n=100:

| lineage | mirror - plain | maps ahead | excluding table_05 |
|---|---|---|---|
| s1 | +6.3 +/- 10.4 | 4/9 | **-3.7** |
| s2 | +1.8 +/- 3.5 | 4/9 | **+3.7** |

Ahead on four of nine maps in both lineages, error bars spanning zero, and the
sign **flips** once the outlier map is removed. Read as a mean it looked like
+6.3; read per map it is one table moving a nine-table average — the same error
this report warns about two sections above, and one I made in-flight before the
rows were in.

That is consistent with the mechanism: `flip_x` swaps the player's and
opponent's zones, so a 1/4/1 map stays 1/4/1 and a 2/2/2 stays 2/2/2. **The
augmentation changes orientation, never the mission mix** — which is the axis
that predicts the agent's deficit. More *real* tables would change it; four
reflections of the existing ones do not.

**One real finding inside the null.** The single map where mirror wins hugely is
table_05 (+85.9 on s1), the collapse map — a differently-trained checkpoint
simply does not collapse there. So the collapse is **fragile and
training-dependent, not a property of the layout**: something a run falls into,
not something the map forces. That makes it a bug worth fixing rather than a
hard table.

And do not
raise `vp_per_objective` to "uncap" the mission: the cap is the rule
(`primary_cap_per_round: 15`), it would void every baseline, and it helps the
bar more than the agent, since the bar currently discards ~25% of its own VP to
the cap against the agent's ~9%.

`squad_march_deny` is deliberately **not** added to `BASELINE_POLICIES`. It ties
`squad_march_shoot` rather than beating it, so it adds no ranking power to the
in-run eval, and every entry there is paid for in wall-clock every epoch.

## Reusable lessons

- **Check whether the scoring function saturates before shaping toward "more".**
  Three rounds of work here have been aimed at raising `held`, a metric the
  mission stops paying for at three.
- **Print the two halves of a difference.** `player_vp` and `opponent_vp` were
  tracked all along and thrown away by the formatter; the entire diagnosis is
  invisible in `vp_margin` alone.
- **Read the per-map rows.** The mean said "27 points behind"; the rows said
  "level on seven maps, catastrophic on two", which is a different bug.
- **A convenient holdout is not a stratified one.** Check that a split matches
  on the dimension that predicts difficulty before calling the difference
  generalisation.
