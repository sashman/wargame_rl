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
refuse — even though the whole army advancing together scores better. And the
training setup pays a bonus for keeping decisions *uncertain*, which held the
agent's play permanently vague; that is why changing what the reward pays for
never moved anything.

**What fixed it.** Not a better reward. A better *hand-written* policy — send
spare squads to the weakest ground rather than attacking the strongest, which is
one line — and then teaching the network to copy it. Copied faithfully enough
(98% of decisions matched), the network scores **115.8** against the benchmark's
**111.8**, ahead on 8 of the 9 unseen maps.

The best network policy on this scenario went from **82.3 to 115.8** in a day.
Worth being clear about what did the work: the copying, not the learning. The
reinforcement-learning step contributed nothing measurable on top.

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
  gives a GAE horizon of **6.9 steps**, so an advance that repays over the
  remaining twelve rounds is discounted to ~4% of face value. Hiding with an
  intact army is what that horizon rewards. `CLAUDE.md` records that the case
  for 0.9 was measured on a different scenario and its refutation retracted.

Two other arms were launched and killed before they answered anything, and are
recorded here so the compute is accounted for: a `deny_mid` (0.75 / 0.5) killed
at epoch ~40 because halving a 4x pay cut still leaves a 2x one, and a combined
`deny_high` + `gamma 0.99` cell killed at epoch ~21 once the teleport audit
above showed *why* the denial arm is null — the advancing squad dies, and no
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

**The obstacle is coordination, not pricing.** A *unilateral* squad advance is
correctly punished, and the per-model gradient is right to refuse it. The bar
survives the same advance because it commits everything at once and trades in
aggregate — 24% casualties for 4.00 objectives held. That is a large, discrete,
coordinated change of strategy, and it is not reachable by local perturbation
from a defensive optimum. **The agent is not mis-trained; it is in a good local
optimum that a gradient cannot leave.**

## Behaviour cloning crosses the gap, and proves the network was never the limit

Two facts measured here are only consistent if the agent sits in a *local*
optimum, and together they say exactly what to do:

- a **unilateral** advance loses the deviating squad 29.4 income -> every
  gradient step toward advancing is **downhill**;
- the **joint** advancing policy earns **30.29** training reward an episode
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

**What this does not claim.** This is not PPO beating the bar. Refinement from
the clone measured neutral-to-slightly-negative at every setting tried
(108.6 against a 109.7 clone at epoch 74; the low-entropy arms hold but do not
climb). The learning algorithm's contribution to the final number is, so far,
zero — and saying otherwise would be the same overreach this report spends four
sections correcting.

Warm-starting PPO from the clone is running at the time of writing (two seeds at
the default learning rate, two at 1e-4 — the clone supplies the policy but not
the critic, so the first updates carry a random value function and could undo
it). The prediction that makes this worth running: because the advancing policy
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
