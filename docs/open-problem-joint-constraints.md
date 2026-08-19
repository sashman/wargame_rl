# Open problem: an RL agent that obeys a joint spacing constraint

**Status: unsolved. Written 2026-08-18 for someone with RL experience and no
knowledge of this project.** It states the setting from scratch, the exact
difficulty, everything already measured, and everything already refuted, so that
suggestions do not repeat work.

---

## 1. The setting

A turn-based tabletop-style wargame, reimplemented as a Gymnasium environment.
Two armies of **25 models** each face off on a continuous **60 × 44** board
(units are inches; positions are real-valued, not grid cells).

- Each army is organised into **5 units of 5 models**. Unit membership is fixed
  for the episode.
- A game lasts **20 battle rounds**. Each round has a **movement** phase and a
  **shooting** phase for each side.
- The board carries **5 or 6 objective markers** and 15–16 pieces of terrain
  (blocking line of sight). Layouts are drawn per episode from a pool of 45
  hand-made tables; 36 are used for training and **9 are held out** for scoring.
- **Scoring:** each turn a side gains `min(15, objectives_controlled × 5)`
  victory points. You *control* an objective when you have strictly more models
  than the enemy within 3" of it.
- The headline metric is **`vp_margin` = own VP − opponent VP** at episode end.

Two consequences of the VP cap that matter for reading any number below:

- Own score **saturates at three objectives**, so anything competent scores
  ~230–275 of a 285 ceiling. All remaining margin comes from **denying** the
  opponent, not from scoring more.
- Because of that, "objectives held" stops discriminating between good policies.

**The opponent** is a fixed scripted policy. The strongest available (call it
`take`) advances each unit toward an objective along **one shared vector for the
whole unit**, and shoots. Against it the entire scripted ladder is at or below
zero, so there is genuine headroom.

## 2. The learning setup

- **PPO**, actor-critic with GAE, clipped surrogate.
- **Network:** a transformer encoder over entity tokens (own models, enemy
  models, objectives, terrain), ~6.5M parameters. One shared trunk; per-model
  action logits.
- **Action space: one discrete action per model, sampled independently.**
  97 actions = 1 "stay" + **16 angle bins × 6 speed bins**. Speeds are 1"–6".
  A model's move is applied exactly (continuous board). All 25 actions are
  emitted from one forward pass and sampled independently.
- **Action masking** exists and is applied inside the network, at both rollout
  and update, so log-probs are consistent.
- **Reward** is a weighted sum of calculators, some **per-model** (each model
  gets its own scalar; PPO consumes the per-model vector) and some **global**
  (broadcast identically to every model). The main ones: potential-based
  progress toward the nearest objective, a per-model payment for standing on an
  objective scaled by control state, per-model kill credit, a formation penalty,
  and a global net-VP term.

## 3. The constraint that is the problem

**Unit coherency.** After every move, each unit of more than one live model must
satisfy all three of:

1. **Chain** — every model within **2"** of at least one *other* model in its unit.
2. **Spread** — every model within **9"** of *every* other model in its unit.
3. **Connectivity** — the graph whose edges are the chain condition must be a
   single connected component. (Not implied by 1 and 2: two pairs 5" apart
   satisfy both and are still not one unit.)

Distances are **base to base** — each model is a disc of radius 0.63", so the
2" chain is 3.26" centre-to-centre.

This is a **joint constraint over the 5 models of a unit**, evaluated on the
configuration *after* all of them have moved.

### Enforcement, and why it is hostile to learning

The game's own rule is: if the move would leave a unit out of coherency, **the
move cannot be made — every model in that unit returns to where it started.**
Implemented as `revert_unit`.

This is a **projection applied after action selection**, and it has the
pathology the safe-RL literature calls **action aliasing**: many distinct joint
actions map to the identical outcome (everyone back at the start), so they share
a return and an advantage, and **the policy gradient inside that whole set is
exactly zero**. Only the entropy bonus acts there. Measured: a policy *trained*
under enforcement ends up **worse** at formation than one never exposed to it
(0.569 units coherent v 0.756–0.886), and loses on held-out tables too.

So the working arrangement is: **train without enforcement**, using a reward
gate (a model outside its unit's coherent body earns no objective income), and
**switch enforcement on at play time**. The question is whether the resulting
policy is actually legal, and how much the rule costs it.

---

## 4. What is actually wrong

### 4.1 The referee cancels a third of all movement

Trained agent under `revert_unit`, instrumented by capturing
(start, intended, final) positions for every movement phase:

| | |
|---|---|
| unit-moves cancelled outright | **33.1%** (scripted policies: 12.1%) |
| **intended movement inches destroyed** | **48.9%** |
| unit-episodes freezing at least once | 91.5% |
| cost | **−25 to −40 vp** (scripted policies pay −4.3) |

**Freezing is an absorbing state.** `P(frozen next | frozen now) = 0.62` against
0.17 after a successful move — because a revert reproduces the same decision:
the policy sees the same board next turn and re-issues the same illegal move.
A *deterministic* scripted policy hard-deadlocks: on one seed it requests the
identical move on all twenty rounds and ends exactly where it deployed.

A revert **cannot repair, only refuse**. A unit split by casualties is already
incoherent when its move begins, so reverting returns it to the split.

### 4.2 The mechanism: a small per-model tail, amplified

**The 2" chain binds; the 9" spread does not.** Median gap to a model's nearest
squadmate is **0.09"**; only **7.8%** exceed 2". Spread breaches are 3–5%.

On a five-model unit:

```
1 − 0.922^5 = 0.32        measured unit freeze rate = 0.331
```

An all-or-nothing revert converts a **7.8% per-model tail into a 33% unit veto**.
Equivalently, unit compliance ≈ `p^k` in per-model compliance `p` for a
`k`-model unit. The observed plateau of ~0.89 unit coherency implies
`p = 0.977`; getting all five units to 0.98 simultaneously needs
**`p = 0.99919`** on a 97-way categorical, every turn, for 25 models.

### 4.3 The behavioural signature

The agent's formation is a **pile plus stragglers**: median gap to nearest
squadmate **0.00"** (models literally base to base) while 5.3% sit beyond 2".
Scripted policies hold an even ~0.4" spacing with a 3.4% tail. Same order of
tail, completely different shape.

And the sharpest anomaly: **the agent stands still on 0.4% of unit-moves; the
scripted policies on 38–57%.** Standing still is *trivially* coherency-legal —
positions do not change, so coherency cannot break — so the scripts collect half
their moves legal for free, while **98.8%** of the agent's cancelled moves were
moves it actually wanted (scripts: 24–25%).

This is why **the coherency rate does not predict the tax**: two policies at
0.790 and 0.800 intended coherency pay **28 vp** and **~0** respectively. A
referee that cancels a move you were not going to make is free.

Reading the policy's own distribution (n=2149 movement decisions):

```
mean policy entropy   3.545 nats of ln(97)=4.575     effective ~35 of 97 actions
median P(stay)        0.0101  ==  exactly uniform, 1/97
mean P(top action)    0.131   -- its best action holds 13% of the mass
```

The mass genuinely is not there; the policy has learned nothing about when to
hold position.

### 4.4 Compliance *degrades* with training

Within every seed measured, unit coherency is **lower at the end of training
than in the middle** (paired, same checkpoint directory, n=10 per point):

| seed | earlier checkpoint | epoch 300 |
|---|---|---|
| A | 0.700 (epoch 145) | **0.659** |
| B | 0.721 (epoch 83) | **0.561** |
| C | 0.844 (epoch 123) | **0.827** |

So the reward gate is not merely *failing to teach* formation — whatever
pressure the rest of the objective applies is actively **eroding** it as the
policy sharpens. This kills "train longer" as a remedy (tested and abandoned on
the strength of this measurement) and it reframes the problem: the question is
not only how to *reach* compliance but why the optimisation **walks away from
it**.

A plausible reading, unverified: the gate withholds objective income from a
model outside its unit's coherent body, but every other term — approach
progress, kill credit, the global net-VP term — keeps paying, and the marginal
value of one more model pushed forward apparently exceeds the withheld income.
If so, the gate is a tax the policy is willing to pay, not a constraint.

---

## 5. Where things stand numerically

Nine **held-out** tables, n=30 episodes per table, error bars across the nine
maps (a map is the unit this generalises over). `free` = no enforcement, which
is where the policy's own intended coherency is readable; `refereed` = the rule
enforced, which is the legality claim.

| policy | free vp | free coherency | refereed vp |
|---|---|---|---|
| do-nothing floor | — | — | −197.3 |
| scripted `march` | −165.5 | 0.772 | −149.7 |
| scripted `march+shoot` | −12.4 | 0.774 | −36.7 |
| **PPO, 3 seeds, best arm** | **+0.2 ± 9.8** | **0.771 ± 0.060** | **−28.9 ± 11.0** |
| **PPO, 3 seeds, other arm** | +5.2 ± 4.2 | 0.674 ± 0.104 | −34.8 ± 13.3 |
| scripted `take` (= the opponent) | +0.2 | 0.806 | −6.4 |
| scripted `deny` | −6.6 | 0.810 | −4.4 |

**Read this carefully:**

- The agent is roughly **level with the scripts without the rule**, and **~25 vp
  behind the best script with it**. The entire gap is the enforcement tax.
- Its intended coherency (0.771) is at the **bottom edge** of the scripted band
  (0.772–0.810) — it is about as compliant as the scripts, and still pays far
  more for it, because of the stay-rate asymmetry in §4.3.
- **The seed spread is larger than any intervention measured.** Three
  from-scratch seeds of the *same* config span **26.0 vp** refereed and
  **0.202** unit coherency, against lever effects of ~6 vp and ~0.10. A single
  seed has inverted the ranking of two arms.

---

## 6. What has been tried, and refuted

Please do not propose these without new evidence. Each was measured here.

| idea | result |
|---|---|
| **Train under enforcement** | Worse formation than never training under it (0.569 v 0.756–0.886) and worse held-out score. Zero gradient inside the projected set. |
| **Revert only the offending models** instead of the whole unit | Ties with full revert on score; the spread condition is collective, so once one model is out, all are, and the two modes coincide. |
| **Clamp** — shorten the move along its own segment until legal | Same ~26 vp cost as the others; cannot fix a pure spread breach, silently degrading to a full revert. Removed. |
| **Unit-level action space** (one shared vector per unit) | The strongest scripted policy *already is one* and reaches only 0.915 coherent. Forcing a trained agent's moves rigid scored **0.444** — rigid translation *preserves* coherency but cannot *restore* it, and casualties split units constantly. |
| **Smaller units** (8 units of 3, so `p^3` not `p^5`) | Cancelled exactly: fewer squadmates means your nearest one is further, so the per-model tail rises 2.36% → 4.02% and predicted freeze is unchanged (+0.003). The apparent gain was a casualty artefact — a unit reduced to one model is coherent by definition. |
| **Add the direction to the unit centroid as an observation** | Worst arm of four: −62.1 refereed. Early coherency lead evaporates by epoch 300. |
| **Lower the entropy bonus** (0.03 → 0.003) | Concentrates the policy exactly as predicted (entropy 3.545 → 1.893, top action 0.131 → 0.488) but makes "stay" **30× rarer**, not commoner. Marginally the better arm on 3 seeds, but does not touch the tax. |
| **Raise the reward for contesting enemy-held ground** | Null: the committing unit dies, and the calculator pays only living models. |
| **Train longer** | Coherency *falls* within every seed measured — see §4.4. More epochs make formation worse, so this is closed. |
| **Autoregressive decoding for a different joint constraint** in this codebase ("every model picks a different shooting target") | Built, PPO-correct, test-pinned; worth +10.5 vp on one of four checkpoints and −0.2/−0.4/−1.5 on the others, then −3.9 vp when trained under, and 7% slower. Removed. |

Also known: a **coherency rate rises whenever an army dies**, since a unit
reduced to one live model is coherent by definition. Any intervention that
increases casualties will *look* like a formation improvement. Always read the
per-model tail, which is invariant to unit size, beside the unit rate.

---

## 7. The question

> **How do you get a policy that emits per-entity actions to satisfy a joint,
> combinatorial, geometric constraint over groups of those entities — well
> enough that a hard post-hoc projection almost never fires — without destroying
> the policy's ability to play the game?**

Sub-questions that would each be useful on their own:

1. **Is the action space the right place to intervene at all?** The unit-level
   space is refuted as stated, but a *hybrid* — a shared unit vector plus a
   bounded per-model offset, with the offset bound chosen so pairwise distances
   can shift by at most the slack — is untried here. The open worry is that
   casualty-split units still cannot re-form under it.

2. **Can the constraint be made differentiable or shaped rather than projected?**
   A penalty proportional to the *distance from the feasible set* (how far the
   intended configuration was from legal) would give a gradient where the revert
   gives none. Nobody has tried it here.

3. **Is the real defect that the policy never holds position?** It stands still
   0.4% of the time against the scripts' 38–57%, and standing still is free
   legality. Nothing in the reward pays for holding ground once taken: the
   progress term is potential-based, so advancing pays and holding pays nothing.
   **The trap:** rewarding stillness as such buys the do-nothing floor at −197.3.
   The term would have to pay for holding ground the side *controls*.

4. **How do you train through variance this large?** Three seeds span 26 vp and
   0.20 coherency. Any lever worth ~6 vp is unmeasurable without many seeds,
   which makes iteration expensive. Is there a lower-variance formulation of the
   objective, or a better estimator, or should the target be the per-model tail
   (a much lower-variance quantity) rather than the unit rate?

5. **Should coherency be a constraint on the policy at all, or a property of the
   parameterisation?** Formation-control literature (leader–follower, virtual
   structures, formation slots) makes the constraint hold by construction. The
   cost is manoeuvre expressiveness, and this project's own evidence is that a
   purely rigid formation loses to a per-model one that breaks the rule.

## 8. Constraints on any proposal

- **RL only.** Behaviour cloning of the scripted policies has been tried and is
  explicitly not the direction wanted, even though it scores well.
- **Changing the action space is expensive** but possible; changing the network
  head is a 1–2 week job that invalidates every existing checkpoint.
- **Adding an observation column is cheap** (one field, one line) but widens the
  per-model token, so existing checkpoints cannot warm-start — a loud, deliberate
  load failure.
- **Adding a reward calculator is cheap** (a class plus a registry entry), but
  the project's history is full of shaping terms that reduced total income for
  the behaviour they were meant to encourage and so suppressed it. Any new term
  should be checked for: is it **per-model and differentiated** (does its value
  actually differ across the choice the model is making)? Does the behaviour it
  wants pay **more in total** than the behaviour it replaces? Can the agent
  **observe** what it keys on?
- **A 300-epoch run is ~2.5 hours** on one GPU; four fit concurrently. Three
  seeds minimum before any result is readable.
