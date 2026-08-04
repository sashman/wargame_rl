# 2026-08-04 — how the reward calculation changed

## TL;DR

Imagine coaching 25 players but only ever shouting one score at the whole team. Nobody
learns what *they* did right. That was the reward: 25 models, one number, averaged.

Five things changed:

1. **Everyone gets their own score now.** Each model is paid for what it did; team-wide
   things (like winning) are still shouted to everyone equally.
2. **Added "hold the point you're standing on."** Every other reward went quiet once
   models stopped moving, so for most of the game all 25 got identical scores again.
3. **Added "you personally shot someone."** Shooting is half the decisions and previously
   paid every model the same whether it fired, missed, or napped.
4. **Deleted rewards everyone already maxed out.** "Models on objectives" scored a perfect
   1.000 for good *and* mediocre policies, so it taught nothing — and it was 83% of the
   early reward.
5. **Fixed a genuine bug: the agent was paid when its own models died.** The code read
   "models killed by the opponent" where it meant "killed by us", and double-counted on top.

**Result:** win rate went from 17% to 93–97%, against a hand-written baseline that scores
67%. **But** these changes shipped together with a batch of PPO fixes, so we cannot say
which one earned it.

**Still wrong:** the agent leaves ~1 model per episode stranded away from its squad, gives
up some ground to chase kills, and — oddly — wiping out the enemy *ends the game early and
costs it points*.

---

**Question:** the 25v25 agent was being paid a single number for the joint behaviour of 25
models, and most of what it was paid for did not distinguish a good policy from a mediocre
one. What changed in the reward calculation, and what does the evidence support?

**Scope.** This report covers the reward side only: `envs/reward/` and the phase blocks of
the env configs. The PPO-side plumbing that consumes it (per-model ratios, per-model value
head, entropy meaning) is in
[mechanism defects](2026-08-04-mechanism-defects.md) and
[the correction](2026-08-04-correction-what-was-actually-broken.md). **They shipped
together**, which is the central confound — see [Confounds](#confounds).

---

## 1. The reward became a vector, not a scalar

`RewardPhaseManager.calculate_reward` previously reduced everything to one float:

```
reward = mean_over_alive_models(per_model_terms) + sum(global_terms)
```

Averaging 25 models into one number leaves each model's action explaining roughly 4% of the
signal it is credited with. Every model was told "the army did well" regardless of whether
it advanced, stood still, or fired into an empty sky.

The manager now also records `last_per_model_reward`, shape `(n_player_models,)`:

- a **per-model** term contributes its own value to its own model
  (`per_model_rewards[i] += contribution`)
- a **global** term is broadcast *whole* to every alive model, not divided among them —
  these are the part of the outcome genuinely not attributable to one model, so every model
  should see the same signal. Dividing by N would shrink the goal term as the army grows
- terminal bonuses (`terminal_success_bonus`, `terminal_vp_bonus`) are treated as global

The scalar return value is **unchanged**, so anything reading the old number still works;
PPO reads the vector (`ppo/lightning.py:661`, consumed at `:290`).

Relative balance between terms is preserved: a per-model term contributes its own value to
one model where it previously contributed `1/N` of the army's sum to the shared scalar, and
a global term contributes its full value in both. An earlier claim that per-model terms
became "25× louder" was checked against the code and is wrong.

### `CurriculumPosition`

Phase progress moved out of the manager into a shared `CurriculumPosition` (`index`,
`epoch_entered`, `consecutive_epochs_above_threshold`). Each env needs its *own* calculators
— they carry per-episode state (`closest_objective`'s previous distance,
`objective_flip_bonus`'s potential) that one env resetting would corrupt for the others —
but all envs must reward the phase the curriculum has actually reached. Sharing the position
object means there is no synchronisation step for a future code path to forget, which is
exactly how the rollout envs came to train on phase 0 for every run to date while
`reward_phase` reported otherwise.

---

## 2. Two new per-model calculators

The credit-assignment change was largely inert without these. Only **3 of 8** calculators
were per-model, and both useful ones go silent once models arrive:

```
after arrival (~step 10 of 40):
  closest_objective progress -> 0   (potential exhausted; exactly 0 on shooting steps)
  group_cohesion             -> 0   (hard 0 inside the limit)
  => all 25 advantages identical for ~30 of 40 steps
```

**`objective_hold`** (per-model) — value keyed on the control state of the objective the
model occupies, reusing the existing `objective_states_from_norms_offset`, which already
implements the OC/count rule VP scoring uses:

| state of the objective the model is in | value |
|---|---|
| player-controlled | 1.0 |
| contested | 0.5 |
| opponent-held | 0.25 |
| not on any objective | 0.0 |

The only term that pays a model while stationary. It pays for *controlling* rather than
merely standing, and supplies a continuous gradient across `neutral → contested → player` —
the tie-break region both `objective_coverage` and `vp_gain` are flat across, since control
is a strict count comparison.

**`model_kills`** (per-model) — `bonus_per_kill × kills made by this model this step`.
Shooting is half the agent's decisions and previously had **no credit path**: the global
`killing` term paid every model identically whether it fired, missed, or stood still.
Required threading attribution that already existed but was discarded —
`StepContext.player_kills_by_model`, shape `(n_player_models,)`, populated from
`PairedShootingResult.attacker_idx`.

Measured: within-step across-model reward spread is **2.3× higher** under the new config
(sd 0.319 vs 0.103).

---

## 3. A bug fixed in the global `killing` term

Two defects in one four-line calculator:

```python
# before
diff = ctx.opponent_models_killed - self._previous_opponent_models_killed
self._previous_opponent_models_killed = ctx.opponent_models_killed
return self.bonus_killing_opponent * diff
```

1. **It read the wrong field.** `opponent_models_killed` means "models the opponent killed"
   — the agent's *own* losses. The `player_` prefix means "by the player", matching
   `player_damage_dealt`. The agent was being paid for being shot.
2. **It telescoped.** The context field is already a per-step count, so subtracting a
   running total made the reward accumulate to the final step and go *negative* on the step
   after any kill.

Now `return self.bonus_killing_opponent * float(ctx.player_models_killed)`. This is
arithmetic, not measurement — it holds regardless of any run.

---

## 4. Three terms dropped from the configs

| dropped | why |
|---|---|
| `models_at_objectives` | **Zero discriminative power.** All three competent scripted baselines saturate it at 1.000 — the 0.57-win policy and the 0.77-win policy score identically. It is a don't-be-random detector, and it was 83% of the old phase-0 reward budget. `objective_hold` strictly dominates it |
| `objective_flip_bonus` | Superseded by `objective_hold`'s continuous control gradient |
| `killing` (global) | Superseded by `model_kills`. Its default bonus is **5.0** paid to *all* N models per kill, so a 25-model wipe was worth 5 × 25 = 125 per model — larger than every other term combined, against an opponent that cannot shoot back |

The calculators remain registered and tested; only the configs stopped using them.

---

## 5. Config restructuring

The old ladder had four rungs, each stack-passable, redundant, or inert (a final phase's
`success_threshold` is never read, because `try_advance` returns at `is_final_phase`). Two
configs replace it, sharing a scenario block and an identical final phase so that comparing
them isolates the curriculum:

**`25v25_single_phase.yaml`** — one phase, `player_ahead_on_vp`:

| term | kind | weight | ≈ episode integral |
|---|---|---|---|
| `closest_objective_v2` | per-model | 1.0 | ≤ 8 |
| `objective_hold` | per-model | 0.25 | +7.5 while holding |
| `model_kills` | per-model | 1.0 (`bonus_per_kill` 2.0) | varies |
| `group_cohesion` | per-model | 0.3 (`group_max_distance` 6.0, `violation_penalty` −0.05) | ≤ 0 |
| `vp_gain` | global | 2.0 | ≈ +6 |
| `objective_coverage` | global | 0.3 | ≈ +6 |

Per-model episode integral ≈ 23, and no term exceeds ~8. That ceiling is deliberate: the
whole gap between a 0.57-win policy and a 0.77-win policy is worth **~3 reward units** at
weight 1.0, so any shaping term integrating above ~10 drowns the thing being taught.

**`25v25_curriculum.yaml`** — same terms at R0 with the goal signal present but not yet
leading (`vp_gain` 1.0, `objective_coverage` 0.4, `objective_hold` 0.4, `model_kills` 0.5),
then an R1 identical to the single-phase config.

R0's gate uses one arithmetic bound: a one-objective stack caps at 19 scoring rounds × 5 VP
= **95 VP = 0.3333 of the 285 theoretical max**, so `player_vp_min` above 1/3 is the
smallest bar a stack provably cannot clear. 0.40 (114 VP) sits above it. This is arithmetic,
so it survives any future policy.

Two parameter notes that are easy to get wrong: `group_cohesion` takes
`group_max_distance` / `violation_penalty` (anything else is a `TypeError` at config load),
and `violation_penalty` is −0.05 rather than −0.2 because at −0.2 a strung-out model eats
−26 per episode, enough to rank `split_evenly` (145 VP) below `greedy_nearest` (132 VP).

`tests/test_curriculum_configs.py` enforces that every phase in both configs keeps
`vp_gain` and at least one per-model calculator — the config-level check for the two defects
that produced the earlier 62% → 47% decline.

---

## 6. What was measured

All figures from 30 held-out episodes on `25v25_single_phase.yaml`, seeds 700 000–700 029,
agent and baselines scored by the same code on identical layouts
(`just measure-checkpoint`, `just measure-baselines … 700000`).

| policy | on obj | win | player VP | opp VP | VP margin | worst cohesion |
|---|---|---|---|---|---|---|
| random | 0.017 | 0.00 | 14.5 | 173.0 | −158.5 | 24.5 |
| greedy_nearest | 1.000 | 0.57 | 132.3 | 115.2 | 17.1 | 10.3 |
| squad_march | 1.000 | 0.67 | 138.8 | 113.3 | 25.5 | 3.8 |
| split_evenly | 1.000 | 0.77 | 145.2 | 105.5 | 39.7 | 17.1 |
| **agent, single-phase** | 0.852 | **0.97** | 141.8 | 52.8 | 89.0 | 12.8 |
| **agent, ladder** | 0.887 | 0.93 | 161.7 | 59.5 | **102.2** | 14.0 |
| squad_march_shoot | 1.000 | **1.00** | 175.5 | 46.3 | 129.2 | 3.8 |

Measured, and attributable to the reward change *only under the confound below*:

- **Win rate 0.17 → 0.93–0.97**, against a movement-only bar of 0.67 and a shooting bar of
  1.00. Both agents beat every movement-only baseline decisively.
- **`entropy/shooting` rises**, 0.11 → 0.74 and 0.13 → 0.60 across two runs. The shooting
  head began near-deterministic and became exploratory — the signature expected from giving
  it a credit path it did not previously have.
- **`target_selection_optimality` 100%** in the traces: every shot at the best
  expected-damage target.
- **Objective drift ratio ≈ 3.0 → 1.00–1.11.** Drift was the failure that motivated the
  whole redesign; `objective_hold` was the term added to flatten it.
- **`entropy/movement` falls** 4.52 → 2.97–3.17 with `clip_fraction` in the healthy
  0.15–0.17 band and `explained_variance` reaching 0.55–0.62.

---

## Confounds

**The reward changes cannot be separated from the PPO mechanism changes.** They shipped in
the same two branches, and four mechanism defects were fixed alongside. The 0.17 → 0.96 jump
is the *joint* effect of per-model credit assignment, the mechanism fixes, the new
calculators, the dropped terms and the config rewrite. Nothing here isolates the
contribution of any one of them, and no ablation was run.

What *is* independently established is narrower and does not depend on any run:

- the `killing` field-name and telescoping defects (arithmetic)
- `models_at_objectives` saturating at 1.000 for every competent baseline (measured on the
  baselines, no learned policy involved)
- the 95 VP = 0.3333 stack bound (arithmetic)

Single seed per config for the curriculum comparison until the repeat; two seeds now.

---

## What the reward still gets wrong

1. **Occupancy is traded away.** Both agents finish at 0.85–0.89 of models on objectives
   where every competent baseline saturates at 1.000. They are buying kills with position.
   Whether that is a defect depends on whether VP margin or board control is the goal —
   they currently disagree.
2. **Stragglers break coherency and the reward barely notices.** Per-step out-of-coherency
   is 5.2–5.5% for the agents versus 5.5–6.1% for `squad_march`/`squad_march_shoot`, so the
   agents are *not* systematically worse during play. The difference is at the end: the
   baselines recover to **0.0%** out of coherency while the agents leave **3.0–3.2%** — about
   0.8 models per episode — stranded, and the worst offender sits **12.7–13.8** from its
   nearest squadmate against a limit of 6.0. `violation_penalty` −0.05 prices this too
   cheaply to fix a straggler that is otherwise scoring.
3. **Tabling the opponent truncates VP.** `is_battle_over` returns immediately on
   `all_eliminated`, so annihilation ends scoring. A traced episode killed all 25 opponents
   by step 26 losing nobody and finished on 85 VP, where an episode that left one enemy
   alive ran the full 40 steps and scored 165–5. `squad_march_shoot` shows the same
   signature (24 steps, 120 VP). No reward term accounts for this, so a policy optimising VP
   is implicitly taught to leave one enemy alive. This is a rules decision, not obviously a
   bug.
4. **The opponent cannot shoot back.** Every kill-related term is measured against a
   defenceless enemy, which flatters `model_kills` and `squad_march_shoot` alike. The 1.00
   shooting bar says more about the opponent policy than about the agent.

## What this does not establish

- That the curriculum helps. Two seeds say the two-rung ladder is indistinguishable from
  the single-phase config at 100 epochs (win 97.0/99.3 then 98.7/98.7; single ahead on VP
  margin both times). The ladder's R0 gate does clear reliably, so the gate design is sound
  — it simply is not buying anything yet.
- Which individual reward term is responsible for any of the improvement.
- That the weights are near-optimal. They were sized from the ~3-unit argument above and
  from one round of episode-integral arithmetic, not from a sweep.
