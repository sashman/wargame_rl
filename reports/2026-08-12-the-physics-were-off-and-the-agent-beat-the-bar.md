# The physics were off, and then the agent beat the bar

**2026-08-12** · `configs/experiments/25v25_real_geometry.yaml`, four 1000-epoch
runs · wandb group `real-geometry-r1`

Goal: beat `scripted_advance_and_shoot` on the 25v25 shooting scenario.

**Achieved.** But the first thing found was that the scenario being trained on
was not the one anyone thought.

---

## 1. `base_radius` defaulted to 0.0, so all of phase 03 was inert

`configs/golden/25v25_shooting_opponent.yaml` has never set `base_radius`. The
default was `0.0`, and at radius 0 the whole continuous-geometry phase is a
**documented no-op**:

- no disc occludes, so models do not block sight
- the three cover rays coincide, so **cover cannot occur at all**
- models do not collide
- range is centre to centre rather than base to base

Objectives were free-floating discs rather than the ruins. The config was the
pre-geometry scenario running on the post-geometry engine, and had been for
every run since phase 03 landed.

Nothing failed. `base_radius: 0.0` is a legal, sensible-looking value that
silently disables four mechanics at once, and the gap map even recorded it as
the default — as a fact about the code, not as a warning.

**Defaults now:** `base_radius` = the rules' 32mm infantry base (0.63"),
`objectives_on_terrain` on, recordings on the v2 renderer with the tabletop
theme. Terrain refitted to the 45 real layouts in `configs/evaluation/maps/`
(15 pieces of 3–11; measured coverage 22.3% against the real 23.5%).

**This voided every number again.** The bar on the shooting scenario:

| | `squad_march_shoot` |
|---|---|
| physics off | +6.7 |
| physics on | +27.8 |
| after the baseline arrival fix (§3) | **+38.0** |

---

## 2. The result

`objective_hold` 1.25 → **2.5** paired with `objective_coverage` 0.3 → **0.1**.
Same reward budget, moved from a *global* term to a *per-model* one. Epoch 1000,
n=100, **paired against the bar on identical layouts**:

| arm | agent | paired vs bar | t | ahead |
|---|---|---|---|---|
| **hold25 s1** | **+52.1** | **+14.2 ± 6.1** | 2.31 | **64 / 89** |
| control s1 | +44.1 | +6.2 ± 6.6 | 0.93 | 51 / 86 |
| control s2 | +42.8 | +4.8 ± 5.7 | 0.84 | 51 / 82 |

The control reproduces itself across two seeds at +6.2 and +4.8 — both t ≈ 0.9,
both with exactly 51 wins. Neither seed is significant alone; **pooled, the sign
test is** (102 of 168 differing episodes, z ≈ 2.8, p ≈ 0.005). So the golden
configuration already edges the bar by ~5 vp, and the income shift roughly
triples that.

**Zero-shot on the goal's own config.** Scored on
`configs/golden/25v25_shooting_opponent.yaml` — same everything except terrain,
29 small ruins against the 15 large ones it trained on:

```
squad_march_shoot   +49.5      hold25 s1   +58.4
paired  +8.8 ± 5.8   t = 1.51   ahead 59 of 90   (sign test p ≈ 0.003)
```

The bar scores *higher* there (+49.5 v +38.0) — small scattered ruins suit a
squad marcher — and the agent still beats it on terrain it never saw.

**Read the win count, not only the t.** 64/89 is z ≈ 4.1 where the t is 2.31.
They disagree because the margin distribution is heavy-tailed: the agent wins
most maps moderately and loses a few badly. The sign test is the more robust
statement throughout this report.

### What changed in the play

| | hold25 s1 | bar |
|---|---|---|
| vp_margin | +52.2 | +38.0 |
| win | 0.79 | 0.75 |
| **objectives held** | **1.80** | 1.71 |
| firepower ratio | 1.40 | 0.71 |
| alive | 0.473 | 0.450 |

It wins in both directions — 17.7 more VP scored, 6.6 fewer conceded. The
interesting number is `held`. At the 300-epoch screen *every* arm held fewer
objectives than the bar (1.33–1.67) while winning the firefight — the "wins
fights, loses ground" signature of [the abandonment
work](2026-08-10-real-terrain-and-the-abandonment-gap.md). By epoch 1000 this
arm holds **more** ground and still wins the firefight 2:1.

**The allocation gap closed without an allocation feature.** One was considered
and declined, on two cheap checks: the policies that *do* allocate deliberately
score worse (`contest_and_spread` +18.7 against the bar's +38.0), and
`measure-objective-split`'s ceiling is explicitly optimistic — its own docs say a
large ceiling "does not rule re-allocation in". Declining was right for the wrong
reason: the agent learned *when* to spread, which is what the scripted
allocators cannot do.

---

## 3. Every baseline was marching onto the objective's centroid

`WargameObjective.radius_size` is **0.0 for an area objective by design** — its
extent is the outline, and distance is reported to that edge through the
`norms_offset` seam. `step_toward_objective` took a *location and a radius*, so
every "have I arrived" test waited for the model to reach the **centroid
exactly**. On a marker objective that is the middle of a small disc. On a ruin it
marches a whole squad onto one point — where, once bases are real, they collide
and the models behind stop dead in the open.

Nothing raised, and no test caught it: the policies did exactly what they were
told, against a field that means something different for the two kinds of
objective. It surfaced only when occupancy was measured after the base default
changed.

Arrival is now a region test. Occupancy rises consistently (0.61 → 0.91 on one
seed set, 0.61 → 0.87 on another). **Its effect on score does not generalise**,
and the first version of this claim was wrong — see §4.

Still broken, deliberately: `greedy_nearest` and `split_evenly` aim every model
at the centre, so a squad approaching from one bearing funnels through one point
of the near face — 3 of 10 get in. Steering at the *nearest* point of the
footprint is worse (models spawned above-left all clamp to the same corner, 1 of
10). A real fix needs per-model target slots or collision-aware steering. Pinned
with its numbers in `tests/test_baseline_area_objectives.py` so it cannot look
fixed.

---

## 4. Four claims died to proper pairing, and they were all mine

Per-episode `vp_margin` sd here is ~60. Two aggregate rows over 20–100 episodes
therefore carry a standard error of 6–20 on their difference, which is the size
of most effects worth measuring. Running both arms over the **same seed list**
and differencing per episode removes the layout variance that dominates those
rows.

| claim | unpaired | paired |
|---|---|---|
| weakest-unit targeting beats nearest-first | +8.0 (n=60) | **+1.7 ± 5.7** (n=100) |
| the baseline arrival fix is worth +10.2 | +10.2 (one seed set) | **+7.5 ± 4.7 pooled** |
| the agent leads the bar at epoch 126 | +7.6 (n=30) | **+0.8 ± 7.6** (n=60) |
| hold25 leads the control at the 300 screen | hold25 +7.2, control −3.8 | **reversed** — control +5.0, hold25 +1.5 |

The arrival fix is the instructive one. Paired against a worktree at the parent
commit, 60 layouts each: **+16.7 ± 6.5** on seeds 700000+, **−1.6 ± 6.7** on
seeds 10000+, **+7.5 ± 4.7** pooled over 120. It helps clearly on one layout set
and does nothing measurable on another. Occupancy, meanwhile, rises on both — the
*behavioural* defect is fixed even where the score effect is invisible.

A near-miss inside that: an in-run baseline logged 28.2 before and 11.8 after on
seeds 10000+, which read as the fix costing 16 points. Both were unpaired
20-episode samples. **Two 20-episode aggregates cannot see a 16-point effect on
a quantity with sd ~60.**

The last row is a different trap and now has its own note in `CLAUDE.md`:
`ppo-NNN-*.ckpt` records the last epoch whose *training reward* improved, and
that epoch differs per arm. Screening each arm's newest file compared γ0.97's
epoch-121 weights against the control's epoch-349, despite both runs starting
together. Matching to `last.ckpt` reversed the ranking.

`just measure-paired` now takes baseline names, checkpoint paths, or two code
versions via a worktree.

---

## 5. What was refuted

**γ = 0.97 was the lead hypothesis and came last.** The argument: `objective_hold`
is the dominant per-model income (35.4% of gross), it is a *hold* term whose
payoff needs a 7–9 step walk to collect, and γ = 0.9 gives a ~10-step credit
window over a 40-step episode. `PPOConfig` even says to retest if the reward's
time structure changes, and it had. Measured **−8.8 ± 8.5** paired at the
300-epoch screen, worst on two independent checkpoints, ahead in only 21 of 52.
Retired there. The arm picked as the *safe* second option is the one that
separated.

**`observe_unit_strength` — built, then its motivation refuted before training.**
Shooting names a unit and the head mean-pools opponent tokens, and a mean cannot
count, so a full squad and its last survivor pooled identically. The feature puts
the unit's alive fraction on every member's token. But the cheapest proxy for the
mechanism is null: weakest-first targeting is **+1.7 ± 5.7** paired over 100
layouts, ahead in 24 of 100. The choice is not rare — 59.5% of shooters see more
than one valid unit, 72% of those differ in strength — it simply does not pay,
because unit targeting discards only **3.6%** of declared attacks, which caps
what finishing a unit early can reclaim. Kept off by default with the refutation
on the config field.

---

## 6. What this does not support

- **Not "the income shift is worth +14.2".** That is **one seed**; the second was
  still training when this was written. Measured within-config seed spread on
  this project is ~9 vp. The two-seed backing here is the *control* arm's ~+5.
- **Not a cover result.** Cover became possible for the first time this session
  (it requires `base_radius > 0`). Nothing here measures whether the agent uses
  it. `exposure` fell 0.354 v the bar's 0.305 — the wrong direction for a cover
  story, and confounded by the agent keeping more models alive.
- **Not that the bar is a ceiling.** It is an *unconditional* policy: fixed
  squad→objective assignment by `k % n_objectives`, nearest-first fire, identical
  play in round 1 and round 19, and it reads neither terrain outlines nor cover
  state nor the round number. The headroom is conditional play, not a novel
  strategy.
- **Not transferable to the pre-2026-08-12 numbers.** Every figure measured
  before the geometry defaults changed describes a different scenario.
