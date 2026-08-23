# Composable missions — the design, and what the design phase found

**Status: design agreed, not built.** Seven independent reviews, one refuted
premise, two verified defects. This records the vocabulary, the seam and the
sequence, so the build can start small and each step can be checked.

## What the design phase actually established

### 1. The cheapest form of the premise is REFUTED

The motivation was that the mission is "hello world" and distorts play: on a
5–6 objective table the fourth objective a side controls pays **zero**, and
`just measure-vp-cap` measured that discarding **10.1%** of `squad_march_take`'s
VP against **1.1%** of the agent's.

That asymmetry runs the **wrong way**. Doubling the cap (held-out nine, n=30,
seeds 700000+, refereed, agent K=3, scripts K=1, `coherent` 0.965–0.967 v 0.947):

| | cap 15 | cap 30 | own VP gained |
|---|---|---|---|
| `squad_march_take` | **+7.6** · held 2.80 | **+10.6** · held 2.80 | **+27.8 (+13.0%)** |
| agent, 3 seeds | **+15.9** · held 2.19 | **+13.7** · held 2.18 | **+3.2 (+2.0%)** |
| **gap** | **+8.33** | **+3.10** | **−5.23, against the agent** |

**Uncapping is worth 8.7x more to the script than to the agent**, because the
agent never reaches three objectives in the first place — `held` 2.19 against a
cap that bites at 3. There is nothing above the cap for it to collect. The
policy ranking is unchanged, so by the pre-registered rule — *a mission that does
not re-rank today's policies cannot teach anything new* — raising the cap is
**rejected before training**.

⚠ **Scope.** This scores FROZEN policies, so `held` cannot move by construction.
It refutes "the current mission costs the agent points it already earns". It does
**not** prove a policy *trained* under a richer mission would behave the same.
What it removes is the specific evidence that motivated this work.

### 2. Two defects, verified, worth fixing regardless

**(a) The agent is shown a control count that is not the one it is scored on.**
There are three independent implementations of objective control. Scoring and
reward use `norms_offset <= obj_radii`, which measures from the model's **base
edge**. `observation_builder.py`'s `inside()` uses
`area.contains_points(model_centres)` — the **centre**. Measured over 2,700
`(objective, step)` slots on the held-out nine: **206 disagree, 7.6%**, 215
models miscounted. `player_count` on the objective token is the one feature every
proposed primitive keys on.

**(b) The first non-default mission silently breaks the curriculum.**
`reward/criteria/player_vp_min.py:19` returns **0** for any
`mission.type != "default"`, so the phase gate collapses to `min_vp`,
`success_rate` pins at 1.0 and phases advance on epoch count alone;
`vp_threshold_for_terminal_bonus` returns `None` on the same branch, silently
disabling the terminal bonus. `reward/calculators/vp_gain.py` divides by
`params["cap_per_turn"]` and **falls back to 15**, so a composed mission that
omits it trains at the wrong reward scale. Found independently by three
reviewers.

### 3. The compat surface is six modules, not 115 configs

**No config in the repo declares a `mission:` block** — all 115 fall through to
`MissionConfig()`. What couples to it is `vp_gain`, `player_vp_min`,
`scripted_squad_march_deny`, `scripted_squad_march_take`, `measure_vp_cap` and
`state/snapshot.py`; two of those branch on the literal string `"default"`.
**That string is what breaks, not the YAML.**

## Corrections to the original brief

- **"Battlefield actions" is IN the spec** — `docs/rules/README.md` lists it under
  *deliberately out of scope*, with a stated procedure: add the chapter first,
  the implementation second.
- **`action` is already this repo's word for the RL action.** `ActionHandler`,
  `action_mask`, `WargameEnvAction`. The spec word is **`task`**.
- **The rules define no board regions.** No "half", no "centre" — only the
  deployment zone. Regions must be `own_zone` / `opponent_zone` / `neither`,
  derived from the zone outlines: 34 of 45 real tables have non-rectangular
  zones and `long_edges` splits the SHORT axis, so a board-half rule means a
  different thing on every table.
- **`Condition -> bool` is the wrong shape.** "Control an objective" pays 5 *per
  objective*, not 5 if any. Counts do not generalise from booleans.

## The vocabulary

`ScoringRule = Selector x Measure x Payout`, dispatched by a `Schedule`,
accumulated through a `Ledger` of cap groups. A mission is a list of rules.

| piece | type | note |
|---|---|---|
| `Selector` | `view -> mask over entities of kind K` | typed to the Measure's entity kind, so illegal pairings are unrepresentable |
| `Measure` | `(view, moment) -> int` | a **count**. Booleans are the {0,1} case |
| `Payout` | `(value_per, cap) -> min(cap, n * value_per)` | integer arithmetic only — this is what keeps bit-identity |
| `Schedule` | `(phase boundary, side, round range)` | timing is not a scalar |
| `Ledger` | cap groups: per round, per battle | the rules' 15/45; the piece with no home in the original brief |

**Tier 0 — free, no new state, no tensor change:** control an objective ·
control more than the opponent · deny/contest · be in a place that is an
objective · unit destroyed · count >= N · region over objectives.

**Tier 1 — pure, but INVISIBLE to the agent** (needs an objective-token column,
which changes tensor width): marked objective · region as a feature · per-model
occupancy. ⚠ Buy **one** width change on the objective token for all of these at
once, never four separately.

**Tier 2 — needs episode state:** per-round and total caps · consecutive hold ·
destroyed-this-turn.

**Tier 3 — refused in v1:** tasks (three absent prerequisites, a new action-space
slice, and a completion-only reward of the shape that does not train here) ·
suppression · control value as a sum.

## The seam

```
types/  <-  domain/  <-  mission/  ->  domain/battle_view (BattleView)
mission/ MUST NOT import env_components/, reward/, renders/, wargame.py
wargame.py -> mission/       (constructs, calls at the timing point)
reward/    -> BattleView.mission  (a MissionView protocol), never mission/
```

`Mission.evaluate(...) -> MissionOutcome(vp, effects)`, where `effects` is
**always empty in v1** and typed `tuple[()]`. That keeps the door open for a
mission that restricts units without granting it — a mission that can forbid
shooting changes the action mask the policy trains against, which is a scenario
change, not a scoring variant.

## The sequence

Each step ships alone, states what it does not yet do, and lands only with a
test **proven to fail on the previous commit**.

0. **Add `configs/golden/25v25_maps_two_mode.yaml` to both golden test lists.**
   None of the three current golden configs draws from a map pool or uses area
   objectives, so the polygon path — 91% of `compute_distances` and the path
   every later step touches — is pinned by nothing. Regenerate on `main` first.
1. **Fix defect (a)**: make the observation read the scoring context. Test: the
   two counts agree on every slot; fails today at 7.6%.
2. **Fix defect (b)**: derived properties on `MissionConfig`
   (`per_round_cap`, `points_per_objective`, `first_scoring_round`); delete both
   `type != "default"` branches. Test: a `type: none` config keeps a real phase
   gate; fails today.
3. **Free throughput, bit-identical**: one shared opponent distance cache
   (−9% step) and a batched `_distances_to_objectives` (verified 0.2441 →
   0.1389 ms). The opponent matrix is currently built **3.19x per step**.
4. **`reset_episode()` on the calculator protocol**, called beside
   `phase_manager.reset_episode()`. `reset()` never touches `_vp_calculator`
   today, so the first stateful mission would leak across episodes.
5. **Value objects** — `ScoringMoment`, `Payout`, `Ledger`, `MissionMemory`.
6. **Re-express `default` as a composition and prove it identical**: golden
   green *unregenerated*, a seeded digest against a worktree at the parent
   commit, and `just measure-baselines` reproducing to the printed digit. The
   digest must include the **phase index and success flag** — VP can be
   bit-identical while the mission is broken.
7. **First new primitive**, in `configs/experiments/`, with its own re-measured
   bar. A new mission inherits none.

## Refused in v1

Tasks · board-mutating effects · unit restrictions · consecutive-hold as a
bespoke ledger (the rules own that mechanism under *secured*, currently absent —
a second private definition of persistence is the `in_cover: bool` failure
again) · a mission suite before any single mission has a control · training
under a raised cap (§1) · editing a golden config.

## Before any mission trains

Two pre-registered rejections, both free:

- **Re-ranking.** Score today's policies under the candidate mission with no
  retraining. Rank correlation with the current mission **>= 0.95 → reject**.
- **Headroom.** A mission where the best script already scores **> 90%** of the
  ceiling is rejected. The current mission saturates at 272–277 of 285, which is
  why it degenerated into a denial game.
