# Play doctrine

`docs/rules/` is this project's authority on what is **legal**. This file is its account of
what is **good**: the higher-level judgements a strong player makes on top of the rules —
which ground to take, how much force to commit to it, when to shoot and at what, and when to
do nothing at all.

Every entry is accountable to two things and only two: the specification in `docs/rules/`, and
the measured record in [CLAUDE.md](../CLAUDE.md) and [reports/](../reports/README.md). A claim
that cannot be stated in the specification's own terms does not belong here, and a claim the
record contradicts is marked `refused` and kept, so it is not proposed again.

## 1. What this is

Three standing warnings. They matter more than any entry below.

- **Doctrine is a hypothesis until it is priced. Nothing here is evidence.** `CLAUDE.md` and
  `reports/` are the record. Where they disagree with an entry, they win and the entry carries
  a `refused` verdict with the reason. An entry has never justified a training run on its own.
- **This game is movement and shooting only.** A claim that needs a phase the environment does
  not have is *recorded and parked*, never quietly adapted into something the environment can
  nearly do. Adaptation is how a heuristic loses the mechanism that made it true.
- **Terms come from [docs/rules/00-glossary.md](rules/00-glossary.md).** The specification names
  no product, publisher, edition or faction, and neither does this file.
  `tests/test_no_ip_references.py` enforces that half automatically. The other half — **no
  attribution of any entry to an outside person, channel or publication** — has no test and is
  review discipline. Do not add one, and do not add a section that would invite one.

## 2. How to read an entry

Every entry is six fixed fields under a stable ID. Cite entries by ID (`D-05`), never by title
— titles get edited.

| field | contents |
|---|---|
| **Claim** | One imperative sentence, in this project's vocabulary only. |
| **Why it is true here** | The mechanism, named in code — symbol, file, constant. Never an appeal to how the game is usually played. |
| **Expressible** | `live` · `partial` · `blind` · `absent` · `quantum`. See below. |
| **Where it lands** | The extension point, by symbol. |
| **Already measured** | The standing result, with date and verdict. **This is the field that stops re-runs.** `Not measured` is a legitimate value and is always written out. |
| **Cheapest test** | The literal command, or `do not test` and why. |

**Expressible** mirrors the gap map's enum on purpose, so the two share a vocabulary:

| value | meaning |
|---|---|
| `live` | The environment can represent this today. |
| `partial` | It works, degraded — usually because there is no unit entity, only `group_id`. |
| `blind` | True on the board, absent from the observation. The agent cannot key on it. |
| `absent` | Needs a mechanic that is not implemented. The entry links its [gap-map](rules/implementation-status.md) row. |
| `quantum` | Legal in principle, forbidden in practice by coherency's allocation granularity. |

**Verdict** is the triage handle, used in the index tables and in § 9:

| value | meaning |
|---|---|
| `act` | Worth building now. Named in the backlog with a pre-registered accept/reject. |
| `price` | Read the existing diagnostics before building anything. |
| `parked` | Blocked. The entry names the blocker so it is not rediscovered. |
| `settled` | Measured. The number is in the entry. |
| `refused` | Measured and rejected, or rejected on the record. Do not propose it again. |

**Lifecycle.** When an entry is measured its verdict flips to `settled` or `refused` and gains
a date and a report link. **The finding itself goes to `CLAUDE.md` and `reports/` as normal.**
This file must never become a second record; if it starts carrying findings that live nowhere
else, that is the failure mode, not a feature.

## 3. The shape of this game

Doctrine only means something against the arithmetic it is played under. Every row below is
derived from a named file, and each one kills or reshapes a heuristic that would otherwise be
imported unexamined.

| Fact | Where | Consequence |
|---|---|---|
| Control is `player_count > opponent_count`, **strictly**. A tie controls nothing. Every model is control value 1 — a headcount. | `env_components/distance_cache.py` · `objective_ownership_from_norms_offset` | **Denial costs a tie, not a win.** Matching their count removes 5 VP from them. Holding costs their count **+ 1**. The body past that threshold is worth exactly zero. |
| `VP = min(15, controlled x 5)` per scoring moment, from round 2. | `mission/vp_calculator.py` · `DefaultVPCalculator` | **Own score saturates at three objectives**, on tables carrying five or six. The fourth pays nothing to you and five to them. **Above the cap the whole game is denial.** |
| VP is scored **at each side's command boundary, before that side moves** — and fires even when `command` sits in `skip_phases`. | `wargame.py` · `_on_before_advance`; `domain/turn_execution.py` | **Your score is decided by the board your opponent leaves you.** To deny, be standing on their ground at the end of *your* turn. No policy in the repo encodes this. |
| The config that trains runs **20 battle rounds**, so ~19 scoring rounds a side. | `configs/golden/25v25_maps_two_mode.yaml` | **No single round is worth more than about 5% of the game.** Every last-round-swing, going-second and endgame-timing heuristic is an artefact of a five-round game and **does not transfer**. This is the largest single transfer failure in this document. |
| Only terrain blocks sight; models never do. Enemy bases block a move; friendly bases may be crossed but not ended on. | [gap map § Terrain](rules/implementation-status.md) · `domain/movement.py` · `resolve_move` | **A screen stops movement, never fire.** Screening is a real mechanic here, and self-gridlock is its price. |
| Every objective on the real tables sits inside a terrain piece. | `CLAUDE.md` § Holding pays | **Holding is hiding.** Standing on an objective pays +0.37 to +0.44 more per model-step at *negative* excess death hazard. "Stay off the point and stay safe" is refuted. |
| There is no melee, so a shooting army has **no reason to close** except to stand on an objective. | [gap map § Charge and fight](rules/implementation-status.md) | ⚠ **Every measurement of a movement feature here is provisional on that.** Closing the distance is priced only by what it captures, never by what it threatens, so any move type whose value is "arrive sooner" is measured in a game that does not yet reward arriving. |
| There is no charge phase and no fight phase. | [gap map § Charge and fight](rules/implementation-status.md) | **Ground is sticky.** Nothing is levered off a point except by being shot off it, so every heuristic built on contact — trading, tarpitting, wrapping — arrives `absent`. |
| Coherency is a 2" chain and a 9" span over a five-model unit. | [rules § Coherency](rules/03-moving.md) | **The allocation quantum is the unit, not the model.** "Send one more body than they have" is not a legal move. Several of the strongest entries below are blocked on precisely this. |
| Weapon range is 12" against a 6" move on a 60x44 board. | `configs/golden/25v25_maps_two_mode.yaml` | Threat is a **two-move quantity**, and on the `long_edges` layouts the armies start 20" apart across the short axis — inside two moves from turn one. |

---

## 4. A — Ground and its triage

| ID | Claim | Expressible | Lands in | Verdict |
|---|---|---|---|---|
| D-01 | Sort every objective into own-zone, contested and hostile before moving anything | `blind` | observation · scripted policy | `price` |
| D-02 | Take the cap's worth of ground and no more; spend the rest on denial | `live` | scripted policy | `act` |
| D-03 | Holding beats raiding — a commitment that does not cross the count threshold changes nothing | `live` | scripted policy | `settled` |
| D-04 | Ground the opponent holds in strength is not a target; the cheapest neutral point is | `live` | scripted policy | `act` |

### D-01 — Sort every objective into own-zone, contested and hostile before moving anything

**Claim.** The three classes are worth different things and are held by different means. Ground
inside your own deployment zone is cheap to keep and expensive to lose; contested ground in the
middle is the pivot; ground inside the opponent's zone is a denial target, not a holding
target.

**Why it is true here.** The real tables place a third of their objectives inside each player's
own zone — 82 in the player's, 82 in the middle, 82 in the opponent's over 246 objectives on 45
tables — so a player begins holding 1.98 of them before a model moves
(`configs/golden/25v25_maps_two_mode.yaml`). That is a materially different mission from the
generated scenarios, which can only place objectives in the contested middle, and no reward
lesson from those is known to carry across.

**Expressible.** `blind`. The objective token carries `location`, and with
`observe_objective_control` also per-side counts and radius — but **nothing tells the agent
which zone an objective is in**. The zone outlines exist in config
(`deployment_outline` / `opponent_deployment_outline`) and are not in the observation.

**Where it lands.** An `observe_*` flag on `WargameEnvConfig` widening the objective token, and
`env_components/observation_builder.py::_objectives_to_obs`. A scripted policy can read the
config directly and needs no observation change.

**Already measured.** Not measured as a policy. The placement asymmetry is measured and is why
`hold_deployment` exists as a floor: standing still ends on 1.63 objectives with 99.8% of the
force alive and still loses every episode, because the opponent takes the middle uncontested
and then comes for the home points.

**Cheapest test.** As a scripted policy first, reading zones from config — no tensor change.
`docs/missions-design.md` Tier 1 warns that region membership is an objective-token widening,
so if it ever becomes an observation, **buy one widening for every such feature at once**.

### D-02 — Take the cap's worth of ground and no more; spend the rest on denial

**Claim.** Hold exactly three objectives. Every unit beyond what those three need goes to
denying the opponent theirs.

**Why it is true here.** `min(15, controlled x 5)` saturates at three. A fourth held objective
is worth **zero** to you and **five** to the opponent, so the same unit is worth 0 or 5
depending only on where it stands.

**Expressible.** `live`.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives` — the per-squad allocation seam.

**Already measured.** The coarse form exists as `ScriptedSquadMarchDenyPolicy`, which banks
`cap // vp_per_objective` points and sends surplus units at what the opponent controls. Against
`squad_march_take`'s flat cheapest-first order it is a **measured null, not a loss**: all 45
generated tables, n=30, seeds 700000+, `take` **+5.9** and `deny` **+5.4**, and paired at n=100
the difference across three layout sets is **+5.7 / −9.2 / +9.8 — it changes sign**, mean +2.1,
with 25–31 of 100 episodes *identical*. ⚠ The reason is D-30: at five units against five or six
objectives **there is no surplus unit**, so the two policies are executing the same plan. This
claim has therefore never actually been tested. `just measure-vp-cap` shows the cap binds on
23.9% of steps for `squad_march_take` and discards 10.1% of its VP, against the agent's 1.1%:
**the scripts are the ones paying the cap tax, and the agent is not even reaching it** — three
objectives on 22.3% of steps against the script's 55.6%.

**Cheapest test.** On the config that *has* a surplus unit —
`just measure-paired squad_march_deny squad_march_take configs/experiments/24v24_maps_spare_squads.yaml 100`
— before building anything new. On the golden config it is untestable by construction.

### D-03 — Holding beats raiding

**Claim.** A commitment that arrives without crossing the count threshold has bought nothing.
Prefer keeping a point you already control to contesting one you will not flip.

**Why it is true here.** Control is a strict comparison, so a raid that arrives at parity
leaves the objective controlled by neither — which denies the opponent 5 VP but costs the
raiding models everything they were earning where they came from.

**Expressible.** `live`.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives`.

**Already measured.** `settled`, from two directions. The 2026-08-11 teleport audit
force-moved a unit onto contested ground and measured **−1.69 of 5 models** and **−29.41 of its
own income** against 4.91 defenders. And three consecutive reward terms built to pay for
contesting left offence **flat or worse** (−50.5, −42, −71.5); the last of them,
`contest_deficit`, worked mechanically — "they hold it by 2+" exclusions fell 43.4% to 3.9% —
and bought nothing. ⚠ Do **not** cite the take-versus-deny ranking as evidence for this: those
two are a measured null on the config that trains (see D-02).

**Cheapest test.** Do not test. The mechanism is measured from three directions.

### D-04 — Ground the opponent holds in strength is not a target

**Claim.** Rank denial targets by how many models it takes to reach parity, ascending. A point
held by four is not a denial target while a point held by one exists.

**Why it is true here.** Parity is the threshold, so the cost of denying a point is exactly the
opponent's count there and the payoff is a flat 5 VP regardless. Cost varies, payoff does not,
so cheapest-first is the whole rule.

**Expressible.** `live` — `observe_objective_control` puts per-objective counts for both sides
on the objective token, and since 2026-08-22 that count is the same one scoring uses
(`objective_counts_from_norms_offset`; before then 7.6% of slots disagreed).

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives`.

**Already measured.** Partly. `contest_and_spread` allocates against *predicted* counts and
skips unwinnable objectives rather than half-garrisoning them, and it lost (0.60/+18.8 against
the bar's 0.77/+39.4) — but it also predicted counts rather than reading them, and bundled
three changes. `closest_objective_v2`'s `contest_deficit` widened the reward-side version of the
same idea and was rejected at −2.7 ± 4.8. **The reward-side form is closed; the decode-side
form is not.**

**Cheapest test.** `just measure-paired <new> squad_march_take configs/golden/25v25_maps_two_mode.yaml 100`.

---

## 5. B — Allocation and commitment

This is the group that aims at the project's one open failure. Read § 8 before proposing
anything from it.

| ID | Claim | Expressible | Lands in | Verdict |
|---|---|---|---|---|
| D-05 | Commit one model more than they have, and no more | `quantum` | scripted policy · decode | `act` |
| D-06 | The allocation quantum is the unit, not the model | `live` | scenario | `settled` |
| D-07 | Choose the cheapest points, not the nearest | `live` | scripted policy | `act` |
| D-08 | A surplus unit goes to the cheapest unheld point, then the cheapest enemy-held one | `live` | scripted policy | `settled` |
| D-09 | Re-allocate against the board as it is, not as it was on departure | `live` | scripted policy | `price` |
| D-10 | Prefer greedy matching to exact matching | `live` | scripted policy | `settled` |

### D-05 — Commit one model more than they have, and no more

**Claim.** The force sent at an objective is the smallest that puts your count strictly above
theirs. Everything past that threshold is surplus and belongs elsewhere.

**Why it is true here.** The marginal model on an already-controlled point changes no score
(§ 3, row 1). The same model on a point where you sit at parity converts 0 VP into 5.

**Expressible.** `quantum`. The 2" chain makes the smallest legal commitment **one unit of
five models**, not one model. With five units against five or six objectives there is nothing
to allocate, which is the whole content of D-06.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives`; and as a follow-on,
`model/common/decoding.py`, where the top-K joint decode already reasons over combinations of
per-model moves and is where +40.5 vp of this game's value currently lives.

**Already measured.** The stacking is the largest recorded here: the agent puts **4.90** models
on its top point where `squad_march_take` puts 2.73, spends 54.4% of model-steps on objectives
against 75.5%, leaves 55.3% of points empty, redistribution ceiling **+2.20** (2026-08-22).
The critic already prices spreading correctly — forking a live game and translating one surplus
unit off the pile onto an empty point gives `dV` **+2.63 ± 0.32** and realised `dVP`
**+3.85 ± 1.81** (2026-08-23), so **reward and value are both right and the failure is search**.
⚠ Two blunt forms are already negative: forced redistribution **−3.6**, and every spare unit
onto cheap ground **−3.2** over 180 paired episodes.

**Cheapest test.** `just measure-paired <new> squad_march_take configs/golden/25v25_maps_two_mode.yaml 100`.
⚠ **Read `vp_margin`, never top-stack occupancy.** The gradient out of the stack is shallow
(+3.85) and the gradient in is steep (−11.52), so any lever moves occupancy a great deal for a
small score change.

### D-06 — The allocation quantum is the unit, not the model

**Claim.** Coherency, not the reward, decides how finely force can be divided. Unit size is
therefore a strategic parameter of the scenario, not a detail of it.

**Why it is true here.** A model outside its unit's 2" chain is out of coherency, and under
`objective_hold.require_coherent` it earns nothing at all — so detaching one body to break a tie
costs its whole income and, at play, risks the referee cancelling the unit's move.

**Expressible.** `live` — as a scenario choice.

**Where it lands.** The scenario: `models[].group_id` and `max_groups`.

**Already measured.** `settled`, 2026-08-22. Five units against five or six objectives pose
**no allocation question**: `squad_march_take` and `squad_march_deny` differ only in what a
spare unit does, and paired at n=100 their difference is +5.7 / −9.2 / +9.8 across three layout
sets — **it changes sign**, and 25–31 episodes in 100 are identical. Eight units of three do
pose it: **+16.0, positive on 3 of 3 sets**, with only 2–5 identical episodes. ⚠ But trained on
the config that does ask, **offence did not move**: agent +15.1 ± 5.6 against the script's
+6.0 ± 3.0, offence −50.5, defence +59.6.

**Cheapest test.** Do not re-test the scenario question; it is answered.
`configs/experiments/24v24_maps_spare_squads.yaml` is the config that asks it.

### D-07 — Choose the cheapest points, not the nearest

**Claim.** Rank a unit's candidate objectives by travel **plus expected contest**, not by
distance. The nearest point is often the one everyone else is already walking to.

**Why it is true here.** Distance is the only term any current shaping pays on
(`closest_objective_v2` prices distance closed), and with 8 units over 5–6 markers leaving 2–3
unassigned each step, a pure-distance rule sends two members of one unit at different targets
on 8.0% of unit-steps against a script's 4.8%.

**Expressible.** `live`.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives`.

**Already measured.** ⚠ Do **not** route this through `closest_objective_v2`. That term has
been nominated four times and produced four empty results — the candidate gate
(`contest_deficit`, rejected at −2.7 ± 4.8), removing the overstack penalty (rejected at
−12.2 ± 5.5), the potential-invariance defect (real; the term nets negative anyway), and the
fallback mechanism (2026-08-23: 43.5% of paid model-steps are already inside their target and
earn zero, 64.2% of objectives are candidates for nobody, net income is progress +0.08 against
a −0.90 penalty). The **cost model** side is untested.

**Cheapest test.** `just measure-paired <new> squad_march_take configs/golden/25v25_maps_two_mode.yaml 100`,
then `just measure-objective-split` for the redistribution ceiling.

### D-08 — A surplus unit goes to the cheapest unheld point, then the cheapest enemy-held one

**Claim.** Order surplus targets: neutral ground first (cheapest to convert), then enemy ground
by ascending defender count. Never a point you already hold.

**Why it is true here.** A neutral point costs one model to flip and pays 5. An enemy point
costs their count to neutralise and pays 5 in denial. A held point pays nothing.

**Expressible.** `live`.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives`.

**Already measured.** `settled` and **negative in its blunt form**: "every spare unit onto cheap
ground" measured **−3.2** on 180 paired episodes, and forced redistribution **−3.6**. Cheap
ground genuinely exists — 2.4 undefended objectives in 96% of movement phases — and taking it
still loses, because own VP saturates at three and units are shot crossing open ground. ⚠ The
overstack penalty that discourages piling **was paying for itself**: removing it measured
**−12.2 ± 5.5, 3 of 3 seeds negative**, with offence +2.9 and defence −15.1. The travel term
did pay more for movement, and the agent conceded fifteen VP for it.

**Cheapest test.** Do not re-run the blunt form. Only worth revisiting bundled with D-26
(timing), which is what neither attempt varied.

### D-09 — Re-allocate against the board as it is, not as it was on departure

**Claim.** A unit's target is recomputed every movement phase against current counts. A target
chosen three phases ago is a target chosen against a board that no longer exists.

**Why it is true here.** Counts change every phase, and a unit walking at a point that has
since been taken in strength is walking at nothing. Against that: abandoning a target under
`closest_objective_v2` returns progress 0.0 and re-anchors, so **switching is free to the reward
and not free to the position** — the shaping cannot price the churn it permits.

**Expressible.** `live`.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives`.

**Already measured.** Confounded. `ScriptedSquadMarchDenyPolicy` recomputes every movement
phase and `ScriptedSquadMarchTakePolicy` does not re-sort, but the two also differ in target
*ordering*, and on the config that trains they are indistinguishable anyway (D-02) — so nothing
isolates recomputation frequency. The travel audit's finding that a target switch is free to the
reward is measured (2026-08-23) and is the mechanism to look at first.

**Cheapest test.** Read `just measure-shaping-gates` output on the training config before
building anything — the switching cost is already instrumented there.

### D-10 — Prefer greedy matching to exact matching

**Claim.** Allocate units to objectives greedily. Do not replace the greedy rule with a globally
optimal assignment.

**Why it is true here.** Optimality is only as good as the cost model, and this game's cost of
sending a unit somewhere is dominated by terms a static matcher does not see: whether the
ground will still be contested on arrival, and how much fire is crossed getting there.

**Expressible.** `live`.

**Where it lands.** `ScriptedAssignmentOptimalPolicy` already exists as the counter-example.

**Already measured.** `refused`, 2026-08-23. `assignment_optimal` is `squad_march_take` with
greedy matching replaced by an exact minimum-cost assignment (subset DP, verified against brute
force on 300 instances): **−26.1 ± 9.4 against greedy's +7.6 ± 3.8**, `held` 2.21 against 2.80.
⚠ This is **not** proof that allocation is at its ceiling — it is one untuned cost model losing
to greedy. But an allocation-aware decode would be replacing a rule that just beat its own exact
counterpart by 33.7 vp. **Re-cost before funding, and tune on the 36 training tables, never on
the nine held out.**

**Cheapest test.** Do not rebuild the optimal matcher. Re-cost the existing one and re-measure.

---

## 6. C — Bodies as obstacles

| ID | Claim | Expressible | Lands in | Verdict |
|---|---|---|---|---|
| D-11 | Bodies deny ground — an enemy model cannot be moved through | `live` | scripted policy | `act` |
| D-12 | Screen the approach, not the marker | `live` | scripted policy | `act` |
| D-13 | The single-model denier is not available here | `quantum` | — | `parked` |

### D-11 — Bodies deny ground

**Claim.** A model standing on a path is a wall. Placing bodies across the route to an objective
buys turns without firing a shot.

**Why it is true here.** `domain/movement.py::resolve_move` treats **enemy bases as blockers**:
the move stops at base contact. It is gated on `base_radius > 0`, which defaults to the rules'
32 mm infantry base, so **this is on in every config**. Friendly bases are `passable` —
crossable but not endable-on — which is what the rules say and which is also why a unit can
gridlock on its own front rank.

**Expressible.** `live`.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives`, or a new subclass placing a
unit on the interval between an opponent unit and the point it is walking to.

**Already measured.** **Not measured.** This is the one implemented mechanic in this document
that nothing here has ever tested. The freezing work (2026-08-22) measured **friendly**
gridlock only: 91.8% of frozen model-steps have a friendly base touching, against 0.22 enemies
per frozen model-step. ⚠ The gap-map row for this read *absent — movement ignores occupancy
entirely* until 2026-08-22; it predated bases, so nothing designed before that date could have
used it.

**Cheapest test.** `just measure-paired <new> squad_march_take configs/golden/25v25_maps_two_mode.yaml 100`.
⚠ **Run `just measure-freezing` on the new policy as part of the test, not after it.** The
freezing result predicts a purposeful screening policy freezes *itself*: a frozen model stays
frozen 89% of the time (absorbing +0.86) and a purposeful policy re-issues the same blocked
order forever. A screen that gridlocks is not a screen.

### D-12 — Screen the approach, not the marker

**Claim.** Bodies placed on the route are worth more than bodies placed on the destination,
because the destination is already contested by whoever is standing on it and the route is not.

**Why it is true here.** Control is evaluated only inside the objective radius, so a model on
the approach contributes nothing to control — but it costs the opponent a move to go around,
and this game is 12" of reach against 6" of move, so a diverted move is a whole turn of range
advantage.

**Expressible.** `live`.

**Where it lands.** As D-11.

**Already measured.** Not measured. ⚠ Note the counter-consideration and do not skip it: this
game is 20 rounds, not 5. Delay is worth much less here than in a five-round game, where a turn
bought is a fifth of the scoring. Price the delay, do not assume it.

**Cheapest test.** As D-11 — the same policy tests both.

### D-13 — The single-model denier is not available here

**Claim.** Control is a headcount, so one body at parity denies 5 VP — but a lone model
detached from its unit is out of coherency, earns nothing, and risks its unit's whole move.

**Why it is true here.** The collision is the entry. Control's threshold is per-model and
coherency's constraint is per-unit, and they disagree about the right unit of force.
`objective_hold.require_coherent` pays a model outside its unit's coherent body **nothing** —
not less, none — and under `enforce_move: revert_unit` at play, one model out of place cancels
the whole unit's move.

**Expressible.** `quantum`.

**Where it lands.** Nowhere, at five-model units. It reopens at smaller unit sizes — see D-06.

**Already measured.** The per-model coherency tail is 7.8% beyond the chain limit, and on a
five-model unit `1 − 0.922^5 = 0.32` against a measured 0.331 unit-veto rate — so an
all-or-nothing revert converts a small per-model tail into a **33% unit veto**. Detaching one
model deliberately walks into that.

**Cheapest test.** Do not test at this unit size.

---

## 7. D — Range, staging and terrain

| ID | Claim | Expressible | Lands in | Verdict |
|---|---|---|---|---|
| D-14 | Threat is move plus range, not range | `blind` | observation · scripted policy | `price` |
| D-15 | Stage outside threat range and arrive in one move | `live` | scripted policy | `price` |
| D-16 | Route through blocked sight, not down the shortest line | `live` | scripted policy | `price` |
| D-17 | Holding is hiding — an objective is cover | `live` | — | `settled` |
| D-18 | Do not build a cover-seeking behaviour | `live` | — | `refused` |

### D-14 — Threat is move plus range, not range

**Claim.** The ground an opponent unit threatens next turn is its move plus its weapon range —
18" here, not 12". Position against that number.

**Why it is true here.** `max_move_speed` is 6.0" and weapon range is 12" on the config that
trains, so a unit that ends a turn 17" from an enemy is inside its threat and a unit at 19" is
not. On the six `long_edges` layouts the armies start 20" apart across the short axis, which is
one inch outside threat from deployment.

**Expressible.** `blind`. Positions of opponent models are observed, and weapon range is a
per-model config field that is **not** in the observation — the model token carries the
firer's own combat stats, not the opponent's.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives`; as an observation, a widening
of the opponent model token.

**Already measured.** Not measured as a policy. The visualisation already exists: the `[R]`
threat-range overlay and `--record-threat-range`.

**Cheapest test.** **Look at a real board before writing anything.**
`just play configs/golden/25v25_maps_two_mode.yaml squad_march_take tabletop R`.

### D-15 — Stage outside threat range and arrive in one move

**Claim.** Hold a unit one move outside the threat band and cross it in a single move, rather
than walking into the band and standing there.

**Why it is true here.** Control is evaluated at the scoring moment (§ 3), not continuously,
so time spent inside the band before arriving buys nothing and costs shots. Movement is a
polar (angle x speed) choice per model, so a unit *can* hold at a chosen distance — but see
below.

**Expressible.** `live`.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives` and the movement step.

**Already measured.** ⚠ **The mirror image of this claim is already the agent's biggest
behavioural defect, so read it carefully.** The trained agent finishes with **52.9% of its army
alive against the scripts' 27.4–30.9% while holding fewer objectives** — it is already too
willing to wait. Staging is a *timing* claim, not a *caution* claim, and any implementation that
cannot tell those apart will make hoarding worse. `held` is the reject signal, not `alive`.

**Cheapest test.** Only as a compound with D-26, which supplies the arrival deadline that makes
staging different from waiting.

### D-16 — Route through blocked sight, not down the shortest line

**Claim.** The path between two points is chosen for what can see it, not for its length.

**Why it is true here.** Only terrain blocks sight, so a route is either exposed or it is not,
and the boards carry 16 pieces each. Measured: the models that die are the ones walking between
points, not the ones standing on them (D-17).

**Expressible.** `live` — terrain outlines are in the observation
(`WargameTerrainObservation.outline`, padded vertices plus a real-vertex count).

**Where it lands.** `ScriptedSquadMarchPolicy` movement, and `domain/los.py` for the sight test.

**Already measured.** ⚠ Attractive and expensive, and the record is against it. The agent
**does not use terrain for cover; it manages range** — established by deleting all terrain
(exposure 0.116 → 0.120) and by doubling weapon range (win rate collapsed to 6.8%), then
reconfirmed with 19.8% of the board hidden, a per-model sight input and priced losses, which
left exposure at 0.092–0.110 across every arm.

**Cheapest test.** Scripted only, and only after D-11/D-12 have been priced. A learned form of
this has already failed twice.

### D-17 — Holding is hiding

**Claim.** Standing on an objective is not a risk taken for reward. It is the safest place on
the board *and* the paying one.

**Why it is true here.** Every one of the 270 markers in `configs/evaluation/maps/` sits inside
a terrain piece, and the environment's see-out/see-into rule exempts the whole footprint of a
piece — so a model on an objective is in cover.

**Expressible.** `live`.

**Where it lands.** Nowhere — it is a fact to reason from, not a behaviour to build.

**Already measured.** `settled`, 2026-08-22. `just measure-hold-hazard` prices the trade per
model-step: standing on an objective pays **+0.37 to +0.44** more, and its excess death hazard
is **negative in 5 of 5 policies** (−0.13% to −1.43%) against a break-even of +3.4% to +6.0%.
**"Hiding is correct play" is refuted** — the agent is leaving return on the table, and the
exposed models are the ones in transit.

**Cheapest test.** Do not test.

### D-18 — Do not build a cover-seeking behaviour

**Claim.** Terrain-seeking as an end in itself is a dead lever here.

**Why it is true here.** D-17: the cover that matters is already where the reward is. A policy
paid to find cover finds cover it is not paid to stand in.

**Expressible.** `live`.

**Where it lands.** Nowhere.

**Already measured.** `refused`. Two rounds of terrain and cover work
([terrain](../reports/2026-08-05-stochastic-terrain-and-cover.md),
[cover](../reports/2026-08-06-cover-signal-reason-geometry.md) — read that one's corrections
before reusing it) left exposure flat across every arm. `observe_threat_count` was a null and
has been removed.

**Cheapest test.** Do not test.

---

## 8. E — Trading, bait and anchors

The group with the fewest survivors. Contact is where most of expert play's trading vocabulary
lives, and there is no contact here.

| ID | Claim | Expressible | Lands in | Verdict |
|---|---|---|---|---|
| D-19 | Build a unit too hard to shift and anchor the game on it | `live` | — | `refused` |
| D-20 | Trade cheap units for expensive ones | `absent` | — | `parked` |
| D-21 | Offer bait and punish the response | `absent` | — | `parked` |
| D-22 | Concede a point rather than lose a unit denying one round of it | `partial` | scripted policy | `price` |

### D-19 — Build a unit too hard to shift and anchor the game on it

**Claim.** *(rejected)* Make one body of models so durable and so well-placed that the opponent
must route around it, and let it hold ground for the whole game.

**Why it is true elsewhere and false here.** ⚠ **This heuristic is already this project's bug.**
An unshiftable body sitting in cover, refusing to move, is exactly what the agent does: 4.90
models on its top point, 52.9% of the army alive against the scripts' 27–31%, and **less ground
held**. Importing this claim would be importing hoarding under a flattering name.

**Expressible.** `live` — and that is the problem.

**Where it lands.** Nowhere. Recorded so it is not nominated.

**Already measured.** `refused` on the whole record: the hoarding finding (2026-08-22), the
allocation gap (2.08–2.30 distinct objectives against a script's 3.28), and the critic probe
(2026-08-23) showing the value function *already* prefers the surplus unit to leave, t = +8.3
over 6 of 6 seed-round cells.

**Cheapest test.** Do not test.

### D-20 — Trade cheap units for expensive ones

**Claim.** *(parked)* Spend low-value models to remove high-value ones.

**Why it does not apply here.** There is no fight phase
([gap map § Charge and fight](rules/implementation-status.md)), and every model on the board
carries the **same profile** — one weapon, one range, one set of stats — so no trade is up or
down. ⚠ Mixed profiles were tried and are a **measured null**: the first arm fired 45 shots a
round against the control's 25 and simply measured its own lethality; held at exactly 25 shots,
roles reproduce the control's paired difference to one decimal. It was the unit count, never the
guns.

**Expressible.** `absent`.

**Cheapest test.** Blocked. Reopens only with the fight phase or genuinely differentiated
profiles.

### D-21 — Offer bait and punish the response

**Claim.** *(parked)* Place a unit somewhere tempting but survivable, and punish whatever comes
for it.

**Why it does not apply here.** Bait needs a fall-back move to be bait rather than a casualty
([gap map § Moving](rules/implementation-status.md): fall-back move, `absent`), and it needs an
opponent that responds. The scripted opponents do not: `scripted_advance_to_objective` parks
permanently once it arrives.

**Expressible.** `absent`.

**Cheapest test.** Blocked on the fall-back move and on a reactive opponent — self-play, roadmap
Phase 4.

### D-22 — Concede a point rather than lose a unit denying one round of it

**Claim.** Denial is worth 5 VP per scoring moment. A unit lost is worth every VP it would have
earned for the rest of the game. Do not trade the second for the first.

**Why it is partial here.** The arithmetic holds — but with ~19 scoring rounds a side, denial
repeats, and a unit that survives to deny for ten rounds is worth 50. The trade is much more
favourable to denial here than in a five-round game, and this is the one place where the long
episode makes an imported heuristic *stronger* rather than void.

**Expressible.** `partial`. Withdrawal exists only as an ordinary move — there is no fall-back
move — but nothing here is in contact, so an ordinary move is a withdrawal.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives`.

**Already measured.** Not measured. ⚠ Related and cautionary: the agent already over-preserves
(D-15, D-19). Any implementation must be gated on `held`, not on `alive`.

**Cheapest test.** `just measure-hold-hazard` and `just measure-objective-split` first — both
already report the per-point cost of holding and what re-allocation could buy.

---

## 9. F — Fire discipline

| ID | Claim | Expressible | Lands in | Verdict |
|---|---|---|---|---|
| D-23 | Shoot what scores, not what threatens | `live` | scripted policy | `price` |
| D-24 | Concentrate until a count changes | `live` | scripted policy · decode | `act` |
| D-25 | Order your firing units by how constrained they are | `live` | scripted policy | `refused` |

### D-23 — Shoot what scores, not what threatens

**Claim.** Target selection is a mission decision, not a threat decision. A unit standing on an
objective is earning points every scoring moment; a unit in the open is not.

**Why it is true here.** VP accrues from control, and control is a headcount, so removing a
model from an objective is worth up to 5 VP per remaining scoring moment while removing one in
transit is worth its future contribution only. With ~19 scoring rounds a side, the difference
compounds far more than it would in a short game.

**Expressible.** `live`. Shooting actions are indexed by enemy `group_id`
(`docs/shooting.md`), and `observe_objective_control` supplies per-objective counts for both
sides, so a policy can see which units are on points.

**Where it lands.** `BaselinePolicy.select_shooting` — a one-file change. Every shooting
baseline currently fires at the **nearest** valid enemy unit.

**Already measured.** Not measured in this form. ⚠ Its nearest measured relative is a null:
weakest-valid-unit targeting over nearest-first is **+1.7 ± 5.7 paired, t = 0.30, ahead in 24 of
100** (`reports/2026-08-12-the-physics-were-off-and-the-agent-beat-the-bar.md`) — down from a
+8.0 first reading that did not pair. So the *ordering* family is a null; whether a
**mission-keyed** criterion behaves differently is genuinely open, because it selects on a
different quantity.

**Cheapest test.** `just measure-paired <new> squad_march_shoot configs/golden/25v25_maps_two_mode.yaml 100`.
State the ceiling in the result: see D-25.

### D-24 — Concentrate until a count changes

**Claim.** Fire is worth something only when it moves an objective's headcount across the
threshold. Damage spread over four units that each keep a model on a point has flipped nothing.

**Why it is true here.** Control is `player_count > opponent_count` over **alive** models, and
a unit reduced to one model still holds. So the value of a volley is a step function of where
it lands, not a linear function of how much it removes. Taking an opponent from 2 to 1 on a
point where you stand with 1 turns their 5 VP into nobody's; taking them from 2 to 0 turns it
into yours; the same casualties spread over five points change no label at all.

**Expressible.** `live`.

**Where it lands.** `BaselinePolicy.select_shooting`, and as a follow-on the joint decode in
`model/common/decoding.py`, which already reasons over combinations of per-model choices.

**Already measured.** Not measured. ⚠ The known constraint is the opposite one and it is
already fixed: attack discarding used to punish concentration hard — 36–40% of a concentrating
unit's volley evaporated when a weapon named a model and shots at an already-dead one silently
vanished. Since the fix, **an attack is discarded only when the whole target unit is destroyed,
measured at 3.6% of declared shots** (`docs/shooting.md`). So concentration is now nearly free
and nothing has tried it.

**Cheapest test.** `just measure-paired <new> squad_march_shoot configs/golden/25v25_maps_two_mode.yaml 100`.

### D-25 — Order your firing units by how constrained they are

**Claim.** *(rejected)* Fire first with the units that have fewest legal targets, so flexible
units can be redirected if a target dies early.

**Why it is capped here.** The entire value of firing order is bounded by how many attacks are
wasted, and **only 3.6% of declared attacks are discarded** — a whole-unit-destroyed condition
(`docs/shooting.md`). Perfect ordering recovers at most that, and the closest measured proxy for
smarter target choice recovered **+1.7 ± 5.7** of it.

**Expressible.** `live`.

**Where it lands.** Nowhere. Recorded so the ceiling is quotable.

**Already measured.** `refused`. See the two numbers above.

**Cheapest test.** Do not test.

---

## 10. G — Tempo and the scoring moment

| ID | Claim | Expressible | Lands in | Verdict |
|---|---|---|---|---|
| D-26 | Contest before the opponent's scoring moment, not before your own | `live` | scripted policy · decode | `act` |
| D-27 | Risk tolerance is a function of the margin | `live` | scripted policy | `price` |
| D-28 | Early rounds buy position, late rounds bank points | `live` | — | `refused` (does not transfer) |
| D-29 | Force a reaction | `absent` | — | `parked` |

### D-26 — Contest before the opponent's scoring moment, not before your own

**Claim.** Your holdings must survive the opponent's turn, and your denial must be in place at
the **end of your own** turn. Those are two different deadlines and they alternate.

**Why it is true here.** VP is computed in `wargame.py::_on_before_advance` when a side leaves
its **command** phase — before that side moves — and it fires even when `command` is listed in
`skip_phases`, because `domain/turn_execution.py` calls the hook on skipped phases precisely so
scoring survives. Two consequences, both sharp:

- Your own score is settled by the board **your opponent left you**. Arriving on a point during
  your own movement phase scores nothing this round.
- To deny, you must be standing on their ground when *their* command boundary fires — that is,
  at the end of *your* turn, not the start of it.

With `turn_order: random` the deadline alternates, so the timing is a state-dependent decision
rather than a fixed offset.

**Expressible.** `live`. `battle_round` and `battle_phase_index` are both in the game token, so
the agent can see where in the cycle it is.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives` for arrival deadlines;
`model/common/decoding.py` if it becomes a decode-time preference.

**Already measured.** **Not measured, and encoded by nothing.** Every scripted policy here
allocates against the current board with no reference to the scoring clock. This is the
cheapest untested claim in the document.

**Cheapest test.** `just measure-paired <new> squad_march_take configs/golden/25v25_maps_two_mode.yaml 100`.
⚠ Read `held` alongside `vp_margin`: `held` is an end-state snapshot and cannot see *which*
points were paid, so a timing lever can move VP with `held` flat — `share_soft` gained +19.3 vp
with `held` unchanged.

### D-27 — Risk tolerance is a function of the margin

**Claim.** Behind on VP, prefer the higher-variance line; ahead, prefer the lower-variance one.
A losing position is not improved by a safe move.

**Why it is true here.** The episode ends on a comparison, not on a total, so variance is worth
whatever it costs only when the mean is losing.

**Expressible.** `live`. `player_vp`, `opponent_vp` and `player_vp_delta` are all in the game
token, so the agent can already condition on margin.

**Where it lands.** `ScriptedSquadMarchPolicy.squad_objectives`, gated on the margin.

**Already measured.** **Not measured.** ⚠ The relevant caution: the outcome noise floor here is
large — `vp_margin` sd is 50.6 *within* a layout against 45.0 *between* layouts, so the dice
already contribute more spread than the scenario does. A deliberate variance lever is being
added on top of that, and n=100 is the minimum that can see it.

**Cheapest test.** `just measure-paired <new> squad_march_take configs/golden/25v25_maps_two_mode.yaml 100`
at n=100, and quote the sign count across tables as well as the t.

### D-28 — Early rounds buy position, late rounds bank points

**Claim.** *(does not transfer)* Accept a low-scoring early round to set up a high-scoring late
one; the last round is worth a disproportionate share of the game.

**Why it fails here.** The config that trains runs **20 battle rounds**, so there are ~19
scoring rounds a side and **no single round is worth more than about 5% of the game**. The
heuristic is an artefact of a five-round game where the final round is a fifth of the scoring
and going second controls it. Here, deferring income is simply losing income.

⚠ This is the largest single transfer failure in this document, and it invalidates a whole
family of imported reasoning: going-second advantage, the staged decisive turn, last-round
objective swaps, and conservative early scoring. **None of them mean anything at 19 scoring
rounds.** They would all reactivate under a five-round scenario, which is why the round count
is a first-class scenario decision rather than a detail.

**Expressible.** `live` — the environment can be configured to a five-round game. That is a
scenario change that voids every baseline on the config.

**Where it lands.** `number_of_battle_rounds`.

**Already measured.** Not as an arm. The gap-map records `number_of_battle_rounds` default 100
against the rules' 5 as a **deliberate divergence**.

**Cheapest test.** `just measure-baselines configs/golden/25v25_maps_two_mode.yaml 100 "" 700000 rounds=5`
— every `measure-*` recipe takes trailing scenario overrides, so the five-round game can be
scored without copying a config. Do it before writing any timing heuristic that assumes a short
game.

### D-29 — Force a reaction

**Claim.** *(parked)* Present a threat the opponent must answer, and take the ground they leave
to answer it.

**Why it does not apply here.** Nothing reacts. `scripted_advance_to_objective` returns `STAY`
once inside an objective radius and does not leave; `scripted_advance_and_shoot` picks its
target uniformly and never revises where a unit is going. A heuristic whose whole mechanism is
the opponent's response cannot be measured against an opponent that has none.

**Expressible.** `absent`.

**Cheapest test.** Blocked on a reactive opponent — self-play, roadmap Phase 4.

---

## 11. H — Force structure and mission shape

These are levers on the **scenario**, not on the policy. They are in this document because the
strongest heuristics in the corpus turn out to be statements about what the mission asks, and
because the record shows the policy-side levers are exhausted where these are not. **D-06 belongs
here too** — unit size is the third scenario lever, and
`configs/experiments/24v24_maps_spare_squads.yaml` is the golden config with only that
changed.

| ID | Claim | Expressible | Lands in | Verdict |
|---|---|---|---|---|
| D-30 | More units than objectives asks no allocation question | `live` | scenario | `settled` |
| D-31 | If you want spread, pay for spread | `live` | mission | `act` |

### D-30 — More units than objectives asks no allocation question

**Claim.** A mission where every unit has an objective to stand on tests marching, not
allocation. To test allocation the force must have to choose.

**Why it is true here.** With five units against five or six markers, `squad_march_take` and
`squad_march_deny` differ only in what a spare unit does — and there is no spare unit.

**Expressible.** `live`.

**Where it lands.** `max_groups` and `models[].group_id`.

**Already measured.** `settled`, 2026-08-22. Paired at n=100 the take-vs-deny difference across
three layout sets is **+5.7 / −9.2 / +9.8 — it changes sign**, mean +2.1, with 25–31 of 100
episodes *identical*. Eight units of three: **+16.0, positive on 3 of 3**, 2–5 identical.
⚠ And trained there, **offence still did not move**. So the scenario was a real defect and
fixing it was not sufficient.

**Cheapest test.** Answered. Do not re-run.

### D-31 — If you want spread, pay for spread

**Claim.** The mission should pay for **distinct points held** and for presence across regions,
not only for a count of controlled objectives. A policy that stacks is optimising the mission it
was given.

**Why it is true here.** `min(15, controlled x 5)` is indifferent between three points held by
five models each and three held by two each, so nothing in the mission distinguishes a spread
army from a stacked one. Every attempt to add that distinction has gone through the **reward**,
which is the pipeline with four consecutive empty results — and the critic probe showed the
value function already prefers spreading, so the signal is not what is missing.

**Expressible.** `live` and **cheap**: distinct-objective counts and region-over-objectives are
**Tier 0** in [docs/missions-design.md](missions-design.md) — no new state, no tensor change.
Region membership as an agent-visible *feature* is Tier 1 and does cost an objective-token
widening; buy one widening for all such features at once.

**Where it lands.** `envs/mission/` — the `Selector x Measure x Payout` build sequence in
`docs/missions-design.md`, which is designed and unbuilt.

**Already measured.** Not measured; nothing in the repo declares a `mission:` block, so all 115
configs run `DefaultVPCalculator`. ⚠ The mission design carries **two pre-registered
rejections** that must be applied before anything trains: a new mission whose policy ranking
correlates at ≥ 0.95 with the current one is rejected, and one where the best script already
scores above 90% of ceiling is rejected.

**Cheapest test.** Score the existing baselines under the candidate mission with
`just measure-baselines` before training anything, and apply both rejections to that table.

---

## 12. J — The move type

The group the environment gained most recently, and the one every coming mechanic
lands in: fall back and charge are both move types, and both will arrive through
whatever seam Advance opened. ⚠ **These are claims about how to play a move type,
not about how to encode one.** The encoding defects are a gap-map matter and live
in [implementation-status.md](rules/implementation-status.md#advance-move).

| ID | Claim | Expressible | Lands in | Verdict |
|---|---|---|---|---|
| D-38 | A unit declares one move type, and the whole unit pays for it | `live` | action space | `settled` |
| D-39 | Advance only when you would have had no shot | `live` | scripted policy | `refused` |
| D-40 | Advance to arrive, not to approach | `live` | scripted policy | `settled` |
| D-41 | The longest move is the most likely to be stopped | `live` | — | `settled` |
| D-42 | An advance is worth what one turn is worth, so it is a short-game move | `live` | scenario | `settled` |
| D-43 | A move type is a lever, not an advantage — do not build a policy that must use one | `live` | — | `settled` |

### D-38 — A unit declares one move type, and the whole unit pays for it

**Claim.** The move type is chosen for the unit. Every model in it moves under that
declaration and every model in it bears the cost.

**Why it is true here.** [Rules § Movement phase](rules/09-movement-phase.md) makes the
move type a unit property, and this action space makes it a per-model choice, resolved
upward afterwards: `ActionHandler._mark_advancing_units` marks every model of any group
in which *one* model chose an advance, so that one model forfeits all five models'
shooting. With 48 of 150 actions in the advance slice, a near-uniform policy triggers
that on `1 - (1 - 0.32)^5` = **85.5%** of five-model unit-turns.

**Expressible.** `live` since 2026-08-23. It was `partial` — enforced by an OR over five
per-model movement actions after the fact — until the declaration was split into its own
phase.

**Where it lands.** `ActionRegistry` · `ActionHandler.apply`.

**Already measured.** `settled`, 2026-08-23, and it **does not bind at convergence**.
Trained policies choose unanimously on **64–81%** of advancing unit-turns and one model
drags four on only **7–11%**, at both `decode_topk` 1 and 3 — so the joint decoder is not
manufacturing the agreement and the policy learned the unit-level structure without being
given it. ⚠ What this does **not** measure is what the same defect costs during
*exploration*, which is where 85.5% applies.

**BUILT 2026-08-23, and the shape of it is the lesson.** The declaration is a `move_type`
slice of two actions in the **command phase**, resolved from the unit's lowest-indexed alive
model. ⚠ The obvious cheaper version — mask the advance rungs for everyone but the leader —
**shatters formation**: a move type and a displacement were the *same action*, so a
leader-only advance caps every other model at `M`, and the scripts advance **5-of-5 with a
within-unit spread of 0.00"** against a 2" chain. Separating the declaration from the
distance is what makes a unit decision safe. Adding fall back or charge now costs **one value
in `move_type`**, not another 48-action slice.

**Cheapest test.** `just measure-advance-use <ckpt> <config> 10 1`.

### D-39 — Advance only when you would have had no shot

**Claim.** An advance costs the unit its entire turn of fire. Take it only when that
fire was worth nothing — when a normal move would have left nothing in range anyway.

**Why it is true here.** `advanced_this_turn` gates both shooting masks and no weapon
here has the ability that would permit firing after an advance, so the forfeit is total.
`model_kills` credits the model that actually fired, so the cost lands per model on the
models that pay it — the trade is priced, and a policy that ignores it is ignoring
something it is charged for.

**Expressible.** `live`.

**Where it lands.** `ScriptedSquadMarchPolicy.advance_when_no_shot`, shipped 2026-08-23 as
the baseline `squad_march_take_advance`. Range only, deliberately: sight can only *remove*
shots, so "nothing in range" is a sufficient condition for "nothing forfeited", and erring
that way declines a few free advances rather than spending a real shot.

**Already measured.** `refused`, 2026-08-23,
[report](../reports/2026-08-23-three-prices-for-the-advance-move.md). Pricing the shooting
is **necessary and not sufficient**: `squad_march_take_advance` is **−18.4 vp paired, 0 of 3
seed bases** against plain `squad_march_take` (−35.0 / −6.3 / −13.8). The heuristic that
prices *nothing* — "run while far, walk once close" — costs its user about **78 vp** in the
2×2, so this is an improvement of 60 vp on a losing move. ⚠ And the mechanism proposed for
the failure (D-14: it ends inside their reach) is **refuted** — advancing moves end inside an
enemy's reach on 4.1% of model-moves against walking's 22.4%. What it costs is whole-episode:
exposure +10.8%, firepower 1.091 → 1.004, `alive` 0.396 → 0.349.

**Cheapest test.** `just measure-paired squad_march_take_advance squad_march_take
configs/experiments/25v25_maps_advance.yaml 100`, on three seed bases.

### D-40 — Advance to arrive, not to approach

**Claim.** Intermediate position buys nothing. The advance that pays is the one that turns
a two-turn approach into a one-turn arrival.

**Why it is true here.** § 3: VP is scored at each side's command boundary, on control,
which is a headcount at that instant. Being three inches closer at the boundary scores
exactly what being nine inches closer scores — nothing. So the extra `D6` is worth its
price only when it crosses the last gap.

**Expressible.** `live`.

**Where it lands.** The squad's advance predicate, beside D-39's clause rather than
instead of it.

**Already measured.** `settled`, 2026-08-23,
[report](../reports/2026-08-23-three-prices-for-the-advance-move.md). The claim is **right and
insufficient**: `squad_march_take_arrive` adds the arrival clause on top of D-39's and more
than halves the loss, **−18.4 → −11.9 vp paired** while firing on 2.2% of unit-turns instead
of 11.2% — but it is still **0 of 3 seed bases** at twenty rounds. See D-42 for why. Also
2026-08-23: only **1.8–4.0%** of a trained policy's advances start and end inside the same
objective, and **17–20%** start inside one and *leave*, which is reallocation. ⚠ Counting
"advances from inside an objective" alone conflates those and reads as waste; the first
version of that statistic did exactly that and was corrected before publication.

**Cheapest test.** `just measure-paired squad_march_take_arrive squad_march_take
configs/experiments/25v25_maps_advance.yaml 100`, on three seed bases.

### D-41 — The longest move is the most likely to be stopped

**Claim.** Price an advance against the chance it is blocked, not only against the
shooting it spends.

**Why it is true here.** Friendly bases may be crossed but not ended on, and for a
deterministic policy freezing is an absorbing state at **+0.86**. An advance is the longest
move in the game and therefore the most exposed to that.

**Expressible.** `live`.

**Where it lands.** Nowhere. A fact to price with — § 14 refuses a fourth movement-side fix
for freezing, and "fix freezing" reduces to "fix allocation".

**Already measured.** `settled`. The advance arm freezes 18–28% and delivers 70–77% of
ordered inches — but the **non-advancing control agent freezes 26.3% and delivers 76.4%**,
so trained agents freeze at that rate because they stack, advance or not. ⚠ The comparison
that made the advance look guilty was against the *scripts* (11%), which was never the
right control. Within-unit distance spread carries the same trap: it is p90 **4–6"** on
advancing unit-turns against a 2" chain, and **the same on walking unit-turns of the same
policy**.

**Cheapest test.** `just measure-freezing`, always beside a non-advancing control of the
same kind — never beside a script.

---

### D-42 — An advance is worth what one turn is worth, so it is a short-game move

**Claim.** What an advance buys is arriving one turn earlier. That is worth a *fraction of
the game*, so its value falls as the game lengthens — and on a twenty-round clock it is
negative.

**Why it is true here.** § 3: VP is scored at each side's command boundary, roughly nineteen
times a side over `number_of_battle_rounds: 20`. Arriving on round 3 instead of round 4 adds
one scoring event out of nineteen — about 5% — against a whole turn of five models' fire
forfeited and an extra turn spent standing forward. At five rounds the same arrival is one
event of four, about 25%, and the arithmetic reverses.

**Expressible.** `live`, and it is a **scenario** property rather than a policy one. Every
`measure-*` recipe takes `rounds=` as a trailing override, so it costs one command.

**Where it lands.** `number_of_battle_rounds`, not any policy. ⚠ Nothing in a movement rule
can fix a move whose gain the clock has priced at zero.

**Already measured.** `settled`, 2026-08-23,
[report](../reports/2026-08-23-three-prices-for-the-advance-move.md). Pre-registered before
the numbers existed. `squad_march_take_arrive` against `squad_march_take`, n=100, three seed
bases, **positive means plain walking wins**: rounds 5 → **−1.7 (3 of 3 to advancing)**,
rounds 10 → +1.3, rounds 20 → **+11.9 (0 of 3)**. Monotone. ⚠ **Absolute vp are not
comparable across horizons** — the five-round outcome sd is 12 against twenty's 91 — so read
it normalised: **+0.14 sd → −0.04 → −0.13**. The five-round game is **not degenerate**:
`hold_deployment` scores −33.1 with `held` 0.79 against the marcher's −0.7 and `held` 2.50.

**Cheapest test.** `just measure-paired squad_march_take_arrive squad_march_take
configs/experiments/25v25_maps_advance.yaml 100 700000 rounds=5`.

---

### D-43 — A move type is a lever, not an advantage; do not build a policy that must use one

**Claim.** Advance is an option with a real trade — further, but no shooting. Whether it is
worth taking is a **situational** judgement, and a policy built to advance is a policy built
to be wrong most of the time. What the environment owes the agent is the *availability* of the
lever; what the agent does with it is learned behaviour.

**Why it is true here.** Its use is niche — it pays when you must reach a specific destination
now — and the arithmetic that decides "now" is the scoring clock (D-42), which at twenty rounds
prices one turn at ~5%. ⚠ And the game is **incomplete**: with no melee, a shooting army has no
reason to close except to capture, so the whole class of reasons a real player advances is
absent. Anything measured about advance today is provisional on a mechanic that does not exist.

**Expressible.** `live`.

**Where it lands.** Nowhere — it is a rule about what NOT to build. ⚠ Specifically: **do not
add scripted policies whose purpose is to advance.** Getting one right is harder than it looks,
and a bar that advances badly is worse than a bar that walks; the record has both, at −78 and
−11.9 vp to their own users.

**Already measured.** `settled`, 2026-08-23,
[report](../reports/2026-08-23-three-prices-for-the-advance-move.md). Three scripted rules,
each pricing more of the trade than the last, all lose to plain walking: **−78**, **−18.4
(0/3)**, **−11.9 (0/3)**. The loss shrinks as usage shrinks and never turns positive, which is
the signature of a move whose value is negative wherever it is *forced*.

⚠ **This retires a gate that was mis-specified.** The standing rule read "a scripted advance
rule that prices the forfeited shooting has to beat `squad_march_take` before anything trains".
That bakes in the assumption above — that advance ought to pay — and so it can never be
satisfied by a correct implementation. The right question is not *does the lever pay* but
**does carrying it cost the agent anything**, which needs no advance-seeking script at all:
train the arm against a `dark_action_slices` control of identical shape and read the paired
difference.

**Cheapest test.** `just measure-advance-use <ckpt> <config> 10 1` on a trained arm — usage
should be *low*, and low usage is not a failure.

---

## 13. I — Parked on absent mechanics

Recorded so they are not rediscovered, and so the wanted-list has one home. Each names the
[gap-map](rules/implementation-status.md) row that blocks it.

| ID | Claim | Blocked by |
|---|---|---|
| D-32 | Deny arrival zones with depth and spacing; keep a screen between the enemy and your rear | Reserves and arrival from off the board are `absent` and **deliberately out of scope** in the specification. |
| D-33 | Break a holder without killing it | Resolve, resolve rolls and all suppression are `absent`. Control is a raw headcount with no state that can turn it off. |
| D-34 | Bank a resource across rounds and spend it on the decisive one | The command-resource economy is **deliberately out of scope** in the specification. |
| D-35 | Give units roles and match each to the ground it is good at | Keywords, unit abilities and weapon abilities are all `absent`; every model carries one profile. ⚠ Mixed profiles were tried and are a measured null once shot count is held equal. |
| D-36 | Control information by choosing what to reveal and when | No alternating deployment; `hidden` and detection range are `absent`; the observation is full-information. |
| D-37 | Spend unit activations on mission work rather than on shooting | Tasks are `absent` and are explicitly refused in the v1 mission design (Tier 3). |

---

## 14. Backlog

**The standing rule: price every entry as a scripted policy before it becomes a reward term, an
observation, a mission or a training run.** A scripted arm costs one inference run and no GPU,
and it is the only way any claim here becomes evidence.

Pre-registration is not ceremony. `CLAUDE.md` records a lever that failed on its own accept
criterion because the criterion had been written for the failure the author was already worried
about rather than the one the lever risked. **Write the reject rule for the failure your change
actually risks**, and write it before you have a number.

### Act — two arms, both zero-GPU

| # | Entries | What it changes | Test | Accept | Reject |
|---|---|---|---|---|---|
| 1 | D-05 · D-07 · D-26 | A `squad_march_take` subclass that caps commitment at the threshold, ranks surplus targets by cost, and times arrival against the scoring clock | `just measure-paired <new> squad_march_take configs/golden/25v25_maps_two_mode.yaml 100` on three seed bases | paired difference > 0 with 3 of 3 seed bases positive, and `held` not lower | difference ≤ 0, **or** `held` falls, **or** `alive` rises with `held` flat |
| 2 | D-11 · D-12 | A policy that places one unit on the interval between an opponent unit and the point it is walking to | as above, **plus** `just measure-freezing` on the new policy and on `squad_march_take` | paired difference > 0 with 3 of 3 positive, and `absorbing` no worse than the control's +0.86 | difference ≤ 0, **or** the screen's own freeze rate exceeds the control's 11% |

Arm 1 aims at the one standing failure, at the place the record says the lever must go: the
**policy**, not the reward. Its refused variant is already named (D-10), so it cannot repeat
`assignment_optimal`'s mistake, and its blunt variants are already named (D-08), so it cannot
repeat those either. What it varies that nothing has varied is **timing**.

Arm 2 is the only implemented mechanic in this document that nothing has ever tested.

⚠ **Do not bundle arms 1 and 2.** They touch the same seam and a bundled change cannot be
attributed — `contest_and_spread` bundled three changes and its refutation cannot say which
one failed.

⚠ **Read `vp_margin` to decide and `held` to rank.** Never top-stack occupancy: the gradient out
of the stack is shallow (+3.85) and steep in (−11.52), so any lever moves occupancy a great deal
for a small score change.

### Price — read the existing diagnostics before building

| Entries | What already exists |
|---|---|
| D-01, D-04 | `just measure-objective-split` — per-objective `(player, opponent)` counts and a redistribution ceiling |
| D-09 | `just measure-shaping-gates` — the target-switching cost is already instrumented |
| D-14, D-15 | the `[R]` threat overlay and `--record-threat-range`; `just play <config> squad_march_take tabletop R` |
| D-20, D-23 | `just measure-matchups` — unit-versus-unit casualties, reach margin and free rounds |
| D-22 | `just measure-hold-hazard` — what holding a point earns against what it costs |
| D-23 | `just measure-income-share` — which calculator pays, and how much is global |
| D-27 | `just measure-noise-floor` — the dice already outspread the scenario; size the arm against that |
| D-28 | `just measure-baselines <config> 100 "" 700000 rounds=5` — score the short game before assuming it |

### Then — the mission lever

D-31 is the one lever that attacks stacking without touching the reward pipeline that has
failed four times, and its cheapest form is Tier 0 in `docs/missions-design.md` — no new state
and no tensor change. It must clear that document's two pre-registered rejections against a
`just measure-baselines` table **before** anything trains.

---

## 15. What this document may never propose

Stated as refusals so they survive a future editor. Each is paid for; the reasoning is in
`CLAUDE.md` and `reports/`.

- **A new reward term aimed at offence.** Three consecutive ones left offence flat or worse
  (−50.5, −42, −71.5). The diagnosis the evidence supports is that the per-model term prices
  only distance closed while the term that prices *outcome* is global and broadcast identically
  to every alive model — no candidacy gate reaches that.
- **Anything routed through `closest_objective_v2`.** Four nominations, four empty results.
- **Coherency enforcement as a training lever**, `repair` included. It is a referee for play: it
  supplies no gradient, because every reverted action produces the identical outcome and the
  policy gradient inside that set is exactly zero.
- **An anti-concentration lever that destroys total income** rather than redistributing it.
  `overstack_penalty_per_extra` and `objective_hold.surplus_value` both halved occupancy because
  the policy read them as "objectives pay less". `crowding_exponent` conserves the pot and works.
  ⚠ And the converse: a term with negative net income is **not** thereby a broken term —
  removing the overstack penalty measured **−12.2 ± 5.5**.
- **A fourth movement-side fix for freezing.** The tangential slide, bisection on travel and a
  descending scan have all been measured away; 75.5% of frozen model-steps have no legal shorter
  move along that heading at all. "Fix freezing" reduces to "fix allocation".
- **A behavioural diagnosis without a clone control.** If a clone of the *winning* policy scores
  near the *losing* one, the statistic is measuring the architecture, not skill. It costs one
  inference run and has already caught two published explanations here.
- **A training run for a claim that has not been priced as a scripted policy first.**

---

## 16. Related

- [docs/rules/](rules/README.md) — what is legal. It wins over this file on any conflict.
- [docs/rules/implementation-status.md](rules/implementation-status.md) — the per-rule gap map
  every `absent` and `parked` entry above links into.
- [docs/opponent-policies.md](opponent-policies.md) — the scripted policies, several of which
  are entries here already implemented. A new one should name the entry it encodes.
- [docs/missions-design.md](missions-design.md) — the unbuilt mission vocabulary D-31 needs.
- [docs/metrics.md](metrics.md) — what each measurement means and how to read it.
- [CLAUDE.md](../CLAUDE.md) and [reports/](../reports/README.md) — the record. Where they
  disagree with an entry, they win.
