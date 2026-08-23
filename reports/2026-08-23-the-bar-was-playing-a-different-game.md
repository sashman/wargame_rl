# The bar was playing a different game

**2026-08-23. No GPU.** Two core-rules changes that land together, because both
change what a legal move is and re-baselining twice would be waste.

## Why these landed together

The Advance move was implemented in #235 and treated as an *arm* — something to
accept or reject against a control without it. That framing was wrong. **Advance
is a core rule and it is staying**, so the question was never "is it better", it
is "how well does the agent play the game that includes it".

Reframing it exposed the real defect: **no scripted baseline and no opponent
policy could advance.** Every one of them predates the feature. So an advancing
agent was being scored against a walking bar, and trained against a walking
opponent.

That is this project's most expensive documented error — *measure the
configuration that ships, not an intermediate one* — and it had already been made
twice. A run had been launched under it and was killed before it cost anything.

## 1. The scripts can advance now

`ActionHandler.best_advance_toward` is the advance counterpart of
`best_action_toward`: nearest angle bin, then the largest advance bin whose
`fraction x (M + roll)` fits. It reads the unit's **own** D6 (`model.advance_roll`)
because the reach is not fixed.

`ScriptedSquadMarchPolicy` advances when a normal full move cannot reach its
objective — **run while far, walk once close** — decided per *squad*, matching how
the env resolves it: one roll per unit, and `advanced_this_turn` marked for every
model of any group that advanced.

`advance_when_out_of_reach = False` reproduces the pre-Advance policy action for
action, so the two are directly comparable on one config.

### What it is worth

Held-out nine, n=10, K=1, same policy code both columns — "off" is the darkened
config, where the slice exists but is masked.

| policy | advance OFF | advance ON | delta |
|---|---|---|---|
| `squad_march` | −165.2 ± 5.7 | −139.7 ± 7.6 | **+25.5** |
| `squad_march_shoot` | −50.7 ± 10.0 | −18.1 ± 13.0 | **+32.6** |
| `squad_march_take` | −12.7 ± 13.1 | **+2.8 ± 4.4** | **+15.5** |
| `squad_march_deny` | −3.7 ± 6.7 | −2.4 ± 9.9 | +1.3 |

**Positive on 4 of 4.** `held` rises 2.46 → 2.82 for `take` and 2.09 → 2.68 for
`shoot`.

⚠ **`squad_march_shoot` gains MOST (+32.6) despite giving up its shooting while
advancing.** The extra distance is worth more than the fire it forgoes at range —
which is the opposite of what the "advance costs you shooting" intuition predicts,
and worth knowing before anyone tunes the trade.

## 2. A move must end unengaged

`09-movement-phase.md` requires a unit to be unengaged **after** moving.
`03-moving.md` is explicit that only the endpoint counts: *"Passing through an
enemy unit's engagement range during a move does not make the moving unit
engaged."*

`back_off_to_unengaged` resolves the move normally — enemies still block at their
true radius — then pulls the endpoint back **along its own heading** until it is
outside every enemy ring, and returns the start when no legal point exists (the
rules' own remedy).

⚠ **The legal set along the ray is NOT an interval**, because a ray can leave one
ring and enter another. It walks back interval by interval rather than bisecting —
the same mistake the movement solver's first rewrite made and was reverted for.

⚠ **The first attempt at this rule was a PATH constraint and was reverted.**
Inflating enemy blockers by the engagement range makes a 2"-thick impassable wall:
review measured **87% of opponent-held objectives with no legal spot at all**.
Passing through is legal; only ending inside is not. Both halves are pinned by
tests.

### What it is worth

Same policy, same seeds, rule on against off:

| | player model-steps engaged after a movement phase |
|---|---|
| rule OFF (today) | **6.01%** |
| rule ON (the fix) | **3.21%** |

**47% of it removed, 2.80pp.** Calibration: the 2026-08-19 corpse bug caused
7.94pp of spurious suppression and fixing it was worth **+7.0 vp**.

The residual 3.21% is legitimate — a model with no legal endpoint stays where it
is, and deployment can place one engaged. Disengaging from contact is a Fall Back
move, which is not implemented.

**The back-off fires on 1.52% of resolved moves, mean pull-back 1.05".** So it is
a targeted correction, not a movement tax.

## ⚠ What this voids

**Every scripted-bar number measured on a config with `n_advance_speed_bins > 0`,
and every agent score compared against one.** The bar moved by +1.3 to +32.6 vp,
and the movement rule changed on every config.

Three golden files were regenerated deliberately:
`reward_golden_25v25_single_phase`, `reward_golden_4v4_two_phases` and
`observation_golden_25v25_single_phase`. The other three are byte-identical, which
is the check that the movement change is targeted rather than global.

## 3. The opponent advances too

⚠ **Correction to an earlier draft of this report.** It said the opponent could not
advance. That was true of the two hand-written opponent policies, and **false of the
measurements above**: the advance configs set `opponent_policy: scripted_baseline`
wrapping `squad_march_take`, so the opponent inherited Advance from
`ScriptedSquadMarchPolicy` in the same change. Both columns of the table are
therefore symmetric — off is neither side advancing, on is both.

`scripted_advance_to_objective` (and `scripted_advance_and_shoot`, which delegates
to it for movement) now advance under the same rule, decided per **unit** — from
the unit's **centroid**, matching the player-side baselines. Measured from the
nearest member instead, it almost never fires: the opponent deploys 3-12" from its
objectives at Move 6.

`advance_when_out_of_reach = False` gives the walking opponent back.

## 4. The opponent's advance columns are zeroed

Each side rolls at the start of its **own** turn, so the opponent's `advance_roll`
and `advanced_this_turn` are zero in round 1 and **one turn stale** thereafter --
they record what it rolled and did on its *last* turn. A stale column is worse than
no column: the network has to learn to ignore it.

⚠ **Issue #237 proposed dropping the two columns and that does not work.** The
player and opponent tokens share a feature width (`model/common/observation.py`
asserts it), so removing two columns from one side alone fails at the tensor.
They are **zeroed** instead. A constant-zero column contributes nothing through the
embedding, so this is informationally identical to dropping it -- and unlike
dropping it, costs no shape change and orphans no checkpoint. Closes #237 by a
different route than it proposed.

## What is NOT done
- **Fall Back** is not implemented, so an engaged model cannot disengage.
- Issue #237 (the opponent's advance columns are stale by one turn) is still open
  and belongs in this same batch.

---

# CORRECTION, 2026-08-23 (same day, after an audit panel)

**Two of the four bar rows were wrong in SIGN, the engagement figure was wrong in the
other direction, and the batch shipped a movement bug.** Two audit panels were pointed
at this report and told to break it. All three findings were reproduced here
independently before being accepted.

## 1. ⚠ The bar table is RETRACTED. Advance HURTS the two strongest scripts.

The published deltas came from n=10 with **no error bar on the delta at all** — each
column's own across-map spread was quoted instead, and `measure_maps` never computes a
paired difference even though `baseline/evaluate.py::paired_difference` exists. Redone
at n=30, paired across the nine held-out tables:

| policy | published (n=10) | **n=30 paired** | t | per-table sign |
|---|---|---|---|---|
| `squad_march` | +25.5 | **+21.1 +/- 5.2** | 4.06 | 8/9 |
| `squad_march_shoot` | +32.6 | **+23.7 +/- 7.4** | 3.22 | 7/9 |
| `squad_march_take` | +15.5 | **-6.5 +/- 8.0** | -0.80 | 4/9 |
| `squad_march_deny` | +1.3 | **-20.0 +/- 7.1** | **-2.82** | **1/9** |

The audit panel reached -18.7 (t=-4.65, 1/9) for `deny` by its own implementation.

**"Worth +1.3 to +32.6, positive on 4 of 4" is false.** The honest statement is that the
naive out-of-reach heuristic helps the movement-only policies, for which advancing is
free because they never shoot, and **costs the allocation-aware ones about 6-20 vp** —
including `squad_march_take`, which *is* the bar.

⚠ **This does not mean Advance is bad. It means THIS HEURISTIC is bad**, and a bar should
use the best scripted play available. Until a better rule exists,
`advance_when_out_of_reach` should default to **False for `take` and `deny`**.

## 2. ⚠ The engagement figure was wrong in the OTHER direction: it is 100%, not 47%

The 6.01% -> 3.21% figure came from an uncommitted script with a hardcoded 2.26" ring.
That constant is fractionally **larger** than the env's own predicate
(`shooting_masks.py`: engaged iff `centre_distance <= engagement_range + 2*base_radius`),
while `back_off_to_unengaged` parks every rescued model at `ring + epsilon`. **So the
test counted every model the rule saved as still engaged.**

Recomputed against the env's own predicate: **7.52% -> 0.00%. All of it removed.**

And the published explanation of the residual — "legitimate: a model with no legal
endpoint stays, and deployment can place one engaged" — is refuted:
`domain/placement.py` enforces `hostile_separation = min_separation + engagement_range`,
so no model ever *starts* a movement phase engaged.

## 3. ⚠ The batch shipped a movement bug: models ended OVERLAPPING

`back_off_to_unengaged` walked the endpoint backwards along its heading **without
re-checking bases**, into ground `resolve_move` had already cleared as
passable-but-not-endable. Measured: **0.18% of friendly pairs ended a movement phase
overlapping, worst penetration 0.68"**, against **0.0000%** with the rule off — violating
the first line of `movement.py`'s own docstring.

**Fixed:** the occupied bases now contribute forbidden spans to the same backward walk,
so one point must satisfy both constraints. Re-measured **0.0000% both ways**.

⚠ **Six unit tests covered the function and not one called `env.step`, so none of them
could ever have seen it** — the composition is where the defect lives. That is verbatim
the defect this project already paid for on the joint decoder, in the same PR that cites
it. `test_no_two_models_end_a_movement_phase_overlapping` now drives the real env.

## What survives

- The **premise**: no scripted baseline or opponent policy could advance, and an advancing
  agent was being scored against a walking bar. That was real and had to be fixed.
- The **endpoint rule**, which works better than claimed.
- The correction in section 3 of the original report (the bar table was already
  symmetric) — the audit panel checked it and rated it SOUND.

## The lesson

**Compute the error bar on the quantity you are claiming, not on its parts.** Both wrong
rows were visible in the published numbers: combining the printed bars naively gives
t = 1.12 and 0.11 for `take` and `deny`, and +1.3 was reported as a positive result
anyway.


---

# SECOND CORRECTION — the heuristic is REJECTED, and the bar never moved

A second audit panel completed the 2x2 the first correction did not. Reproduced here
independently. `25v25_maps_advance_refereed`, held-out nine, n=10, `squad_march_take` on
both sides, vp_margin to the player:

| | opponent walks | opponent advances |
|---|---|---|
| **player walks** | **-4.1** | +72.7 |
| **player advances** | **-81.8** | -3.6 |

- **Using the heuristic costs its USER about 78 vp.**
- **The bar never moved.** Both-advance (-3.6) is indistinguishable from both-walk (-4.1).
- ⚠ **The published "+15.5 to the bar" was two self-inflicted wounds cancelling.** Both
  sides adopted the same bad heuristic in the same change and hurt themselves by the same
  amount, so the diagonal looked like a gain.

**NEVER MEASURE A SYMMETRIC CHANGE WITH BOTH SIDES CHANGED AT ONCE.** The whole bar table
is void, not merely mis-signed: it compared one diagonal of a 2x2 against the other.

Two further findings from the same panel, both reproduced:

- ⚠ **The OFF column was measured on DIFFERENT CODE.** It came from the bridge run, taken
  before the endpoint rule existed; the ON column includes it. So the published deltas were
  the sum of two changes shipping together, never one.
- ⚠ **"`squad_march_shoot` gains most despite forgoing its shooting" is FALSE — it forgoes
  nothing.** Declared shots: 8,132 walking against **8,375 advancing**, 3% *more*. At
  weapon range 12 with objectives 20-40" away, the squads only advance while already out of
  range. The mechanism the highlight asserted does not exist in the data.

## What changes in the code

`advance_when_out_of_reach` now defaults to **False** on both the player baselines and the
opponent policy, pinned by a test. **The mechanism stays** -- `best_advance_toward` is
correct and a scripted bar that cannot use a core rule is not a bar. What is rejected is
this heuristic, which never prices the forfeited shooting.

## What still stands

- The **premise**: no script or opponent policy could advance. Still true, still worth
  fixing, and the mechanism to fix it now exists.
- The **endpoint rule**: 7.52% -> 0.00%, all of it removed.
- The **overlap fix**, and the behavioural test that catches it.
