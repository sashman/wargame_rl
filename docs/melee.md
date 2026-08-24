# Melee: the charge phase and the fight phase

`melee.enabled` defaults to **False**, and off it is an exact no-op — no slice
registered, no dice drawn, no observation column, `skip_phases` untouched. Every
golden config, every observation golden and every reward golden is bit-identical
across the whole feature, verified by a seeded episode digest against `main`: 3
configs × 3 policies × 2 seeds, hashing per-step reward, every shooting result,
final VP and every model's position and wounds. All nine cells match; all nine
differ from each other, which is what makes the check non-vacuous.

Turning it on is **not a tuning change**. It steps a phase that was skipped, so
`max_turns` changes, and it creates a board state this environment has never
entered. Every baseline and every agent score on that config is void.

## The fact the whole design turns on

Engagement was measured at **0.0000%** of model-pairs over 60,520 observations,
and the obvious reading — that contact is unreachable — is wrong. The minimum
edge-to-edge gap is **1.000008740"** against an engagement range of 1.0:
`back_off_to_unengaged` walks every mover on both seats to `min(lo) - 1e-6`, so
the army parks **8.7 micro-inches** outside contact and always has.

A charge therefore does not need to cross a gap. It needs the **exemption** from
that back-off, which is one branch in `ActionHandler.apply`. Every distance
argument that assumed a real gap had to be crossed was answering a question that
does not arise.

## What is implemented

| | |
|---|---|
| **Charge phase** | Stepped when `melee.enabled`. The `movement` slice is valid there, so **no new movement actions** — which is also what keeps an arm *pairable* against its control. |
| **2D6** | One roll per unit at the start of the side's turn, visible in the observation the policy conditions on, masking every speed bin that would travel further. |
| **Eligibility** | Within `melee.charge_range` (12") of an enemy unit, unengaged, and did not advance or fall back this turn. |
| **The exemption** | A charge move, and only a charge move, may end inside an enemy's engagement range. |
| **The referee** | A charge that does not end legally **is not made at all**: every model of the unit returns to where it started. Two conditions — the unit is coherent, and it is engaged with exactly ONE enemy unit. |
| **Fight** | Engaged models strike the unit they are in contact with, on the boundary leaving the fight phase, before attrition. The attack sequence is `resolve_attack`, shared verbatim with shooting so the two cannot resolve the same dice differently. No cover — `12-fight-phase.md` grants none. |
| **Order** | Charging units first, then the rest, each in group order; active player before the opposing one. |
| **Fall Back** | An engaged unit that moves forfeits its shooting and its charge for the turn. The *geometry* was always right — `back_off_to_unengaged` is the fall-back constraint — what was missing was the cost. |
| **Targets must be unengaged** | A model engaged in melee cannot be shot at. |
| **Consolidate** | Objective mode only, env-resolved, no agent action. |

## What is outstanding

⚠ **This list is the point of this document.** Every entry has a row in
[docs/rules/implementation-status.md](rules/implementation-status.md) carrying a
`` `DEFERRED: <name>` `` tag, and
`tests/test_implementation_status.py::test_every_deferred_gap_is_still_a_gap`
**fails the day one of them is implemented** without the register being updated —
so a gap cannot close quietly and leave the register lying about it. Verified
non-vacuous: adding a `_pile_in` definition trips it by name.

- `fight.pile_in` — no 3" close-up. A unit engaged at the edge of engagement
  range fights from there, and a unit that loses its nearest model cannot close.
- `fight.overrun` — a unit that destroys its target cannot reach a new one.
- `fight.passing` — a unit whose targets all die simply does not fight. With no
  pile-in there is nothing to wait for.
- `fight.alternating_activation` — the order is fixed. The rules alternate
  between players and return to the Strikes First sub-step whenever a new
  Strikes First unit becomes eligible; that needs a per-unit sequencing
  decision, which needs its own action space.
- `fight.fighting_after_death` — a destroyed model is removed at once. A no-op
  only while every model has one wound.
- `fight.select_weapon` — a model fights with `melee_weapons[0]`.
- `charge.target_declaration` — ⚠ **measured, not preferred.** The approved plan
  had a `charge_target` slice; a declaration must be an action, the declaring
  model then cannot move, and a model left behind while its squadmates charge
  breaks the 2" chain — against `evaluate_coherency` a five-model unit whose
  declarer stays put is coherent at 2" and **incoherent at 4", 8" and 12"**.
  Since an incoherent charge is reverted whole, the slice would have made almost
  every charge fail. The target is derived from where the unit ends instead, so
  *charge A, land on B* is currently legal where the rules forbid it.
- `charge.blind_declaration` — the 2D6 is visible before the charge is made,
  exactly as `advance_roll` is. Legality is gated on the roll, so a declaration
  taken before it would have no legal distance to take.
- `consolidate.ongoing`, `consolidate.engaging` — see below.
- `consolidate.select_objective` — the env takes the nearest objective in range;
  the rules let the player choose.
- `fallback.declared_move_type`, `fallback.reckless_break`,
  `fallback.blocks_charge`.

## Two things that will surprise you

⚠ **Consolidation almost never fires, and that is the RULE.** The three modes
are assessed in order and the first match is **compulsory**: a unit still engaged
is in Ongoing mode, a unit with any enemy within 3" is in Engaging mode, and
neither reaches Objective. What is left is a unit that killed everything near it.
Reading a near-zero consolidation rate as a bug would be reading the rule.

⚠ **A charge that clips a second unit fails.** "Engaged with exactly one enemy
unit" is both after-moving conditions at once while a charge has a single target,
and `11-charge-phase.md` calls this out as what makes a charge fail even on a
long roll.

## Lethality

⚠ **"No lethality-neutral melee profile is expressible" is FALSE.** That claim
swept `melee_skill` and `attacks` and left `strength`/`ap` at their defaults.
`wound_roll_threshold` returns **6** whenever `2 × strength <= toughness`, so S1
against T3 wounds on 6+: **`A1 / MS6+ / S1 / AP2` = 0.02315** against a neutral
target of 0.02415, within 4.1%.

`MeleeWeaponProfile`'s defaults are **not** that profile — they are an ordinary
weapon. A scenario that wants to measure the *mechanic* rather than the *damage*
must say so in its config.

## Measuring it

⚠ **A vp gate is unpowered by construction here.** At n=3 the one-sided
half-width is **19.05 vp**; at n=6, **9.32**. A lethality-neutral melee mechanic
is designed to move vp by less than that, and no number of seeds fixes a gate
that is tighter than its own estimator — the old "≥ −8 on 3 of 3" rule passes a
do-nothing feature **44%** of the time.

Pre-register **mechanism** counts instead, every one of which moves from a hard
floor of zero and is detectable at n=3: engagement rate (floor 0.0000%), charges
declared, charges succeeding, fights resolved, models locked per turn. Then run
the vp comparison at **n ≥ 6** with an explicit PASS / FAIL / **UNDERPOWERED**
trichotomy, and be willing to report UNDERPOWERED.

Zero-cost screens first: `just measure-throughput` on a charge config; score
`squad_march_take` (which never charges) on it to isolate what merely *having*
the phase costs; and `just measure-freezing` split by engaged-at-phase-start.
