# 04-02 Execution Summary

Wave 2 of phase 04: the schedule, the arena, the ledger and the two recipes.
The rating pipeline works end to end — **and its gate failed on the reference
scenario, which is the headline.**

## Shipped

| Module | Owns |
|---|---|
| `rating/entrant.py` | `Entrant` — a name plus a **factory** for a selector, because a network entrant must be sized from the env it plays in |
| `rating/schedule.py` | `Leg`, `FOUR_LEGS`, `config_for_leg`, `with_opponent`, seeds, `pairings` |
| `rating/arena.py` | `LegResult`, `play_leg`, `play_pairing` — the only module in `rating/` that touches a live env |
| `rating/ledger.py` | `canonical_scenario`, `fingerprint`, append-only persistence, `RatingScenarioMismatch` |
| `rating/table.py` | `design_from_legs`, `rate`, `format_table` |

Recipes: `just measure-elo`, `just elo-table`, `just measure-seat-parity`.

**`envs/baseline/evaluate.py` is unmodified**, as designed. The arena installs
entrant B with `set_opponent_policy` and calls `evaluate_selector` — opponent
identity is state on the env, not a parameter of the scoring loop.
`test_a_leg_is_scored_by_evaluate_selector` asserts the arena's margins equal
`evaluate_selector`'s on the same seeds, so the two can never drift.

## The pipeline works

First table, on `configs/dev/4v4_two_phases.yaml`, 20 layouts:

```
squad_march_shoot   +132   [+119, +145]   +38.9 vp   1.000 coherent   80 games
random                +0       [+0, +0]   -38.9 vp        -           80 games
```

Cross-check against the linear approximation: `ΔR ≈ 173.7 × 38.9 / 50 = 135`
against a fitted **+132**. The bridge between a rating and a `measure-paired`
number holds.

## ⚠ The seat-parity gate FAILS on the reference scenario

`configs/golden/25v25_shooting_opponent.yaml`, `squad_march_shoot` on both
seats, 30 layouts × 4 legs:

| Leg | mean margin | wins |
|---|---|---|
| zone 1, A first | **−40.8** | 0.40 |
| zone 1, B first | −27.5 | 0.40 |
| zone 2, A first | −15.2 | 0.50 |
| zone 2, B first | −14.8 | 0.57 |
| **aggregate** | **−24.6 ± 9.4 (1 se)** | |

The same policy on both seats loses from the player seat by **24.6 vp**, at
2.6 standard errors. That is larger than most effects this repo has ever
measured, and every Elo number on this scenario would carry it. **Rating this
config is blocked until it is explained.**

### Localised to shooting, by a controlled second measurement

The same check with `squad_march`, which does not shoot:

| Leg | mean margin |
|---|---|
| zone 1, A first | +27.5 |
| zone 1, B first | −31.3 |
| zone 2, A first | +33.2 |
| zone 2, B first | −0.5 |
| **aggregate** | **+7.2 ± 9.1** — fair |

Same board, same seeds, same schedule; the only difference is whether the
policies shoot. So the asymmetry is **in the shooting phase**, not in
placement, movement, VP scoring or the mirror's movement path.

The leg breakdown says more than the aggregate. Without shooting, moving first
is worth a great deal (+27.5 / +33.2 as A-first against −31.3 / −0.5), which is
what a race for objectives should look like — `h_turn` fits at **+59.2 Elo**.
Turn shooting on and the A-first legs *invert*, to −40.8 / −15.2. So shooting
does not merely add a constant penalty: it **reverses the first-mover
advantage**, which is the signature of an ordering effect inside the exchange
rather than a static imbalance.

### What has been ruled out

- **Weapons.** Both armies carry the identical `*rifle` anchor.
- **The opponent's action mask.** `_opponent_action_mask` (`wargame.py:928`) is
  a faithful mirror of `build_observation`'s, uses `_opponent_max_ranges`, and
  applies the range/LOS/engagement refinement because
  `ScriptedBaselineOpponentPolicy` derives `shoots = True` from
  `squad_march_shoot` overriding `select_shooting`.
- **Movement, placement and VP.** The non-shooting control is fair.

### The leading hypothesis, untested

`step()` applies the player's action and *then* calls
`run_after_player_action`, which runs the opponent's whole turn — so within one
step the player's volley always resolves before the opponent's, and casualties
are removed in between. `turn_order` changes which side the *clock* calls
`player_1`, not the order in which `step()` executes the two. If that is it,
the effect is a genuine environment asymmetry rather than anything to do with
rating, and it would also mean **the bar `squad_march_shoot` has never been
measured on equal terms with the opponent it is quoted against.**

Not investigated further here: root-causing it is its own piece of work, and
two clean measurements localising it is the right place to hand over.

## A latent domain bug, found by playing both zones

`objective_placement` read the free strip between the deployment zones as
`(deployment_zone.x_max, opponent_deployment_zone.x_min)` — correct only while
the player is the left-hand army. Every shipped config puts it there, so the
assumption was invisible. Give the player the right-hand zone and those two
numbers are the *outer* edges of the board: the range inverts and numpy raises
`high - low < 0` at reset.

**Any config deploying the player on the right hit this.** Fixed by ordering
the two zones by their left edge and taking the gap between them
(`_band_between_zones`). Both golden gates stay bit-identical, since no shipped
config exercises the path.

The regression test needed two attempts and the first was worthless: it used
`configs/dev/tiny.yaml`, which **fixes its objectives**, so
`objective_placement` never ran and the test passed with the bug reintroduced.
It now clears `objectives` to force a draw, and is verified to fail when the
fix is reverted.

## Corrections to `docs/elo.md` carried out of wave 1

The fingerprint must exclude `turn_order` and `opponent_policy` and sort the
zone and army pairs — the doc's literal instruction puts the four legs of one
pairing into four different ledgers.
`test_all_four_leg_configs_fingerprint_identically` is the proof.

## State

- `just format && just lint` clean; 1631 tests pass with the same four
  pre-existing failures (the known LOS-symmetry bug, and three CUDA tests that
  pass under `CUDA_VISIBLE_DEVICES=""`).
- Committed as `6da4771` on `feature/elo-rating-subsystem`.
- **`ratings/` is empty on purpose.** The only tables produced so far are on a
  dev fixture config and on a scenario whose seat-parity gate fails; neither
  belongs in a committed ledger.

## For 04-03

1. **Do not rate the golden config until seat parity is resolved.** The gate is
   `just measure-seat-parity <config> <policy> <n>`.
2. Wave 3 (the mirror's observation path and the `model` opponent policy) is
   unaffected by the finding and can proceed — it is what lets checkpoints be
   entered, and `just measure-paired ckptA ckptB` works the moment the `model`
   key exists.
3. `opponent_config_for` currently raises `NotImplementedError` for a
   checkpoint entrant, naming wave 3. That is the seam to fill.
