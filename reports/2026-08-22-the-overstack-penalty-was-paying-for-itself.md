# The overstack penalty was paying for itself. Removing it costs 12 vp.

**2026-08-22.** Three seeds, 300 epochs, `ent_coef` 0.003, scored on
`24v24_maps_spare_squads_refereed.yaml` at K=3, n=30, held-out nine, seeds
700000+. **Paired** — the only change is one reward scalar, so no tensor shape
moves and the two arms start from bit-identical weights at the same seed.

## The proposal, and who made it

`closest_objective_v2` is the only per-model term that pays a model to move
*between* objectives, and on this config its net income is **negative**:

| component | agent | `squad_march_take` |
|---|---|---|
| progress | +0.08 | +0.23 |
| **overstack_penalty** | **−0.90** | **−0.58** |
| **total** | **−0.82** | **−0.35** |

The whole negative is the penalty, running at `overstack_penalty_per_extra:
0.01`. Three of seven independent reviewers converged on removing it, and the
argument was clean: `overstack_penalty_per_extra` and
`objective_hold.surplus_value` are both recorded as halving objective occupancy
because they **destroy total income** rather than redistribute it, and
`crowding_exponent` at a=1 — which conserves the pot — is the one lever in that
family that has ever worked here. So the term that is supposed to pay for
movement was being more than cancelled by an income-destroying knob layered on
top of it.

The ledger check confirmed the mechanism exactly: setting it to 0.0 flips
`closest_objective_v2` from **−0.30 to +0.29**, with every other calculator
bit-identical.

## The result

| seed | control (0.01) | no penalty | paired difference | `held` |
|---|---|---|---|---|
| s1 | +5.6 | −15.6 | **−21.2** | 2.00 → 1.76 |
| s2 | +19.0 | +16.7 | −2.3 | 2.39 → 2.26 |
| s3 | +23.2 | +10.0 | −13.2 | 2.18 → 2.13 |
| **mean** | +15.9 | +3.7 | **−12.2 ± 5.5**, t = −2.23, **3/3 negative** | 2.19 → **2.05** |

Coherency is unmoved (0.966 both). `alive` rises slightly (0.526 → 0.543).

**REJECTED.** Removing the penalty is worse on every seed.

## Why — and this is the part worth keeping

| | control | no penalty | change |
|---|---|---|---|
| own VP | 163.3 | 166.2 | **offence +2.9** |
| conceded | 147.3 | 162.5 | **defence −15.1** |

**The travel term did pay more for movement, exactly as predicted — and the
agent conceded fifteen VP for it.** The penalty was doing real work that its
income ledger could not show: discouraging stacking made models spread out to
*deny*, and removing it let them concentrate and give ground away. `held` fell
too, so the extra movement did not even buy the objectives it was supposed to.

## The lesson, which is bigger than the arm

⚠ **A term with negative net income is not thereby a broken term.** The
reasoning that produced this arm ran from the ledger to the conclusion: the term
nets −0.26 an episode, income-destroying anti-concentration levers are a
documented failure class, therefore remove it. The ledger was right. The
inference was not. **What a term costs is visible in `measure-income-share`;
what it prevents is not**, and here what it prevented was worth **+15 vp of
defence** against the +2.9 of offence it suppressed.

The existing rule — *"an anti-concentration lever must REDISTRIBUTE reward, not
destroy it"* — was derived from levers that halved objective occupancy. It does
not license removing a small one that is not doing that. This penalty is 1/5 the
magnitude of the one that failed, and it sits alongside `crowding_exponent`
rather than instead of it.

## Do not re-run

- **Removing `overstack_penalty_per_extra` from the golden lineage.** −12.2 ± 5.5
  paired, 3/3 seeds, `held` down 0.14.
- **Arguing from a calculator's net income to whether it should exist.** Read the
  differential the term creates across the choices a model is making, and if you
  cannot, measure the arm rather than the ledger.
