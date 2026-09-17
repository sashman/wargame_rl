# Pre-registration: curriculum rung D2d — PPO from the escort clone with its critic fitted and the KL anchor

Written 2026-09-17 18:20, **before any PPO round with both changes
exists**, on branch `feature/curriculum-d2` (PR #377). Issue #380,
parent question #340, rung **D2** — fourth arm.

## The one change, per comparison

Relative to D2c (#378, the anchor alone): the starting checkpoint is
the critic-fitted clone `escort-c3b-1200-s0-critic.pt` (held-out
explained variance −0.17 → 0.74, policy bit-identical). Relative to D2b
(#379, the critic alone): `--kl-ref-coef 10 --kl-ref-target 0.03`.
Everything else is D2's recipe (three seeds, 122,880 rounds, 128 per
update, `--ent-coef 0.003`), Wandb `curriculum-d2` tag `d2d`.

## Why, and what is already known in-run

At 40k rounds, before any final read: D2 (neither) and D2b (critic
alone) sit at 2–17% in-run from the clone's 95%; D2c (anchor alone)
sits at 97–100%. The critic was not the mechanism of the collapse on
this trainer; the anchor is what keeps the plan. D2c's expected verdict
is HOLDS. The question that remains is IMPROVES — whether reward makes
the anchored policy better than its start — and a sound critic is the
obvious thing an anchored run needs to do that. This arm is the 2×2's
fourth cell, read beside the other three.

## Comparator

The clone paired per episode (0.960 / +59.6 / ordering 1.00); the escort
(0.990 / +63.5); D2, D2b and D2c beside every row.

## Criteria

D2's, unchanged: IMPROVES / HOLDS / DESTROYS at n=180 on 700000+, NULL
only if the escort fails. Readouts as D2b's and D2c's together.

## Launch

Three seeds launched by a detached script the moment D2's three exit
(the twelve-trainer limit), reads queued behind its own exit.

## What I expect (a guess, written so it can be wrong)

HOLDS 3/3 and IMPROVES on none, as D2c: the anchor at this coefficient
pins the policy to the clone and the critic has nothing to steer. If
it IMPROVES on any seed, the anchor's coefficient is the next sweep; if
D2c holds and D2d does not, the fitted critic is a liability under an
anchor, which would be worth a rule.
