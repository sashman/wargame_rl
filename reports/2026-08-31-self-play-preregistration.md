# Pre-registration — does self-play move the agent, or only fix the reference?

**Written 2026-08-31, before any number in it exists.** The pool, the sampler and
the training wiring ship and are exercised end to end on
`configs/dev/4v4_two_phases.yaml`. **No self-play run has been trained**, so
every figure quoted below is a prior from another arm, and is labelled as one.

## The question, and the one it is NOT

Self-play supports two separate claims, and this project has a history of
running an arm that measures the second while reporting the first.

- **REFERENCE.** Training against one fixed opponent teaches the agent to beat
  that opponent. The receipt is already on file: the same agent scores **+61.4**
  against the weakest scripted opponent and **+20.8** against the strongest,
  while being **−75.9** behind the best script in the first case and **−9.5** in
  the second. Absolute score measures the opponent. A pool replaces the fixed
  reference with a moving, honest one.
- **STRENGTH.** A pool makes the agent better.

REFERENCE is nearly a definition and needs no run. **STRENGTH is the claim this
arm tests, and the standing evidence predicts it fails.** The agent's deficit is
allocation — it holds 1.9–2.1 objectives against the scripts' 2.9–3.9, its board
is static after round 8, and `measure-critic-probe` found the critic already
prices spreading correctly (**dV +2.63 ± 0.32, t=+8.3**, 6 of 6 cells) while the
policy does not do it. That is a **search** failure. A pool changes the opponent
distribution; it does not change the search. Three consecutive reward terms have
left offence flat or worse (−50.5, −42, −71.5), and this is the first lever tried
that does not even claim to address it.

Registering that prediction now is the point of this document. If offence moves,
the prediction was wrong and that is a finding; if it does not, this is a null
that was called in advance rather than a disappointment.

## Preconditions — all three must pass before the run starts

**All three now pass.** Two were found by writing this document; the third was
the standing gate nobody had run on this config.

1. **Seat parity on the training config — RUN 2026-08-31, and it PASSES.**
   `squad_march_take` on both seats, `25v25_maps_two_mode.yaml`, 120 layouts:
   **aggregate +6.5 ± 6.1 vp (1 se), t=1.07**, against a threshold of 2 se.
   For contrast the gate **fails** on `25v25_shooting_opponent.yaml` by
   **−24.6 ± 9.4**. So the learner's seat is not a structural advantage here,
   and a snapshot seated as the opponent is playing approximately the same game
   — which matters because the learner only ever trains the player seat
   (`learner_side` does not exist; `model/common/self_play.py:120` records it as
   a later phase).

   ⚠ **Quote it with its interval, not as "fair".** The 95% bound is roughly
   **[−5.5, +18.5]**, so a gap up to ~19 vp in the *player's* favour is not
   excluded — and that is the direction that would flatter a self-play run,
   inflating PFSP's `p` with seating rather than skill. It is comfortably short
   of the −24.6 that condemned the other config, and that is the claim.

   ⚠ **A first pass at n=30 read +19.1 ± 11.2 and was nearly a fail.** It did
   not survive quadrupling the layouts: the estimate fell to +6.5 as the error
   halved, and the leg disagreement (+10.7 / +27.5) collapsed to +0.2 / +12.8.
   **n=30 cannot run this gate** — the repo's own n=100 rule applies here too,
   and at n=30 the threshold sits at 22.4 vp, so the −24.6 that failed elsewhere
   would have been about a coin-flip to catch.

   ⚠ **On a map-pool config this number is the SEAT AND THE SIDE OF THE TABLE
   lumped together**, because the drawn outlines stay bound to the seats and the
   zone axis cannot separate them. For this precondition that is the right
   quantity — a snapshot on the opponent seat suffers both — but it is not a
   seat term and must not be quoted as one. Terrain is 180-degree rotation
   invariant on only **34 of 45** tables, so a real side asymmetry is inside it.
2. **Group-id aliasing — FIXED in this branch.** `group_span` floored, so an
   army could split into more units than `max_groups` has one-hot columns, and
   `_group_ids_to_one_hot` **clips** rather than raising. Measured live: 15
   models at `max_groups: 6` split into 8 units, ids 0..7, clipped to 0..5 —
   units 6 and 7 encoded as one column while `unit_count` sized the shooting
   slice at the true 8. Not on the training config (25 at cap 5 is clean), but
   it is on the asymmetric experiments, which is the regime self-play eventually
   wants. Rounding up is bit-identical wherever the cap divides the army — every
   golden, evaluation and dev config — and both goldens verify unchanged.
3. **The rating zone axis — REFUSED for map pools in this branch.** Irrelevant
   to training and load-bearing for the evaluation half below.

## Design

| | |
|---|---|
| config | `configs/golden/25v25_maps_two_mode.yaml`, the config that trains |
| recipe | the documented one — `ent_coef` 0.003, `--no-tf32` |
| arm | `--self-play --pfsp-mode uniform --snapshot-every-n-epochs 25 --pool-capacity 8 --pool-anchor squad_march_take` |
| control | the same recipe with `--self-play` omitted |
| pairing | **paired at init** — `train.py:303` seeds before `:374` constructs, and self-play off constructs no scheduler at all, so the two arms start from bit-identical weights |
| epochs | 300 to screen mechanism, **1000 to decide** |
| seeds | 3 to screen, **6 to decide** (see the power check) |
| scoring | held-out nine, n=30, seeds 700000+, **refereed** eval configs, K=3 verified decode |

**`uniform` is the arm, not `hard`.** A pool changes training on its own and the
schedule is a separate claim; `hard` and `even` are a second experiment that
should not start until this one has an answer. Running them together would leave
a null unattributable between the two.

## How to run it — needs a GPU

```
just train-self-play-screen 300 1     # arm + control, seed 1
just train-self-play-screen 300 2
just train-self-play-screen 300 3
```

Two concurrent runs per seed (~3.8 GB VRAM each): the arm and its control, with
**byte-identical flags apart from `--self-play`**.

⚠ **Do not launch the control through `just train-coherency-baseline`.** It adds
`--record-every-n-epochs 10`, and the Justfile's own note says a differing
recording cadence is not something to assume is free. The pair is a paired
estimator only while the flags match; that is why both sides go through one
recipe.

Then score, once the screen has cleared mechanism and the 6x1000 runs exist:

```
just measure-checkpoint <ckpt> configs/evaluation/25v25_maps_take_opponent_refereed.yaml 30 "" 3
just measure-baselines  configs/evaluation/25v25_maps_take_opponent_refereed.yaml 30 "" 700000
```

### What the screen has to show before the deciding run starts

- Snapshots appear in `checkpoints/<run>/pool/` and load back as an opponent.
- `self_play/pool_size` reaches capacity and `self_play/mean_opponent_epoch`
  keeps rising — a pool collapsed onto its newest member is a mechanism failure
  a score cannot distinguish from a null.
- `vp_margin` does not fall off a cliff against the control.
- Nothing raises.

⚠ **SIGKILL writes no snapshot** — the pool hook is a Lightning callback, the
same trap as `last.ckpt`, and SIGKILL is the prescribed way to stop these
trainers. Score a killed run from its highest `ppo-NNN-*.ckpt`, and check the
pool directory before believing a pool-size figure.

⚠ **No verdict comes out of this stage.** Three seeds cannot resolve a
difference under ~28 vp (see the power check), so a score from it is not quoted.

## What this does NOT void, which is more than expected

⚠ The original design (`docs/self-play.md`, superseded draft) states that a
self-play config is a new scenario and that **every** baseline on it must be
re-measured. **That over-scopes itself, and the code says so.** The scheduler
seats only `_ensure_rollout_envs()` (`model/ppo/lightning.py:691`); `_eval_envs`
is a separate list built in `lightning_base.py:215` and is never touched. So
self-play changes **who the learner trains against and nothing else**. Held-out
scoring runs on the `configs/evaluation/` family against their own fixed scripted
opponents, exactly as every published row did.

**So the existing five-row table is the comparator, and no bar needs
re-measuring.** That is unusual here and worth stating plainly, because the
default assumption in this repo — earned four times over — is that a change of
this size voids the baselines.

The one thing it does void is the *interpretation* of the in-run `eval/baseline_*`
keys, which are still measured against the config's own opponent and no longer
describe what the learner faced.

## Primary readout

**Paired per-seed `vp_margin` difference, arm − control, on
`25v25_maps_take_opponent_refereed.yaml`** — the primary documented row, where
the agent currently reads **+19.4** against `squad_march_take`'s **+6.5** (the
2026-08-24 reissue at `f741e14`; ⚠ always stamp a revision on a quoted table).

Reported with its sd, its correlation, and a per-table sign count. If the
correlation is negative, say so and fall back to unpaired.

### Secondary, and the ones that decide *what happened*

- **The offence/defence split.** Offence is what this arm is predicted not to
  move. It is an identity, not a decomposition — read it as bookkeeping.
- **`held`**, and **`held` by round at 2/5/8/12/16/20.** The agent's allocation
  is fixed by round 2 and gains +0.18 objectives over the remaining eighteen
  rounds against the script's +0.53 by round 8. If a pool changes anything about
  search, this curve is where it shows.
- **Coherency**, unconditionally. A `vp_margin` alone is a result plus an
  unstated claim that the moves earning it were legal.
- **`self_play/pool_size` and `self_play/mean_opponent_epoch`.** A pool collapsed
  onto its newest member is a mechanism failure that a score cannot distinguish
  from a null.

## Criteria, power-checked before being written down

⚠ The last arm here committed a per-seed bound **tighter than its own
estimator's noise**: a lever costing exactly zero failed it 56% of the time. The
check below is that mistake not repeated.

Best available prior for the per-seed paired sd on a map-pool config is **11.3**,
from the advance-lever arm. That gives:

| seeds | SE | t crit | minimum detectable difference |
|---|---|---|---|
| 3 | 6.52 | 4.303 | **28.1 vp** |
| 4 | 5.65 | 3.182 | 18.0 vp |
| **6** | 4.61 | 2.571 | **11.9 vp** |
| 9 | 3.77 | 2.306 | 8.7 vp |

**Three seeds cannot resolve anything smaller than 28 vp**, which is larger than
every arm difference ever measured here bar two. So:

- **The 3-seed 300-epoch stage is a MECHANISM screen and returns no verdict.** It
  passes if snapshots write and load back, the pool spans the run rather than
  collapsing, `vp_margin` does not fall off a cliff, and nothing raises. A score
  from it is not quoted.
- **ACCEPT** (STRENGTH supported): paired difference **≥ +12 vp** at 6 seeds and
  1000 epochs, with **≥ 5 of 6 seeds positive**.
- **REJECT**: paired difference **≤ −12 vp**, or ≤ 4 of 6 seeds positive with a
  negative mean.
- **NULL**: anything between. ⚠ A null here is the **predicted** outcome and is
  the third possible result, not a failed experiment. It would establish that a
  pool is worth having for REFERENCE and not for STRENGTH — which is a reason to
  keep the feature and stop spending GPU on it.
- **UNDERPOWERED**: |difference| < 12 with signs flipping. Distinguished from
  NULL by the sign count, and reported as such rather than as a null. The advance
  lever read `+2.2 ± 6.5` with flipping signs at 300 epochs and `−16.3 ± 8.9`
  with all three negative at 1000 — **a 300-epoch reading pointed the opposite
  way from the 1000-epoch one**, so no verdict is taken before 1000.

## Two ways this arm could measure something other than the agent

Stated in advance because both have happened here before.

1. ⚠ **The opponent decodes at K=1 while every score is quoted at K=3.** Joint
   constrained decoding is worth **+40.5 vp** at play. So the learner trains
   against a version of itself roughly 40 vp weaker than its own scored self —
   the pool is systematically softer than it looks. This is defensible on cost
   (`K^k` forwards per unit per rollout step) but it is a **decision**, and a
   ledger refuses one entrant name under two decodes for exactly this reason. If
   the arm reads NULL, the softened pool is a live alternative explanation and
   this pre-registration does not separate them.
2. ⚠ **Snapshots are written from a Lightning hook, and `SIGKILL` writes
   nothing** — while SIGKILL is the prescribed way to stop these trainers. A pool
   is routinely up to `snapshot_every_n_epochs` behind, exactly as `last.ckpt` is
   up to 25 epochs stale. Score a killed run from its highest `ppo-NNN-*.ckpt`,
   and check the pool directory before believing a pool-size figure.

## The evaluation half, which is available now and needs no training

Independent of the arm above, and the part of this work with the better expected
return: **the first rating ledger**. `ratings/` is empty and no rating has ever
been published.

The rating subsystem does not need self-play, a model change, or any training —
on a symmetric config today's network already plays either seat, and
`tests/test_swap_invariance.py` pins that the mirrored observation is the other
seat's, with a falsifier. Entrants: `random` (anchor), `squad_march_take`,
`squad_march_deny`, `squad_march_shoot`, `contest_and_spread`, and the six
`two_mode` checkpoints. **Three entrants minimum** — `h_seat` is identified
through a cycle in the pairing graph — plus a self-pairing, which roughly halves
that term's standard error.

Two constraints on where it can be played:

- ⚠ **Not on a map-pool config.** A drawn table carries its own deployment
  outlines and they stay bound to the seats across the zone swap, so `h_zone`
  would be fitted from noise. `config_for_leg` now refuses this rather than
  producing the number. That rules out the `maps` family and points the first
  ledger at `25v25_shooting_opponent.yaml` or `25v25_cover_control.yaml`, which
  deploy under their own rectangles.
- ⚠ **`25v25_shooting_opponent.yaml` is the config that FAILS seat parity** by
  −24.6 ± 9.4 vp. `h_seat` absorbs it, but the fitted term assumes the advantage
  is constant in Elo across pairs, which `h_turn` — measured to change sign with
  shooting — suggests may be false. **Cross-check the fitted `h_seat` against the
  gate's own aggregate**, and quote the interval, never the point.

⚠ **One provenance caveat, from the superseded design and still live:** if the
size-agnostic policy work is ever executed, it changes weight shapes and **no
checkpoint rated before it can be loaded after it**. The scripted rows survive;
the checkpoint rows become unreproducible. Neither doc shipped and the planning
phase was renamed, so this is a risk to note rather than a reason to wait.

## What is NOT claimed here

- That self-play is the right next spend. It is a mechanism that ships and has
  never been run; this document says how to run it so the result means something.
- That `hard` or `even` beat `uniform`. Untested, and deliberately out of scope.
- That the pool prevents forgetting. The anchor and the `uniform_floor` exist for
  that and neither has been measured.
- Anything about the opponent seat. `learner_side` does not exist, so every
  statement here is about a learner pinned to the player seat.
