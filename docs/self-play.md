# Self-Play: A Pool, and Who To Draw From It

Training against one fixed opponent teaches the agent to beat that opponent.
This repo has the receipt: the same agent measured **+61.4** against the weakest
scripted opponent and **+20.8** against the strongest, while being **−75.9
behind the best script** in the first case and **−9.5** in the second. Absolute
score measures the opponent.

Self-play replaces the fixed opponent with the learner's own frozen past. The
opponent then improves as the learner does, and "did this get better" is asked
against a moving, honest reference.

- Related: [elo.md](elo.md) (where `p` comes from) ·
  [opponent-policies.md](opponent-policies.md) (who can sit on the other side) ·
  [metrics.md](metrics.md)

**Status:** the pool, the sampler and the training wiring ship and are exercised
end to end on `configs/dev/4v4_two_phases.yaml`. **No self-play run has been
trained**, so nothing here carries a measured number yet.

---

## 1. Off is a no-op, and that is a property

`--self-play` is off by default, and when it is off **no scheduler object
exists**. Not "exists but is unused" — is not constructed. So no stream is drawn
from, and a control run is bit-identical to one on a build without the feature.

That is the `augment_start` standard, and it is the difference between an arm
that can be compared to its control and one that cannot: a single unused draw
from a shared generator shifts every layout and every dice roll after it.

The opponent stream has **its own seed band**, `1_100_000`, disjoint from
rollout (0), in-run baselines (10k), in-run eval (500k), held-out scoring
(700k), behaviour cloning (800k) and ratings (900k). Turning self-play on
changes who the learner plays, not which boards it plays on.

## 2. The pool

`rating/pool.py`. An ordered, capped collection of frozen opponents — a
checkpoint path, the epoch it was taken at, and its rating once it has one.

Three decisions, each a way pools go wrong:

- **The anchor is never evicted.** Entry zero is a fixed reference, by default
  the scripted `squad_march_take`. A pool of nothing but recent selves can drift
  as a whole — every member beating the one before it while the lot of them get
  worse against anything outside. The anchor is also what the rating scale is
  pinned to.
- **Eviction thins uniformly; it does not drop the oldest.** Keeping the most
  recent *k* snapshots keeps the part of the run the learner is *least* likely
  to have forgotten. Thinning removes the member whose neighbours are closest
  together in epochs — the most redundant one — so the pool keeps spanning the
  whole run.
- **Capacity is a disk budget before it is a statistical one.** A snapshot is a
  full checkpoint, `checkpoints/` is the only copy of any weights this project
  has, and checkpoints are already not uploaded to Wandb because each run pushed
  ~591 MB and filled the quota. Default 8.

⚠ **Snapshots are written from a Lightning hook, so `SIGKILL` writes nothing** —
and SIGKILL is the prescribed way to stop these trainers. A pool is routinely up
to `snapshot_every_n_epochs` behind the run that produced it, exactly as
`last.ckpt` is up to 25 epochs stale for the same reason.

## 3. Who to draw

`rating/pfsp.py`. Sample a snapshot with probability proportional to a function
of `p`, the learner's chance of beating it.

| mode | weight | picks |
|---|---|---|
| `hard` | `(1 − p)²` | the opponents the learner loses to |
| `even` | `p(1 − p)` | the ones it is level with |
| `uniform` | `1` | **the control** |

`hard` squares rather than using `1 − p` because a linear weight still spends a
third of its games on opponents it beats 2:1. `even` is not the same ordering:
a `p = 0.1` opponent is the *hardest* and the *least* even.

**Run `uniform` as the control before believing any `hard` or `even` number.** A
pool changes training on its own; the schedule is a separate claim, and this
repo's most expensive class of error is a comparison against the wrong control.

⚠ **The anchor must be able to use the features the config enables.** It is the
pool's permanent floor and is never evicted, so it sets what "no better than
where we started" means. On a melee config the default `squad_march_take` has a
`select_charge` that returns STAY, so the floor would be a policy that **never
charges** — and the learner could climb the whole ladder without meeting a
charge, while the config's own opponent charges every turn.
`squad_march_take_charge` is the anchor there, and `just train-self-play-screen`
takes it as an argument. The same question has to be asked of any future move
type: a floor that cannot use a rule is not a floor for a game that has it.

A `uniform_floor` (default 0.1) keeps any snapshot from being starved — an
opponent that is never sampled cannot catch the learner forgetting how to beat
it, and forgetting is what a pool exists to prevent. A pool the learner has
entirely outgrown has no signal to prioritise on and falls back to uniform
rather than dividing by zero; that is the normal state just before a new
snapshot is taken.

`p` comes from `elo.win_probability` — the **rating table**, not a second
win-rate counter kept alongside it. One estimator, fitted from every game either
policy has played.

⚠ **Until the in-run ladder landed, that table was empty, and all three modes
were the same policy.** `SnapshotPool.sample` reads an unrated entry as an even
match — `p = 0.5`, deliberately, since refusing to play an unrated snapshot is a
deadlock — and *nothing called `rate()`*. With every member at `p = 0.5`,
`pfsp_weights` returns a uniform vector for `hard`, `even` and `uniform` alike,
verified directly. So `sampling: "hard"`, the **default**, drew uniformly.

What that means for anyone reading a self-play result:

- Any `hard` or `even` arm run before the ladder measured `uniform` under
  another name, and a `hard`-vs-`uniform` comparison would have been a **null
  by construction**. Nothing has been run, so nothing is voided.
- `uniform` is unaffected — it ignores ratings by definition. The pre-registered
  screen names `uniform` as its arm, so it is measuring what it says it is.
- The modes are live now, which makes "does the schedule matter" a real second
  experiment rather than a comparison of a thing with itself.

### The learner's own rating

`self_play/learner_elo`. The rollout is already a rated match — the learner
plays a seated pool member every epoch — and its result was being thrown away.
Each finished rollout episode is banked as a game against whichever entry that
env was seated with, and the learner's rating steps once per epoch by Elo's own
rule, `R + K (s − E[s])`, with `s` a `margin_score` rather than a win indicator
for the reason the fit uses one: win rate cannot resolve differences under ~7pp
here.

Three choices worth stating, because each is a place this could have measured
something else:

- **One batched update per epoch, not one per game.** Elo updates do not
  commute, and an epoch's games are played in parallel across rollout envs, so a
  sequential update would make the rating depend on which env happened to finish
  an episode first — not a fact about the policy.
- **A snapshot inherits the learner's rating at the moment it is frozen**, and
  frozen weights never re-rate afterwards. That is what makes the pool a
  *ladder* rather than a bag: a later self that beats an earlier one gains
  points against a fixed reference. The anchor is the origin and never moves.
- **The rating is updated before the snapshot is written**, so a snapshot
  carries the epoch it was actually frozen at rather than being one epoch stale.

⚠ **It is a ladder, not a rating on the published scale.** It says how far the
learner has pulled ahead of its own history and one scripted anchor. Two runs'
ladders are not comparable to each other unless they share an anchor *and* a
history, which two runs never do. `just measure-elo` is what puts policies on
one scale.

## 4. The seat, which is the live risk

⚠ **The learner only ever trains the player seat.** A `learner_side` reset
option is a later phase. So a snapshot seated as the *opponent* is playing a
seat it never practised.

That is harmless exactly when the two seats are the same game — and on
`configs/golden/25v25_shooting_opponent.yaml` they are not, by **24.6 vp**. So:

- **Do not start a self-play run on a scenario whose
  `just measure-seat-parity` gate fails.**
- The probability PFSP weights on is the one for *the game about to be played*,
  so `OpponentScheduler.rate` takes `h_seat` from the same fit and passes it as
  the seating advantage. Weighting on the seat-neutral number instead would
  spend the run's games on a bias rather than on skill.

See [elo.md § Open gaps](elo.md#open-gaps).

## 5. What a snapshot is, and the mistake it shipped with

A snapshot is `self.state_dict()` of the **Lightning module**, not of the inner
`ppo_model`.

⚠ This is not cosmetic. `NetworkOpponentPolicy` reads a snapshot through
`convert_state_dict`, which looks for `policy_net.` or
`ppo_model.policy_network.` — the prefixes a real Lightning checkpoint carries.
Saving the inner module gives bare `policy_network.` keys and **raises on load**,
which is what a three-epoch smoke run found. It raised loudly only because that
path loads *strict*; the same mistake through `_apply_warm_start_weights`, which
uses `strict=False`, loads **nothing** and reports a warm start.
`test_a_snapshot_loads_back_as_an_opponent` is the regression, and it steps the
env rather than merely constructing the policy.

The opponent's decode is `decode_topk: 1` by default. Joint constrained decoding
is worth +40.5 vp at play time, but it costs `K^k` forward-model evaluations per
unit and the opponent would pay that on every step of every rollout. Whatever it
is set to, it is recorded: **a ledger refuses one entrant name under two
decodes**, and it treats *sampled* and *greedy* as different decodes too, since
self-play rollouts draw from the policy where every scoring path here takes its
argmax.

## 6. Running one

```
uv run train.py --env-config-path <config> --self-play \
    --snapshot-every-n-epochs 25 --pool-capacity 8 \
    --pool-anchor squad_march_take --pfsp-mode hard --seed 1
```

Snapshots land in `checkpoints/<run>/pool/`, beside that run's checkpoints —
`<run>` being the full name, timestamp and `--run-suffix` included. ⚠ **It was
the run *base* until 2026-09-01**, so the pool went to a sibling directory and
every self-play run on one env config wrote the same filenames into it. A pool
entry holds a path loaded lazily at seating time, so two **concurrent** runs
would have seated each other's weights as their own past selves, silently.
Sequential runs were unaffected — the pool is in-memory and never scans its
directory. Pinned by `tests/test_train_run_name.py`. The pool logs its size and the mean
epoch of the opponents drawn, so a pool that has collapsed onto its newest member
is visible in the dashboard rather than only in the score.

⚠ **Run `just measure-seat-parity` on the scenario first.** § 4 is the reason.
It costs minutes and no GPU. Run 2026-08-31 on two configs: it **fails** on
`25v25_shooting_opponent.yaml` (**−24.6 ± 9.4 vp**) and **passes** on
`25v25_maps_two_mode.yaml`, the config that trains (**+6.5 ± 6.1 vp**,
`squad_march_take` on both seats, 120 layouts). ⚠ **Run it at n ≥ 100** — the
same gate at n=30 read +19.1 ± 11.2, within 15% of failing, and that estimate did
not survive quadrupling the layouts.

### What a self-play run does and does not void

**It changes who the learner trains against, and nothing else.** The scheduler
seats only `_ensure_rollout_envs()` (`model/ppo/lightning.py:691`); `_eval_envs`
is a separate list built in `lightning_base.py:215` and is never touched. So
held-out scoring runs on the `configs/evaluation/` family against their own fixed
scripted opponents, exactly as every published row did.

**The existing agent table is therefore the comparator, and no bar needs
re-measuring.** That is worth stating because the default assumption in this
repo — earned four times over — is that a change of this size voids the
baselines. An earlier draft of this document said so too, and it over-scoped
itself.

What it *does* void is the reading of the in-run `eval/baseline_*` keys: those
are still measured against the config's own opponent and no longer describe what
the learner faced.

The arm, its criteria and its power check are pre-registered in
[reports/2026-08-31-self-play-preregistration.md](../reports/2026-08-31-self-play-preregistration.md),
written before any number existed.

---

## Costs and open work

- **The opponent force keeps its own coherency tracker when asked**, so an
  entrant seated as B carries the legality column. It is **opt-in**
  (`track_opponent_coherency`, default off) and the rating arena switches it on
  for its own legs — it costs an extra coherency evaluation per opponent
  movement phase and nothing outside a rated leg reads it, so no training or
  scoring path pays for it. Off, no tracker object exists at all.
- ⚠ **WP-4 is deliberately NOT run, and the refactor stays gated.** The
  throughput gate on the re-entrant `active_side` / `observation_for` / `apply`
  refactor needs a `model` opponent measured inside a rollout, and D-03 says to
  report it either way. **No figure is recorded, on purpose:** the env and the
  action space are both being changed elsewhere, and a throughput number is a
  property of the scenario that produced it. This repo's own rule is to *measure
  what ships*, and what ships here is not settled — a figure taken now would be
  quoted later against an env that no longer exists, which is worse than none.

  When it is run, **the aggregate will not answer the question.** The refactor
  removes the *second, mirrored observation build* the opponent seat needs; it
  cannot touch a forward pass. So "a network opponent is expensive" is not on
  its own an argument for it — the gate has to split the opponent's turn into
  **building its view** and **choosing its move**, and only the first of those is
  what the refactor removes. `measure-throughput` does not make that split
  today; adding it is the first step of running WP-4.

  Until then **do not attempt the refactor**. If a model opponent does prove
  expensive, the first thing to try is batching its forward pass across rollout
  envs — the trick `_run_episodes_batched` already uses for evaluation.
- **No self-play run has been trained.** Everything above is mechanism.
- **`learner_side`** — training the opponent seat — is Phase 05, following the
  `augment_start` precedent.
