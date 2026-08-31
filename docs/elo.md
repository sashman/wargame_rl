# Rating Policies on One Scale

Every score in this repo is quoted against a *particular* opponent, and
[CLAUDE.md](../CLAUDE.md) records what that costs: the same agent measured
**+61.4** against one opponent and **+20.8** against another, while being
**−75.9 behind the best script** in the first case and **−9.5** in the second.
Absolute score measures the opponent, not the agent, so "did this get better?"
has no answer that survives changing who it played.

A rating fixes that by construction. Put every policy — scripted baselines and
learned checkpoints alike — into one pool, play them against each other, and fit
a single number per policy such that the whole table of results is as likely as
possible. The result is comparable across opponents because *every* opponent is
in the fit.

- Related: [metrics.md](metrics.md) (what each measured column means) ·
  [self-play.md](self-play.md) (what consumes a rating) ·
  [opponent-policies.md](opponent-policies.md) (who can sit on the other side) ·
  [ratings/README.md](../ratings/README.md) (the ledgers themselves)

**Status:** the subsystem ships and is exercised end to end; **no rating is
published**, because the reference scenario fails the seat-parity precondition.
See [Open gaps](#open-gaps).

---

## 1. The scale: a margin, not a win

Standard Elo takes an outcome `S ∈ {0, ½, 1}`. That is the wrong input here, and
the repo already paid to learn why: **win rate cannot resolve differences under
~7pp** on these configs. TF32 cost 8.5 vp_margin on both seeds and moved win rate
only 0.705 → 0.65 — inside the noise, invisible. A rating built on win/draw/loss
inherits that blindness exactly.

So the margin enters through the **score**, not through the update rule:

```
s = 1 / (1 + exp(−m / s_m))        m = own VP − enemy VP, from A's side
```

This is not a redefinition of Elo. With `s_m` fitted so `s ≈ P(win | margin m)`,
the noisy 0/1 outcome has been replaced by its own conditional expectation —
same expected rating, strictly lower variance. A rating point still means a win
probability.

Three properties earn their keep (`rating/score.py`):

- `s(0) = 0.5` **exactly**, so a VP tie is a draw with no special case.
- It **saturates**. Per-episode `vp_margin` sd is 45–50 on 25v25, so blowouts are
  common and an unbounded score would let one layout dominate a rating.
- `margin_scale ≤ 0` degrades to the win indicator — the check that this
  *generalises* plain Elo rather than replacing it.

`DEFAULT_MARGIN_SCALE` is **50.0**, pinned rather than fitted per run so a rating
reproduces. `fit_margin_scale` re-derives it from a recorded corpus, and the
fitted value belongs in the ledger beside the ratings.

## 2. The fit: Bradley–Terry with the board's two advantages

Ratings are **fitted, not accumulated**. Sequential `K`-factor updates exist
because online chess cannot re-fit; here the pool is fixed and any match can be
replayed, so a sequential rule would only make a rating depend on the *order*
games were played in — gratuitous irreproducibility in a project whose training
is otherwise bit-reproducible.

```
E[s] = 1 / (1 + 10^(−(R_A − R_B + σz·h_zone + σt·h_turn) / 400))

  σz = +1 if A deployed in zone 1, −1 if zone 2
  σt = +1 if A moved first,        −1 otherwise
```

`h_zone` and `h_turn` are the **deployment-zone and first-turn advantages in Elo
points**, shared across every pair. They are not a nuisance correction: the fit
*reports* them, which is a result in its own right and available before any
self-play training happens. On the dev fixture the first-turn term comes out
around **+16 Elo**; the seat-parity work found it **changes sign with shooting**
— movement-only, going first is worth **+59.2 Elo**.

Both terms are identified **only because the schedule varies the two axes
independently**. A schedule that moved them together would confound them into one
number from which neither is recoverable; `fit_ratings` **refuses** such a design
rather than regularising it, because a table reporting a zone advantage it could
not have measured is worse than no table.

Implementation notes (`rating/elo.py`):

- Newton–Raphson on a convex objective, ~30 lines of numpy on an `(n+2)`-square
  Hessian. **No scipy**, deliberately.
- A weak Gaussian prior (`σ = 400`) keeps an undefeated entrant from running to
  infinity — `squad_march_shoot` has genuinely scored a 1.00 win rate here — and
  keeps the Hessian positive definite under perfect separation.
- One entrant is **anchored** at 0 (default `random`). Without a fixed reference
  "zero Elo" means nothing, and `rate()` raises if the anchor did not play.

### The interval comes from a bootstrap over layouts

Resample **layouts**, not rows: the four legs played on one layout are one piece
of evidence, not four. Default 500 resamples.

⚠ The original design justified the bootstrap by arguing the quasi-likelihood
Hessian *understates* the error. **The premise is right and the conclusion is
backwards** — any `[0,1]`-valued score has variance at most `p(1−p)`, so the
Hessian interval is if anything too *wide*, and measured it came out **118 wide
against the bootstrap's 27**. The bootstrap stands; only the reason changes.
What the Hessian cannot see is **dependence between rows** — the same layout, the
same armies, four correlated legs.

**Quote the interval, not the point.** A rating without one is the same failure
as a `success_rate` with no floor and no bar.

## 3. The schedule: four legs, as config transforms

A rated pairing plays every combination of the two axes the board is imbalanced
on:

| Leg | A's zone | Moves first |
|-----|----------|-------------|
| 1   | zone 1   | A           |
| 2   | zone 1   | B           |
| 3   | zone 2   | A           |
| 4   | zone 2   | B           |

**Neither axis needs environment code.** `turn_order` is already a config field
read only by `_resolve_player_side`; A's zone is a swap of `deployment_zone` ↔
`opponent_deployment_zone`; and `scripted_baseline` already seats a player-side
baseline on the opponent seat. A rated leg turned out to be a **config transform,
not new env code** — which is why the whole subsystem landed with nothing changed
under `envs/` except one latent bug it exposed.

`config_for_leg` **raises** rather than producing a config on which the zone axis
does nothing. Three ways that happens, all silent: a `None` deployment zone (the
battle factory derives the defaults, so swapping two `None`s is a no-op), fixed
model positions (which override the zone entirely), and **a map pool**. If the
axis were inert, `h_zone` would fit noise and report it as a number.

⚠ **The map-pool case is why this section needed revising, and it means the
`maps` family cannot be rated as things stand.** A drawn table carries its own
deployment *outlines* (`TerrainMapConfig.deployment`), and placement accepts
against those; the config's `deployment_zone` rectangles survive only as the
sampling bounds. The outlines stay bound to the player and opponent seats across
the swap, so swapping the rectangles moves the box and not the zone — and
swapping the boxes alone samples one army against the *other* side's outline.
The schedule was written on 2026-08-17, when the rectangles *were* the zones;
the tables were regenerated with their own polygons on 2026-08-20 and this was
not revisited. Rating a pool config in between would have fitted `h_zone` from
noise **on the config that trains**.

The refusal keys on `map_pool` itself rather than on whether the pool's maps
carry outlines: `rating/` may import `envs/types` and nothing else, so it cannot
read the map files. A pool whose maps all left `deployment` unset would be safe
and is refused anyway — no shipped pool is like that (54 of 54 map files carry
one), and over-refusing costs a config while under-refusing costs a published
number. The first ledger therefore goes on a config that deploys under its own
rectangles, such as `25v25_shooting_opponent.yaml` or `25v25_cover_control.yaml`
— ⚠ noting that the first of those is the config that **fails** seat parity.

Layout seeds come from the **900 000** band, disjoint from rollout (0), in-run
baselines (10k), in-run eval (500k), held-out scoring (700k) and behaviour
cloning (800k) — so a rated match is played on layouts nothing else in the repo
reports on. Combat seeds are offset by 1 000 000 and held **fixed across a
pairing's four legs**: the dice are a bigger source of spread than the scenario
(sd 50.6 within a layout against 45.0 between), so pinning them is what makes the
four legs a comparison rather than four samples.

### The arena wraps the scoring loop, it does not reimplement it

`rating/arena.py` is the **only** module in the package that touches a live
`WargameEnv`, and it calls `evaluate_selector` unmodified. Opponent identity is
not a parameter of the scoring loop — it is *state on the env*, and it is the
arena's knowledge. Two implementations of "score a policy over seeds" drifting
apart is the exact defect `measure_paired_policies` documents guarding against.

The unit of work is the **leg**, not the layout: one env per leg, reused across
every layout, so a network entrant loads its checkpoint once rather than once per
layout.

## 4. The ledger and its fingerprint

One file per scenario under [`ratings/`](../ratings/README.md), named by a
16-hex-character digest of the scenario.

**The ledger stores raw per-layout legs, not fitted ratings.** Three reasons,
each sufficient on its own: the bootstrap resamples layouts and needs the rows;
adding one entrant would otherwise mean replaying every pairing; and
recalibrating the margin scale would mean replaying everything. `just elo-table`
fits on read.

### The fingerprint

A rating means something only *within* one scenario, so **two fingerprints in one
ledger is refused, not warned about**. A warning in a log is what the TF32 and
`last.ckpt` episodes show gets ignored.

⚠ **The fingerprint must EXCLUDE `turn_order` and `opponent_policy`, and SORT the
zone and army pairs.** The original design said only "excluding rendering and
logging fields" — fingerprinting the resolved config that way puts the four legs
of one pairing into four different ledgers and breaks the feature outright.
`turn_order` is a leg axis; `opponent_policy` is entrant B; the zone pair is
sorted so a zone swap leaves the fingerprint unchanged while a genuinely
different board does not.

Rule of thumb for anything added later: *if it changes what happens on the board,
it belongs in the fingerprint.*

Every leg carries a **`code_revision`**. Three open bugs
(`polygons_contain_points`' padded outlines, the LOS asymmetry, `_cover_mask`'s
hidden models) all touch the board every rating is measured on; fixing any of
them shifts every number in every ledger, and the revision field is how a stale
ledger is recognised rather than silently re-fitted.

## 5. Seating a checkpoint on the other side

`model` is the opponent-policy key that plays a checkpoint. It is split across
the layer boundary on purpose: `envs/opponent/selector_policy.py` is torch-free
and seats any `ActionSelector`; `model/opponent/network_policy.py` subclasses it
and registers under `model` at import. Registration flows **downward** into the
lower layer's registry, so `model → envs` stays one-way. A single
`envs/opponent/model_policy.py` importing `net.py` would be a dependency
inversion *and* a real import cycle.

Two things that bit, and are now pinned by tests:

- **`from_env` cannot size the opponent's network.** It calls `env.reset()` and
  reads `env._action_handler` — but the policy is built *inside*
  `WargameEnv.__init__`, so a reset there re-enters a half-built env and consumes
  the layout RNG; and read through the mirror, `_action_handler` falls through to
  the **player's** handler, sizing with the wrong action count, silently, on a
  symmetric config. Hence `spec_from_observation` + `from_spec`, sized lazily on
  first `select_action`.
- **`shoots` is derived, not declared** — from whether the opponent's action
  handler has a shooting slice. Same "cannot forget" discipline as
  `ScriptedBaselineOpponentPolicy`'s `select_shooting` identity check.

## 6. Recipes

```
just measure-seat-parity <env_config> [policy] [n_layouts]   # the precondition
just measure-elo <env_config> [n_layouts] <entrant> ...      # play legs, append
just elo-table <env_config>                                  # fit and print
```

An entrant is a baseline registry name or a path to a `.ckpt`, resolved by the
single `wargame_rl/wargame/selectors.py` resolver.

---

## Preconditions

**Enforced in code:** `require_symmetric` refuses a scenario with unequal armies,
raising in `play_leg`. This is not fastidiousness — `net.py`'s
`_alive_feature_index` counts backwards assuming the trailing expected-damage
block is exactly `n_opponents` wide, and `_alive_from_features` falls back to
**all-alive** when the index lands out of range. It degrades silently rather than
raising, and this check is the only thing between that and a plausible-looking
wrong number.

⚠ **Not enforced in code: seat parity.** See below.

## Open gaps

> **Closed in wave 4.** The seat is now a fitted term (`h_seat`) and the
> opponent force carries its own coherency column. What remains open below is
> what the fix does *not* cover -- read it before quoting either.

### The seats are not a balanced axis, and the gate is advisory

The four legs balance zone and turn order. They do **not** balance the engine
seat: entrant A always sits on the player seat and entrant B always rides in
`opponent_policy`. `pairings()` lists each unordered pair once **in input order**,
so the entrant named *first* on the command line takes the player seat in every
one of its pairings and the entrant named *last* never takes it at all.

That is harmless if and only if the two seats are the same game. On
`configs/golden/25v25_shooting_opponent.yaml` **they are not** — one policy
played from both seats over the balanced four legs loses from the player seat by
**−24.6 ± 9.4 vp**, and the residual survives averaging turn order out
([report](../reports/2026-08-19-the-two-seats-are-not-the-same-game.md)). On such
a config, ratings are confounded by command-line position.

`just measure-seat-parity` is the gate, and **nothing calls it**. It is a
precondition a human is expected to check before rating a scenario. Options, none
taken yet: refuse to rate a scenario with no recorded passing gate; record the
gate's result in the ledger beside `code_revision`; or add a seat term to the
model and balance it in the schedule — which doubles the legs, and the current
`pairings()` docstring's claim that ordered pairs "add no information" is only
true when the seats are equal.

### An entrant seated only as B has no coherency figure

`evaluate_selector` measures the player seat, so `LegResult.coherency_rate` is
entrant A's. Combined with the seating above, the last-named entrant comes back
with `-` in the coherency column — a real gap against this repo's rule that **no
score is quoted without coherency**, since a `vp_margin` alone is a result plus an
unstated claim that the moves earning it were legal.

**Closed in wave 4.** The opponent force keeps its own `CoherencyTracker`, and
`mean_coherency` fills the column from whichever seat an entrant occupied. It is
**opt-in** — `track_opponent_coherency`, default off, switched on by `play_leg`
because a rated leg is the only caller that needs it. The flag is dropped from
the scenario fingerprint: it is instrumentation and changes no outcome, and
leaving it in would fingerprint every rated leg differently from the same
scenario measured any other way.

⚠ Read it with the standing warning: a coherency *rate* rises whenever an army
dies, since a unit reduced to one model is coherent by definition. The
opponent's `models_out_of_coherency` is carried for the same reason the
player's is.

### The neutral cadence is not neutral

The control used to refute the scoring-cadence explanation for the seat gap is
sampled at step boundaries, so it always lands just after the opponent's turn.
The refutation stands — the neutral gap is the *larger*, which is the wrong
direction for the cadence to be the cause — but the −0.331 figure is an upper
bound.

---

## Corrections to the original design

Recorded because the history is the evidence.

1. **The fingerprint** must exclude the leg axes and the entrant, and sort the
   zone pair. Fingerprinting the resolved config, as first written, breaks the
   feature outright. Found by a test.
2. **The bootstrap's justification** was backwards; the bootstrap itself was
   right. See [§2](#the-interval-comes-from-a-bootstrap-over-layouts).
3. **A confounded schedule is refused, not regularised.**
4. **`opponent_max_ranges` is on `BattleView`, not just `WargameEnv`.** The
   design argued it should stay off the protocol because the replay adapter reads
   a `GameStateSnapshot` and could not supply it. The threat-overlay work
   (#224) put it on the protocol and had the adapter derive it from the snapshot,
   so the argument no longer holds. `WargameEnv.player_side` remains env-only.
5. **The scoring cadence is not the seat gap's cause** — proposed here, refuted
   by its own control one commit later.
