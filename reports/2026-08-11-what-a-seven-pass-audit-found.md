# What a seven-pass audit found, and the shape of the errors

**2026-08-11** · seven parallel review agents over `experiment/objective-income`
· no GPU time

This is not an experiment report. It records **how** a day's conclusions were
checked and what the checking found, because the pattern in the errors turned out
to be more reusable than any of the individual corrections.

The corrections themselves are written where they belong and are **not repeated
here**:

- [the spread-objectives report](2026-08-11-spreading-objectives-lowers-the-bar-not-the-abandonment.md)
  carries three correction sections, in place, beside the claims they retract
- `.planning/HANDOVER-2026-08-11.md` carries the state
- `docs/rules/implementation-status.md` carries the rules divergences

## The finding that generalises

**Every error was in how data was bucketed or paired. None was in the code under
test.** Seven passes, four retracted claims, and not one of them was a bug in the
environment, the reward, or the network. The probes all ran correctly and measured
something other than what their author believed.

| what was claimed | what the probe actually measured |
|---|---|
| "walking pays half of standing still" | movement steps against a bucket that was two-thirds *shooting* steps |
| "the policy leaves +3.26 reward on the table" | the **team scalar**, while PPO trains on the per-model vector — where the same deviation *costs* the movers 29.4 |
| "+21.1 vp_margin from a free squad" | a `zip` of two lists that diverged after the first skipped episode |
| "targets re-base 0.273 times a step" | one unavoidable initialisation per model per episode, counted as churn |
| "the globals are a floor that discourages walking" | a term added *identically* to walking and to loitering |

The common shape: **a quantity was compared across a boundary that also separated
something else.** Phase, currency, episode alignment, initialisation. In each case
the arithmetic was right and the comparison was not.

Two rules fall out, and they are cheap:

1. **Bucket a rate by whether the event could have been otherwise.** A rate that
   includes an event with no alternative is not a rate, it is a constant plus
   noise.
2. **In a phase-alternating environment where reward accrues unevenly across
   phases, any bucket keyed on a phase-specific event is secretly a phase
   indicator.** Split by phase, and subtract the broadcast terms, before
   comparing two activities. `last_per_model_reward[i]` is own-terms plus
   `shared_reward`, so the split is free.

## What the seven passes were, and what each caught

Composition mattered more than count. Three of the seven found things no other
pass would have.

| pass | what it uniquely caught |
|---|---|
| **Experimental statistician** | The phase artefact — the largest error of the day, upstream of most of the rest |
| **Adversarial falsifier** | Per-model vs scalar currency, and the broken pairing |
| **Determinism / measurement integrity** | A silent no-op on the serial rollout path; independently reproduced the golden digests from a parent-commit worktree |
| **Game-rules / domain** | That the augmentation starts 52/200 episodes with models already dead, and the placement-vs-reward coherency mismatch that became its own fix |
| **RL algorithms** | That γ = 0.9 gives a ~6.9-step credit window against a 6.6–8.7 step manoeuvre — so the "local optimum" framing may be unfalsifiable as posed |
| **Devil's advocate (strategy)** | That the bar is a target the experimenter moves — 64 vp of range in five days — and that the 45 held-out maps have never produced a reported number |
| **Conventions / docs drift** | Live docs asserting behaviour that had become false, including a stated "critical invariant" |

The two adversarial roles were worth their seats. The falsifier attacks specific
claims; the devil's advocate attacks the premise of the work. They do not
overlap, and the second produced the only argument that questioned the goal
rather than the method.

## The base rate is the argument for doing this

Eight self-retractions in eight days, every one found in-house. That is good
hygiene and a bad prior: **at roughly one retraction per day, a fresh claim has
no better than even odds of surviving the week.** Three training arms and one
environment feature were built on claims that did not survive.

The audit cost about an hour of wall-clock and no GPU. The arms it would have
prevented cost roughly seven GPU-hours *each*.

The corollary is not "audit everything". It is that **the cheap half of this
project has been the under-sampled half** — a probe costs minutes, an arm costs
hours, and the probes were the ones going unchecked.

## What was verified and survived

Worth recording, because a survived attack is a result:

- **Target pinning is not worth building** — mid-walk churn is 0.063/step, and
  96% of switches involve the `fallback_to_nearest` selection a pin would not
  touch. Independently reproduced at 0.066.
- **Agent and baseline are scored on identical layouts everywhere it is
  claimed.** The `eval/vp_margin` versus `eval/baseline_*` trap is correctly
  documented and not violated.
- **`PeriodicLastCheckpoint` behaves**, with one gap now recorded: SIGKILL
  triggers none of its three write paths, so a killed run's `last.ckpt` is up to
  25 epochs stale.
- **The augmentation cannot leak into a measurement.** Every `reset` call site
  was enumerated; no evaluation path passes `augment_start`.

## Outstanding

Not yet actioned, in rough order of what would bite first:

- [ ] `tests/test_polygon_terrain_and_area_objectives.py:421,469` read configs in
      `configs/experiments/`, which `configs/README.md` defines as *deleted once
      answered* — and the spread question **has** been answered. Deleting those
      arms per the README's own instruction breaks `just test`.
- [ ] No diagnostic that the start-state augmentation ever fired.
      `place_for_episode` discards the return value and nothing logs a rate, so
      a null result would be indistinguishable from "it never ran". Moot while no
      config enables it.
- [ ] `docs/metrics.md` has no section for `just measure-income-share`, though
      every other analysis script has one — and this one carries a real trap: a
      share of *mean* income does not bound a term's influence on a *choice*.
- [ ] `_AUGMENT_START` is a shared module-level mutable dict.
- [ ] The augmentation picks its objective uniformly, spending about a third of
      its dose on the point the opponent concedes.
- [ ] `terminate_on_success: true` plus an augmented start would end episodes at
      t≈1 with a near-full terminal bonus. Latent — no shipped config sets it.

## What this does not support

- **Not "the environment was broken".** It was not. Every retracted claim came
  from analysis code, not from the simulation.
- **Not "seven agents is the right number".** Three of the seven found the
  material defects. The value was in *role diversity* — a statistician, a
  rules lawyer and a strategist look at different things — rather than in count.
- **Not a verdict on the abandonment finding itself.** The observation survives:
  the agent wins the firefight ~2:1, keeps more models alive, and sends zero
  models to ~37% of objectives. Every *explanation* offered for it on 2026-08-10
  and 2026-08-11 has been withdrawn.
