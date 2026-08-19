# Phase 4: Elo-Based Self-Play - Context

**Gathered:** 2026-08-19
**Status:** Ready for planning
**Design:** [docs/elo.md](../../../docs/elo.md), [docs/self-play.md](../../../docs/self-play.md)

<domain>
## Phase Boundary

Put scripted baselines and learned checkpoints on **one rating scale**, so "did this
get better" has an answer that does not depend on which opponent it happened to be
measured against — then use that scale to drive self-play opponent sampling.

Phase numbering follows `docs/goals-and-roadmap.md` § Phase 4 (Opponent AI &
Self-Play), not the v2.0 milestone's internal 01/02.

**Elo goes first and self-play consumes it.** `docs/elo.md` notes a rating table needs
no model change on a symmetric config. Exploration established it needs **no env
change either**: the four-leg schedule's two axes are already config fields, and the
`scripted_baseline` opponent policy already ships. So a complete rating table over the
nine scripted baselines is reachable with zero changes to `envs/` or `model/`.

**In scope:** the rating mathematics (margin score, Bradley–Terry fit with explicit
zone and first-turn advantage terms, layout bootstrap); the four-leg schedule as
config transforms; an arena that wraps `evaluate_selector`; an append-only ledger
keyed by a scenario fingerprint that refuses to mix; consolidation of the four
duplicate policy-or-checkpoint resolvers; the mirror extended to the observation path;
the `model` opponent policy; the snapshot pool and PFSP sampling.

**Out of scope:** the **size-agnostic policy** work (taking enemy model and unit counts
out of weight shapes). It is required only for *asymmetric* armies; every golden config
here is symmetric 25v25, so it buys nothing yet and costs a full retrain. See
[docs/size-agnostic-policy.md](../../../docs/size-agnostic-policy.md). Also out:
coordinate canonicalisation (a sample-efficiency claim needing its own screen), melee,
morale.

**Gated, not scheduled:** the re-entrant `active_side` / `observation_for` / `apply`
core. It is sequenced *after* a throughput measurement that can cancel it.
</domain>

<decisions>
## Implementation Decisions

### Sequencing
- **D-01:** **Elo before self-play.** The rating subsystem is standalone, needs no
  retrain, and produces a result (the zone and first-turn advantages in Elo points)
  before any training happens. Self-play then reads `p` from the rating table rather
  than maintaining a second win-rate estimate.
- **D-02:** The size-agnostic policy work is **deferred, not cancelled**. Consequence
  accepted: it changes weight shapes, so no checkpoint rated before it can be loaded
  after it. Those ledger rows stay as history, marked unreproducible.
- **D-03:** The two-sided env refactor is **gated on a throughput measurement**
  (WP-4), which must be reported either way. Do not refactor on an unmeasured
  assumption.

### Architecture
- **D-04:** Two new packages, both **peers of `model/`, above `envs/`**:
  `wargame_rl/wargame/rating/` and the selector consolidation in
  `model/common/selector.py`. Rating must drive a live env *and* reach checkpoints;
  `model → envs` is currently one-way (verified: zero references to `wargame.model`
  anywhere under `envs/`), so rating cannot sit beside `envs/baseline/` without
  inverting it.
- **D-05:** Inside `rating/`, the split that matters is **torch-free / env-free**, not
  play/fit. `rating/score.py` and `rating/elo.py` import **numpy only** — no project
  imports at all — so the rating mathematics is unit-testable on synthetic arrays.
  `rating/arena.py` is the **only** module in the package that imports `WargameEnv`.
- **D-06:** **`scipy` stays out.** The fit is Newton–Raphson on a convex objective,
  ~30 lines of numpy on a (n+2)-square Hessian. `math.erf` covers anything Gaussian.
- **D-07:** The arena **wraps** `evaluate_selector` via `set_opponent_policy`; zero
  lines change in `envs/baseline/evaluate.py`. Opponent identity is not a parameter of
  the scoring loop — it is state on the env, and it is the arena's knowledge.
  `measure_paired_policies.py:69` already documents why a second scoring loop is the
  wrong answer.
- **D-08:** The unit of work is the **leg**, not the layout. `docs/elo.md`'s
  `play_match(..., layout_seed, ...)` discards env reuse and reimplements the loop; a
  network entrant must load its checkpoint once per leg, not once per layout.

### The schedule
- **D-09:** A leg is a **config transform**, not new env code. Who moves first is
  `turn_order: player | opponent` (`types/config/env.py:429`); A's zone is a swap of
  `deployment_zone` ↔ `opponent_deployment_zone`. Entrant A always sits on the player
  seat; entrant B rides in `opponent_policy`.
- **D-10:** `config_for_leg` **raises** on a `None` deployment zone and on
  `has_fixed_model_positions` / `has_fixed_opponent_positions`. `battle_factory.py:185`
  derives zone defaults, so swapping two `None`s is a silent no-op — `h_zone` would
  then fit noise and report it as a number. Refusing is better.
- **D-11:** One env per leg, four total, reused across all layouts. Never mutate a
  live env's config: `turn_order` happens to be read only at reset, but `_skip_phases`,
  `max_turns` and both `ActionHandler`s are cached at `__init__`.

### The ledger
- **D-12:** The fingerprint is over the **scenario**, and must **exclude** `turn_order`
  and `opponent_policy` and **canonicalise** the zone pair and army pair (sorted).
  `docs/elo.md` says only "excluding rendering and logging fields" — that puts the four
  legs of one table into four different ledgers. Guard:
  `test_all_four_leg_configs_fingerprint_identically`.
- **D-13:** The ledger stores **raw per-layout legs, not fitted ratings**. The
  bootstrap resamples layouts and needs them; adding one entrant would otherwise mean
  replaying every pairing; recalibrating `s_m` would mean replaying everything.
- **D-14:** Mixing two fingerprints is **refused**, not warned about. CLAUDE.md records
  what happens to warnings (TF32, `last.ckpt`).
- **D-15:** `ratings/` is **committed**, unlike `checkpoints/` and `recordings/` — it
  is a durable measurement artefact in the class of `reports/`. Add it to the
  `.claude/hooks/docs_check.py` exemption list.
- **D-16:** New seed band **900_000**, disjoint from rollout 0 / baselines 10k / eval
  500k / held-out 700k / clone 800k.

### The `model` opponent policy
- **D-17:** Split in two. `envs/opponent/selector_policy.py::SelectorOpponentPolicy`
  is torch-free and seats any `ActionSelector` on the opponent side;
  `model/opponent/network_policy.py::NetworkOpponentPolicy` subclasses it and calls
  `register_policy("model", ...)` at import. Registration flows **downward** into the
  lower layer's registry, preserving `model → envs`. A single
  `envs/opponent/model_policy.py` importing `net.py` would be a DDD violation *and* a
  real import cycle (`net.py` imports `envs.wargame`).
- **D-18:** `TransformerNetwork.from_env` is split into `spec_from_observation` +
  `from_spec`, and the network is sized **lazily on first `select_action`**.
  `from_env` (`net.py:648`) calls `env.reset()` and reads `env._action_handler`
  (`:657`). `build_opponent_policy` runs at `wargame.py:296` *inside*
  `WargameEnv.__init__`, so a `from_env` call there recurses into the half-built
  policy and consumes the layout RNG; and `env._action_handler` read through the
  mirror's `__getattr__` falls through to the **player's** handler, silently sizing the
  opponent network with the wrong `n_actions`.
- **D-19:** `shoots` is **derived**, not declared:
  `env.opponent_action_handler.shooting_slice is not None`. Same "cannot forget"
  discipline as `ScriptedBaselineOpponentPolicy`'s `select_shooting` identity check.
- **D-20:** Pass `action_registry=None` to `build_observation` and **splice in** the
  mask the env already computed (`_opponent_action_mask`, `wargame.py:928`). Letting
  `build_observation` recompute it runs `compute_unit_shooting_masks`' LOS pass twice
  per shooting phase.
- **D-21:** `opponent_max_ranges` goes on **`WargameEnv`, not `BattleView`**,
  contradicting `docs/self-play.md`. Its only consumer is the mirror, which is cast to
  `WargameEnv`; widening the view would push a property onto the replay adapter that
  `GameStateSnapshot` cannot supply. Same reasoning that produced `DebugView`. Add
  `WargameEnv.player_side` likewise.
- **D-22:** A symmetry precondition is **raised in `__init__`**:
  `number_of_wargame_models == number_of_opponent_models` and equal unit counts.
  `net.py:301` `_alive_feature_index = feature_dim - n_opponents - 10` counts
  backwards and `_alive_from_features` falls back to **all-alive** when out of range —
  it degrades silently. This check is the only thing between that and a
  plausible-looking wrong number.

### Claude's Discretion
Newton step count and convergence tolerance; bootstrap resample count (default ~500);
the exact `s_m` starting value for calibration; table column formatting; whether the
edge cases in `canonical_scenario` are dropped by an explicit deny-list or an
allow-list.
</decisions>

<canonical_refs>
## Canonical References

- `docs/elo.md` — the rating design. **Corrections D-12, D-13 and D-08 apply.**
- `docs/self-play.md` — the self-play design. **Correction D-21 applies**, and
  `last_player_shooting_results` on the mirror is not needed (only renderers read it,
  and the same doc says the mirror is not a rendering surface).
- `docs/ddd-envs.md` § Dependency direction — the rule D-04 and D-17 are argued
  against. Note `baseline/` and `debug/` are documented exceptions that may hold a
  `WargameEnv`.
- `docs/metrics.md` § Coherency, § The noise floor, § Pairing beats sample size —
  every reported row carries `coherent` and `adrift`; per-episode `vp_margin` sd is
  45–50, so pairing is the difference between a result and an artefact.
- `docs/opponent-policies.md` § Planned Policies — the reserved `model` key.
- `CLAUDE.md` § Training Runs — the seed-band table, the decode rules, and the
  standing instruction never to quote a score without saying how it was decoded.
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ActionSelector` (`envs/baseline/evaluate.py:32`) — the alias that already unifies
  scripted policies and checkpoints. Everything in this phase is built on it.
- `evaluate_selector` (`evaluate.py:251`) — plays a leg unchanged. Already returns
  `vp_margin_per_episode`, `win_per_episode` and `objectives_held_per_episode` in seed
  order, which are exactly the raw rows the fit and the bootstrap need.
- `set_opponent_policy` (`wargame.py:326`) — already exists, and `reset()` never
  touches `_opponent_policy`, so an arena-seated opponent persists across seeds.
- `turn_order` (`types/config/env.py:429`) — the first-turn axis, already a config
  field. `docs/elo.md` plans a mechanism for this; it exists.
- `ScriptedBaselineOpponentPolicy` + `_MirroredEnv`
  (`envs/opponent/scripted_baseline_policy.py`) — the seat swap, shipped in PR #203.
- `_opponent_action_mask` (`wargame.py:928`) — a fully rules-legal opponent mask
  including range and LOS, already computed and passed to `select_action`.
- `env._opponent_action_handler` (`wargame.py:287`) — the opponent seat already has a
  correctly-sized action space, including its shooting slice.
- `decode_joint_coherent` (`model/common/decoding.py:166`) — its env reads are
  `config`, `game_clock_state`, `opponent_models`, `player_action_handler`,
  `player_models`, `rules_quantities`. **Every one is already mirrored or genuinely
  shared**, so joint constrained decoding works for a network opponent unchanged.
- `standard_error` / `paired_difference` / `mean_of_measured` (`evaluate.py:127-159`).
- The sha256-over-`tobytes` idiom in `tests/test_start_state_augmentation.py:104`.

### Established Patterns
- Scripts are **raw `sys.argv`**, not Typer — `python -m scripts.<name>`, module
  docstring as `--help`, optional args arriving as **empty strings**.
- Justfile recipes: a comment block of rationale, then
  `@uv run python -m scripts.<name> {{a}} "{{b}}"`. Variadic params must come last
  (`+entrants`).
- Reset options are parsed **inline** with no schema, and `augment_start` is the
  precedent to copy: opt-in, and it **draws nothing when not requested**, so a config
  carrying the field produces bit-identical layouts.
- `_MirroredEnv.__getattr__` reads `self.__dict__.get("_env")`, not `self._env` —
  load-bearing, because `copy.deepcopy` reconstructs without `__init__` and Lightning
  deep-copies the env in `save_hyperparameters` (PR #204).

### Integration Points
- `envs/baseline/evaluate.py` — **untouched**. The arena wraps it.
- `model/common/factory.py::create_environment` — gains one `# noqa: F401` import so
  the `model` opponent key registers. Every training, evaluation and scoring path goes
  through it.
- `model/common/lightning_base.py::on_train_start` (`:137`) — already the only place
  scripted opponents are scored inside training, on its own seed base. The natural
  hook for logging Elo against the frozen anchors.
- The four duplicate policy-or-checkpoint resolvers — `scripts/measure_maps.py:101`,
  `debug.py:61`, `scripts/measure_paired_policies.py:54`,
  `scripts/measure_income_share.py:63` — **with two different precedences**
  (path-first vs registry-first). Consolidate rather than add a fifth.
</code_context>

<specifics>
## Specific Ideas

- **Seat parity must be measured before any rating is trusted.** The reward,
  coherency and exposure trackers are player-side only, and
  `terminate_on_player_elimination` is player-only. Those are measurement-only or
  default-off, so the game *should* be seat-symmetric — but that is a claim. Put
  `squad_march_shoot` on both seats over the balanced four legs and assert the
  aggregate margin is ≈ 0. A non-zero result is a bug found, and it blocks the phase.

- **`turn_order: random` consumes a layout-RNG draw.** `_resolve_player_side`
  (`wargame.py:1022`) draws at `:1029` and runs at `:781`, *before* the map-pool draw
  and `place_for_episode`. `configs/golden/25v25_shooting_opponent.yaml:162` sets
  `random`. So pinning turn order for a leg **shifts the layout stream off that
  config's own baseline seeds**. The four legs agree with each other, which is what
  the fit needs — but the ΔElo ↔ `vp_margin` cross-check is then across different
  layouts. Say so in the recipe help; pin it with
  `test_the_four_legs_share_terrain_and_objectives`.

- **The tripwire test has a hole worth documenting rather than fixing.**
  `tests/test_scripted_baseline_opponent.py:156` intersects reads with a
  `SIDE_SPECIFIC` set, so a *newly invented* side-specific name nobody adds to that set
  falls through silently. It catches new *reads* of known names, not new names. The
  real guard is the swap-invariance tensor comparison, which cannot be fooled. Say so
  in the docstring so the next person does not over-trust the enumeration.

- Ratings are only comparable at a fixed decode. Record `decode_topk` per ledger row
  and refuse a table that mixes decodes — CLAUDE.md's standing rule is never to quote
  a score without saying how it was decoded.
</specifics>

<deferred>
## Deferred Ideas

- **Size-agnostic policy** (`U` and `P` out of the weight shapes, padded collation).
  Needed only for asymmetric armies. Its two latent bugs — the `group_span` pooling
  desync and the group-id aliasing, both invisible at 25 models / `max_groups: 5` —
  are worth fixing on their own schedule; `group_span` rounding up fixes both and is
  bit-identical on all 37 shipped configs.
- **Coordinate canonicalisation** (reflecting the board so "my zone" is always the
  left). A sample-efficiency claim, so it needs its own screened arm.
- **The re-entrant symmetric core** — gated on WP-4's number.
- **`learner_side` reset option** — Phase 05, following the `augment_start` precedent
  verbatim.
- **`EpisodeProvenance` opponent slot** — not needed. `provenance.config` already
  carries `opponent_policy.params.checkpoint`, so a `model`-opponent episode seated by
  config *is* reproducible. The arena does not record by default; to record a rated
  match, seat B via config rather than `set_opponent_policy`.

### Reviewed Todos (not folded)
- The open bugs in memory (`polygons_contain_points` padded-outline bug, LOS
  asymmetry, `_cover_mask` hidden-model cover) all touch the same board every rating
  is measured on. None blocks this phase, but each shifts every number when fixed —
  which is exactly what the ledger's code-revision field is for.
</deferred>

---
