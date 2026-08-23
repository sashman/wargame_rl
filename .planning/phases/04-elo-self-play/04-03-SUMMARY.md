# 04-03 Execution Summary

Wave 3: the mirror reaches the observation path, and a checkpoint can play the
opponent seat. Both golden gates stay bit-identical.

## Shipped

| Module | Owns |
|---|---|
| `envs/opponent/mirror.py` | `MirroredEnv`, moved out of `scripted_baseline_policy` and extended from 5 overrides to 15 |
| `envs/opponent/selector_policy.py` | `SelectorOpponentPolicy` — seats any `ActionSelector` on the opponent side, torch-free |
| `model/opponent/network_policy.py` | `NetworkOpponentPolicy`, registering the reserved `model` key |
| `model/net.py` | `NetworkSpec`, `spec_from_observation`, `from_spec`; `from_env` reimplemented on top |
| `envs/wargame.py` | `player_side` and `opponent_max_ranges` accessors |

The compat alias `_MirroredEnv` was **not** kept: the only consumer was one
test, so it now imports the real name and there is one name for the class.

## The layering, and what it cost

`envs` importing `model` would be a dependency inversion *and* a real import
cycle, since `net.py` imports `envs.wargame`. So the policy is split: the
env-layer half seats any selector and is torch-free; the model-layer half
subclasses it and calls `register_policy("model", ...)` at import.

Registration therefore flows **downward**, and
`model/common/factory.create_environment` performs that import — every
training, evaluation and scoring path goes through it. A direct
`WargameEnv(config)` on a `model` config raises with a message naming the
import that fixes it, which is the whole price of keeping the arrow one-way.
`scripts/measure_elo.py` imports it explicitly for the same reason: `rating`
never imports `model`, so pulling a registration into the arena would have
inverted *that* arrow instead.

## Three things that would have been silent

**`from_env` cannot size the opponent's network.** It calls `env.reset()` and
reads `env._action_handler`. `build_opponent_policy` runs *inside*
`WargameEnv.__init__`, so a reset there re-enters a half-built env — and would
consume the layout RNG, shifting every seeded episode. Worse, `_action_handler`
read through the mirror's `__getattr__` falls through to the **player's**
handler, sizing the opponent network with the wrong action count, which on a
symmetric config is the same width and therefore invisible. Fixed by
`spec_from_observation` + `from_spec` and lazy sizing;
`test_a_model_opponent_draws_nothing_from_the_layout_rng` pins the first half
and `test_the_network_is_sized_from_the_opponent_seat` the second.

**Unequal armies degrade silently.** `_alive_feature_index` counts back from the
trailing expected-damage block and `_alive_from_features` falls back to treating
**every row as alive** when the index lands out of range — so a network on the
opponent seat would read casualties as live models and never raise.
`require_equal_armies` refuses at construction, checking both model count and
unit count.

**The strict prefix conversion is deliberate.** `train.py`'s warm start uses
`strict=False` with no prefix rewriting, so a wrong prefix loads *nothing* and
trains a random network while reporting success. This path uses
`convert_state_dict`, which raises.

## The swap-invariance test, and three attempts to make it real

`tests/test_swap_invariance.py` builds `E`, builds `E'` with the two armies and
zones exchanged, and asserts the mirrored observation on `E` equals `E'`'s own,
tensor for tensor. Getting it to mean anything took three corrections, each of
which would have left a passing but vacuous test:

1. **Seeding both envs alike is not enough.** `place_for_episode` fills each
   army into its own zone, and the zones differ between the two configs, so one
   seed gives two different boards rather than one board with roles exchanged.
   Positions are copied across instead.
2. **The game tensor carries shared state.** Round, phase and both scores had to
   be synced, or the comparison failed on progress rather than on sides.
3. **Objective distances are cached on the models.** `build_observation`
   recomputes them for the player side only when handed a distance cache, so
   copying locations without refreshing left `E'` describing where its models
   used to be.

`test_the_comparison_would_notice_an_unmirrored_side` is the sensitivity
control: handing the real env where the mirror belongs must produce a different
observation, or the whole file proves nothing.

## The tripwire grew, and was verified

`test_scripted_baseline_opponent.py` now scans `observation_builder.py` and
`decoding.py` as well as the baseline package, matches `view.` as well as
`env.` — the observation builder names its parameter `view`, so an `env.`-only
pattern read none of the code the guarantee had just been extended to cover —
and classifies ten more names. **Verified sensitive**: deleting the mirror's
`player_max_ranges` fails it via `observation_builder`'s read.

Its docstring now states the hole plainly: the assertion intersects with
`SIDE_SPECIFIC`, so a *newly invented* side-specific name falls through in
silence. Swap invariance is the guard that cannot be fooled that way.

## Demonstrated end to end

Two checkpoints written, then rated against the scripted anchors on
`configs/dev/4v4_two_phases.yaml`, 8 layouts — a checkpoint genuinely playing
the opponent seat:

```
entrant                Elo      95% interval   vp margin   coherent   games
squad_march_shoot     +310      [+303, +318]      +134.5      1.000      64
netA                   +86        [+77, +95]       -16.6          -      64
random                  +0          [+0, +0]      -117.9      1.000      64
```

Ordering is what it should be: an untrained network beats random actions and
loses heavily to a competent script.

## ⚠ Known gap: an entrant seated only as B has no coherency figure

Visible in the table above — `netA` shows `-`. `evaluate_selector` measures the
player seat, and `pairings` lists each pair once in input order, so the entrant
named **last** on the command line is entrant B in every one of its pairings and
comes back with no coherency at all. That is a real gap against this repo's rule
that no score is quoted without it. Closing it needs the arena to track the
opponent's coherency, which the env does not currently measure for that side.
Recorded in `rating/table.mean_coherency`'s docstring; not fixed here.

## State

- `just format && just lint` clean; **1654 tests pass**, with the same four
  pre-existing failures (the known LOS-symmetry bug and three CUDA tests that
  pass under `CUDA_VISIBLE_DEVICES=""`).
- `test_reward_golden` and `test_observation_golden` **bit-identical**.
- `ratings/` deliberately empty: the only tables produced are on a dev fixture
  and on a scenario whose seat-parity gate fails.

## For whoever picks this up

1. **Rating `configs/golden/25v25_shooting_opponent.yaml` is still blocked** by
   the seat asymmetry — see `reports/2026-08-19-the-two-seats-are-not-the-same-game.md`.
   `just measure-seat-parity` is the gate and should be run on any config before
   it is rated.
2. WP-4 is the throughput spike: measure a `model` opponent in a rollout with
   `just measure-throughput`. That number decides whether the re-entrant
   `active_side` / `observation_for` / `apply` refactor happens at all.
3. The coherency gap above should be closed before any rating is published.
