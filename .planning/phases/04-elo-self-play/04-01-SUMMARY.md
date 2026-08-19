# 04-01 Execution Summary

Wave 1 of phase 04. Selector consolidation and the rating mathematics. Nothing
here plays a game, and nothing under `envs/` changed.

## `wargame_rl/wargame/selectors.py` — one resolver

Placed at the top level beside `envs/`, `model/` and `types.py`, **not** in
`model/common/` as first planned. `model/common/__init__.py` imports
`checkpoint_callback`, which pulls in pytorch-lightning and therefore torch, so
importing any submodule of that package executes it — which would have silently
undone `debug.py`'s deferred-torch behaviour. A single top-level module has the
same shape as the existing `wargame/types.py`.

```python
@dataclass(frozen=True, slots=True)
class ResolvedSelector:
    select: ActionSelector
    label: str
    kind: Literal["baseline", "checkpoint"]
    source: str | None
    network: TransformerNetwork | None

def is_checkpoint(spec: str) -> bool: ...
def label_for(checkpoint_path: str) -> str: ...
def build_action_selector(
    spec: str, env: WargameEnv, decode_topk: int = 1, decode_stay: bool = False,
) -> ResolvedSelector: ...
```

**Precedence is path-first**, the `measure_maps` behaviour. `is_checkpoint`
reads the `.ckpt` **suffix** rather than the filesystem, so a mistyped path is
reported as a missing checkpoint instead of an unknown baseline — the two
mistakes need different fixes.

Torch is imported inside `_resolve_checkpoint`, not at module scope.
`test_resolving_a_baseline_does_not_import_torch` asserts that in a subprocess,
because the rest of that test file has already imported torch into the parent.

### Call sites migrated — **seven, not four**

The plan named four. Two more turned up by grep (`scripts/measure_coherency.py`,
`scripts/render_coherency_figures.py`) and a seventh was caught by **mypy**, not
by grep: `scripts/behaviour_clone.py:362` imported `build_action_selector` from
`measure_maps` and unpacked it as a tuple.

| File | Was |
|---|---|
| `scripts/measure_maps.py` | `build_action_selector` (path-first, took a config) |
| `debug.py` | `build_selector_for` (path-first, `typer.BadParameter`) |
| `scripts/measure_paired_policies.py` | `_selector_for` (registry-first) |
| `scripts/measure_income_share.py` | `_selector_for_policy` (registry-first) |
| `scripts/measure_objective_split.py` | inline (registry-first) |
| `scripts/measure_coherency.py` | inline (registry-first) |
| `scripts/render_coherency_figures.py` | direct `build_selector` |

`scripts/measure_checkpoint.build_selector` survives as a thin wrapper returning
`(select, network)`, so its own callers and every `just` recipe are unchanged.

**Two deliberate behaviour changes**, both worth knowing:

1. **Labels are unified on `label_for`.** `debug.py` used `path.parent.name` and
   two scripts used `split("/")[-2]`. `label_for` falls back to the directory
   name, so it differs only when a run carries a `--run-suffix` — where it is
   strictly better, because the scenario part is identical across the arms of a
   screen and the suffix is the only part that says which arm a row is.
2. **`measure_maps` builds one env per map instead of two.** The old
   `build_action_selector` constructed an env purely to size the network and
   discarded it, then `main` built another for scoring.

## `wargame_rl/wargame/rating/score.py`

`margin_score`, `fit_margin_scale`, `DEFAULT_MARGIN_SCALE = 50.0` (illustrative,
pinned, to be re-fitted per scenario). Computed through a stable sigmoid — the
naive `1/(1+exp(-m/s))` overflows on the blowouts this scenario actually
produces, and `test_a_huge_negative_margin_does_not_overflow` runs under
`np.errstate(over="raise")` so a regression fails rather than warns.

`fit_margin_scale` refuses an all-one-way corpus rather than returning a number
fitted from no information.

## `wargame_rl/wargame/rating/elo.py`

`Design`, `RatingFit`, `RatingTable`, `fit_ratings`, `bootstrap_ratings`,
`win_probability`. Newton–Raphson on the concave objective, dense `(n+2)`-square
Hessian, one pinned anchor and a Gaussian prior (σ=400) **on the rating block
only** — the two advantage terms are structural quantities of the scenario, not
entrants to be shrunk. No scipy.

### Three findings

**1. A confounded schedule is refused, not regularised.** Nothing shrinks
`h_zone` and `h_turn`, so a schedule varying the two axes together leaves them
identified only up to their sum and Newton hits a singular Hessian.
`_require_identifiable` rank-checks the advantage block up front and raises
naming the fix ("play all four legs"). Regularising instead would return a
plausible split of a quantity the data never separated, and a table reporting a
deployment-zone advantage it could not have measured is worse than no table.

**2. `docs/elo.md`'s stated reason for the bootstrap is wrong, and the test
found it.** The doc argues that with a fractional score the quasi-likelihood
Hessian **understates** the error. The premise is right and the conclusion is
backwards: any `[0,1]`-valued score has variance at most `p(1-p)`, so the
Bernoulli assumption bounds the *marginal* variance from above and is
conservative on its own — the same fact as the Rao-Blackwellisation argument
that motivates the score. Measured on synthetic data with independent rows, the
Hessian interval came out **118 wide against the bootstrap's 27**.

What the Hessian genuinely cannot see is **dependence between rows** — the four
legs on one layout share terrain, objectives and dice — and that is unbounded in
the direction that matters. The test is now
`test_the_layout_bootstrap_is_wider_than_resampling_rows`: the same games,
bootstrapped twice, differing only in whether layouts are grouped. **The
conclusion — bootstrap over layouts — is unchanged; only the reason is.**

**3. A self-pairing is the cleanest estimator of the two advantages.**
`_design_matrix` uses `np.add.at`, so an entrant against itself accumulates to
zero rather than leaving a stray `+1`. The rating difference is then identically
zero by construction, and whatever margin survives the balanced four legs is the
seat advantage and nothing else. This is what WP-2's seat-parity gate rests on,
and `test_a_self_pairing_measures_the_seat_advantage_alone` pins it.

## `tests/test_import_direction.py`

AST-walked, not grepped, so a docstring mentioning torch does not fail it. Three
rules: `envs/` imports neither `model/`, `rating/` nor torch; `rating/` imports
neither `model/` nor torch; `score.py` and `elo.py` import nothing from
`wargame_rl` at all. **Verified sensitive** — appending `import torch` to
`envs/opponent/registry.py` fails it, and removing it restores 147 green.

## State

- `just format && just lint` clean (ruff + mypy strict, 291 files).
- New tests: 10 + 8 + 12 + 147 = 177, all green.
- Pre-existing failures unrelated to this wave and confirmed on a stashed tree:
  `test_terrain_los_symmetry` (the known open LOS asymmetry bug) and three CUDA
  tests that pass under `CUDA_VISIBLE_DEVICES=""` (sm_61 GPU unsupported by this
  torch build).
- Smoke-tested: `measure_baselines`, `measure_objective_split`,
  `measure_paired_policies` on `configs/dev/tiny.yaml` all print as before.
- The checkpoint branch is covered by unit test only — every `.ckpt` on this
  machine is a stale artefact with a pre-PPO key prefix, and
  `convert_state_dict` rejects them exactly as it did before this wave.

## For 04-02

- `Design.layout` is the bootstrap unit; the arena must return per-layout rows.
- `fit_ratings(design, entrants, anchor=...)` requires the anchor to be an
  entrant — the ledger should refuse a table with no scripted anchor in it.
- `build_action_selector` takes an **env**, not a config, so the arena must
  build the leg's env first and resolve entrants against it.
