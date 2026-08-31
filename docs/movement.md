# Movement & Action Space

## Union Action Space

Every model shares a single flat discrete action space. The space is partitioned into contiguous **slices**, each owned by an action type and tagged with the battle phases where it is valid. An `ActionRegistry` manages these slices and produces phase-aware boolean masks so the agent (and opponent policies) never select illegal actions.

Current slices:

| Slice | Indices | Valid phases | Registered when |
|-------|---------|--------------|-----------------|
| `stay` | `0` | All phases | always |
| `movement` | `1 .. N×S` | Movement phase only | always |
| `shooting` | `N×S+1 .. N×S+T` | Shooting phase only | opponents configured |
| `advance` | after `shooting`, `N×A` wide | Movement phase only | `n_advance_speed_bins > 0` |
| `move_type` | last, 2 wide | **Command phase only** | `n_advance_speed_bins > 0` |

With the defaults (`n_movement_angles=16`, `n_speed_bins=6`) and no opponents, the total action space is **97** (1 stay + 96 movement). With opponents, shooting target indices are appended (see [shooting.md](shooting.md)). On the 25v25 advance scenario it is **152** (1 + 96 + 5 + 48 + 2), against **102** with `n_advance_speed_bins: 0`.

⚠ The last two are registered **after** shooting so that no pre-existing action index moves — `decode_action` is angle-major, speed-minor, so widening `n_speed_bins` instead would renumber the movement slice and `_apply_warm_start_weights` loads with `strict=False`, meaning every old checkpoint would load and be silently wrong.

### Action Masking

During each step, the environment generates a `(n_models, n_actions)` boolean mask based on the current `BattlePhase`. The mask is:

- Attached to the observation (`WargameEnvObservation.action_mask`).
- Threaded through the observation tensor pipeline as tensor 5, a `torch.bool` tensor (`model/common/observation.py`).
- Applied by **PPO** inside `TransformerNetwork.policy_from_encoded`, which fills masked logits with `-inf` before the categorical distribution is formed — so sampling, log-probs, and entropy all see only legal actions.
- Applied by `ArgmaxAgent` (`model/common/argmax_agent.py`) during greedy selection, which sets invalid logits to `-inf` before the argmax, and restricts random exploration to valid indices. This is the path `simulate.py` and `measure-phase-gates` take.

### Extending with New Phases

To add actions for a new phase (e.g. charging), register a new slice in `ActionHandler.__init__`:

```python
self._registry.register(
    "charging",
    n_charging_actions,
    frozenset({BattlePhase.charge}),
)
```

This appends the new actions after the existing slices. The mask generation, observation pipeline, and network output layer automatically account for the larger `n_actions` — no other wiring changes are needed beyond implementing the action application logic itself. Shooting already follows this pattern (see [shooting.md](shooting.md)).

An action can be valid in multiple phases by including them in the `valid_phases` frozenset (e.g. `stay` is valid in all phases).

⚠ **A new MOVE TYPE is not a new slice.** Fall back and charge are move types, not
phases, and the pattern above is the wrong one for them: it is what Advance did
first, and it cost 48 actions plus a bespoke unit-resolution rule. A move type is
**one more value in `move_type`** (§ Move types below) — the declaration machinery,
the unit resolution and the legality mask are already general. Reach for a new slice
only when a phase needs genuinely new *targets*, as shooting does.

⚠ **The charge, as shipped, took a third route and added NO actions at all**
(`melee.enabled`, default off — see [melee.md](melee.md)). The charge is a phase in
`BattlePhase`, so the `movement` slice was simply made valid there and the 2D6 masks
its speed bins, exactly as `advance_roll` masks the advance rungs. That keeps the
parameter shape identical, which is the only way an arm here is **pairable** against
its control. Fall back needed no action either: it is inferred from a unit that began
its move engaged. So of the three move types named above, one is a `move_type` value
and two are not — reach for whichever costs no actions.

## Movement Encoding

Each model's movement action is a single integer from `1` to `n_movement_angles × n_speed_bins` (index `0` is the phase-universal stay action):

| Action | Meaning |
|--------|---------|
| `0` | Stay (no movement) |
| `1 .. N×S` | Move with a specific (angle, speed) pair |

For movement actions, the angle and speed indices are decoded as:

```
angle_idx = (action - 1) // n_speed_bins
speed_idx = (action - 1) %  n_speed_bins
```

## Move types

The rules make the **move type** a unit's choice, and this action space makes it one
too — declared once, in the command phase, rather than inferred from what individual
models did.

| | |
|---|---|
| **Who decides** | the unit's lowest-indexed **alive** model (its leader) |
| **When** | the command phase, before anything moves |
| **How** | `move_type` slice: `MOVE_TYPE_NORMAL`, `MOVE_TYPE_ADVANCE` |
| **Default** | `STAY` in the command phase declares `normal` |
| **Cost** | declaring an advance sets `advanced_this_turn` for the whole unit immediately, forfeiting its shooting — the rules attach that to the move *type*, not to the distance travelled |

`STAY` declaring `normal` is what keeps every policy written before the declaration
existed working unchanged; a non-advancing script scores bit-identically across the
change.

⚠ **`n_advance_speed_bins > 0` requires the command phase.** A config that lists
`command` in `skip_phases` is rejected at construction — otherwise the rungs are
registered and no declaration is ever legal, so a run would spend hours measuring a
feature it never had.

### Advance rungs are absolute

An advance bin is a **fixed distance above the model's Move**:

```
distance(bin) = M + (bin + 1) × (6 / n_advance_speed_bins)
```

At `M = 6` with three bins that is **8" / 10" / 12"**. The unit's D6
(`model.advance_roll`, one roll per unit at the start of its turn) decides which
rungs are **legal**, via `ActionHandler.advance_legality`, masked on both seats. It
does not decide what an action *means*.

Two properties follow, and both were defects of the earlier `fraction × (M + roll)`
encoding:

- **No dominated actions.** Every rung is beyond a normal move's reach, so no action
  can spend the unit's shooting for a distance a walk already delivers. The old
  ladder had ~50% of the slice dominated in expectation.
- **Stationary semantics.** An index means the same displacement every turn. The old
  ladder was the only slice in the game whose meaning changed turn to turn, so a
  policy had to read `advance_roll` to know what its own action did.

⚠ A rung is legal only for a unit that **declared**, *and* only within `M + roll`.
The declaration gate is what makes the move type a unit decision; separating it from
the distance is what keeps the unit able to move as a body. A leader-only rule that
fused the two would cap every squadmate at `M` and shatter the 2" chain.

⚠ At three bins a roll of 1 leaves **no** legal rung. The rules would permit a 7"
advance and the ladder cannot express it — a resolution limit, not a bug, since a 1"
gain never repays a turn of fire.

⚠ **Advance is a lever, not an advantage.** See
[play-doctrine D-43](play-doctrine.md); do not build policies whose purpose is to
advance, and do not read low usage as a failure.

## Direction

Angles are evenly spaced around the full circle starting at 0 radians (east / +x) and going counter-clockwise:

```
angle = 2π × angle_idx / n_movement_angles
```

With 16 angular bins, each bin is 22.5° apart:

| Index | Angle | Direction |
|-------|-------|-----------|
| 0 | 0° | East |
| 1 | 22.5° | ENE |
| 2 | 45° | NE |
| 3 | 67.5° | NNE |
| 4 | 90° | North |
| 5 | 112.5° | NNW |
| 6 | 135° | NW |
| 7 | 157.5° | WNW |
| 8 | 180° | West |
| 9 | 202.5° | WSW |
| 10 | 225° | SW |
| 11 | 247.5° | SSW |
| 12 | 270° | South |
| 13 | 292.5° | SSE |
| 14 | 315° | SE |
| 15 | 337.5° | ESE |

Board `y` increases *downward* (row 0 renders at the top), so the compass labels above read correctly only on a y-up plot: index 4 (+90°) displaces a model toward the bottom of the rendered board.

## Speed

Speed bins are linearly spaced from `max_move_speed / n_speed_bins` up to `max_move_speed`:

```
speed = max_move_speed × (speed_idx + 1) / n_speed_bins
```

With the defaults (`max_move_speed=6`, `n_speed_bins=6`), the available speeds are 1, 2, 3, 4, 5, 6 cells per step.

## Displacement Calculation

The board is continuous, so the displacement is applied **exactly**:

```
dx = speed × cos(angle)
dy = speed × sin(angle)
```

This used to be `round(...)`, snapping every move to a whole cell, and it was
destroying information rather than approximating it. On the 25v25 action space
the 96 movement actions collapsed to **80 distinct outcomes** — 16 pairs the
policy could not have told apart in the one head that steers — and a "speed 1"
diagonal travelled 1.414 against an orthogonal move's 1.000, so the cheapest way
to cover ground was to face diagonally.

All displacements are pre-computed at environment initialization for efficiency.
The board edge is clamped into the displacement to `[r, r] .. [width - r, height - r]`
for a model of base radius `r`, **before** collisions are resolved.

### Seeing the discretisation

`just debug` makes the bins visible: click open ground with a model selected and it is
ordered there, but the ghost is drawn at the **decoded landing point** rather than at the
click. The gap between the two is exactly the angle-and-speed quantisation the policy is
choosing under. An order the action mask refuses is drawn too, in the casualty colour, with
the reason — a dead model has only `STAY`.

![The debug renderer with four hand-authored orders pending: three player models and one opponent, one of them refused because the model is a casualty](images/debug-hand-authored-orders.png)

## Collision

With `base_radius > 0` a model occupies ground, and moves are resolved against
the other models (`domain/movement.py`):

| | rule |
|---|---|
| enemy base | blocks the path — the move stops at contact |
| friendly base | may be crossed, but not **ended on** |
| resolution order | sequential, by model index |

The asymmetry is deliberate: walking through an enemy line is the one thing
models physically cannot do, while a squad moving as a body would gridlock on
its own front rank if friendlies blocked too. Sequential resolution gives model 0
a documented right of way, and that is the price of a board that is the same
every time the same actions are played.

At `base_radius: 0.0` — the default — none of this applies and movement is
exactly what was asked for, which is what keeps every result measured before
models had bases reproducible.

> **A tangential slide was tried and measured worse.** Blocked models otherwise
> queue radially behind whoever reached the objective first, so a sideways step
> around the obstruction is the obvious fix. Measured on the polygon scenario at
> n=30 on identical layouts, `squad_march_shoot` went 0.70 / +20.6 vp_margin to
> 0.57 / +1.0 with the slide in. A *fully* blocked model has its whole move left
> to spend, so the slide becomes a full-length swing away from the objective:
> models drift laterally, stay in the open longer, and are shot. The real fix is
> on the policy side — distinct target slots around an objective, rather than
> aiming every model at the centre.

## Ending unengaged

A move must leave the moving unit **unengaged** (`docs/rules/09-movement-phase.md`),
and `docs/rules/03-moving.md` is explicit that only the endpoint counts:

> Passing through an enemy unit's engagement range during a move does **not** make
> the moving unit engaged. Only where it *ends* matters.

`domain/movement.py::back_off_to_unengaged` applies this **after** `resolve_move`,
so enemies still block at their true base radius and the path is unchanged. If the
resolved endpoint lies inside any enemy's engagement ring
(`engagement_range + both base radii`), it is pulled back along its own heading
until it clears every ring. When no legal point exists short of the start, the
model does not move — the rules' own remedy.

⚠ The legal set along the ray is **not an interval**: a ray can leave one ring and
enter another, so the back-off walks ring by ring rather than bisecting.

⚠ Applying the engagement range as a *path* constraint instead — by inflating
enemy blocker radii — was tried and reverted. It turns an end-state rule into an
impassable wall and left 87% of opponent-held objectives with no legal spot at all.

## Configuration

Movement parameters are set via `WargameEnvConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_movement_angles` | `16` | Number of angular bins (22.5° apart) |
| `n_advance_speed_bins` | `0` | Advance rungs, as their own slice appended after shooting, plus a 2-wide `move_type` slice for the declaration. Rungs are **absolute** (`M + (bin+1)×(6/bins)`), gated by the unit's D6 through a per-model mask. 0 registers nothing, draws no dice and changes no action index. ⚠ **> 0 requires `command` NOT in `skip_phases`**, enforced at construction |
| `dark_action_slices` | `[]` | Slice names registered at full width but valid in **no** phase, so every one of their actions is masked all episode. Exists to restore *pairing* to an action-space arm: the arm and its control then share a parameter shape and start from bit-identical weights. ⚠ It does not make an existing control reusable — a narrower head consumes less RNG at init, so the control must be retrained with the slice darkened. ⚠ Darkening `"advance"` darkens the `move_type` slice too, so the rungs become unreachable by the same switch: a rung is legal only where a declaration was made |
| `n_speed_bins` | `6` | Number of discrete speed levels |
| `max_move_speed` | `6.0` | Maximum distance a model can move per step, in inches. The scenario-wide default for the rules' **Move (M)** characteristic |
| `ModelConfig.move` | `None` | Per-model override of `max_move_speed`, in inches. `None` takes the scenario value |
| `base_radius` | `0.0` | Model base radius, in inches. `0.63` is the rules' 32mm infantry base |

These can be overridden in YAML environment config files:

```yaml
n_movement_angles: 16
n_speed_bins: 6
max_move_speed: 6.0
base_radius: 0.63
```

## Per-Model Speed

`ModelConfig.move` is the rules' **Move (M)** characteristic, in inches. A model
that sets it uses its own maximum in place of `max_move_speed`:

```yaml
models:
  - { group_id: 0, move: 10.0 }   # a fast squad
  - { group_id: 1 }               # takes max_move_speed
```

The action space is **uniform across models**: the bin *count* never changes, so
"speed bin 3 of 6" means "half of my own maximum" whatever that maximum is, and
the network architecture is untouched. Each side's handler is built from its own
model list, so the two armies can move at different speeds.

Two things to know:

- **A uniformly-fast army is byte-identical to one that never set the field.**
  The shared displacement table is kept verbatim in that case rather than being
  re-derived as fractions × M, because the two are not the same float: at M = 6,
  `6.0 / 6` is exactly `1.0` while `linspace(1/6, 1, 6)[0] * 6` is
  `0.9999999999999999`.
- **M is not in the observation yet.** With every model equally fast the network
  does not need it, but under genuinely differing speeds the same action index
  would mean different distances to different models with nothing in the tensor
  saying so. Adding it widens the per-model token, which orphans every
  checkpoint — see `docs/rules/implementation-status.md`.
