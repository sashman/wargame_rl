# Phase 5: Shooting Resolution - Research

**Researched:** 2026-04-06
**Domain:** Tabletop shooting resolution (D6 dice mechanics), RL observation extension
**Confidence:** HIGH

## Summary

Phase 5 replaces the Phase 4 shooting no-op with a full tabletop attack sequence: hit roll → wound roll → save → damage. The implementation is entirely internal to the existing codebase — no new dependencies are required. Numpy's `random.Generator` handles D6 rolls, config types gain weapon/defense stats, a new pure domain module resolves attacks, and the observation tensor grows to include combat-relevant features plus precomputed expected damage per attacker–target pair.

The core complexity lies in three areas: (1) correctly implementing the wound roll comparison table with its edge cases, (2) extending the observation tensor pipeline without breaking the existing feature layout, and (3) wiring the resolution into both player and opponent action paths so both sides can inflict damage. All other pieces (config extension, shooting mask checks, `take_damage` entry point) are straightforward extensions of established patterns.

**Primary recommendation:** Create `domain/shooting.py` as a pure domain service with vectorized D6 rolls, wire it into `ActionHandler.apply` and `_apply_opponent_action`, extend config and observation types with defaults that maintain backward compatibility.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Shooting uses stochastic D6 rolls matching tabletop rules. Hit roll (D6 ≥ BS), wound roll (S vs T table), saving throw (D6 + AP ≥ Sv). Unmodified 6 always succeeds, unmodified 1 always fails.
- **D-02:** Use a dedicated `numpy.random.Generator` seeded per episode for all combat rolls. Stochastic in training, deterministic for testing and replay with fixed seed.
- **D-03:** Damage applied via existing `take_damage(amount)`. Excess damage from a single attack is lost (no carry-over).
- **D-04:** Wound roll threshold follows the standard tabletop S vs T comparison table (2+ through 6+).
- **D-05:** Extend `WeaponProfile` with: `attacks: int`, `ballistic_skill: int`, `strength: int`, `ap: int`, `damage: int`.
- **D-06:** Default weapon profile: `attacks=2, ballistic_skill=3, strength=4, ap=1, damage=1` (~50% chance of dealing ≥1 wound).
- **D-07:** Existing `range` field unchanged. Resolution fields have defaults matching D-06 so Phase 4 configs get a standard combat profile.
- **D-08:** Add `toughness: int` and `save: int` to `ModelConfig`.
- **D-09:** Defaults: `toughness=3, save=4`. Symmetric ~50% wound-per-action baseline.
- **D-10:** No invulnerable save. Save uses only `Sv + AP`.
- **D-11:** Toughness and save added to `WargameModel.stats` dict, wired from `ModelConfig` via `battle_factory`.
- **D-12:** Per player model: add normalized weapon stats to observation. Per opponent model: add normalized `toughness` and `save`.
- **D-13:** Add precomputed expected damage per (attacker, target) pair as observation feature.
- **D-14:** ~50% baseline means expected damage feature starts clean and interpretable.
- **D-15:** `advanced_this_turn: bool` flag on `WargameModel`. Always `False` until v3.0. Shooting mask checks it.
- **D-16:** Engagement range check in shooting mask. Never triggers until v3.0.
- **D-17:** Both flags reset per turn by domain layer.

### Claude's Discretion
- Whether shooting resolution lives as a pure domain function in `domain/shooting.py` or as part of `env_components/`
- Exact normalization scheme for weapon/defense observation features (match existing patterns in `observation.py`)
- Whether expected damage is precomputed at episode init and cached, or recomputed each step
- Internal structure of the D6 roll functions (vectorized numpy vs per-attack loop)
- How `advanced_this_turn` reset is wired into the turn execution pipeline
- Exact engagement range threshold value to stub (e.g. 0 or 1 grid cell)

### Deferred Ideas (OUT OF SCOPE)
- Multi-weapon sub-steps (all weapons fire at one target per action)
- Invulnerable saves (v4.0)
- Full probability matrices for UI/explainability (v9.0)
- Pointer-network attention for shooting logits
- Cover saves (v2.0)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SHOT-01 | Models can select a shoot action targeting an enemy within weapon range | Resolution code decodes target index from shooting slice (Phase 4 established), calls resolution pipeline |
| SHOT-02 | Shooting resolves via hit roll → wound roll → save → damage | `domain/shooting.py` implements full attack sequence per D-01 through D-04 |
| SHOT-04 | Models that advanced cannot shoot | `advanced_this_turn` flag on `WargameModel` checked in `compute_shooting_masks`; always False until v3.0 |
| SHOT-05 | Models in engagement range cannot shoot | Engagement range check in `compute_shooting_masks`; never triggers until v3.0 |
| SHOT-06 | Weapon profiles configurable per model (range, attacks, BS, strength, AP, damage) | Extend `WeaponProfile` + `ModelConfig` with defaults per D-05 through D-09 |
| OBS-02 | Agent observation includes weapon profiles or combat-relevant stats | Extend observation tensor with weapon/defense features + expected damage per D-12 through D-14 |
</phase_requirements>

## Project Constraints (from .cursor/rules/)

- **Domain purity:** `domain/` must not import from `env_components`, `reward`, or `renders` (per `ddd-envs.md`). Resolution logic belongs in `domain/`.
- **Config defaults:** All new fields must have backward-compatible defaults so existing YAML configs keep working (`Field(default=...)`).
- **Type hints:** All public functions typed (mypy strict). Use `from __future__ import annotations`, built-in generics.
- **Testing:** Arrange–Act–Assert structure. Deterministic with fixed seeds. Avoid mocks — use real dependencies.
- **Validation pipeline:** `just validate` (format + lint + test) before pushing.
- **Line length:** 88, double quotes, Ruff format.
- **Pass dependencies in:** Don't construct them inside classes.
- **Tooling:** Use `just` for all commands. `uv add` for deps.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | (existing dep) | D6 dice rolls via `numpy.random.Generator`, vectorized roll arrays | Already used throughout; `Generator.integers(1, 7, size=n)` for D6 rolls |
| pydantic | (existing dep) | Config model extension (`WeaponProfile`, `ModelConfig`) | All config uses Pydantic `BaseModel` with `Field(...)` |

### Supporting
No new dependencies required. Phase 5 is entirely internal codebase work using existing numpy, pydantic, and gymnasium infrastructure.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `numpy.random.Generator.integers` | `random.randint` | numpy Generator is seedable per-instance, vectorizable, already the project pattern |
| Per-attack Python loop | Vectorized numpy rolls | Vectorized is faster for multi-attack weapons; use numpy batch rolls |

## Architecture Patterns

### Recommended Domain Extension

```
wargame_rl/wargame/envs/
├── domain/
│   ├── shooting.py           # NEW: pure resolution functions
│   └── entities.py           # EXTEND: advanced_this_turn, engagement_range_of
├── env_components/
│   ├── actions.py            # EXTEND: replace shooting no-op with resolution call
│   ├── observation_builder.py # EXTEND: pass weapon/defense data to obs
│   └── shooting_masks.py     # EXTEND: advanced + engagement checks
├── types/
│   ├── config.py             # EXTEND: WeaponProfile fields, ModelConfig fields
│   └── model_observation.py  # EXTEND: weapon/defense observation fields
```

### Pattern 1: Pure Domain Resolution Function
**What:** `resolve_shooting(attacker_weapon, target_stats, rng) -> ShootingResult` — a pure function that takes weapon profile, target stats, and RNG, returns a dataclass with rolls and damage dealt.
**When to use:** The domain layer owns all game rules; resolution is a game rule.
**Recommendation:** Place in `domain/shooting.py`. This module has no Gym imports, no env_components imports — only numpy and domain types.

```python
@dataclass(frozen=True, slots=True)
class ShootingResult:
    """Outcome of one model's shooting action against one target."""
    hits: int
    wounds: int
    unsaved: int
    damage_dealt: int

def resolve_shooting(
    attacks: int,
    ballistic_skill: int,
    strength: int,
    ap: int,
    damage: int,
    target_toughness: int,
    target_save: int,
    rng: np.random.Generator,
) -> ShootingResult:
    ...
```

### Pattern 2: Wiring Into ActionHandler.apply
**What:** Replace the `continue` on shooting-slice actions with a call to `resolve_shooting` followed by `target.take_damage(result.damage_dealt)`.
**When to use:** `ActionHandler.apply` already recognizes shooting-slice indices.
**Recommendation:** The `apply` method needs access to opponent models and their stats. Pass them as a new keyword-only parameter when phase is shooting. Alternatively, create a separate `resolve_shooting_action` function that `WargameEnv._apply_player_action` and `_apply_opponent_action` call after `ActionHandler.apply`.

The cleaner approach: keep `ActionHandler.apply` focused on movement. Extract shooting resolution into a separate call that `WargameEnv` makes during the shooting phase. This avoids bloating `ActionHandler` with domain logic it shouldn't own.

```python
# In WargameEnv._apply_player_action:
if phase == BattlePhase.shooting:
    resolve_player_shooting(action, self.wargame_models, self.opponent_models, rng)
else:
    self._action_handler.apply(action, ...)
```

### Pattern 3: Observation Feature Extension
**What:** Extend `_models_to_features` to include weapon stats (player models) and defense stats (all models).
**When to use:** OBS-02 requires combat stats in the agent's observation.
**Key insight:** The feature vector is a horizontal `hstack` of normalized blocks. Weapon stats join the hstack for player models; defense stats for opponent models. The feature_dim computation must be updated to account for new columns.

### Pattern 4: Expected Damage Precomputation
**What:** Compute P(hit) × P(wound) × P(fail_save) × damage × attacks for each (attacker, target) pair.
**When to use:** OBS-02 / D-13 — gives the agent explicit efficiency information.
**Recommendation:** Recompute each step rather than caching. Reasons: (1) targets take wounds and die during the episode, changing valid pairs; (2) computation is O(n_player × n_opponent) scalar arithmetic — negligible cost; (3) avoids stale cache bugs.

The expected damage matrix shape is `(n_player, n_opponent)` with dtype float32. Flatten or include as an additional observation block.

### Anti-Patterns to Avoid
- **Resolution logic in env_components:** Resolution is a game rule (domain). Putting it in `env_components/actions.py` violates the dependency direction documented in `ddd-envs.md`.
- **Mutating model state inside `ActionHandler.apply`:** The handler should dispatch, not resolve. Keep domain mutation (take_damage) at the env orchestration level or in a domain service.
- **Global numpy RNG for combat rolls:** Must use the per-episode `numpy.random.Generator` for reproducibility. The env already has `self.np_random` from Gymnasium, but D-02 specifies a dedicated combat Generator seeded per episode.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| D6 dice rolling | Custom random function | `numpy.random.Generator.integers(1, 7, size=n)` | Seedable, vectorized, correct uniform distribution |
| Wound roll threshold | Chained if/elif | Lookup function with explicit S vs T comparison | Tabletop table is well-defined; a single function with the 5 cases is clearer |
| Expected damage probability | Manual probability math | Closed-form: `P_hit × P_wound × P_fail_save × damage × attacks` | Each probability is a simple threshold on D6: `P = (7 - threshold) / 6` |
| Config validation | Manual checks | Pydantic `Field(gt=0, ge=1)` constraints | Existing pattern, catches bad configs at load time |

## Common Pitfalls

### Pitfall 1: Wound Roll Table Comparison Order
**What goes wrong:** Getting the S vs T comparison order wrong. E.g., checking `S > T` before `S >= 2*T` would give 3+ when it should be 2+.
**Why it happens:** The five cases overlap — `S >= 2*T` is a subset of `S > T`. Checking in the wrong order picks the weaker threshold.
**How to avoid:** Always check from most favourable to least: `S >= 2*T` → `S > T` → `S == T` → `S < T` → `S <= T/2`. The `T/2` comparison also needs integer handling (floor division or floating-point).
**Warning signs:** Unit test where S=6, T=3 should be 2+ but gets 3+.

### Pitfall 2: Integer Division in T/2 Comparison
**What goes wrong:** `S <= T/2` with integer division: if T=3, is the threshold `S <= 1` or `S <= 1.5`? The tabletop rule is `S ≤ T/2` where T/2 rounds down (or effectively uses real division). T=3 → T/2=1.5 → S must be ≤ 1 for 6+ (since S is always an integer ≥ 1, only S=1 qualifies). But T=4 → T/2=2 → S ≤ 2 qualifies.
**Why it happens:** Python's `//` operator does floor division, but `S <= T // 2` gives wrong results for odd T. E.g. T=5 → `T//2 = 2`, so S=2 → `2 <= 2` → True → 6+. But tabletop: T=5, T/2=2.5, S=2 → `2 ≤ 2.5` → True → 6+. Both agree here. T=3, T//2=1, S=1 → `1 <= 1` → True → 6+. Tabletop: T=3, T/2=1.5, S=1 → `1 ≤ 1.5` → True → 6+. They agree! The comparison `S <= T // 2` is equivalent to `2*S <= T` for positive integers. Use `2 * S <= T` to avoid any ambiguity.
**How to avoid:** Use the equivalent integer form: `2 * strength <= toughness` for the 6+ case, `2 * toughness <= strength` for the 2+ case. No division, no rounding issues.
**Warning signs:** Parameterized test matrix covering all 5 threshold bands.

### Pitfall 3: Unmodified 1 Always Fails / 6 Always Succeeds
**What goes wrong:** Applying modifiers before checking natural 1/6. The tabletop rule is that an *unmodified* roll of 1 always fails and an *unmodified* roll of 6 always succeeds, regardless of modifiers.
**Why it happens:** Phase 5 has no modifiers (no cover, no heavy weapon bonus, no hit penalty), so this can't actually manifest yet. But the code should be structured to handle it correctly when modifiers arrive.
**How to avoid:** Check the raw die roll against 1 and 6 first, then apply modifiers and check threshold. For Phase 5 (no modifiers), this simplifies to: roll the die, if 1 → fail, if 6 → succeed, else compare roll ≥ threshold. Structure the code so this logic is explicit.
**Warning signs:** None in Phase 5 (no modifiers), but test natural 1 and 6 explicitly.

### Pitfall 4: Observation Tensor Width Change Breaks Checkpoints
**What goes wrong:** Adding weapon/defense features changes `feature_dim`, making old checkpoints incompatible.
**Why it happens:** The transformer's input projection expects a fixed feature_dim. New features change it.
**How to avoid:** This is expected and documented (Phase 2 CONTEXT.md D-09: "checkpoint compat is allowed to break"). Don't try to preserve checkpoint compatibility — document the break.
**Warning signs:** Old checkpoint load fails with shape mismatch (expected behaviour).

### Pitfall 5: RNG Instance Management
**What goes wrong:** Using `env.np_random` (Gymnasium's RNG) instead of a dedicated combat Generator. Or creating the Generator in the wrong place so it's not reset per episode.
**Why it happens:** Gymnasium provides `self.np_random` from `super().reset(seed=seed)`. Using it for combat rolls would couple combat stochasticity with movement/placement randomness.
**How to avoid:** Create a `numpy.random.Generator(numpy.random.PCG64(seed))` at `reset()` time using the episode seed (or a derived seed). Store it on the env as `self._combat_rng`. Pass it to all resolution calls. For testing, provide a fixed seed to get deterministic rolls.
**Warning signs:** Non-reproducible test results when combat is involved.

### Pitfall 6: Opponent Shooting Path
**What goes wrong:** Only wiring player shooting, forgetting that `_apply_opponent_action` also needs to resolve shooting.
**Why it happens:** The opponent path uses a separate `ActionHandler` and calls `_opponent_action_handler.apply()`. Both paths need to call into the same resolution logic.
**How to avoid:** Extract resolution into a shared function that both player and opponent paths call. The opponent policy already selects shooting actions during the shooting phase — the env just needs to resolve them the same way.
**Warning signs:** Opponent shooting actions are no-ops (deal no damage).

## Code Examples

### Wound Roll Threshold (Integer-Safe)

```python
def wound_roll_threshold(strength: int, toughness: int) -> int:
    """Return the minimum D6 roll needed to wound (2-6)."""
    if 2 * toughness <= strength:   # S >= 2T → 2+
        return 2
    if strength > toughness:        # S > T → 3+
        return 3
    if strength == toughness:       # S == T → 4+
        return 4
    if strength < toughness:        # S < T → 5+
        if 2 * strength <= toughness:  # S <= T/2 → 6+
            return 6
        return 5
    return 4  # unreachable for positive ints
```

### Vectorized D6 Resolution

```python
def resolve_shooting(
    attacks: int,
    ballistic_skill: int,
    strength: int,
    ap: int,
    damage: int,
    target_toughness: int,
    target_save: int,
    rng: np.random.Generator,
) -> ShootingResult:
    """Resolve one model's shooting against one target."""
    # Hit rolls
    hit_rolls = rng.integers(1, 7, size=attacks)
    hits = int(np.sum((hit_rolls >= ballistic_skill) | (hit_rolls == 6))
               - np.sum((hit_rolls == 1) & (ballistic_skill <= 1)))
    # Simplified for no-modifier Phase 5:
    hits = int(np.sum(
        (hit_rolls != 1) & ((hit_rolls >= ballistic_skill) | (hit_rolls == 6))
    ))

    if hits == 0:
        return ShootingResult(hits=0, wounds=0, unsaved=0, damage_dealt=0)

    # Wound rolls
    wound_threshold = wound_roll_threshold(strength, target_toughness)
    wound_rolls = rng.integers(1, 7, size=hits)
    wounds = int(np.sum(
        (wound_rolls != 1) & ((wound_rolls >= wound_threshold) | (wound_rolls == 6))
    ))

    if wounds == 0:
        return ShootingResult(hits=hits, wounds=0, unsaved=0, damage_dealt=0)

    # Save rolls
    modified_save = target_save + ap  # AP increases required save roll
    save_rolls = rng.integers(1, 7, size=wounds)
    saves = int(np.sum(
        (save_rolls != 1) & (save_rolls >= modified_save)
    ))
    unsaved = wounds - saves

    if unsaved <= 0:
        return ShootingResult(hits=hits, wounds=wounds, unsaved=0, damage_dealt=0)

    damage_dealt = unsaved * damage
    return ShootingResult(
        hits=hits, wounds=wounds, unsaved=unsaved, damage_dealt=damage_dealt
    )
```

### Expected Damage Computation (Closed-Form)

```python
def expected_damage(
    attacks: int,
    ballistic_skill: int,
    strength: int,
    ap: int,
    damage: int,
    target_toughness: int,
    target_save: int,
) -> float:
    """Expected damage from one model shooting at one target (analytical)."""
    p_hit = (7 - ballistic_skill) / 6.0
    wound_threshold = wound_roll_threshold(strength, target_toughness)
    p_wound = (7 - wound_threshold) / 6.0
    modified_save = target_save + ap
    p_save = max(0.0, (7 - modified_save) / 6.0) if modified_save <= 6 else 0.0
    p_fail_save = 1.0 - p_save
    return attacks * p_hit * p_wound * p_fail_save * damage
```

### Config Extension Pattern

```python
class WeaponProfile(BaseModel):
    """Weapon stat block."""
    range: int = Field(gt=0, description="Maximum range in grid cells")
    attacks: int = Field(default=2, gt=0, description="Number of hit rolls per action")
    ballistic_skill: int = Field(default=3, ge=2, le=6, description="D6 roll needed to hit (e.g. 3 means 3+)")
    strength: int = Field(default=4, gt=0, description="For wound roll comparison vs target toughness")
    ap: int = Field(default=1, ge=0, description="Armour penetration (worsens target save by this amount)")
    damage: int = Field(default=1, gt=0, description="Wounds per failed save")

class ModelConfig(BaseModel):
    # ... existing fields ...
    toughness: int = Field(default=3, gt=0, description="Wound roll comparison stat")
    save: int = Field(default=4, ge=2, le=7, description="Base armour save (e.g. 4 means 4+, lower is better)")
```

### Feature Dim Extension Pattern

```python
# In _observation_to_numpy / _models_to_features:
# Existing: 2 (loc) + n_obj*2 (dists) + max_groups (one-hot) + 1 (closest) + 3 (alive, wound_ratio, max_w_norm)
# New player features: +5 (attacks, bs, strength, ap, damage) normalized
# New all-model features: +2 (toughness, save) normalized
# Expected damage: separate block, shape (n_player, n_opponent) or flattened
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Shooting no-op (Phase 4) | Full resolution pipeline (Phase 5) | This phase | Shooting actions now have consequences |
| WeaponProfile with range only | Full weapon stat block | This phase | Configs can now specify diverse weapons |
| Observation without combat stats | Combat features + expected damage | This phase | Agent has explicit combat information |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `just test` |
| Full suite command | `just validate` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SHOT-01 | Shoot action selects target, calls resolution | integration | `uv run pytest tests/test_shooting_resolution.py::TestShootingIntegration -x` | ❌ Wave 0 |
| SHOT-02 | Hit→wound→save→damage pipeline | unit | `uv run pytest tests/test_shooting_resolution.py::TestResolveShootingFunction -x` | ❌ Wave 0 |
| SHOT-04 | Advanced models cannot shoot | unit | `uv run pytest tests/test_shooting_resolution.py::TestShootingMaskExtensions -x` | ❌ Wave 0 |
| SHOT-05 | Engagement range prevents shooting | unit | `uv run pytest tests/test_shooting_resolution.py::TestShootingMaskExtensions -x` | ❌ Wave 0 |
| SHOT-06 | Configurable weapon profiles | unit | `uv run pytest tests/test_shooting_resolution.py::TestConfigExtensions -x` | ❌ Wave 0 |
| OBS-02 | Weapon/defense stats in observation | unit+integration | `uv run pytest tests/test_shooting_resolution.py::TestObservationExtension -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `just test`
- **Per wave merge:** `just validate`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_shooting_resolution.py` — new test file covering SHOT-01, SHOT-02, SHOT-04, SHOT-05, SHOT-06, OBS-02
- [ ] No new fixtures needed — extend existing patterns from `test_shooting_action.py` and `test_wounds.py`
- [ ] Framework install: not needed — pytest is already configured

## Open Questions

1. **Feature dim for expected damage observation**
   - What we know: Expected damage is per (attacker, target) pair, shape `(n_player, n_opponent)`. Existing per-model features are per-model vectors.
   - What's unclear: Should expected damage be flattened into the player model feature vector (each player model gets `n_opponent` extra features), or passed as a separate tensor in the observation pipeline?
   - Recommendation: Flatten into the per-player-model feature vector. Each player model's row gets `n_opponent` additional columns with the expected damage against each opponent. This keeps the single-tensor-per-entity pattern. For opponent models, the features include their toughness and save but not expected damage (they are targets, not attackers in the player's decision space).

2. **Opponent observation features: weapon stats or defense stats?**
   - What we know: D-12 says "Per opponent model: add normalized toughness and save." It doesn't mention adding weapon stats for opponent models.
   - What's unclear: Should opponent models also show their weapon stats in the observation? The agent might benefit from knowing opponent offensive capability.
   - Recommendation: Follow D-12 literally — only toughness and save for opponents in Phase 5. Opponent weapon stats can be added in Phase 6 when combat reward gives the agent reason to care about incoming damage.

3. **Save value of 7+ (no armour)**
   - What we know: Some tabletop models have no armour (effectively 7+ save, which always fails). Pydantic validation should allow `save=7`.
   - Recommendation: Allow `save` in range [2, 7] where 7 means "no armour save." The resolution code naturally handles this: `modified_save = 7 + ap` → always > 6 → save always fails.

## Discretion Recommendations

Based on the Claude's Discretion items from CONTEXT.md:

1. **Resolution location:** Place in `domain/shooting.py`. This follows the DDD pattern — resolution is a game rule, domain owns game rules. The function is pure (takes stats + RNG, returns result). The env orchestrates by calling it.

2. **Normalization scheme:** Match existing patterns in `_models_to_features`. Weapon stats normalize to [0, 1] using reasonable maxima: `attacks/10`, `bs/6`, `strength/10`, `ap/6`, `damage/10`, `toughness/10`, `save/7`. Expected damage normalizes to [0, 1] by dividing by a reasonable max (e.g. 10 damage).

3. **Expected damage computation timing:** Recompute each step. Cost is trivial (scalar arithmetic per pair), and targets change state mid-episode (take wounds, die). Computing per step ensures the agent always sees current information.

4. **D6 roll structure:** Vectorized numpy. Roll all attacks in one `rng.integers(1, 7, size=n_attacks)` call, filter hits with boolean masking, roll wounds for the hits, filter, roll saves for wounds. Three numpy calls per resolution — fast and clear.

5. **`advanced_this_turn` reset:** Add to `WargameModel.reset_for_episode()` (already exists for wound reset). For per-turn reset, the turn execution pipeline (`run_after_player_action` / `run_until_player_phase`) should call a reset on models at turn boundaries. Since advance doesn't exist yet, the flag is always False — the reset is structural prep.

6. **Engagement range threshold stub:** Use `ENGAGEMENT_RANGE = 1` (1 grid cell) as the constant. The check in `compute_shooting_masks` would be `distance(M, K) <= ENGAGEMENT_RANGE`, but since no mechanic places models in engagement range yet, it never triggers. The constant is a named value in `domain/shooting.py` for easy discovery when v3.0 implements engagement.

## Sources

### Primary (HIGH confidence)
- `docs/tabletop-rules-reference.md` — Attack sequence, wound roll table, save mechanics, engagement rules
- `docs/ddd-envs.md` — Domain layer architecture, extension patterns, dependency direction
- `.planning/codebase/CONVENTIONS.md` — Naming, registry, config, testing patterns
- `.planning/phases/04-shooting-action-space/04-CONTEXT.md` — Phase 4 decisions constraining Phase 5
- `.planning/phases/01-wounds-elimination/01-CONTEXT.md` — `take_damage` entry point

### Secondary (MEDIUM confidence)
- Numpy `random.Generator.integers` documentation — D6 roll semantics (well-known API)

### Tertiary (LOW confidence)
- None — all findings are from codebase inspection and project documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new deps, existing numpy/pydantic patterns
- Architecture: HIGH — DDD patterns well-documented, extension points clear from prior phases
- Pitfalls: HIGH — wound roll table edge cases verified against tabletop rules reference, observation extension pattern established in Phase 2
- Resolution mechanics: HIGH — fully specified in D-01 through D-04, verified against `docs/tabletop-rules-reference.md`
- Observation extension: HIGH — pattern established in Phase 2 (alive/wounds), feature_dim computation is explicit

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable — internal codebase patterns, no external dependency churn)
