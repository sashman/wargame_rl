# Architecture Research: Combat, Terrain/LOS, and Self-Play in a DDD Gymnasium Wargame

**Domain:** Brownfield RL tabletop wargame (discrete grid, multi-unit control, curriculum rewards)
**Researched:** 2026-04-02
**Scope:** How shooting/melee, terrain and line-of-sight, and multi-agent self-play are typically structured in grid-based RL environments, and how they should integrate with this repo’s **domain / `BattleView` / registry** layout.

**Overall confidence:** **HIGH** for integration with the existing codebase (grounded in `docs/ddd-envs.md` and `.planning/codebase/ARCHITECTURE.md`). **MEDIUM** for ecosystem-wide “typical RL” patterns (PettingZoo/AEC vs single-agent Gym is a product choice; algorithms vary).

---

## 1. Executive synthesis

Grid-based tactical RL environments usually separate **(A) authoritative simulation** (who can act, what resolves, what changes state) from **(B) observation construction** (what each policy sees) and **(C) training orchestration** (one vs two policies, frozen copies, league play). Combat and terrain belong almost entirely in **A**, implemented as **pure domain services** over the battle aggregate, with **LOS and cover** implemented as **query services** over a static or slowly changing **board layer** (terrain grid), not inside reward or render code.

Multi-agent self-play in turn-based games most often uses either **PettingZoo AEC** (natural fit for alternating turns) or a **single Gym env** that internally runs two sides with one training policy and a **frozen or scripted opponent**—the latter matches this project’s current scripted/random opponent hooks and avoids rewriting the Lightning stack around MARL APIs.

This codebase already has the right seams: **`GameClock` / `turn_execution`** for phase order, **`ActionRegistry`** for phase-valid slices and masks, **`BattleView`** for read-only consumers, and **registries** for rewards and opponents. New mechanics should extend the **aggregate and domain services**, widen **`BattleView`** and **`StepContext`** where observability or rewards need new facts, and keep **RL training** agnostic beyond observation/action shape changes.

---

## 2. Typical components in grid-based combat / terrain / self-play RL

### 2.1 Simulation core (authoritative rules)

| Component | Responsibility | Typical placement |
|-----------|----------------|-------------------|
| **Board / terrain layer** | Per-cell type (open, difficult, blocking, cover height, etc.); may be fixed per episode or scenario | Domain value object + factory from config; *not* in Gym facade |
| **Movement resolver** | Legal destinations, collision, terrain movement cost | Domain service (already: displacement + clamp; extend for difficult terrain) |
| **LOS / cover queries** | Can shooter see target? Cover modifier? | Domain **query service**: inputs = terrain + model positions (+ optional geometry rules); **no** dependency on reward/render |
| **Combat resolver** | Target selection validity, hit/wound resolution, simultaneous vs ordered resolution | Domain service invoked from phase execution after actions applied |
| **Unit state** | Wounds, eliminated, weapon stats, engagement range | Fields on `WargameModel` (or related entity); mutations only via aggregate rules |

**Opinionated choice:** Implement **LOS once** as a small `los.py` (or `visibility.py`) module under `domain/` that operates on `(terrain_grid, positions)` and is called by shooting/charge rules and optionally by **observation builders** (for “visible enemies” channels). Avoid duplicating Bresenham in reward calculators.

**Ecosystem note:** Tile tactics and roguelikes commonly use **Bresenham**, **supercover**, or **shadow casting** for LOS; symmetry and “corner peeking” rules are design choices—document them in domain tests (MEDIUM confidence: common practice on grid games; see e.g. tile-based LOS discussions on GameDev Stack Exchange).

### 2.2 Action space and phase integration

| Component | Responsibility | Fit to this repo |
|-----------|----------------|------------------|
| **Phase-gated action slices** | Shooting/melee/charge actions only valid in matching `BattlePhase` | `ActionRegistry` + masks (`docs/movement.md` already describes extension) |
| **Action decoding** | Map discrete index → target id, weapon, or “stay” | `ActionHandler` in `env_components/actions.py` |
| **Legal action mask** | Exclude out-of-range, no LOS, wrong phase, eliminated models | Same pipeline as movement; may need **target enumeration** order fixed for stable indices |

**Data shape pattern:** Many grid RL games use **fixed max targets** (pad invalid slots and mask) or **factored actions** (move direction + discrete target slot). This project’s transformer-over-entities stack favors **consistent ordering** of candidate targets in observation and mask generation.

### 2.3 Observations and information hiding

| Component | Responsibility | Fit |
|-----------|----------------|-----|
| **Per-agent or per-side view** | Fog-of-war, visible units only | Extend `observation_builder` from `BattleView`; enforce **observation honesty** (`PROJECT.md`) in one place |
| **Combat-relevant features** | Range bands, LOS bit, cover, wounds | Tensorized from view + **read-only** queries (LOS service reading same terrain as domain) |

**Anti-pattern:** Computing LOS or “can shoot” only inside a reward calculator while the agent observation shows full state—creates exploitable train/test mismatch.

### 2.4 Reward and curriculum

| Component | Responsibility | Fit |
|-----------|----------------|-----|
| **Reward calculators** | Damage dealt, objectives, penalties | New registered calculators; consume `BattleView` + `StepContext` |
| **Success criteria** | Elimination, VP, hold objectives | New registered criteria |
| **Phase ordering** | Simple battlefield skills before full VP | Already documented in `docs/reward-phases.md` |

**Pattern:** Expose **step deltas** (e.g. `player_vp_delta`-style) on the battle or info for shaping; keep calculators dumb and declarative.

### 2.5 Opponent AI and self-play

| Pattern | Description | When to use |
|---------|-------------|-------------|
| **Scripted / heuristic opponent** | Fixed policy API | Already: `opponent/` registry |
| **Frozen snapshot** | Same network weights, no grad | Simple self-play; periodic refresh |
| **League / pool** | Mix of past checkpoints | Stronger non-stationarity, more engineering |
| **Symmetric two-policy training** | Two optimizers or role swapping | Heavier; often deferred |

**Ecosystem note:** **PettingZoo** AEC API matches turn-based games; tutorials cover DQN + curriculum + self-play for board-style envs (MEDIUM confidence: Farama/PettingZoo docs and AgileRL tutorial references). **This repo** can stay **single-agent Gym** longer by treating “opponent” as environment-side policy with optional **frozen RL policy** implementing the same `OpponentPolicy` interface—avoids adopting PettingZoo until a milestone explicitly requires it.

---

## 3. Integration with existing DDD architecture

### 3.1 Dependency direction (preserve)

```
domain/  →  types/  (config, timing)
env (wargame.py)  →  domain/, env_components/, reward/, renders/, opponent/
reward/, renders/, opponent/  →  BattleView (+ types), not concrete Battle internals
```

- **Combat resolution, terrain, LOS:** add **domain modules** and **aggregate fields**; **do not** import `env_components` from `domain/`.
- **Gym action space size / masks:** `env_components/actions.py` only.
- **What the agent sees:** `env_components/observation_builder.py`, using `BattleView` + domain queries as needed.
- **Shaping and eval success:** `reward/` registries; extend `StepContext` for combat/terrain summaries as promised in `docs/reward-phases.md`.

### 3.2 `BattleView` protocol evolution

Add **read-only** surface area for anything reward/render/observation needs:

- Terrain: e.g. `terrain_grid` or accessor for cell type at `(x, y)`.
- Combat: e.g. per-model wounds, eliminated flags, last combat event summary (for logging/shaping).
- Clock: already exposes phase; ensure shooting/fight phases are meaningful when not skipped.

Implementations: `WargameEnv` delegates to `_battle` / domain.

### 3.3 Registries (unchanged pattern)

| Subsystem | Registry | Extension |
|-----------|----------|-----------|
| Reward | `reward/calculators/registry.py`, `criteria/registry.py` | New keys: `damage_dealt`, `models_remaining`, etc. |
| Opponent | `opponent/registry.py` | New policy: `frozen_rl_checkpoint`, `mirror_agent` |
| Mission | `mission/registry.py` | If scoring ties to kills/terrain—only if rules require |

### 3.4 Turn and phase execution

`domain/turn_execution.py` and `game_clock.py` should own **when** combat steps run relative to movement. Opponent turns already run through the same pipeline; **symmetric rules** (same action space layout for both sides) simplify eventual true self-play.

---

## 4. Component boundaries (what talks to what)

| From | To | Allowed |
|------|-----|---------|
| `Battle` / aggregate | Domain services (LOS, combat, movement) | Yes—mutations only through aggregate methods |
| Domain services | `Battle`, terrain value objects, `types/` | Yes |
| Domain | `reward/`, `renders/`, `env_components` | **No** |
| `ActionHandler` | `Battle`, `GameClock`, domain combat/movement | Yes |
| `observation_builder` | `BattleView`, domain **read-only** queries | Yes |
| `RewardPhaseManager` | `BattleView`, `StepContext` | Yes |
| `OpponentPolicy` | Env reference (for masks/state) per existing pattern | Yes—keep thin; prefer `BattleView` if refactoring allows |
| Lightning / `Agent` | `gym.Env` API only | No direct domain imports |

---

## 5. Data flow (explicit direction)

**Step (training rollouts):**

1. **Agent** outputs discrete actions per model (possibly larger space: move/shoot/melee/stay).
2. **`WargameEnv.step`** → **`ActionHandler`** validates against mask, decodes slice → domain intent (target id, etc.).
3. **Domain** applies movement or combat resolution → updates **`Battle`** (positions, wounds, eliminations, terrain occupancy if dynamic).
4. **`turn_execution`** advances **phases/rounds**, runs **opponent policy** (scripted, random, or frozen RL) with same masking rules.
5. **Mission/VP hooks** run at clock boundaries (existing `_on_before_advance` pattern).
6. **`DistanceCache`** / optional **connectivity or LOS cache** rebuilt or invalidated for the step.
7. **`StepContext`** assembled (extend with combat/terrain deltas).
8. **`RewardPhaseManager.calculate_reward(view, ctx)`** → scalar + breakdown.
9. **`build_observation(view)`** → agent tensor; **info** may carry diagnostics for logging.

**Self-play variant:** Between (3) and (4), opponent actions come from a **second policy head** or **frozen copy** with the **same observation builder** applied from the opponent’s perspective if using fog-of-war; if full observability is intentional for research, document it as a **different env mode** to avoid silent honesty violations.

---

## 6. Suggested build order (dependencies)

Order minimizes rework and respects existing DDD seams:

1. **Terrain board + placement**
   - Config → factory → `Battle` + `BattleView`.
   - Movement cost and blocking only (no combat yet).
   - **Unlocks:** difficult ground, observation channels, render layers.

2. **LOS / cover query service (domain)**
   - Pure functions over terrain + positions.
   - Unit tests without Gym.
   - **Unlocks:** shooting validity, observation masking, future AI heuristics.

3. **Wounds / elimination + aggregate invariants**
   - Models removed or marked dead; termination hooks (`termination.py`).
   - **Unlocks:** meaningful combat outcomes and simpler masks.

4. **Shooting (then melee) action slices + resolver**
   - Register slices in `ActionRegistry`; implement resolver in domain; wire `ActionHandler`.
   - Extend masks for range/LOS/target legality.
   - **Unlocks:** first full “contest” dynamics.

5. **Reward calculators + success criteria + curriculum phases**
   - Register damage/objective/hold criteria.
   - **Unlocks:** trainable combat behaviour without changing algo core.

6. **Observation honesty pass**
   - Align fog/visibility with LOS service.
   - **Unlocks:** transferable policies and fair eval.

7. **Opponent escalation: scripted → frozen checkpoint self-play**
   - New opponent registry type; training loop loads frozen weights.
   - Optional later: PettingZoo AEC or league if single-env approach plateaus.

**Parallelizable:** Render support for terrain and LOS debug overlays can trail (1–2) by a step; **must** use `BattleView` only.

---

## 7. Pitfalls that affect architecture (concise)

| Pitfall | Consequence | Prevention |
|---------|-------------|------------|
| LOS logic only in rewards or renderer | Divergent rules vs actual shooting | Single domain LOS module called by rules + obs |
| Exploding action space without ordering | Unstable masks and attention alignment | Fixed target ordering or hierarchical actions |
| Full-state observations + “human-like” claim | Cheating policies | Explicit per-side observation modes in config |
| Combat in `wargame.py` | Untestable god object | Keep facade thin; domain services own rules |
| Self-play without periodic refresh | Oscillation / exploitation cycles | Checkpoint pool or scheduled opponent updates |

---

## 8. Sources and confidence

| Claim | Confidence | Source |
|-------|------------|--------|
| DDD layering and extension points for this repo | HIGH | `docs/ddd-envs.md`, `.planning/codebase/ARCHITECTURE.md`, `docs/movement.md`, `docs/reward-phases.md` |
| Action slices + masks for new phases | HIGH | `docs/movement.md` (documented extension pattern) |
| PettingZoo AEC for turn-based MARL / self-play tutorials | MEDIUM | PettingZoo / Farama ecosystem, AgileRL tutorial references (web search summary) |
| LOS algorithms on grids (Bresenham, shadow casting, symmetry caveats) | MEDIUM | Game dev literature / Stack Exchange (web search summary); exact algorithm is a **design** choice |

---

## 9. Roadmap consumer checklist

- [x] **Components defined** with boundaries (simulation core, actions, obs, reward, opponent/self-play).
- [x] **Data flow direction** explicit (agent → env → domain → clock/opponent → reward → obs).
- [x] **Build order** stated with dependencies (terrain → LOS → wounds → combat actions → reward → honesty → self-play).

**Open point for a later phase:** Whether to adopt **PettingZoo** for first-class two-agent training or keep **Gym + frozen opponent** until combat/terrain stabilize (tradeoff: API churn vs native MARL tooling).
