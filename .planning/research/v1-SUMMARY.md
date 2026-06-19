# Project Research Summary

**Project:** Wargame RL
**Domain:** Brownfield RL tabletop wargame (discrete grid, multi-unit control, curriculum rewards) expanding into combat, terrain/LOS, and self-play
**Researched:** 2026-04-02
**Confidence:** MEDIUM–HIGH (HIGH for stack/architecture fit to this repo; MEDIUM for ecosystem-wide RL tactical-env norms)

## Executive Summary

This milestone is **not** a stack pivot: Gymnasium, PyTorch Lightning, PPO (default) + DQN, transformer-over-entities, Pydantic YAML configs, Wandb, and UV remain the right foundation. The upcoming work is **domain logic and training patterns**—combat resolution, terrain grids, LOS queries, and opponent escalation—implemented mostly as **pure Python/NumPy** domain services under the existing DDD layout (`domain/`, `BattleView`, registries), not as a game engine or heavy MARL framework.

Experts building credible tactical RL environments typically separate **authoritative simulation** (terrain, LOS, wounds, legal actions) from **observation construction** and **training orchestration** (single-agent Gym with environment-side opponent vs PettingZoo AEC). This repo should **extend aggregate and `BattleView`**, keep **one LOS/combat source of truth** callable from rules and observations, and grow **action spaces with explicit legal masks** derived from the same domain rules. Self-play should favor a **frozen checkpoint pool** and scripted baselines before considering league play or PettingZoo.

Primary risks are **reward shaping diverging from real mission outcomes**, **action-space explosion without factoring**, **subtle LOS/terrain bugs that RL exploits**, **curriculum ordering that fights new mechanics**, and **naive self-play cycling**. Mitigations: sparse anchors on VP/success criteria, factored actions + mask unit tests, golden/property tests for LOS, phased calculators aligned with `docs/reward-phases.md`, and opponent pools with fixed scripted eval suites—not Elo alone.

## Key Findings

### Recommended Stack

**Keep:** Gymnasium 1.x, PyTorch + Lightning 2.5+, PPO + DQN, transformer policy, Pydantic/pydantic-yaml, Wandb, UV. No OpenSpiel, RLlib multi-agent, or Unity/Godot for grid dice math.

**Add (mostly zero new dependencies):** NumPy for vectorized combat dice; Bresenham-style LOS in-domain; `torch.profiler` for GPU profiling alongside existing pyinstrument. **Optional:** Wandb Sweeps (or Optuna + Wandb) for hyperparameter search; PettingZoo only if a milestone explicitly wants first-class two-agent APIs.

**Key insight:** Upcoming features are **domain + reward shaping**, not infrastructure churn. The main fork is **extend Gym + `OpponentPolicy` (frozen checkpoint)** vs **adopt PettingZoo**—defer until combat/terrain stabilize.

Detail: [STACK.md](./STACK.md).

**Core technologies:**

- **Gymnasium + existing Lightning loop:** Stable training surface; opponent stays env-side until P4 needs a hard two-agent contract.
- **NumPy:** Batch RNG and threshold comparisons for hit/wound/save/damage—no external combat library.
- **Domain modules + `BattleView`:** Terrain grid, LOS queries, wounds—keeps reward/render honest and testable.

### Expected Features

**Must have (table stakes):** Damage/durability/removal; ranged attacks with range limits; **LOS aligned with what the agent may know**; legal actions or masks; combat-aware (non-purely-sparse) reward; two-sided game state API path; turn/phase ordering consistent with actions; **terrain minimum: blocking and/or cover** for credible tactics.

**Should have (differentiators):** Self-play + checkpoint pool/league; faithful tabletop attack pipeline; fog-of-war; curriculum across mechanics (already partially present); procedural maps; multi-unit coordination objectives; transformer entity encoding (already architectural).

**Defer (v2+ / anti-features):** Perfect info while claiming hidden rules; full weapon keyword matrix day one; army building; real-time sim; 3+ player FFA; unshaped sparse combat-only training; simultaneous resolution without a written spec.

**MVP slice (for requirements):** (1) wounds + elimination, (2) shooting with range + LOS + masked actions, (3) per-model action type (move/shoot/pass), (4) combat-aware reward tied to objectives, (5) blocking + cover terrain interacting with movement and shooting.

Detail: [FEATURES.md](./FEATURES.md).

### Architecture Approach

Combat, terrain, and LOS live in **domain services** and the **battle aggregate**; **LOS implemented once** (e.g. `domain/los.py`) and reused by shooting rules, masks, and (when enabled) observation filtering—not duplicated in reward or Pygame. **`ActionRegistry` / `ActionHandler`** own space size and masks; **observation_builder** reads `BattleView` + read-only queries; **reward registries** stay declarative with extended `StepContext`. **Self-play:** new `OpponentPolicy` types (e.g. frozen RL checkpoint) before rewriting around PettingZoo.

**Major components:**

1. **Board / terrain layer** — Per-cell types from config; movement cost and blocking; observation channel.
2. **LOS / cover query service** — Grid ray trace over terrain + positions; cover modifiers for saves.
3. **Combat resolver + unit state** — Wounds, elimination, ordered resolution; invoked from phase execution.
4. **Phase-gated actions + masks** — Shooting/melee slices; fixed target ordering for transformer alignment.
5. **Opponent escalation** — Scripted → frozen snapshots → pool/Elo; symmetric rules for both sides.

**Recommended build order (dependency-minimizing):** terrain board + placement (blocking/difficult) → LOS/cover query (tests without Gym) → wounds/elimination + alive masks → shooting (then melee) + `ActionHandler` wiring → reward calculators + curriculum phases → observation honesty pass → frozen-checkpoint self-play.

Detail: [ARCHITECTURE.md](./ARCHITECTURE.md).

### Critical Pitfalls

1. **Reward shaping hijacks the mission** — Keep sparse anchors (VP, terminal outcomes); eval vs **fixed scripted** opponents before advancing phases; log decomposed combat metrics (P2, ties to P1 dashboards).
2. **Action space explosion / bad masks** — Factor action type vs arguments; unit-test mask ↔ domain legality bidirectionally; grow complexity incrementally (P2, P5 melee adds branches).
3. **LOS/terrain bugs as learnable exploits** — Golden/property tests on known boards; renderer uses same queries as rules; watch edge positioning and obs/render disagreement (P2–P4).
4. **Curriculum fights new mechanics** — Extend reward-phase discipline: movement/group competence before heavy combat mix; dedicated phases per new calculator (P2, P1 eval harness).
5. **Naive self-play cycling** — Frozen **pool**, periodic snapshots, mixture with scripted/random; anchor Elo with fixed baselines, not self-play-only (P4).
6. **Two-agent refactor leaks info or breaks Markov property** — Per-side observation contract from `BattleView` + tests before shipping P4 (design touchpoints P2–P3 if visibility matters).
7. **Scale: attention cost and missing PE** — P1 positional encoding before large squads; P6 profile env + network; curriculum from smaller battles upward.

Detail: [PITFALLS.md](./PITFALLS.md).

## Implications for Roadmap

Official phase themes in [docs/goals-and-roadmap.md](../../docs/goals-and-roadmap.md) (P1–P6) align with research, with one **ordering tension**: architecture and feature dependency graphs favor **terrain + LOS primitives before production-quality shooting**, while a linear “P2 combat then P3 terrain” plan risks rework and weak tactical credibility. **Recommendation:** overlap or partially **front-load** a minimal terrain grid and LOS query service **before or in the same wave as** ranged combat going trainable—not full procedural maps on day one.

### Phase 1: Foundation & instrumentation (P1)

**Rationale:** Scaling and debugging depend on metrics and representation before combat complexity lands.
**Delivers:** Per-model speed, transformer positional encoding, improved dashboards, Wandb Sweeps (or Optuna) as needed.
**Addresses:** Active `PROJECT.md` items; pitfall #7 (scale), #1 (instrumentation for reward hijack detection).
**Avoids:** Training large squads on an under-specified entity encoding.

### Phase 2: Board primitives & LOS service (early P3 + domain spine)

**Rationale:** ARCHITECTURE and FEATURES agree credible shooting pulls terrain/LOS; implement **blocking/difficult** and **Bresenham LOS + cover query** in `domain/` with unit tests.
**Delivers:** Terrain in config/factory, `BattleView` surface, movement extension, LOS module used by nothing else yet—or immediately by mask prototypes.
**Addresses:** Table stakes terrain minimum; pitfall #3 (single source of truth).
**Avoids:** LOS only in reward or renderer.

### Phase 3: Wounds, elimination, and alive-aware pipeline (P2 prep)

**Rationale:** Shooting without durable state changes is not a MDP upgrade; masks and obs must handle **elimination** without ghost exploits.
**Delivers:** Wound model, termination hooks, padded obs + alive flags, sync domain list ↔ obs builder.
**Addresses:** Table stakes damage/removal; pitfall #9.
**Avoids:** Mid-episode shape bugs and pad exploitation.

### Phase 4: Ranged combat, action factoring, combat rewards (P2 core)

**Rationale:** Depends on LOS legality + wounds; unlocks first adversarial “contest.”
**Delivers:** Phase-gated shoot actions, domain resolver (start simpler than full dice pipeline if needed), registered calculators/criteria, curriculum phases.
**Addresses:** MVP slice items 2–4; stack choice (NumPy resolution).
**Avoids:** Pitfalls #1–2 (shaping + action explosion); defer full keyword fidelity to later.

### Phase 5: Terrain depth, cover curriculum, observation honesty

**Rationale:** Cover modifiers, richer maps, and **fog/visibility** aligned with LOS service; separate from “grid exists.”
**Delivers:** Cover in resolution, optional procedural maps, per-side obs modes documented in config.
**Addresses:** Differentiators; pitfall #3, #10 (terrain visible but unrewarded).
**Avoids:** Perfect-info cheat vs stated honesty.

### Phase 6: Two-agent API & self-play (P4)

**Rationale:** Requires stable rules + obs contract for both sides.
**Delivers:** Frozen checkpoint `OpponentPolicy`, pool sampling, Elo logging (pure Python), optional PettingZoo only if justified.
**Addresses:** Self-play differentiator; stack pattern (no RLlib).
**Avoids:** Pitfalls #5–6 (cycling, info leaks).

### Phase 7: Melee, morale, command (P5)

**Rationale:** Depends on stable positioning, engagement, and shooting stress on action pipeline.
**Delivers:** Melee/charge/fight hooks, phased enablement, integration tests for `skip_phases` vs full rules.
**Avoids:** Pitfall #11 (train/test phase mismatch).

### Phase 8: Scale & polish (P6)

**Rationale:** 10+ models, batched inference, throughput.
**Delivers:** Profiling-driven batching, optional local obs windows if attention cost bites.
**Avoids:** Pitfall #7.

### Phase Ordering Rationale

- **Simulation before training tricks:** Domain terrain/LOS/wounds before heavy self-play.
- **One query module for LOS** drives rules, masks, and later fog—reduces exploit surface.
- **P4 after** combat + honesty stabilizes avoids refactoring two policies against moving rules.
- **P1 PE and metrics** early prevents false negatives when P2–P6 expand state and action space.

### Research Flags

**Likely need `/gsd-research-phase` or a design pass during planning:**

- **Partial observability:** Per-cell vs per-unit visibility, memory, and config surface (FEATURES gaps).
- **PettingZoo vs Gym + frozen opponent:** API and Lightning integration tradeoff (ARCHITECTURE open point).
- **Charge/melee stochasticity:** 2D6 vs abstraction for RL stability (FEATURES gap).
- **Single shared policy vs per-side policies** for self-play (FEATURES / P4).

**Standard patterns (skip extra research unless integration surprises):**

- Bresenham LOS on a grid; NumPy dice pipelines; Wandb Sweeps; frozen checkpoint pools; reward registry extensions per `docs/reward-phases.md`.

## Confidence Assessment

| Area         | Confidence   | Notes                                                                 |
| ------------ | ------------ | --------------------------------------------------------------------- |
| Stack        | **HIGH**     | Grounded in current repo; explicit avoid list; minimal new deps       |
| Features     | **MEDIUM–HIGH** | Strong fit to `PROJECT.md` / rules docs; ecosystem “paper norm” varies |
| Architecture | **HIGH**     | Matches `docs/ddd-envs.md` and codebase seams; PettingZoo is optional |
| Pitfalls     | **MEDIUM–HIGH** | Repo-specific items HIGH; general RL self-play/LOS MEDIUM             |

**Overall confidence:** **MEDIUM–HIGH**

### Gaps to Address

- Exact fog-of-war and per-side observation contract—resolve before locking P4.
- Whether to simplify the attack pipeline for first trainable combat vs full tabletop fidelity—product/curriculum choice.
- Charge phase randomness vs deterministic training aids—feasibility when P5 scopes.
- Confirm roadmap ordering: **minimal terrain+LOS before full P2** vs strict P2-then-P3—stakeholders should align with this synthesis to avoid duplicate LOS implementations.

## Sources

### Primary (HIGH confidence for this repo)

- [.planning/PROJECT.md](../PROJECT.md) — requirements and scope
- [docs/goals-and-roadmap.md](../../docs/goals-and-roadmap.md) — phase themes (P1–P6)
- [docs/tabletop-rules-reference.md](../../docs/tabletop-rules-reference.md) — rules mapping
- [docs/ddd-envs.md](../../docs/ddd-envs.md), [docs/reward-phases.md](../../docs/reward-phases.md) — layering and curriculum
- [.planning/codebase/ARCHITECTURE.md](../codebase/ARCHITECTURE.md), [.planning/codebase/CONCERNS.md](../codebase/CONCERNS.md) — integration and known risks

### Secondary (MEDIUM confidence)

- Parallel research artifacts: [STACK.md](./STACK.md), [FEATURES.md](./FEATURES.md), [ARCHITECTURE.md](./ARCHITECTURE.md), [PITFALLS.md](./PITFALLS.md)
- RL tactical / self-play literature and grid-LOS practice (summarized in research files; verify per implementation)

---
*Research completed: 2026-04-02*
*Ready for roadmap: yes*
