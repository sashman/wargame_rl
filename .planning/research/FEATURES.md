# Feature Landscape — RL Tabletop / Grid Wargame Environments

**Domain:** Turn-based (or phase-based) tactical combat on a discrete grid, trained with RL
**Researched:** 2026-04-02
**Project lens:** Brownfield milestone — combat, terrain, opponent AI, advanced mechanics atop existing movement, objectives, curriculum, PPO/DQN, scripted opponents (`docs/goals-and-roadmap.md`, `docs/tabletop-rules-reference.md`).

**Overall confidence:** **MEDIUM–HIGH** for categorisation relative to this repo’s rules target and public RL tactical-env patterns; **MEDIUM** for “what every paper builds” (research prototypes vary widely).

---

## How mechanics usually show up in RL wargame-style envs

| Mechanic | Typical implementation | RL-facing concern |
|----------|------------------------|------------------|
| **Shooting** | Discrete target choice + range check + stochastic or deterministic damage | Exploding combinatorial action space; needs masking and/or factorisation |
| **LOS** | Grid raycast, visibility graph, or hex line algorithms; terrain blocks or soft-blocks | Must align with observation (honest partial observability vs omniscient cheat) |
| **Terrain** | Per-cell tags: open, difficult, blocking, cover; may affect movement cost, saves, visibility | Observation encoding and map variety for generalisation |
| **Self-play** | Two-player env API + same policy both sides, or frozen checkpoints / population | Training stability (non-stationarity), role conditioning, league/Elo optional |
| **Melee** | Adjacency / engagement range, often separate phase; no LOS or shorter range | Credit assignment across phases; coupling with movement (charge, pile-in) |
| **Morale** | Threshold triggers (e.g. half strength), random or deterministic pass/fail, debuffs | Delayed consequences; sparse signal unless shaped |

Recent adjacent work (e.g. hex-and-counter wargames with RL, grid tactical envs with fog/LOS/cover) reinforces that **terrain + visibility + combat resolution** appear together when the stated goal is tactical depth rather than abstract board control. Self-play appears as a **training methodology** once a symmetric two-player interface exists, not as a substitute for environment mechanics.

---

## Table stakes

Features a **credible** “wargame RL environment” (grid, ranged combat, two sides) is expected to expose or simulate. Omitting these tends to make “tactical combat” claims weak or the MDP degenerate.

| Feature | Why expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Damage / durability / removal** | Combat must change who can act and who controls space | Low–Med | Wounds, HP, or binary alive; elimination updates observation and action slots |
| **Ranged attacks with range limits** | Defines shooting as a distinct interaction from movement | Med | Often one target per model per step initially |
| **Visibility / LOS for ranged fire** | Without it, “shooting” collapses to range-only arcades | Med | Grid raycast or Bresenham-style; must match what the agent is allowed to know |
| **Legal action specification** | Invalid shoot/move must be disallowed or masked | Med | Gymnasium `action_masks`, invalid-action penalties, or structured action spaces |
| **Combat-aware reward (non-purely sparse)** | RL needs gradients when objectives are long-horizon | Med | Damage dealt/taken, eliminations, shaping balanced with mission VP |
| **Two-sided game state** | Wargames are adversarial; single-agent-vs-environment is a stepping stone | Med–High | API for two policies or player/opponent turns; your roadmap’s Phase 4 |
| **Turn / phase ordering consistent with actions** | Avoids ambiguous simultaneous resolution | Med | You already have multi-phase turns; stakes rise when shooting/melee go live |
| **Terrain minimum viable set** | “Tactics” without terrain is often just geometry | Med | At least **blocking** (movement and/or LOS) **or** **cover** (defensive modifier)—ideally both for rules fidelity |

**Dependency note:** Shooting **depends on** damage model + (for credibility) LOS + legal targets. Terrain **interacts with** movement (already present) and **should** interact with LOS/cover once shooting exists.

---

## Differentiators

Valued or research-novel relative to a minimal “units shoot each other on a grid” demo. Not all are required for a useful training sandbox.

| Feature | Value proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Self-play + checkpoint pool / league** | Automatic curriculum of stronger opponents; common in competitive RL | High | Requires stable two-agent loop; optional Elo/version tracking |
| **Faithful attack pipeline** | Hit → wound → save → damage (per `tabletop-rules-reference.md`) | High | Richer but slower to learn; good for sim fidelity, harder for first RL convergence |
| **Partial observability / fog-of-war** | Matches human information; stresses robust policies | Med–High | Tension with “debuggability”; must match LOS rules |
| **Curriculum across mechanics** | Phased rewards and success criteria (you already have reward phases) | Med | Differentiator vs one-shot sparse mission rewards |
| **Charge + fight sub-mechanics** | Pile-in, consolidate, fight order — tabletop flavour | High | Often phased after basic melee |
| **Morale / battleshock** | Cascading debuffs, OC suppression | Med–High | Sparse signal; needs careful shaping and phase hooks (Command phase) |
| **Command points / stratagems** | Extra action economy and tactical timing | High | Large design space; usually late milestone |
| **Procedural maps + scenario library** | Generalisation beyond hand-tuned boards | Med–High | Pairs well with terrain types |
| **Multi-unit coordination objectives** | Group cohesion + focus fire + objective VP (you partially have this) | Med | Team RL (MAPPO, QMIX) is optional future stack |
| **Transformer / entity-centric encoders** | Variable units/objectives; attention over battlefield | Med | Architectural differentiator, not a rules feature |

---

## Anti-features

Deliberately **not** building these (or not building them *early*) avoids scope explosion, misleading training signal, or contradictions with project constraints.

| Anti-feature | Why avoid | What to do instead |
|--------------|-----------|-------------------|
| **Perfect information while modelling hidden rules** | Violates stated principle “observation honesty” | Expose only visible enemies/cells; document belief-state if needed later |
| **Full weapon keyword matrix on day one** | Explodes rules surface (Indirect, Devastating, etc.) | Start with one weapon profile + optional toggles per scenario |
| **Army list building / points balancing** | Out of scope per `PROJECT.md` | Fixed scenarios in YAML; curated force lists |
| **Real-time continuous simulation** | Contradicts discrete grid + turn-based goal | Keep phase/step discrete |
| **3+ player diplomacy / FFA** | Out of scope; explodes joint action spaces | Strict two-player (or two-team) APIs |
| **Unshaped sparse combat** | Long episodes, vanishing gradients | Pair every new mechanic with calculators / phase-aware rewards |
| **Simultaneous move resolution without spec** | Ambiguous collisions and fairness | Prefer explicit turn order or documented simultaneous rules |

---

## Feature dependencies

```text
Wounds / elimination ─────────────────┐
                                      ├──► Shooting (credible ranged combat)
LOS / visibility ─────────────────────┤
Legal targets + action structure ────┘

Terrain: blocking ──────► LOS blocks, movement blocks
Terrain: difficult ──────► Movement costs (extends existing movement)
Terrain: cover ──────────► Shooting resolution modifier

Two-agent env API ───────► Self-play, Elo, frozen opponents
Shooting + melee + morale ► Full phase loop (command → … → fight) with real actions
```

**Ordering rationale (aligns with your roadmap):**
Combat resolution (wounds, shoot, LOS, actions) **before** heavy terrain variety is acceptable for prototypes, but **production-quality shooting** almost always pulls terrain in for LOS/cover. Opponent AI at strength **requires** a two-sided stepping API; self-play layers on top. Melee and morale **depend on** stable positioning, engagement concepts, and usually shooting already stressing the observation/action pipeline.

---

## MVP recommendation (for requirements downstream)

**Prioritise (table stakes slice):**

1. Functional wounds + elimination
2. Shooting with range + LOS + masked/valid actions
3. Action-type selection per model (move / shoot / pass)
4. Combat-aware reward components (damage, losses, tie-in to VP/objectives)
5. Terrain: blocking + cover (minimum set that interacts with movement and shooting)

**Defer as differentiators until the slice trains:** full dice attack sequence fidelity, stratagems, full morale chain, charge fight sequencing, league self-play.

---

## Sources

- Project: `.planning/PROJECT.md`, `docs/goals-and-roadmap.md`, `docs/tabletop-rules-reference.md` — **HIGH** confidence for intended feature set and rules mapping.
- Ecosystem (illustrative, not exhaustive): arXiv “Hex and Counter Wargames” (RL + terrain/unit interactions); open tactical Gymnasium-style projects (e.g. grid combat with LOS/fog/cover patterns cited in community repos); SPIRAL / self-play MARL literature for two-player training framing — **MEDIUM** confidence for “what researchers often add when going beyond toy combat.”

---

## Gaps / phase-specific follow-ups

- Exact partial-observability design (per-cell vs per-unit visibility, memory) — needs a dedicated design pass when LOS lands.
- Whether charge phase is stochastic (2D6) or abstracted for RL stability — feasibility trade-off.
- Single shared policy vs separate policies per side for self-play — architecture decision in Phase 4.
