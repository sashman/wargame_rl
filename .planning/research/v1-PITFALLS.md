# Domain Pitfalls: RL Tabletop Wargame (Combat, Terrain, Self-Play, Scale)

**Project:** Wargame RL (brownfield: movement, objectives, reward phases, PPO/DQN, scripted opponents)
**Researched:** 2026-04-02
**Scope:** Mistakes projects like this commonly make when adding **combat**, **terrain/LOS**, **self-play**, and **larger scenarios** — tied to this repo’s roadmap phases and reward architecture.

**Overall confidence:** **MEDIUM–HIGH** for RL/game-design patterns (literature + community consensus); **HIGH** for pitfalls that directly reference this codebase’s stated constraints (`PROJECT.md`, `CONCERNS.md`, `reward-phases.md`).

---

## How to Read This Doc

Roadmap phases referenced below match [docs/goals-and-roadmap.md](../docs/goals-and-roadmap.md):

| Phase | Theme |
|-------|--------|
| **P1** | Foundation (metrics, per-model speed, transformer PE, sweeps) |
| **P2** | Combat: shooting, wounds, LOS, action-type selection, combat rewards |
| **P3** | Terrain (cover, blocking, difficult, map gen) |
| **P4** | Two-agent env, self-play, Elo |
| **P5** | Melee, morale, command abilities |
| **P6** | Scale (10+ models, batched inference), polish |

For each pitfall: **warning signs**, **prevention**, **primary phase(s)** to address it.

---

## Critical Pitfalls

### 1. Reward shaping hijacks the real game objective

**What goes wrong:** Agents maximise dense combat shaping (damage dealt, kills) while ignoring objectives, VP, or survival — or the opposite: never learning to fight because movement/objective rewards dominate. Shaped rewards can become a *different game* than the mission you think you encoded.

**Why it happens:** Combat introduces many local signals; curriculum phases can advance while the policy is a “reward hacker” for the current calculator mix. Per-phase averaging of per-model rewards ([reward-phases.md](../../docs/reward-phases.md)) can also dilute or skew gradients when some models rarely shoot.

**Consequences:** Policies look good on Wandb reward curves but lose vs scripted opponents, collapse when weights change, or fail when you add terrain/LOS (distribution shift).

**Warning signs:**

- Reward components anticorrelate with win rate / VP / episode success criteria.
- Policies differ sharply when you zero one calculator weight.
- Success criteria met with trivial tactics (e.g. hide and plink) that break in full rules.

**Prevention:**

- Keep a **sparse anchor** on true outcomes (terminal win/loss, VP, or `player_vp_min`-style criteria) even in early combat curricula; use shaping as gradient hint, not sole objective.
- Add **combat-specific success criteria** and/or evaluation episodes against **fixed scripted opponents** before advancing phases — not only in-distribution self-play metrics.
- Log **decomposed metrics** (damage dealt/taken, models lost, objective control time) as first-class dashboards (aligns with P1 metrics work).

**Primary phases:** **P2** (design combat calculators + criteria); **P1** (instrumentation); **P4** (adversarial evaluation).

**Sources:** Reward shaping vs true objective is a recurring theme in strategy-game RL (e.g. discussion of shaping and sparse terminal signals in RTS-style settings); **MEDIUM** — pattern is well established, exact citations vary by benchmark.

---

### 2. Action space explosion without factored structure

**What goes wrong:** Combining per-model `(move | shoot | stay) × (direction/speed | target)` blows up discrete branches; training becomes sample-inefficient or numerically unstable. Masking invalid actions incorrectly causes **illegal-action bugs** or **exploitable “do nothing” basins**.

**Why it happens:** Tabletop phases are sequential; a naive Cartesian product of choices per model per phase ignores legality (range, LOS, phase).

**Consequences:** Long wall-clock times, policies that never discover shooting, or agents that exploit implementation bugs in masks.

**Warning signs:**

- Random policy baseline rarely takes valid shoot actions.
- Gradient norms or entropy collapse when action dims jump.
- High fraction of “no-op” or masked-out logits in traces.

**Prevention:**

- **Factor** actions: e.g. type head + argument head (or separate phase-specific spaces) with **explicit legal-action masks** derived from domain rules (same source of truth as `BattleView`).
- Unit-test **mask consistency**: every masked action is illegal in domain; every illegal action is masked.
- Grow complexity incrementally (roadmap already says: test mechanics in isolation).

**Primary phases:** **P2** (action-type selection); **P5** (melee adds another branch).

**Sources:** Common in discrete RL for multi-unit games; **MEDIUM** (practice consensus).

---

### 3. LOS / terrain bugs become “learnable exploits”

**What goes wrong:** Off-by-one LOS, wrong cover occlusion, or asymmetric rules (player vs opponent) create arbitrary advantages. RL will **exploit** them harder than humans.

**Why it happens:** Grid LOS and cover are easy to get subtly wrong; two-sided play doubles the surface area.

**Consequences:** Policies overfit to simulator quirks; behaviour looks absurd in renderings; fixes after training invalidate checkpoints.

**Warning signs:**

- Agent systematically positions on “weird” cells or board edges.
- Performance drops when you change only LOS/cover implementation.
- Disagreement between renderer visualization and combat resolution.

**Prevention:**

- Property-based or golden-file tests: known board layouts → expected LOS/cover outcomes.
- Cross-check **observation honesty** (`PROJECT.md`: no perfect info) with **actual information given** — e.g. if opponent is hidden, obs must not leak it via LOS side channels.
- Fix **renderer and rules** from the same primitive (query domain/BattleView, don’t duplicate ray logic in Pygame only).

**Primary phases:** **P2** (LOS); **P3** (cover/blocking); **P4** (two-sided consistency).

**Sources:** General RL/simulation robustness; **MEDIUM**.

---

### 4. Curriculum phase order fights new mechanics

**What goes wrong:** Jumping to “engage opponents” or VP-only phases before the policy can **move and group** yields flat learning. Conversely, staying too long on movement-only phases **entrenches** tactics that are bad once shooting exists (e.g. clumping in open ground).

**Why it happens:** Reward phases are powerful; their ordering is easy to mis-tune when you add calculators (`damage_dealt`, `models_lost`, etc.).

**Consequences:** Wasted GPU; false conclusion that “combat doesn’t train.”

**Warning signs:**

- Success rate stuck with success criteria that require combat but movement metrics still poor.
- Large regression when enabling shooting mid-run without a new phase.

**Prevention:**

- Follow the documented intent: **objectives/grouping before VP-heavy “win the game”** ([reward-phases.md](../../docs/reward-phases.md)); extend the same idea for combat: **shooting competence against static or scripted targets** before full adversarial mix.
- When adding calculators, add **dedicated phases** and thresholds; avoid changing weights of many calculators at once.

**Primary phases:** **P2** (new calculators + criteria); touches **P1** (eval harness).

**Sources:** Project’s own reward-phase docs; **HIGH** for this repo.

---

### 5. Naive self-play: cycling, exploitability, and non-transitivity

**What goes wrong:** Training against the **current** policy only can produce **cyclic** or **brittle** strategies: each version beats the previous but is easily exploited by other styles. In imperfect information or rich observation games, best-response dynamics need not converge to low-exploitability Nash-like play.

**Why it happens:** Homogeneous self-play is the default “turn on two copies” approach; without a **population**, **historical snapshots**, or **diverse opponents**, the environment’s Nash set may be large or non-transitive.

**Consequences:** Misleading Elo curves; agents that fail vs humans or scripted heuristics; research-time spent tuning a policy that only beats itself.

**Warning signs:**

- Win rate vs checkpoint **50–100 epochs ago** swings wildly.
- Policy beats latest snapshot but loses to **older** snapshots or simple scripts.
- Same hyperparameters produce different “styles” on different seeds with no clear improvement.

**Prevention:**

- Use a **frozen opponent pool** (uniform or biased sampling), **periodic snapshots**, and/or **fpsp-style** mixture of self and historical policies — align with roadmap items (freeze copies, Elo across versions).
- Keep **scripted and random opponents** in the training mixture for grounding.
- Track **exploitability proxies**: performance vs fixed suite + vs population (not only vs current).

**Primary phases:** **P4**; instrumentation starts **P1**.

**Sources:** Surveys and recent work on self-play limitations and alternatives (e.g. arXiv survey on self-play in RL, discussions of cycling / exploitability in competitive settings); **MEDIUM** — specifics depend on game class.

---

### 6. Two-agent refactor breaks single-agent assumptions

**What goes wrong:** Observation space, turn order, opponent step hook, and info dict semantics are built for **one learning side**. Bolting on a second trainable agent without re-auditing spaces causes **duplicate keys**, **wrong partial observability**, or **non-Markov** tuples from hidden opponent RNG.

**Why it happens:** `WargameEnv` and training loop assume a single policy; opponent is scripted today.

**Consequences:** Subtle bugs (learning from opponent’s privileged info, wrong reward attribution), flaky self-play.

**Warning signs:**

- Opponent actions appear in learner obs without fog-of-war rules.
- Same env yields different Markov properties when `opponent_policy` swaps.

**Prevention:**

- Treat **BattleView + info filtering** as the contract for *both* sides; add conformance tests for “what P1/P2 can see.”
- Separate **environment truth** from **per-agent observations** explicitly before P4 ships.

**Primary phases:** **P4** (core); design review in **P2–P3** if LOS/terrain affect visibility.

**Sources:** Multi-agent RL partial observability best practices; **MEDIUM**.

---

### 7. Scaling scenarios: attention, sequence length, and episode cost

**What goes wrong:** More models → longer sequences for the transformer, **quadratic attention cost**, larger action sampling batches, and **longer episodes**. Training throughput collapses; PPO rollouts dominate wall time.

**Why it happens:** Roadmap targets **10+ models per side** with batched inference ([CONCERNS.md](../codebase/CONCERNS.md) already flags **missing positional encoding** as a scaling risk).

**Consequences:** Need to cut batch size or truncating context hurts tactics; without PE, permutation-sensitive tactics are harder to learn.

**Warning signs:**

- GPU util drops while CPU rolls envs.
- Step time grows superlinearly with model count.
- Policies trained on small squads fail to generalize even with linear width increase.

**Prevention:**

- **P1:** Add positional encoding (or unambiguous slot identity) before relying on large squads.
- **P6:** Batched forward across units; profile env step; consider **local windows** or **entity grouping** in obs if full attention is too costly.
- Curriculum: train smaller battles first, then scale board/units.

**Primary phases:** **P6** (throughput); **P1** (representation); **P2–P5** (avoid encoding choices that don’t scale).

**Sources:** CONCERNS.md scaling limits; standard transformer scaling; **HIGH** for this repo’s stated gap.

---

## Moderate Pitfalls

### 8. Stochastic combat variance hides learning signal

**What goes wrong:** High-variance hit rolls make returns noisy; PPO/DQN credit assignment across long turns gets worse.

**Prevention:** Configurable variance (deterministic eval mode); enough episodes in eval; optional variance reduction in curriculum (e.g. higher hit chance early).

**Primary phases:** **P2**; eval in **P1**.

---

### 9. Elimination changes entity sets mid-episode

**What goes wrong:** If tensor shapes assume fixed `n_models`, removing casualties breaks observation or masks. If padding ghosts exist, agents learn pad exploits.

**Prevention:** Masked padding with explicit **alive** flags; tests for step-after-elimination; keep **domain entity list** and **obs builder** in sync.

**Primary phases:** **P2** (wounds/elimination); **P6** (batching must respect masks).

---

### 10. Terrain added only as observation, not as reward curriculum

**What goes wrong:** Agent sees cover but reward still pushes open-field behaviour from pre-terrain policies.

**Prevention:** Short phases that reward survival under fire or cover usage (careful with shaping — tie back to pitfall #1); scripted shooting opponents to force pressure.

**Primary phases:** **P3**; reward design with **P2** hooks.

---

### 11. Melee + shooting + morale interaction overload (late game)

**What goes wrong:** Phase ordering in tabletop rules interacts (charge, fight, morale); skipping phases (`skip_phases`) while training causes **train/test mismatch** when full phases enable.

**Prevention:** Explicit configs for “training skeleton” vs “full rules”; regression tests when `skip_phases: []`.

**Primary phases:** **P5**; continuity with **P2–P3**.

---

### 12. Elo misinterpretation

**What goes wrong:** Elo across only self-play snapshots measures **relative** strength within a closed pool, not absolute skill.

**Prevention:** Anchor to **fixed scripts** and human-readable baselines; report both.

**Primary phases:** **P4**.

---

## Minor Pitfalls

### 13. Duplicate reward sources (legacy env fields vs phases)

**What goes wrong:** `CONCERNS.md` notes legacy `terminal_*_bonus` on env config vs per-phase fields — same class of issue can recur for combat terminal bonuses.

**Prevention:** Single source of truth in phase config; migration + tests.

**Primary phases:** **P2** (when adding terminal combat bonuses).

---

### 14. Observation fields that don’t change (pre-combat)

**What goes wrong:** Wounds in obs while non-functional trains the network on **noise**; sudden activation in P2 changes feature semantics.

**Prevention:** Feature flags or versioned obs; or zero/consistent semantics until mechanics live.

**Primary phases:** **P2** (per CONCERNS.md).

---

## Phase-Specific Warning Matrix

| Phase | Likely pitfall focus | Mitigation summary |
|-------|---------------------|-------------------|
| **P1** | Insufficient metrics; transformer without PE | Dashboards for combat/VP/group; add PE before large-scale sweeps |
| **P2** | Reward hijack; action explosion; LOS bugs; elimination/masks | Factored actions + tests; shaped + sparse mix; golden LOS; alive masks |
| **P3** | Cover/LOS interaction; exploit bugs; obs without curriculum | Test matrices; renderer/rules single source; short terrain-aware curricula |
| **P4** | Self-play cycling; two-agent info leaks; Elo hubris | Opponent pool + scripts; per-agent obs contract; fixed baselines |
| **P5** | Rule combinatorics; phase-skip mismatch | Full-phase integration tests; gradual enablement |
| **P6** | Throughput; context length; generalization across squad sizes | Profile env + network; curriculum scale-up; optional local obs windows |

---

## Sources & Confidence

| Topic | Confidence | Notes |
|-------|------------|--------|
| Reward phases / curriculum ordering | **HIGH** | [docs/reward-phases.md](../../docs/reward-phases.md), this repo |
| Wounds/obs, transformer PE, scaling | **HIGH** | [.planning/codebase/CONCERNS.md](../codebase/CONCERNS.md) |
| Self-play cycling / exploitability | **MEDIUM** | RL self-play literature (surveys, competitive games); verify per implementation |
| LOS/terrain exploit sensitivity | **MEDIUM** | Simulation + RL robustness consensus |
| Action factoring / legal masks | **MEDIUM** | Discrete multi-action RL practice |

---

## “What Might We Have Missed?”

- **Human-style rule exceptions** (auras, stratagems) can explode edge cases — keep registry-driven rules testable.
- **Credit assignment across battle rounds** when episodes get long may push toward **option-critic** or **hierarchical** approaches — out of current roadmap but flagged if P5 makes episodes very long.
- **Determinism vs seeding** for reproducibility when adding stochastic combat — CI should fix seeds.

---

*Pitfalls research for subsequent milestone (combat, terrain, self-play, scale). Downstream: use this when ordering roadmap phases and defining acceptance tests per phase.*
