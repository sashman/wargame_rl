# Phase 5: Shooting Resolution - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 05-shooting-resolution
**Areas discussed:** Resolution stochasticity, Model defensive stats, Observation features (OBS-02), Advance/engagement restrictions

---

## Resolution Stochasticity

| Option | Description | Selected |
|--------|-------------|----------|
| Stochastic with seeded RNG | Real D6 rolls per tabletop rules, dedicated seeded generator for reproducibility | ✓ |
| Pure expected value | Deterministic fractional damage, no rolls | |
| Stochastic with configurable toggle | Default to dice, config flag for expected-value mode | |

**User's choice:** Stochastic with seeded RNG
**Notes:** Matches the game, PPO handles variance, testable with fixed seeds.

---

## Model Defensive Stats

| Option | Description | Selected |
|--------|-------------|----------|
| T and Sv on ModelConfig, no invulnerable save | Add toughness and save with infantry defaults, skip invulnerable saves | ✓ |
| T, Sv, and invulnerable save on ModelConfig | Add all three now for full save step from the start | |

**User's choice:** T and Sv only, no invulnerable save
**Notes:** Keeps Phase 5 focused on core pipeline. Invulnerable saves come with v4.0 weapon keywords.

### Follow-up: Default Stat Calibration

User requested defaults calibrated so predicted chance to successfully wound is ~50%.

| Stat | Default | Probability Step |
|------|---------|-----------------|
| BS (weapon) | 3+ | 4/6 = 66.7% hit |
| S (weapon) | 4 | |
| T (model) | 3 | S > T → 3+ = 4/6 = 66.7% wound |
| Sv (model) | 4+ | |
| AP (weapon) | 1 | Sv 4+ with AP -1 → need 5+ → 4/6 = 66.7% fail save |
| Attacks (weapon) | 2 | Per attack: 29.6% |
| **Per action** | | **P(≥1 wound) = 50.4%** |

**User's choice:** Accepted this stat block. Noted it makes the expected damage observation feature cleanly default to ~50%.

---

## Observation Features (OBS-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Raw defensive stats only | T, Sv per opponent; weapon stats per player model; transformer learns interactions | |
| Raw stats + precomputed expected damage | All raw stats plus expected damage per (attacker, target) pair | ✓ |
| Precomputed expected damage only | Single scalar per pair, skip raw stats | |

**User's choice:** Raw stats plus precomputed expected damage
**Notes:** Gives the transformer both raw mechanics and derived efficiency signal. Mirrors what a human player calculates mentally.

---

## Advance/Engagement Restrictions

| Option | Description | Selected |
|--------|-------------|----------|
| Add tracking infrastructure defaulting to always-eligible | Per-model flags that never fire until v3.0 adds movement types | ✓ |
| Defer entirely until v3.0 | No infrastructure, document as pending | |
| Stub with config-level flags | Config toggles for testing without real mechanics | |

**User's choice:** Add tracking infrastructure that defaults to always-eligible
**Notes:** Formally satisfies SHOT-04/SHOT-05 success criteria. Minimal cost, v3.0 just flips the flags.

---

## Claude's Discretion

- Resolution code location (domain vs env_components)
- Normalization scheme for weapon/defense observation features
- Expected damage caching strategy
- D6 roll implementation structure (vectorized vs loop)
- Engagement range stub threshold value
- `advanced_this_turn` reset wiring

## Deferred Ideas

- Multi-weapon sub-steps (from Phase 4, still deferred)
- Invulnerable saves (v4.0)
- Full probability matrices for explainability (v9.0)
- Pointer-network attention (from Phase 4, still deferred)
- Cover saves (v2.0)
