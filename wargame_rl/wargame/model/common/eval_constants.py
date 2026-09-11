"""The seed bands and the scripted bar every training loop measures against.

Plain constants, in a module that imports nothing, so a driver that never
builds a Lightning module (`train_per_model.py`) can read them without
loading Lightning or the phase facade. `lightning_base.py` re-exports them.

Seed bands across the repo: rollout 0 (the phase facade's fixed base) and
`seed x 100` (the per-model driver), baselines 10k, in-run eval 500k,
held-out 700k, cloning 800k, ratings 900k, self-play opponents 1.1M.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.evaluation.constants import EVAL_WAVE_SIZE

# The three baselines every run logs. The shooting one is the bar that matters
# against an opponent that shoots back; the middle rungs live in
# scripts/measure_baselines.py.
BASELINE_POLICIES = ("random", "squad_march", "squad_march_shoot")
BASELINE_EPISODES = 20
# Held out from the rollout bands so baselines never share training layouts.
BASELINE_SEED_BASE = 10_000
# Evaluation layouts, disjoint from both training and baseline seeds. Fixed
# across epochs on purpose: objective placement dominates episode variance, so
# resampling every epoch makes a curve mostly report which maps were drawn.
EVAL_SEED_BASE = 500_000

__all__ = [
    "BASELINE_EPISODES",
    "BASELINE_POLICIES",
    "BASELINE_SEED_BASE",
    "EVAL_SEED_BASE",
    "EVAL_WAVE_SIZE",
]
