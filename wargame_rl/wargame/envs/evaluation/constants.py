"""Evaluation constants both facades share, importable without torch."""

from __future__ import annotations

# Episodes evaluated in lockstep per wave, on both facades. Sixteen is what
# the whole-phase batched eval settled on; the per-model runner drops an env
# from the wave when its episode ends, so the size only bounds the batch.
EVAL_WAVE_SIZE = 16

__all__ = ["EVAL_WAVE_SIZE"]
