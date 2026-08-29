"""Training against a pool of the learner's own frozen selves.

The pool and the sampler live in `rating/` and are numpy-only. This module is
the part that needs torch and a live env: freezing a snapshot to disk, and
seating the sampled opponent on the rollout envs.

**Off is a no-op, and that is a property rather than an intention.** With
`enabled` false nothing here runs, no snapshot is written, and -- critically --
**no random number is drawn**. `augment_start` is the precedent: a training loop
that forgets to ask trains the control, which is bit-identical and therefore
obvious, where a version that drew even one unused number would shift every
layout and dice stream and make an arm incomparable to its own control.

The opponent stream is **separate** from the layout and dice streams for the
same reason. Turning self-play on must change who the learner plays, not which
boards it plays on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from wargame_rl.wargame.envs.opponent.registry import build_opponent_policy
from wargame_rl.wargame.envs.types import OpponentPolicyConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.rating.pool import Snapshot, SnapshotPool, spans

# Disjoint from every seed band this repo already uses -- rollout 0, in-run
# baselines 10k, in-run eval 500k, held-out 700k, cloning 800k, ratings 900k.
# Its own band is what keeps opponent choice from perturbing anything else.
SELF_PLAY_SEED_BASE = 1_100_000


class SelfPlayConfig(BaseModel):
    """When to snapshot, how many to keep, and who to play.

    Every field defaults to the value that makes self-play absent rather than
    merely quiet, so a config carrying this block trains the control.

    ⚠ It lives on the **training** config, not on `WargameEnvConfig`. Two
    reasons, either sufficient: `envs/` may not know about training, and the
    rating ledger fingerprints the env config -- so a self-play block there
    would split one scenario's ledger across every schedule ever tried, unless
    explicitly dropped, which is the failure the fingerprint's own design notes
    warn about.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    snapshot_every_n_epochs: int = Field(default=25, gt=0)
    # A snapshot is a full checkpoint on local disk, and `checkpoints/` is the
    # only copy of any weights here -- capacity is a disk budget before it is a
    # statistical choice.
    pool_capacity: int = Field(default=8, ge=1)
    # Entry zero, never evicted. A scripted baseline by default, so the pool has
    # a floor that cannot drift with the learner.
    anchor: str = "squad_march_take"
    sampling: Literal["hard", "even", "uniform"] = "hard"
    uniform_floor: float = Field(default=0.1, ge=0.0, le=1.0)
    # The opponent's decode. Left at 1 because joint constrained decoding costs
    # `K^k` forward-model evaluations per unit, and the opponent pays it on
    # every step of every rollout.
    decode_topk: int = Field(default=1, ge=1)


class OpponentScheduler:
    """Owns the pool, its RNG stream, and seating an opponent on a live env.

    Constructed only when self-play is enabled. Nothing in this class is
    reachable otherwise, which is what makes "off is bit-identical" checkable by
    reading rather than by measuring.
    """

    def __init__(
        self,
        config: SelfPlayConfig,
        snapshot_dir: Path,
        seed: int = 0,
    ) -> None:
        if not config.enabled:
            raise ValueError(
                "OpponentScheduler is for enabled self-play only; when it is off "
                "nothing should construct one, so that no stream is drawn from"
            )
        self._config = config
        self._snapshot_dir = snapshot_dir
        # Its own generator, seeded off the run seed but drawn separately, so
        # turning self-play on changes who the learner plays and not which
        # boards it plays on.
        self._rng = np.random.default_rng(SELF_PLAY_SEED_BASE + seed)
        self._pool = SnapshotPool(
            Snapshot(name=config.anchor, checkpoint="", epoch=0),
            capacity=config.pool_capacity,
        )
        self._learner_rating = 0.0
        self._seat_advantage = 0.0

    @property
    def pool(self) -> SnapshotPool:
        """The frozen opponents, oldest first, anchor at index zero."""
        return self._pool

    def rate(
        self,
        ratings: dict[str, float],
        learner_rating: float,
        seat_advantage: float = 0.0,
    ) -> None:
        """Take fitted ratings from a table, so sampling can prioritise.

        `seat_advantage` is `h_seat` from the same fit. It belongs here because
        the learner always plays the **player** seat -- `learner_side` is a
        later phase -- so the probability that matters is the one for the game
        about to be played, not the seat-neutral comparison.
        """
        self._pool.rate(ratings)
        self._learner_rating = learner_rating
        self._seat_advantage = seat_advantage

    def should_snapshot(self, epoch: int) -> bool:
        """Whether this epoch is a snapshot epoch."""
        return epoch > 0 and epoch % self._config.snapshot_every_n_epochs == 0

    def snapshot(self, epoch: int, state_dict: Mapping[str, Any]) -> Snapshot:
        """Freeze the learner's current weights into the pool.

        ⚠ Written from a Lightning hook, so **`SIGKILL` writes nothing** -- the
        same hazard `last.ckpt` has, and SIGKILL is the prescribed way to stop
        these trainers. A pool is therefore routinely up to
        `snapshot_every_n_epochs` behind the run that produced it.
        """
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self._snapshot_dir / f"snapshot-epoch-{epoch:04d}.ckpt"
        torch.save({"state_dict": state_dict}, path)
        entry = Snapshot(name=f"epoch_{epoch}", checkpoint=str(path), epoch=epoch)
        self._pool.add(entry)
        earliest, latest = spans(self._pool.entries)
        logger.info(
            "self-play pool: {} entries spanning epochs {}-{}, newest {}",
            len(self._pool.entries),
            earliest,
            latest,
            entry.name,
        )
        return entry

    def seat(self, envs: Sequence[WargameEnv]) -> list[Snapshot]:
        """Draw one opponent per rollout env and install it.

        Per env rather than per run, so a single epoch's batch spans the pool
        instead of betting the whole epoch on one draw -- the same reason the
        rollout uses several envs at all.

        `set_opponent_policy` is installed rather than configured, because
        `reset()` never touches `_opponent_policy`: the seating survives every
        episode until it is replaced.
        """
        drawn: list[Snapshot] = []
        for env in envs:
            entry = self._pool.sample(
                self._learner_rating,
                self._rng,
                mode=self._config.sampling,
                uniform_floor=self._config.uniform_floor,
                advantage=self._seat_advantage,
            )
            env.set_opponent_policy(
                build_opponent_policy(self._opponent_config(entry), env)
            )
            drawn.append(entry)
        return drawn

    def _opponent_config(self, entry: Snapshot) -> OpponentPolicyConfig:
        """How a pool entry is seated -- the same two shapes the arena uses."""
        if entry.is_anchor and not entry.checkpoint:
            return OpponentPolicyConfig(
                type="scripted_baseline", params={"baseline": entry.name}
            )
        return OpponentPolicyConfig(
            type="model",
            params={
                "checkpoint": entry.checkpoint,
                "decode_topk": self._config.decode_topk,
            },
        )
