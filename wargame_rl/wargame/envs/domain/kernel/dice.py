"""Dice as a port: the domain asks for a roll and does not own the generator.

Every die in the game is one call shape -- `rng.integers(1, 7, size=n)` on a
numpy generator -- in five places: the hit, wound and save rolls of the shared
attack sequence (`shooting.py:resolve_attack`), the advance D6 and the charge
2D6. None of those callers knows *why* it is rolling, so nothing outside the
process can answer a roll: an external roller, a replay of recorded dice, a
fixed outcome for a test. This module is the port those answer through.

The existing domain functions keep their `rng: np.random.Generator` parameter
untouched -- the phase facade and every golden depend on that stream to the
bit. `RollerAdapter` is what a caller hands them instead: it has the one method
they call (`integers`) and forwards every draw to a `DiceSource` together with
the purpose the caller tagged it with.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from wargame_rl.wargame.envs.types.game_timing import PlayerSide


class DicePurpose(str, Enum):
    """What a die is for -- the one thing a raw generator cannot say."""

    shooting = "shooting"
    melee = "melee"
    advance = "advance"
    charge = "charge"


@dataclass(frozen=True, slots=True)
class DiceCall:
    """The context of one draw: purpose, and who is rolling for what.

    `unit` and `model` index the roller's own force; `side` is the clock seat
    whose turn it is. Every field but the purpose is optional because a source
    that answers by purpose alone (a seeded generator) never reads them.
    """

    purpose: DicePurpose
    side: PlayerSide | None = None
    battle_round: int | None = None
    unit: int | None = None
    model: int | None = None


class DiceSource(Protocol):
    """Where rolls come from.

    `draw` is the attack sequence's need -- `n` faces in `[low, high)`, the
    numpy shape -- and the two roll methods are the movement rules' need: one
    unit's advance D6 and charge 2D6, in inches, answered per unit so a source
    can be asked exactly when a unit declares.
    """

    def draw(
        self,
        call: DiceCall,
        low: int,
        high: int | None,
        size: int | tuple[int, ...] | None,
    ) -> np.ndarray:
        """Integer dice in `[low, high)`, `size` of them."""
        ...

    def advance_roll(self, side: PlayerSide, battle_round: int, unit: int) -> float:
        """This unit's advance D6 for the turn, in inches."""
        ...

    def charge_roll(self, side: PlayerSide, battle_round: int, unit: int) -> float:
        """This unit's charge 2D6 for the turn, in inches."""
        ...


class RollerAdapter:
    """A `DiceSource` wearing the generator's face.

    The domain's attack sequence takes `rng: np.random.Generator` and calls
    nothing on it but `integers(low, high, size=...)`. This object answers that
    one call by forwarding to the source with the purpose it was built with, so
    the shared sequence stays untouched and every roll it makes is attributed.
    """

    def __init__(self, source: DiceSource, call: DiceCall) -> None:
        self._source = source
        self._call = call

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """The generator's signature, forwarded with this roll's purpose."""
        return self._source.draw(self._call, low, high, size)
