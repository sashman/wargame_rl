"""The default dice: the phase facade's three seeded streams, reproduced exactly.

`WargameEnv` draws combat dice from `default_rng(combat_seed)`, the advance D6
from `default_rng(combat_seed + 1_000_003)` and the charge 2D6 from
`default_rng(combat_seed + 2_000_003)`, and it rolls a whole side's advance and
charge dice at the start of that side's turn, one draw per unit in the order
units first appear in the side's model list -- dead models' units included.

This source reproduces all of that so a script played through the per-model
facade meets the same dice the phase facade dealt it, which is what the bridge
test asserts. It is a *reproduction policy*, not a rule, which is why it lives
here and not in `domain/`: the port it implements says nothing about when a
side's block is drawn, only that a unit's roll can be asked for by name. The
block is drawn the first time any unit of a `(round, side)` asks, so asking per
unit at declaration and asking for the whole side at turn start read the same
stream.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from wargame_rl.wargame.envs.domain.kernel.dice import DiceCall
from wargame_rl.wargame.envs.types.game_timing import PlayerSide

ADVANCE_STREAM_OFFSET = 1_000_003
CHARGE_STREAM_OFFSET = 2_000_003


class SeededDice:
    """A `DiceSource` over the phase facade's three generators."""

    def __init__(
        self,
        combat_seed: int,
        *,
        units_by_side: Callable[[PlayerSide], list[int]],
        has_advance: bool,
        has_melee: bool,
    ) -> None:
        self._combat = np.random.default_rng(combat_seed)
        self._advance = np.random.default_rng(combat_seed + ADVANCE_STREAM_OFFSET)
        self._charge = np.random.default_rng(combat_seed + CHARGE_STREAM_OFFSET)
        self._units_by_side = units_by_side
        self._has_advance = has_advance
        self._has_melee = has_melee
        self._advance_rolls: dict[tuple[int, PlayerSide], dict[int, float]] = {}
        self._charge_rolls: dict[tuple[int, PlayerSide], dict[int, float]] = {}

    def draw(
        self,
        call: DiceCall,
        low: int,
        high: int | None,
        size: int | tuple[int, ...] | None,
    ) -> np.ndarray:
        """Combat dice, whatever the purpose: one stream, as the phase facade has."""
        drawn: np.ndarray = self._combat.integers(low, high, size=size)
        return drawn

    def advance_roll(self, side: PlayerSide, battle_round: int, unit: int) -> float:
        """This unit's D6, from the side's block drawn on the side's first ask."""
        if not self._has_advance:
            return 0.0
        key = (battle_round, side)
        block = self._advance_rolls.get(key)
        if block is None:
            block = {
                group: float(self._advance.integers(1, 7))
                for group in self._units_by_side(side)
            }
            self._advance_rolls[key] = block
        return block.get(unit, 0.0)

    def charge_roll(self, side: PlayerSide, battle_round: int, unit: int) -> float:
        """This unit's 2D6, from the side's block drawn on the side's first ask."""
        if not self._has_melee:
            return 0.0
        key = (battle_round, side)
        block = self._charge_rolls.get(key)
        if block is None:
            block = {
                group: float(self._charge.integers(1, 7) + self._charge.integers(1, 7))
                for group in self._units_by_side(side)
            }
            self._charge_rolls[key] = block
        return block.get(unit, 0.0)

    @property
    def state(self) -> dict[str, Any]:
        """The three bit-generator states, for asserting stream position."""
        return {
            "combat": dict(self._combat.bit_generator.state),
            "advance": dict(self._advance.bit_generator.state),
            "charge": dict(self._charge.bit_generator.state),
        }
