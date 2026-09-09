"""The stat lines an attack sequence reads, as protocols the config satisfies structurally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class AttackStats(Protocol):
    """The stats an attack sequence needs once its SKILL is already decided.

    Everything after the hit roll is identical for a bow and a blade, so this is
    what a melee weapon and a ranged one share. The skill itself is not here: a
    ranged weapon carries `ballistic_skill` and is modified by cover, a melee
    weapon carries `melee_skill` and is not, so it is passed in.
    """

    @property
    def attacks(self) -> int: ...
    @property
    def strength(self) -> int: ...
    @property
    def ap(self) -> int: ...
    @property
    def damage(self) -> int: ...


@runtime_checkable
class WeaponStats(AttackStats, Protocol):
    """Structural protocol for weapon stats used in resolution.

    Satisfied by ``WeaponProfile`` (Pydantic, types layer) without importing it,
    keeping the domain layer dependency-free.
    """

    @property
    def ballistic_skill(self) -> int: ...


@runtime_checkable
class MeleeStats(Protocol):
    """A melee weapon's stat line, structurally — the domain imports no config."""

    @property
    def attacks(self) -> int: ...
    @property
    def melee_skill(self) -> int: ...
    @property
    def strength(self) -> int: ...
    @property
    def ap(self) -> int: ...
    @property
    def damage(self) -> int: ...


@dataclass(frozen=True, slots=True)
class DefenderStats:
    """Target defensive stats needed for wound roll and save."""

    toughness: int
    save: int


@dataclass(frozen=True, slots=True)
class ShootingResult:
    """Outcome of one model's shooting action against one target."""

    hits: int
    wounds: int
    unsaved: int
    damage_dealt: int
