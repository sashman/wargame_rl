"""Which model of the target unit takes the next attack: the defender's choice."""

from __future__ import annotations

from wargame_rl.wargame.envs.domain.kernel.entities import WargameModel


def allocate_target(members: list[WargameModel]) -> WargameModel | None:
    """Pick which model of the target unit takes the next attack.

    The defender's choice, per the attack sequence: *"Select model. Pick a model
    in the current allocation group -- a model that has already lost Wounds if
    one is available."* Preferring the wounded model concentrates damage rather
    than spreading it, which is what stops a unit fielding a line of
    one-wound-remaining survivors.

    Allocation groups themselves are not modelled: they split a unit by
    CHARACTER and by distinct (W, Sv, InSv), and this project has one profile
    per army and no characters, so every unit is a single group. Adding the
    machinery would be a type with no second case.

    Returns None when the unit is destroyed, which is the only condition under
    which the rules discard an attack.
    """
    wounded = [m for m in members if m.is_alive and m.has_lost_wounds]
    if wounded:
        return wounded[0]
    alive = [m for m in members if m.is_alive]
    return alive[0] if alive else None
