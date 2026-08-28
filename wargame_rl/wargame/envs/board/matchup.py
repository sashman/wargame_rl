"""Which of our units beat which of theirs, before a model has moved.

The per-model expected-damage matrix already exists and already ships
(`domain/shooting.py::expected_damage_matrix`, hstacked onto every player model
token). What it cannot answer is the question a player asks while reading the
opponent's list: *which of my units wants to meet which of theirs*. That is a
unit-level quantity, and it is a reduction of the per-model one rather than a
second calculation -- computed any other way the two could disagree, and the
network's input would stop matching the table a human reads.

⚠ **This is a PRE-GAME tool and it has no positions.** Range never enters the
damage scalar. A matchup number that pretended to know range-to-target would be
answering a question about *ground* -- which is a different tool, wanting a
board and a turn -- and answering it from a stat line, which cannot. So reach
appears only as its own columns: `reach_margin` and `free_rounds`, plus an
exchange ratio quoted at two distances instead of blended into one.

⚠ **On the config that trains, this table is 1x1.** Both armies in
`configs/golden/25v25_maps_two_mode.yaml` are one profile. It says something
only where the profiles differ -- `30v15_fast_horde_vs_elite`, the mixed-role
arms -- and reading a 1x1 table as a finding is the way to oversell it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wargame_rl.wargame.envs.domain.shooting import (
    DefenderStats,
    expected_damage_matrix,
)
from wargame_rl.wargame.envs.types.config.entities import ModelConfig


@dataclass(frozen=True, slots=True)
class UnitProfile:
    """One unit's stat line, as a pre-game read.

    A unit here is a `group_id`, which is this project's name for the rules'
    unit. Every member is assumed to share the profile of the first member --
    see `unit_profiles`, which refuses a unit where that is false rather than
    averaging it away.
    """

    group_id: int
    n_models: int
    move: float
    weapon_range: float
    attacks: int
    ballistic_skill: int
    strength: int
    ap: int
    damage: int
    toughness: int
    save: int
    max_wounds: int

    @property
    def defender(self) -> DefenderStats:
        """This unit viewed as a target."""
        return DefenderStats(toughness=self.toughness, save=self.save)

    @property
    def attacker_row(self) -> tuple[int, int, int, int, int]:
        """This unit's model viewed as `expected_damage_matrix`'s attacker row."""
        return (
            self.attacks,
            self.ballistic_skill,
            self.strength,
            self.ap,
            self.damage,
        )

    @property
    def wounds_pool(self) -> int:
        """Every wound the unit has to lose."""
        return self.n_models * self.max_wounds


@dataclass(frozen=True, slots=True)
class Matchup:
    """What one round of one unit's fire does to another, in the open."""

    attacker: UnitProfile
    defender: UnitProfile
    wounds_per_round: float
    casualties_per_round: float
    overkill_share: float
    rounds_to_halve: float
    reach_margin: float
    free_rounds: float


def unit_profiles(
    model_configs: list[ModelConfig] | None,
    n_models: int,
    default_move: float,
) -> tuple[UnitProfile, ...]:
    """One `UnitProfile` per `group_id`, in ascending group order.

    `default_move` is the scenario's `max_move_speed`, used for any model whose
    `ModelConfig.move` is None -- which is every model on most configs.

    ⚠ **The FIRST weapon is the unit's weapon**, matching
    `observation_builder`'s `cfg.weapons[0]`. `max_weapon_ranges` instead takes
    the longest range over all weapons, and the two disagree the day a config
    carries a second gun; nothing does today. The observation is the convention
    to match, because this table exists to explain what the network was shown.
    A model with no weapon contributes a zero attacker row, which
    `expected_damage_matrix` already reads as "no such shooter".
    """
    if not model_configs:
        return ()
    if len(model_configs) != n_models:
        raise ValueError(f"{len(model_configs)} model configs for {n_models} models")
    by_group: dict[int, list[ModelConfig]] = {}
    for config in model_configs:
        by_group.setdefault(config.group_id, []).append(config)

    profiles: list[UnitProfile] = []
    for group_id in sorted(by_group):
        members = by_group[group_id]
        leader = members[0]
        weapon = leader.weapons[0] if leader.weapons else None
        _reject_mixed_unit(group_id, members)
        profiles.append(
            UnitProfile(
                group_id=group_id,
                n_models=len(members),
                move=float(leader.move if leader.move is not None else default_move),
                weapon_range=float(weapon.range) if weapon else 0.0,
                attacks=int(weapon.attacks) if weapon else 0,
                ballistic_skill=int(weapon.ballistic_skill) if weapon else 6,
                strength=int(weapon.strength) if weapon else 0,
                ap=int(weapon.ap) if weapon else 0,
                damage=int(weapon.damage) if weapon else 1,
                toughness=int(leader.toughness),
                save=int(leader.save),
                max_wounds=int(leader.max_wounds),
            )
        )
    return tuple(profiles)


def _reject_mixed_unit(group_id: int, members: list[ModelConfig]) -> None:
    """Refuse a unit whose members do not share a profile.

    Averaging a mixed unit would produce a stat line no model has, and every
    number downstream would then describe a fiction. The rules give a unit one
    profile; a config that does otherwise is a scenario this tool cannot read,
    and saying so is better than quietly reporting the mean.
    """
    leader = members[0]
    for member in members[1:]:
        same = (
            member.toughness == leader.toughness
            and member.save == leader.save
            and member.max_wounds == leader.max_wounds
            and [w.model_dump() for w in member.weapons]
            == [w.model_dump() for w in leader.weapons]
            and member.move == leader.move
        )
        if not same:
            raise ValueError(
                f"unit {group_id} mixes model profiles; "
                "board.matchup reads one profile per unit"
            )


def _casualty_scale(damage: int, max_wounds: int) -> float:
    """Models removed per point of expected damage.

    `expected_damage` returns **damage points and does not clip them** to what a
    model has left, and its own docstring says so. Two separate things then
    stand between that number and a body on the floor, and only one of them is
    the clip everyone reaches for first:

    * **Excess damage is lost.** Damage does not spill between models, so a
      Damage-3 hit on a Wounds-1 model removes one model and wastes two points.
    * **A model may take more than one hit.** A Wounds-3 model absorbs three
      Damage-1 hits before it goes down.

    Both are the same arithmetic seen from either end: a model costs
    `ceil(max_wounds / damage)` hits and each hit costs `damage` points, so a
    model costs `damage * ceil(max_wounds / damage)` points however the two
    numbers are arranged. Exact rather than approximate, because expected damage
    is *linear* in the weapon's Damage.

    ⚠ Every shipped config is `damage: 1 / max_wounds: 1`, where this is
    identically 1.0, and it agrees with the old `min(damage, max_wounds) /
    damage` at **every** `max_wounds: 1` profile -- so no measured number moves.
    It diverges only against a multi-wound target, which nothing has yet: that
    form reported 0.37 *wounds* as 0.37 *casualties* against a Wounds-3 model,
    a threefold overstatement, and its own regression test asserted that one
    Damage-3 hit removes two Wounds-2 models.
    """
    if damage <= 0 or max_wounds <= 0:
        return 0.0
    hits_per_model = -(-max_wounds // damage)  # ceil, in integers
    return 1.0 / float(damage * hits_per_model)


def matchup_matrix(
    attackers: tuple[UnitProfile, ...],
    defenders: tuple[UnitProfile, ...],
) -> np.ndarray:
    """`(n_attackers, n_defenders)` expected casualties per round, in the open.

    **Sum on the attacker axis, mean on the defender axis.** Every model in the
    firing unit shoots, so their expectations add; one representative model of
    the target unit takes the maths, so the defender axis is a profile lookup
    and not a sum. Getting that backwards reads `n * m` times the truth on a
    unit-versus-unit cell and is the easiest mistake here.

    Because a unit is one profile, the sum is `n_models x` the single-model
    entry from `expected_damage_matrix` -- which is how this stays a reduction
    of the shipped matrix rather than a second implementation of it.

    Casualties are capped at the defending unit's model count: fire that would
    remove more models than exist has removed the unit and no more. Cover is not
    applied, matching `expected_damage_matrix`, so every entry is the
    expectation against a target in the open.
    """
    caps = np.array([d.n_models for d in defenders], dtype=np.float32)
    _wounds, casualties = _wounds_and_casualties(attackers, defenders)
    clipped: np.ndarray = np.minimum(casualties, caps[np.newaxis, :])
    return clipped.astype(np.float32)


def _free_rounds(attacker: UnitProfile, defender: UnitProfile) -> float:
    """Rounds of unanswered fire while the shorter-ranged unit closes.

    Kinematic and position-free: the gap the defender must close is the reach
    difference, and both units contribute their Move to closing it. On
    `30v15_fast_horde_vs_elite` the elite's 24" against the horde's 12" at Move
    6 and Move 12 gives `(24 - 12) / (12 + 6) = 0.67` -- two thirds of one
    round, which is the correct and usefully small answer, and exactly the kind
    of thing a single blended scalar would bury.
    """
    margin = attacker.weapon_range - defender.weapon_range
    closing = attacker.move + defender.move
    if margin <= 0.0 or closing <= 0.0:
        return 0.0
    return float(margin / closing)


def matchup_table(
    attackers: tuple[UnitProfile, ...],
    defenders: tuple[UnitProfile, ...],
) -> tuple[tuple[Matchup, ...], ...]:
    """Every attacker against every defender, one `Matchup` per cell."""
    wounds, uncapped = _wounds_and_casualties(attackers, defenders)
    casualties = matchup_matrix(attackers, defenders)
    rows: list[tuple[Matchup, ...]] = []
    for i, attacker in enumerate(attackers):
        row: list[Matchup] = []
        for j, defender in enumerate(defenders):
            taken = float(casualties[i, j])
            raw = float(uncapped[i, j])
            row.append(
                Matchup(
                    attacker=attacker,
                    defender=defender,
                    wounds_per_round=float(wounds[i, j]),
                    casualties_per_round=taken,
                    overkill_share=(raw - taken) / raw if raw > 0.0 else 0.0,
                    rounds_to_halve=(
                        (defender.n_models / 2.0) / taken
                        if taken > 0.0
                        else float("inf")
                    ),
                    reach_margin=attacker.weapon_range - defender.weapon_range,
                    free_rounds=_free_rounds(attacker, defender),
                )
            )
        rows.append(tuple(row))
    return tuple(rows)


def _wounds_and_casualties(
    attackers: tuple[UnitProfile, ...], defenders: tuple[UnitProfile, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """`(wounds, casualties)` per round, neither capped at the unit's size.

    One call to `expected_damage_matrix` serves both, so the wound figure the
    table prints and the casualty figure it ranks on cannot come from different
    arithmetic.
    """
    shape = (len(attackers), len(defenders))
    if not attackers or not defenders:
        zero = np.zeros(shape, dtype=np.float32)
        return zero, zero.copy()
    per_model = expected_damage_matrix(
        np.array([a.attacker_row for a in attackers], dtype=np.int64),
        np.array([(d.toughness, d.save) for d in defenders], dtype=np.int64),
    )
    counts = np.array([a.n_models for a in attackers], dtype=np.float32)
    wounds = (per_model * counts[:, np.newaxis]).astype(np.float32)
    scales = np.array(
        [
            [_casualty_scale(a.damage, d.max_wounds) for d in defenders]
            for a in attackers
        ],
        dtype=np.float32,
    )
    return wounds, (wounds * scales).astype(np.float32)


def exchange_ratio(a: UnitProfile, b: UnitProfile) -> tuple[float, float]:
    """`(at_long_range, at_short_range)` -- a's casualties per b's, per round.

    Two numbers rather than one, because the answer changes with the distance
    and a blended scalar hides which side chose it. At long range only the
    longer-ranged unit fires, so the ratio is infinite for whoever owns the
    reach; at short range both fire and the ratio is the honest trade.

    `inf` means "b cannot answer at all", which is a real state of this game and
    not a division accident.
    """
    a_hits = float(matchup_matrix((a,), (b,))[0, 0])
    b_hits = float(matchup_matrix((b,), (a,))[0, 0])
    short = a_hits / b_hits if b_hits > 0.0 else float("inf")
    if a.weapon_range > b.weapon_range:
        long = float("inf") if a_hits > 0.0 else 0.0
    elif b.weapon_range > a.weapon_range:
        long = 0.0
    else:
        long = short
    return long, short
