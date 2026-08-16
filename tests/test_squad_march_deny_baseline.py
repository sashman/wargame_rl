"""The `squad_march_deny` baseline: bank the VP cap, then spend the surplus.

VP is ``min(cap_per_turn, controlled * vp_per_objective)`` -- ``min(15, held *
5)`` on the shipped mission -- so the cap binds at **three** objectives while
the real tables carry five or six. A fourth objective you control is worth zero
additional VP, and above the cap the only remaining gradient in ``vp_margin`` is
denial: an objective taken off the opponent removes 5 from *their* score every
round.

`squad_march_shoot` allocates by a fixed ``k % n`` it never revises, so its
denial is incidental. This policy commits exactly ``cap // vp_per_objective``
squads to holding and sends the rest at whatever the opponent controls.

Measured on the nine held-out tables it holds **2.99** objectives against the
bar's 4.00 and still scores ahead, because opponent VP falls. That is the whole
claim, and these tests pin the allocation behaviour it rests on -- on hand-built
geometry, so a regression names its own cause rather than moving an aggregate.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.baseline.scripted_squad_march import (
    ScriptedSquadMarchPolicy,
)
from wargame_rl.wargame.envs.baseline.scripted_squad_march_deny import (
    ScriptedSquadMarchDenyPolicy,
)
from wargame_rl.wargame.envs.baseline.scripted_squad_march_take import (
    ScriptedSquadMarchTakePolicy,
)
from wargame_rl.wargame.envs.types import (
    MissionConfig,
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import WeaponProfile
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

WEAPONS = [WeaponProfile(range=12, attacks=1)]


def _make_env(cap_per_turn: int = 10) -> WargameEnv:
    """Three squads, four objectives, opponents massed on the far two.

    Objectives 0 and 1 are empty and near the player; 2 and 3 are held by four
    opponents each. With `cap_per_turn` 10 and 5 VP an objective, exactly two
    squads are needed to saturate own VP, which leaves one to deny with -- the
    surplus this policy exists to spend.
    """
    config = WargameEnvConfig(
        render_mode=None,
        board_width=60,
        board_height=40,
        number_of_wargame_models=6,
        number_of_opponent_models=8,
        number_of_objectives=4,
        objective_radius_size=3,
        number_of_battle_rounds=6,
        max_groups=3,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        models=[
            ModelConfig(x=6, y=10 + 8 * (i // 2), group_id=i // 2, weapons=WEAPONS)
            for i in range(6)
        ],
        # Four opponents on objective 2, four on objective 3.
        opponent_models=[
            ModelConfig(
                x=45 if i < 4 else 52, y=20 + (i % 2), group_id=i // 4, weapons=WEAPONS
            )
            for i in range(8)
        ],
        objectives=[
            ObjectiveConfig(x=12, y=10),
            ObjectiveConfig(x=12, y=30),
            ObjectiveConfig(x=45, y=20),
            ObjectiveConfig(x=52, y=20),
        ],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
        mission=MissionConfig(
            type="default", params={"cap_per_turn": cap_per_turn, "vp_per_objective": 5}
        ),
    )
    return WargameEnv(config=config)


def _targets(policy: ScriptedSquadMarchPolicy, env: WargameEnv) -> list[int]:
    """Objective index assigned to each squad, in group order."""
    models = env.player_models
    group_ids = sorted({model.group_id for model in models})
    chosen = policy.squad_objectives(models, env, group_ids)
    return [env.objectives.index(objective) for objective in chosen]


def test_squad_march_assignment_is_unchanged_by_the_seam() -> None:
    """`squad_march` must still be squad *k* -> objective *k mod n*.

    The overridable `squad_objectives` seam was extracted from this class so the
    denial policy could reuse its coherency-preserving movement. It is the bar
    every result in `reports/` is quoted against, so the refactor has to leave it
    bit-identical -- this is the regression guard on that.
    """
    env = _make_env()
    env.reset(seed=0)

    assert _targets(ScriptedSquadMarchPolicy(), env) == [0, 1, 2]


def test_surplus_squads_are_sent_at_what_the_opponent_holds() -> None:
    """Two squads bank the cap; the third goes for an opponent-held objective."""
    env = _make_env(cap_per_turn=10)
    env.reset(seed=0)

    targets = _targets(ScriptedSquadMarchDenyPolicy(), env)

    # Objectives 0 and 1 are uncontested, so they are the cheapest to hold and
    # take the first two squads. The remaining squad must be on 2 or 3 -- the
    # only objectives the opponent controls.
    assert {0, 1}.issubset(set(targets)), f"cap not banked: {targets}"
    assert any(index in (2, 3) for index in targets), f"nothing denied: {targets}"


def test_how_many_squads_hold_follows_the_mission_cap() -> None:
    """`needed` is read off the mission, not hardcoded.

    At a cap of 5 a single objective saturates own VP, so two of the three
    squads become surplus and both should be denying.
    """
    env = _make_env(cap_per_turn=5)
    env.reset(seed=0)

    targets = _targets(ScriptedSquadMarchDenyPolicy(), env)

    denying = [index for index in targets if index in (2, 3)]
    assert len(denying) == 2, f"expected two denying squads, got {targets}"


def test_every_squad_gets_an_objective() -> None:
    """No squad may be left unassigned, whatever the counts are."""
    for cap in (5, 10, 15, 30):
        env = _make_env(cap_per_turn=cap)
        env.reset(seed=0)
        targets = _targets(ScriptedSquadMarchDenyPolicy(), env)
        assert len(targets) == 3, f"cap {cap}: {targets}"
        assert all(0 <= index < 4 for index in targets), f"cap {cap}: {targets}"


def test_take_sends_the_surplus_at_the_weakest_ground_not_the_strongest() -> None:
    """The one thing that separates `take` from `deny`, and it is worth ~5 vp.

    `deny` sends its surplus at what the opponent *controls*, and its `held`
    sits at exactly 3.00 -- the raids never flip anything, so they deny nothing.
    `take` sends the surplus at the *weakest*-held ground, which flips and then
    denies for the rest of the game: 116.7 v 112.3 vp_margin on the held-out
    tables, with `held` 4.02 v 3.00.

    Objectives 0 and 1 are empty, 2 and 3 hold four opponents each. With a cap of
    5 one squad banks the cap and two are surplus -- and with only two free
    objectives for three squads, one squad must end up on defended ground even
    under `take`. What separates the policies is *how many*: `take` fills the
    free ground first and commits no more than it has to, while `deny` goes
    looking for the defended ground on purpose.
    """
    env = _make_env(cap_per_turn=5)
    env.reset(seed=0)

    take = _targets(ScriptedSquadMarchTakePolicy(), env)
    deny = _targets(ScriptedSquadMarchDenyPolicy(), env)

    defended = (2, 3)
    assert {0, 1}.issubset(set(take)), f"take left free ground unclaimed: {take}"
    take_on_defended = [index for index in take if index in defended]
    deny_on_defended = [index for index in deny if index in defended]
    assert len(take_on_defended) < len(deny_on_defended), (
        f"take should commit less to defended ground: take={take} deny={deny}"
    )
