"""A unit that fought shuffles up to 3" afterwards — in Objective mode only.

`docs/rules/12-fight-phase.md` § Consolidate step. The three modes are assessed
in order and the first match is **compulsory**, so most of these tests are about
what does NOT happen: a unit still engaged is in Ongoing mode and a unit with an
enemy within 3" is in Engaging mode, and neither may consolidate onto an
objective instead. Both are `DEFERRED`, so for those units the step is a no-op.

⚠ Contact is set BY HAND, as in `tests/test_fight_phase.py`. Every mover on both
seats is walked back out of engagement, so no sequence of legal non-charge moves
reaches the state these tests are about.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.domain.consolidate import consolidate_objective
from wargame_rl.wargame.envs.domain.entities import WargameObjective
from wargame_rl.wargame.envs.domain.shooting import (
    DefenderStats,
    expected_attack_damage,
)
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.state.events import (
    _apply_model_delta,
    _compute_model_delta,
)
from wargame_rl.wargame.envs.state.snapshot import GameStateSnapshot
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.envs.wargame_model import WargameModel
from wargame_rl.wargame.model.common.factory import create_environment

LETHAL = MeleeWeaponProfile(attacks=12, melee_skill=2, strength=10, ap=6, damage=4)


def _model(x: float, y: float = 10.0, group: int = 0) -> WargameModel:
    return WargameModel(
        location=np.array([x, y]),
        stats={"toughness": 3, "save": 4, "max_wounds": 1, "current_wounds": 1},
        distances_to_objectives=np.zeros(1),
        group_id=group,
        base_radius=0.0,
    )


def _consolidate(
    models: list[WargameModel],
    enemies: list[WargameModel],
    objectives: list[WargameObjective],
) -> list[int]:
    return consolidate_objective(
        models,
        enemies,
        objectives,
        eligible_units={0},
        objective_offsets=lambda: (
            compute_distances(models, objectives).model_obj_norms_offset
        ),
        max_distance=3.0,
        engagement_range=1.0,
        base_radius=0.0,
        board=(60.0, 44.0),
        coherency_nearest=2.0,
        coherency_furthest=9.0,
    )


def _objective(x: float, radius: float = 1.0) -> WargameObjective:
    return WargameObjective(location=np.array([x, 10.0]), radius_size=radius)


def test_a_unit_that_fought_and_is_now_clear_walks_onto_the_objective() -> None:
    models = [_model(10.0), _model(10.5)]
    enemies = [_model(40.0, group=1)]
    objectives = [_objective(12.5)]

    assert _consolidate(models, enemies, objectives) == [0]

    offsets = compute_distances(models, objectives).model_obj_norms_offset
    assert (offsets[:, 0] <= 1.0).any(), "nobody ended in range of the objective"


def test_ongoing_mode_pre_empts_objective_mode() -> None:
    """Still engaged, so the mode is Ongoing — and Ongoing is deferred."""
    models = [_model(10.0), _model(10.5)]
    enemies = [_model(11.2, group=1)]  # inside the 1" engagement range
    objectives = [_objective(12.5)]
    before = [m.location.copy() for m in models]

    assert _consolidate(models, enemies, objectives) == []
    assert all(np.array_equal(b, m.location) for b, m in zip(before, models))


def test_engaging_mode_pre_empts_objective_mode() -> None:
    """Unengaged but within 3" of an enemy, so the mode is Engaging."""
    models = [_model(10.0), _model(10.5)]
    enemies = [_model(12.5, group=1)]  # 2.0" from the nearest model
    objectives = [_objective(12.0)]
    before = [m.location.copy() for m in models]

    assert _consolidate(models, enemies, objectives) == []
    assert all(np.array_equal(b, m.location) for b, m in zip(before, models))


def test_an_objective_further_than_three_inches_cannot_be_consolidated_onto() -> None:
    models = [_model(10.0), _model(10.5)]
    enemies = [_model(40.0, group=1)]
    objectives = [_objective(20.0)]
    before = [m.location.copy() for m in models]

    assert _consolidate(models, enemies, objectives) == []
    assert all(np.array_equal(b, m.location) for b, m in zip(before, models))


def test_a_unit_that_did_not_fight_does_not_consolidate() -> None:
    """Eligibility is "was eligible to fight this phase", not "is near a point"."""
    models = [_model(10.0), _model(10.5)]
    enemies = [_model(40.0, group=1)]
    objectives = [_objective(12.5)]
    before = [m.location.copy() for m in models]

    moved = consolidate_objective(
        models,
        enemies,
        objectives,
        eligible_units=set(),
        objective_offsets=lambda: (
            compute_distances(models, objectives).model_obj_norms_offset
        ),
        max_distance=3.0,
        engagement_range=1.0,
        base_radius=0.0,
        board=(60.0, 44.0),
        coherency_nearest=2.0,
        coherency_furthest=9.0,
    )
    assert moved == []
    assert all(np.array_equal(b, m.location) for b, m in zip(before, models))


def test_a_unit_that_came_out_of_the_fight_strung_out_cannot_consolidate() -> None:
    """All-or-nothing at the UNIT, per `03-moving.md`.

    The trailing model is 6" behind — past the 2" chain — and a 3" walk does not
    close that, so the end state is still incoherent and the whole move is
    reverted rather than the offending model.

    ⚠ This, not a move that *breaks* the chain, is what the coherency guard
    catches. Every model walks toward the same point, so an Objective-mode
    consolidation is **contractive**: it can only bring a unit closer together.
    The guard is still the rule, and still worth having — melee casualties are
    what leave a unit strung out in the first place.
    """
    models = [_model(10.0), _model(4.0)]
    enemies = [_model(40.0, group=1)]
    objectives = [_objective(12.5)]
    before = [m.location.copy() for m in models]

    assert _consolidate(models, enemies, objectives) == []
    assert all(np.array_equal(b, m.location) for b, m in zip(before, models))


def test_a_model_already_in_range_does_not_shuffle() -> None:
    """The rules only bind a model that *moves*, and moving buys it nothing."""
    models = [_model(12.0), _model(10.6)]
    enemies = [_model(40.0, group=1)]
    objectives = [_objective(12.5)]
    arrived = models[0].location.copy()

    assert _consolidate(models, enemies, objectives) == [0]
    assert np.array_equal(arrived, models[0].location)


def _env() -> WargameEnv:
    """One player model that kills in melee, against one that cannot fight back."""
    config = WargameEnvConfig(
        number_of_wargame_models=1,
        number_of_opponent_models=1,
        number_of_objectives=1,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[ModelConfig(melee_weapons=[LETHAL])],
        opponent_models=[ModelConfig(melee_weapons=[])],
        melee=MeleeConfig(enabled=True),
        skip_phases=[BattlePhase.shooting],
    )
    env = create_environment(config)
    env.reset(seed=7)
    return env


def test_consolidation_happens_through_env_step() -> None:
    """The whole path: locked in melee, kill the target, walk onto the point."""
    env = _env()
    player = env.wargame_models[0]
    opponent = env.opponent_models[0]
    player.location = np.array([10.0, 10.0], dtype=player.location.dtype)
    opponent.location = np.array([11.0, 10.0], dtype=opponent.location.dtype)
    objective = env.objectives[0]
    objective.location = np.array([7.5, 10.0], dtype=objective.location.dtype)
    radius = objective.radius_size

    # ⚠ Eight, not three. The command phase is stepped now -- a charge is
    # declared there -- so a round costs more agent steps and a budget sized to
    # the old phase layout never reaches the fight boundary.
    for _ in range(8):
        env.step(WargameEnvAction(actions=[STAY_ACTION]))
        if not opponent.is_alive:
            break

    assert not opponent.is_alive, "the melee did not kill the target"
    offset = compute_distances([player], [objective]).model_obj_norms_offset[0, 0]
    assert offset <= radius, "the survivor did not consolidate onto the objective"


def test_a_melee_recording_carries_the_blows_and_the_flags() -> None:
    """Schema 2.7: melee results and the two per-turn flags reach the snapshot.

    Separate lists from the shooting ones, because the renderer draws a tracer
    for a shot and a clash marker for a blow — see `scene.py::_draw_clashes`.
    """
    # Arrange
    env = _env()
    player = env.wargame_models[0]
    opponent = env.opponent_models[0]
    player.location = np.array([10.0, 10.0], dtype=player.location.dtype)
    opponent.location = np.array([11.0, 10.0], dtype=opponent.location.dtype)

    # Act
    for _ in range(8):
        env.step(WargameEnvAction(actions=[STAY_ACTION]))
        snapshot = env.to_snapshot()
        if snapshot.player_melee_results:
            break

    # Assert
    assert snapshot.schema_version == "2.7"
    assert snapshot.player_melee_results, "the melee never reached the snapshot"
    blow = snapshot.player_melee_results[0]
    assert blow.expected_damage > 0.0, "melee expectation was not computed"
    assert not blow.in_cover, "12-fight-phase.md grants no cover in melee"
    assert snapshot.player_models[0].charged_this_turn is not None
    assert snapshot.player_models[0].fell_back_this_turn is not None


def test_a_charge_survives_the_DELTA_codec() -> None:
    """Schema 2.7's two flags must reach a replay, not just a full snapshot.

    ⚠ **This replaces a tautology.** The test here asserted
    `charged_this_turn in (True, False)`, which no implementation can fail, and
    under it both fields were added to `ModelSnapshot` and to NEITHER side of
    the delta codec in `state/events.py`. Full snapshots carried them; every
    delta dropped them silently, so a replay reconstructed from an event log
    carried the anchor's value forever and a charge could never read True.
    Found by an expert panel.
    """
    # Arrange: two snapshots of the same model differing ONLY in the melee flags.
    env = _env()
    env.reset(seed=13)
    before = env.to_snapshot()
    for model in env.wargame_models:
        model.charged_this_turn = True
        model.fell_back_this_turn = True
    after = env.to_snapshot()

    # Act
    delta = _compute_model_delta(0, before.player_models[0], after.player_models[0])
    assert delta is not None, "the codec saw no change at all"
    restored = _apply_model_delta(before.player_models[0], delta)

    # Assert
    assert restored.charged_this_turn, "a charge did not survive the delta codec"
    assert restored.fell_back_this_turn, "a fall back did not survive the delta codec"


def test_a_pre_2_7_recording_still_loads() -> None:
    """The new fields are all defaulted, so an old recording is not orphaned."""
    # Arrange
    env = _env()
    env.reset(seed=11)
    payload = env.to_snapshot().model_dump()
    payload["schema_version"] = "2.6"
    for key in (
        "player_melee_results",
        "opponent_melee_results",
    ):
        payload.pop(key)
    for side in ("player_models", "opponent_models"):
        for model in payload[side]:
            model.pop("charged_this_turn")
            model.pop("fell_back_this_turn")

    # Act
    restored = GameStateSnapshot.model_validate(payload)

    # Assert
    assert restored.player_melee_results == []
    assert not restored.player_models[0].charged_this_turn
    assert not restored.player_models[0].fell_back_this_turn


def test_the_shipped_melee_config_is_lethality_neutral() -> None:
    """`configs/experiments/25v25_maps_melee.yaml` measures the MECHANIC.

    ⚠ **`MeleeWeaponProfile`'s defaults are an ordinary weapon, not a cold
    one**, so a scenario that wants to price the charge rather than the blade
    has to say so. `wound_roll_threshold` returns 6 whenever
    `2 x strength <= toughness`, which is what makes `A1 / MS6+ / S1 / AP2`
    land at 0.0232 against the ~0.0242 an engaged model forfeits in shooting.
    Pinned because a later edit to the profile would silently turn a mechanic
    arm into a lethality arm, and the score would move for the other reason.
    """
    # Arrange
    config = load_env_config("configs/experiments/25v25_maps_melee.yaml")
    player_models = config.models or []
    opponent_models = config.opponent_models or []
    assert player_models and opponent_models
    blade = player_models[0].melee_weapons[0]
    defender = DefenderStats(
        toughness=player_models[0].toughness, save=player_models[0].save
    )

    # Act
    per_fight = expected_attack_damage(blade.melee_skill, blade, defender)

    # Assert
    assert config.melee.enabled
    assert BattlePhase.charge not in (config.skip_phases or [])
    assert BattlePhase.fight in (config.skip_phases or [])
    # ⚠ Pinned as a NUMBER, not as a claim about neutrality. The 0.02415 target
    # this was chosen against is half of `0.163 x 0.296296` -- half the damage an
    # AVERAGE model's shooting is worth -- and an engaged model is not average:
    # it stands within an inch of an enemy, so the shooting it forfeits is close
    # to certain. Against that conditional the blade returns about a tenth of
    # what it costs. Lethality-NEGLIGIBLE, which is still the right choice for
    # measuring the mechanic. See docs/melee.md.
    assert per_fight == pytest.approx(0.0232, abs=0.0005)
    assert all(m.melee_weapons for m in player_models)
    assert all(m.melee_weapons for m in opponent_models)


def test_melee_ON_with_no_charges_is_the_SAME_GAME_as_melee_off() -> None:
    """The gate the seeded-episode digest could not see, because it runs melee OFF.

    ⚠ **This caught a real defect and the digest could not.** Before the corpse
    fix, `25v25_maps_melee.yaml` and `25v25_maps_two_mode.yaml` scored
    differently on 8 of 12 seeds for a policy that never charges — an enemy
    casualty lying beside one of my models was shielding its whole unit from
    shooting. The no-op digest proves melee OFF is byte-identical to `main`; it
    is structurally blind to melee ON.

    This is also the better cross-config bridge: 60 agent steps against 40, a
    different `skip_phases`, and the same game to the point.

    ⚠ **Two things have to be forced, and neither is cosmetic: both seats must
    be non-charging, and `engagement_range` must match across the bridge.**
    The shipped melee config seats `squad_march_take_charge` — it has to, or the
    arm trains in the unilateral cell of a mechanic whose value is the asymmetry
    between the seats. So "no charges" here means overriding the opponent too;
    without that this test compares a charging opponent against a walking one and
    measures the seat rather than melee. It failed exactly that way the moment
    the seat was corrected, which is the test working.
    """
    # Arrange
    seeds = range(700000, 700006)
    policy_name = "squad_march_take"

    # Act
    scores = {}
    for path in (
        "configs/experiments/25v25_maps_melee.yaml",
        "configs/golden/25v25_maps_two_mode.yaml",
    ):
        # ⚠ **`engagement_range` must be PINNED across the bridge**, and this is
        # not incidental. The melee configs adopted the rules' 2" on
        # 2026-08-26; `two_mode` keeps the repo default of 1". That scalar gates
        # which SHOTS are legal, so without this the two configs are a different
        # game for a reason that has nothing to do with melee, and the test
        # fails while measuring the wrong thing. It failed exactly that way the
        # hour the value changed, which is the test working twice.
        config = load_env_config(path, engagement_range="1.0")
        opponent = config.opponent_policy
        assert opponent is not None
        opponent.params = dict(opponent.params or {})
        opponent.params["baseline"] = policy_name
        env = create_environment(config)
        policy = build_baseline_policy(policy_name)
        margins = []
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                action = policy.select_action(
                    env.wargame_models, env, action_mask=observation.action_mask
                )
                observation, _r, terminated, truncated, _i = env.step(action)
            margins.append(env.player_vp - env.opponent_vp)
        scores[path] = margins

    # Assert
    melee_on, melee_off = scores.values()
    assert melee_on == melee_off, (
        "a policy that never charges scored differently with melee on: "
        f"{melee_on} vs {melee_off}"
    )
