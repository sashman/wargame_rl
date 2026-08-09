"""Config-level checks for the 25v25 training configs.

These encode the design rules that came out of the diagnostic, as assertions
rather than comments. The failure they guard against is concrete: win rate fell
62% -> 47% when the curriculum advanced to a phase that rewarded occupancy and
carried no VP term, while `success_rate` held at ~80%.

The configs come in pairs — a single-phase control and a ladder that ends on the
same phase — so a comparison between the two measures the ladder and nothing
else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager
from wargame_rl.wargame.envs.types.config import WargameEnvConfig

CONFIG_ROOT = Path("configs") / "golden"

# Each pair is (single-phase control, matching ladder). The invariants below
# apply to every config; `test_paired_configs_share_a_final_phase` applies to
# the pairing.
CONFIG_PAIRS = [
    (
        CONFIG_ROOT / "25v25_single_phase.yaml",
        CONFIG_ROOT / "25v25_curriculum.yaml",
    ),
]

CONFIG_PATHS = [
    *(path for pair in CONFIG_PAIRS for path in pair),
    # The shooting-opponent scenario: the control it was developed against, and
    # the configuration that beat the scripted bar on it. Both single-phase,
    # with no ladder to pair with. The closed experiment arms around them were
    # deleted once their questions were answered -- `git log` on
    # configs/ restores any of them.
    CONFIG_ROOT / "25v25_cover_control.yaml",
    CONFIG_ROOT / "25v25_shooting_opponent.yaml",
]

# A one-objective stack caps at 19 scoring rounds x 5 VP = 95 of a 285
# theoretical max. Any VP gate at or below this is passable without ever
# contesting a second objective.
ONE_OBJECTIVE_STACK_FRACTION = 95 / 285


def _load(path: Path) -> WargameEnvConfig:
    config: WargameEnvConfig = parse_yaml_raw_as(WargameEnvConfig, path.read_text())
    return config


@pytest.fixture(params=CONFIG_PATHS, ids=lambda p: p.stem)
def config(request: pytest.FixtureRequest) -> WargameEnvConfig:
    path: Path = request.param
    return _load(path)


def test_config_loads_and_builds_its_phases(config: WargameEnvConfig) -> None:
    """Every calculator name and parameter is valid.

    `build_calculator` passes params straight to the constructor, so a
    misspelled parameter is a TypeError at load rather than a silent default.
    """
    manager = RewardPhaseManager.from_configs(config.reward_phases)

    assert manager.phases


def test_every_phase_keeps_the_goal_signal(config: WargameEnvConfig) -> None:
    """No rung trains away from winning.

    This is the rule the old ladder broke: its `mass_on_objectives` phase
    rewarded occupancy and dropped VP entirely.
    """
    for phase in config.reward_phases:
        types = {c.type for c in phase.reward_calculators}
        assert "vp_gain" in types, f"phase '{phase.name}' has no VP signal"


def test_every_phase_carries_per_model_credit(config: WargameEnvConfig) -> None:
    """Each phase has at least one per-model calculator.

    Only 3 of 8 calculators are per-model; the rest are broadcast identically
    to all 25 models. A phase built purely from global terms hands every model
    the same reward, undoing per-model credit assignment by configuration.
    """
    manager = RewardPhaseManager.from_configs(config.reward_phases)

    for phase in manager.phases:
        assert phase.per_model_calculators, (
            f"phase '{phase.name}' is entirely global, so all models share "
            "one reward and the advantage cannot differentiate them"
        )


def test_no_phase_relies_on_a_saturating_occupancy_term(
    config: WargameEnvConfig,
) -> None:
    """`models_at_objectives` cannot rank policies and is excluded.

    Every competent scripted baseline saturates it at 1.000, so it scores a
    0.53-win policy and a 0.78-win policy identically.
    """
    for phase in config.reward_phases:
        types = {c.type for c in phase.reward_calculators}
        assert "models_at_objectives" not in types


def test_vp_gates_are_above_the_one_objective_stack_ceiling(
    config: WargameEnvConfig,
) -> None:
    """A VP gate must require contesting more than one objective.

    Objective discs hold 29 cells and models do not collide, so parking the
    whole army on one point is legal and saturates every occupancy metric.
    """
    for phase in config.reward_phases:
        if phase.success_criteria.type != "player_vp_min":
            continue
        fraction = phase.success_criteria.params["fraction_of_max"]
        assert fraction > ONE_OBJECTIVE_STACK_FRACTION, (
            f"phase '{phase.name}' gate of {fraction} is clearable by a stack "
            f"on one objective ({ONE_OBJECTIVE_STACK_FRACTION:.4f})"
        )


def test_terminal_bonus_is_delivered_at_full_strength(
    config: WargameEnvConfig,
) -> None:
    """`terminate_on_success` must stay false wherever a terminal bonus is set.

    With it true the bonus is scaled by the remaining-turn fraction, which
    collapses to 1/max_turns when episodes run to the limit.
    """
    for phase in config.reward_phases:
        if phase.terminal_success_bonus != 0.0:
            assert not phase.terminate_on_success, phase.name


def test_group_cohesion_parameters_are_named_correctly(
    config: WargameEnvConfig,
) -> None:
    """Guards a config-only failure mode: params go straight to the constructor.

    `max_dist` or `penalty` would raise at load; a silently wrong name cannot
    happen, but pinning the expected keys documents them.
    """
    for phase in config.reward_phases:
        for calculator in phase.reward_calculators:
            if calculator.type != "group_cohesion":
                continue
            assert set(calculator.params) <= {
                "group_max_distance",
                "violation_penalty",
            }


@pytest.mark.parametrize(
    ("control_path", "ladder_path"), CONFIG_PAIRS, ids=lambda p: p.stem
)
def test_paired_configs_share_a_final_phase(
    control_path: Path, ladder_path: Path
) -> None:
    """The control and the ladder converge on the same reward.

    They must differ only in how the policy gets there, or a comparison
    between them measures two things at once.
    """
    single = _load(control_path)
    curriculum = _load(ladder_path)

    assert single.reward_phases[-1].reward_calculators == (
        curriculum.reward_phases[-1].reward_calculators
    )
    assert single.reward_phases[-1].success_criteria == (
        curriculum.reward_phases[-1].success_criteria
    )


@pytest.mark.parametrize(
    ("control_path", "ladder_path"), CONFIG_PAIRS, ids=lambda p: p.stem
)
def test_paired_configs_share_a_scenario(control_path: Path, ladder_path: Path) -> None:
    """A pair differs in reward phases only — never in the world.

    The opponent, the terrain and the board are what make one run harder than
    another. If a pair drifts on any of them, the ladder comparison silently
    becomes a comparison of two different games.
    """
    single = _load(control_path)
    curriculum = _load(ladder_path)

    assert single.opponent_policy == curriculum.opponent_policy
    assert single.terrain == curriculum.terrain
    assert single.random_terrain == curriculum.random_terrain
    assert single.number_of_opponent_models == curriculum.number_of_opponent_models
    assert (single.board_width, single.board_height) == (
        curriculum.board_width,
        curriculum.board_height,
    )
