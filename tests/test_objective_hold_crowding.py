"""`objective_hold`'s crowding exponent: pay an objective a pot, not a wage.

The measured failure this exists for: at 1000 epochs the agent ends with 15.8
models alive and stands 12.9 of them on an objective defended by 0.25 opponents,
while losing the second objective 4.2 to 2.7. Under the flat default the
thirteenth model on a point earns exactly what the first does, so no model ever
has a private reason to leave.

`crowding_exponent=0.0` is the default and must stay bit-identical, because every
existing config and checkpoint assumes it.

The property that separates this from `surplus_value` — and the reason to expect
a different result — is **pot conservation at a = 1**: total pay across a point's
occupants is its value no matter how many stand there, so spreading onto a second
point raises total income rather than lowering it.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.calculators.objective_hold import (
    ObjectiveHoldCalculator,
)
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

OBJECTIVE_A = (20, 20)
OBJECTIVE_B = (36, 36)
# Distinct cells all strictly inside radius 3 of a centre. Stacking six models
# along one axis would put the fifth at distance 4 and silently outside the disc,
# which reads as "crowding is not counted" rather than as a broken fixture.
_INSIDE_OFFSETS = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)]


def _make_env(n_on_a: int, n_on_b: int) -> WargameEnv:
    """Six player models split between two far-apart, undefended objectives.

    Every opponent is parked in a corner off both discs, so both objectives are
    player-controlled and the only thing varying is how the player split.
    """
    if n_on_a + n_on_b > 6:
        raise ValueError("only six player models exist")

    models = [
        ModelConfig(x=OBJECTIVE_A[0] + dx, y=OBJECTIVE_A[1] + dy, group_id=0)
        for dx, dy in _INSIDE_OFFSETS[:n_on_a]
    ]
    models += [
        ModelConfig(x=OBJECTIVE_B[0] + dx, y=OBJECTIVE_B[1] + dy, group_id=0)
        for dx, dy in _INSIDE_OFFSETS[:n_on_b]
    ]
    # Any remaining models sit off both objectives and are paid nothing.
    models += [ModelConfig(x=2, y=2, group_id=0) for _ in range(6 - len(models))]

    config = WargameEnvConfig(
        render_mode=None,
        board_width=44,
        board_height=44,
        number_of_wargame_models=6,
        number_of_opponent_models=2,
        number_of_objectives=2,
        objective_radius_size=3,
        number_of_battle_rounds=6,
        max_groups=1,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        models=models,
        opponent_models=[ModelConfig(x=1, y=42, group_id=0) for _ in range(2)],
        objectives=[
            ObjectiveConfig(x=OBJECTIVE_A[0], y=OBJECTIVE_A[1]),
            ObjectiveConfig(x=OBJECTIVE_B[0], y=OBJECTIVE_B[1]),
        ],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
    )
    return WargameEnv(config=config)


def _context(env: WargameEnv) -> StepContext:
    """A fresh StepContext, which is also the calculators' per-step cache key."""
    return StepContext(
        distance_cache=compute_distances(env.wargame_models, env.objectives),
        current_turn=1,
        max_turns=env.max_turns,
        board_width=env.board_width,
        board_height=env.board_height,
    )


def _rewards(
    crowding_exponent: float,
    n_on_a: int,
    n_on_b: int = 0,
    weight: float = 1.0,
) -> list[float]:
    """Per-model `objective_hold` values for the given split."""
    env = _make_env(n_on_a, n_on_b)
    env.reset(seed=0)
    calculator = ObjectiveHoldCalculator(
        weight=weight, crowding_exponent=crowding_exponent
    )
    context = _context(env)
    # `calculate` returns the value *before* the weight -- `phase_manager`
    # multiplies it in -- so the arm calibrations have to apply it here.
    return [
        weight * calculator.calculate(i, model, env, context)
        for i, model in enumerate(env.wargame_models)
    ]


def test_default_exponent_pays_every_occupant_in_full() -> None:
    """Backward compatibility: 0.0 must not change a single number."""
    flat = _rewards(crowding_exponent=0.0, n_on_a=4)

    on_objective = [r for r in flat if r > 0.0]
    assert len(on_objective) == 4
    assert len(set(on_objective)) == 1, f"default is no longer uniform: {flat}"


@pytest.mark.parametrize(
    ("occupants", "expected"),
    # value 1.0 (player-controlled) divided by occupants ** 1.0.
    [(1, 1.0), (2, 0.5), (4, 0.25)],
)
def test_full_sharing_divides_the_value_by_the_occupant_count(
    occupants: int, expected: float
) -> None:
    """At a = 1 the point pays a pot, so each occupant's share is value / k."""
    rewards = _rewards(crowding_exponent=1.0, n_on_a=occupants)

    paid = [r for r in rewards if r > 0.0]
    assert len(paid) == occupants
    assert paid == pytest.approx([expected] * occupants)


@pytest.mark.parametrize("occupants", [1, 2, 3, 5])
def test_the_pot_is_conserved_at_full_sharing(occupants: int) -> None:
    """The property that distinguishes this from `surplus_value`.

    Total pay across a point's occupants is its value regardless of how many
    stand there — so unlike a discount, crowding moves reward *between* models
    rather than destroying it, and holding a second point strictly raises the
    total.
    """
    rewards = _rewards(crowding_exponent=1.0, n_on_a=occupants)

    assert sum(rewards) == pytest.approx(1.0)


def test_spreading_across_two_points_beats_stacking_on_one() -> None:
    """Six models on one point earn 1.0 in total; 3+3 across two earns 2.0."""
    stacked = sum(_rewards(crowding_exponent=1.0, n_on_a=6))
    spread = sum(_rewards(crowding_exponent=1.0, n_on_a=3, n_on_b=3))

    assert spread == pytest.approx(2 * stacked)


def test_moving_to_the_emptier_point_pays_the_mover_more() -> None:
    """The individual incentive, which is what the policy actually optimises.

    Pot conservation makes the *team* better off; this asserts the marginal model
    is better off too, or no single model would ever move.
    """
    before = _rewards(crowding_exponent=1.0, n_on_a=5, n_on_b=1)
    after = _rewards(crowding_exponent=1.0, n_on_a=4, n_on_b=2)

    crowded_share = max(before)  # the lone model on B, not the mover
    mover_before = min(r for r in before if r > 0.0)
    mover_after = min(r for r in after if r > 0.0)
    assert mover_after > mover_before, f"{before} -> {after}"
    assert crowded_share > 0.0


@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0, 2.0])
def test_crowding_never_pays_more_as_the_point_fills(exponent: float) -> None:
    """Monotonicity: an exponent must not invert the ordering it exists to set."""
    sparse = max(_rewards(crowding_exponent=exponent, n_on_a=2))
    crowded = max(_rewards(crowding_exponent=exponent, n_on_a=5))

    assert crowded <= sparse


def test_soft_exponent_sits_between_flat_and_full_sharing() -> None:
    """a = 0.5 is the fallback arm; it must actually be intermediate."""
    flat = max(_rewards(crowding_exponent=0.0, n_on_a=4))
    soft = max(_rewards(crowding_exponent=0.5, n_on_a=4))
    full = max(_rewards(crowding_exponent=1.0, n_on_a=4))

    assert full < soft < flat


def test_arm_weights_price_the_target_occupancy_identically() -> None:
    """Both arms are calibrated so 5 models on a point pay today's 0.25/step.

    The arms are meant to differ only in how sharply crowding is punished, not in
    the price of correct play. If a weight drifts, they stop being one axis.
    """
    share = max(_rewards(crowding_exponent=1.0, n_on_a=5, weight=1.25))
    share_soft = max(_rewards(crowding_exponent=0.5, n_on_a=5, weight=0.56))

    assert share == pytest.approx(0.25, abs=0.01)
    assert share_soft == pytest.approx(0.25, abs=0.01)


def test_occupancy_is_recomputed_each_step_not_frozen_after_the_first() -> None:
    """Its own cache key, for the reason `_within_quota` needs one.

    `_objective_values` stamps `_cached_ctx` before occupancy is ever built, so a
    shared key would price every later step at step one's crowd.
    """
    env = _make_env(4, 0)
    env.reset(seed=0)
    calculator = ObjectiveHoldCalculator(weight=1.0, crowding_exponent=1.0)

    first = _context(env)
    calculator.calculate(0, env.wargame_models[0], env, first)
    occupancy_after_first = calculator._cached_occupancy

    # A model leaves the objective; the survivors' share must rise.
    env.wargame_models[3].location = np.array([2, 2])
    second = _context(env)
    reward = calculator.calculate(0, env.wargame_models[0], env, second)

    assert calculator._cached_occupancy is not occupancy_after_first
    assert reward == pytest.approx(1.0 / 3.0)


def test_a_negative_exponent_is_rejected_at_construction() -> None:
    """It would pay *more* for crowding — validate at init, not at runtime."""
    with pytest.raises(ValueError, match="crowding_exponent"):
        ObjectiveHoldCalculator(weight=1.0, crowding_exponent=-1.0)


def test_dead_models_do_not_count_toward_the_crowd() -> None:
    """Casualties must free up the share, or the pot shrinks as models die."""
    env = _make_env(4, 0)
    env.reset(seed=0)
    calculator = ObjectiveHoldCalculator(weight=1.0, crowding_exponent=1.0)

    for model in env.wargame_models[2:4]:
        model.stats["current_wounds"] = 0

    context = StepContext(
        distance_cache=compute_distances(
            env.wargame_models,
            env.objectives,
            alive_mask=np.array([m.is_alive for m in env.wargame_models]),
        ),
        current_turn=1,
        max_turns=env.max_turns,
        board_width=env.board_width,
        board_height=env.board_height,
    )
    reward = calculator.calculate(0, env.wargame_models[0], env, context)

    assert reward == pytest.approx(0.5), "two live occupants should split the pot"
