"""What breaks when the two armies are not the same.

Nothing in the config validates the sides equal, and a 25 v 18 board with
different profiles already builds, batches and plays. What did not work is
subtler than an exception, which is why these tests exist:

**Unit membership was recovered by arithmetic.** The shooting head pooled
opponent tokens into unit tokens with ``index = arange // ceil(n / n_units)``
while the shooting *mask* and the damage resolution both key on ``group_id``.
The two agree only while every squad is the same size, so 25/5 hid it
completely and squads of 4/4/4/3/3 pool a model into one unit's token and
resolve its wounds in another — silently, with correct-looking logits.

**Move was not a model characteristic.** One scenario-level `max_move_speed`
gave both armies the same speed, so "different armies" could differ in
everything but how far they walk.

**Objective control counts were normalised per side.** Control is a raw count
comparison, so scaling each side by its own establishment made the two columns
incomparable exactly when the establishments differ.

Every case here is a no-op at parity — that is the property batch 1 claims, and
`tests/test_observation_golden.py` is what pins it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION, ActionHandler
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import (
    observation_to_tensor,
    observations_to_tensor_batch,
)
from wargame_rl.wargame.model.net import EncodedState, TransformerNetwork

RIFLE = WeaponProfile(range=12, attacks=1)
PLASMA = WeaponProfile(range=24, attacks=1, strength=6, ap=2)

# 18 models in squads of 4/4/4/3/3 -- the smallest layout where the arithmetic
# span (ceil(18/5) = 4) and the real group ids disagree.
UNEVEN_GROUPS = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]


def _asymmetric_config(**overrides: object) -> WargameEnvConfig:
    """25 tough short-ranged models against 18 fragile long-ranged ones."""
    return WargameEnvConfig(
        render_mode=None,
        number_of_battle_rounds=4,
        board_width=40,
        board_height=40,
        number_of_wargame_models=25,
        number_of_opponent_models=18,
        number_of_objectives=2,
        max_groups=5,
        models=[
            ModelConfig(group_id=i // 5, toughness=5, save=3, weapons=[RIFLE])
            for i in range(25)
        ],
        opponent_models=[
            ModelConfig(group_id=g, toughness=3, save=4, weapons=[PLASMA])
            for g in UNEVEN_GROUPS
        ],
        opponent_policy=OpponentPolicyConfig(
            type="scripted_advance_to_objective", params={}
        ),
        **overrides,  # type: ignore[arg-type]
    )


def _encode(env: WargameEnv) -> tuple[TransformerNetwork, EncodedState]:
    torch.manual_seed(0)
    net = TransformerNetwork.from_env(env, is_policy=True)
    observation, _ = env.reset(seed=11)
    tensors = observation_to_tensor(observation)
    return net, net.encode_state([t.unsqueeze(0) for t in tensors[:5]])


class TestUnitPooling:
    def test_uneven_squads_pool_by_group_id_not_by_arithmetic(self) -> None:
        """The regression. On `main` the derived index is the arithmetic one."""
        env = WargameEnv(config=_asymmetric_config())
        net, state = _encode(env)

        assert state.opponent_unit_index is not None
        derived = state.opponent_unit_index[0].cpu().tolist()

        assert derived == UNEVEN_GROUPS

        n_units = net.shooting_slice_end - net.shooting_slice_start  # type: ignore[operator]
        span = max(1, math.ceil(len(UNEVEN_GROUPS) / n_units))
        arithmetic = [min(i // span, n_units - 1) for i in range(len(UNEVEN_GROUPS))]
        # The point of the test: the two genuinely differ here, so a green
        # assertion above is not the old behaviour passing under a new name.
        assert arithmetic != UNEVEN_GROUPS

    def test_even_squads_are_unchanged(self) -> None:
        """At 25v25 the two derivations coincide, which is why this was dormant."""
        env = WargameEnv(
            config=WargameEnvConfig(
                render_mode=None,
                number_of_battle_rounds=4,
                number_of_wargame_models=25,
                number_of_opponent_models=25,
                max_groups=5,
                opponent_policy=OpponentPolicyConfig(
                    type="scripted_advance_to_objective", params={}
                ),
            )
        )
        _net, state = _encode(env)
        assert state.opponent_unit_index is not None
        expected = [model.group_id for model in env.opponent_models]
        assert state.opponent_unit_index[0].cpu().tolist() == expected
        assert expected == [i // 5 for i in range(25)]

    def test_a_dead_model_still_contributes_nothing(self) -> None:
        """Pooling by the real index must keep excluding the dead."""
        latents = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(1, 6, 6)
        index = torch.tensor([[0, 0, 0, 1, 1, 1]])
        alive = torch.tensor([[True, True, False, True, True, True]])
        pooled = TransformerNetwork._pool_opponents_into_units(latents, 2, alive, index)
        assert torch.allclose(pooled[0, 0], latents[0, :2].mean(dim=0))


class TestPerModelMove:
    def test_a_faster_model_travels_further_for_the_same_action(self) -> None:
        config = _asymmetric_config()
        assert config.models is not None
        config.models[0].move = 12.0
        env = WargameEnv(config=config)
        handler = env.player_action_handler

        action = handler.encode_action(angle_idx=0, speed_idx=handler._n_speed_bins - 1)
        fast = np.linalg.norm(handler.decode_action(action, model_idx=0))
        default = np.linalg.norm(handler.decode_action(action, model_idx=1))

        assert fast == pytest.approx(12.0)
        assert default == pytest.approx(config.max_move_speed)

    @pytest.mark.parametrize("move", [None, 6.0])
    def test_a_uniform_army_is_byte_identical_to_one_that_never_set_move(
        self, move: float | None
    ) -> None:
        """`move` equal to the scenario speed must not perturb a single bin.

        Rebuilding the table as fractions x M would: at M = 6, `6.0 / 6` is
        exactly 1.0 while `linspace(1/6, 1, 6)[0] * 6` is 0.9999999999999999.
        """
        baseline = ActionHandler(WargameEnvConfig(render_mode=None), n_models=4)
        handler = ActionHandler(
            WargameEnvConfig(render_mode=None), n_models=4, model_moves=[move] * 4
        )
        for action in range(1 + baseline.n_move_actions):
            np.testing.assert_array_equal(
                baseline.decode_action(action),
                handler.decode_action(action, model_idx=2),
            )

    def test_a_squad_marches_at_its_slowest_member(self) -> None:
        """One shared vector is what keeps a marching squad's formation rigid."""
        config = _asymmetric_config()
        assert config.models is not None
        config.models[0].move = 2.0
        env = WargameEnv(config=config)
        assert float(env.player_action_handler.move_speeds.min()) == pytest.approx(2.0)


class TestObjectiveControlCounts:
    def test_the_two_count_columns_share_a_scale(self) -> None:
        """Ten of ours must not read as fewer than nine of theirs."""
        config = _asymmetric_config(observe_objective_control=True)
        env = WargameEnv(config=config)
        observation, _ = env.reset(seed=5)

        objective = env.objectives[0]
        for index, model in enumerate(env.player_models):
            if index < 10:
                model.location = np.array(objective.location, dtype=float)
        for index, model in enumerate(env.opponent_models):
            if index < 9:
                model.location = np.array(objective.location, dtype=float)

        token = env.observation.objectives[0]
        assert token.player_count is not None and token.opponent_count is not None
        assert token.player_count > token.opponent_count


class TestItPlays:
    def test_an_asymmetric_board_builds_batches_and_plays(self) -> None:
        env = WargameEnv(config=_asymmetric_config())
        observation, _ = env.reset(seed=3)

        player, opponent = observation_to_tensor(observation)[2:4]
        assert player.shape[0] == 25
        assert opponent.shape[0] == 18
        assert player.shape[1] == opponent.shape[1]

        batch = observations_to_tensor_batch([observation, observation])
        assert batch[2].shape[0] == 2

        torch.manual_seed(0)
        net = TransformerNetwork.from_env(env, is_policy=True)
        logits = net([t.unsqueeze(0) for t in observation_to_tensor(observation)[:5]])
        assert logits.shape == (1, 25, env.n_actions)

        done = False
        steps = 0
        while not done and steps < 20:
            action = WargameEnvAction(actions=[STAY_ACTION] * 25)
            _obs, _reward, done, _trunc, _info = env.step(action)
            steps += 1
        assert steps > 0


def test_a_group_id_the_one_hot_cannot_hold_is_rejected() -> None:
    """Clipping would fold two units into one column, silently."""
    with pytest.raises(ValueError, match="max_groups"):
        WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=4,
            max_groups=2,
            models=[ModelConfig(group_id=g) for g in (0, 1, 2, 2)],
        )


class TestSquadMarchSpeed:
    """A squad's march step is capped by ITS OWN slowest member, not the army's.

    Identical while every model is equally fast, and silently wrong the moment
    they are not: one slow squad capping a fast one means a scripted bar cannot
    use a speed a learned policy can, which flatters the agent against a hobbled
    bar. Found by screening a mixed-profile config, where the fast squad never
    moved at its own speed.
    """

    def test_a_fast_squad_is_not_capped_by_a_slow_one(self) -> None:
        from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy

        config = _asymmetric_config()
        assert config.models is not None
        for index in range(5):  # squad 0 is fast
            config.models[index].move = 12.0
        for index in range(5, 10):  # squad 1 is slow
            config.models[index].move = 2.0
        env = WargameEnv(config=config)
        observation, _ = env.reset(seed=13)

        policy = build_baseline_policy("squad_march")
        before = [model.location.copy() for model in env.player_models]
        env.step(
            policy.select_action(
                env.player_models, env, action_mask=observation.action_mask
            )
        )
        moved = [
            float(np.linalg.norm(model.location - start))
            for model, start in zip(env.player_models, before)
        ]

        fast = max(moved[:5])
        slow = max(moved[5:10])
        # The fast squad must be able to exceed the slow squad's whole budget.
        assert fast > 2.0 + 1e-9, f"fast squad capped at the army minimum ({fast:.2f})"
        assert slow <= 2.0 + 1e-9, f"slow squad exceeded its own Move ({slow:.2f})"
