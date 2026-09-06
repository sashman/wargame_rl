"""The set network's load-bearing properties: one set of weights serves any
scenario size, padding and death are masked rather than dropped, and the
pointers respect the env's own legality."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wargame_rl.wargame.envs.per_model import PerModelEnv, StepKind
from wargame_rl.wargame.envs.per_model.observation import build_token_observation
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.per_model import (
    SetAgent,
    SetNetwork,
    SetNetworkConfig,
    collate,
)

SMALL_TRUNK = SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)


def _config(n_models: int, n_units: int, shooting: bool = False) -> WargameEnvConfig:
    rifle = [WeaponProfile(range=12, attacks=1)]
    per_unit = n_models // n_units
    squads = [
        ModelConfig(
            group_id=min(i // per_unit, n_units - 1), weapons=rifle, max_wounds=1
        )
        for i in range(n_models)
    ]
    skip = [
        BattlePhase.command,
        BattlePhase.charge,
        BattlePhase.pile_in,
        BattlePhase.fight,
        BattlePhase.consolidate,
    ]
    if not shooting:
        skip.append(BattlePhase.shooting)
    return WargameEnvConfig(
        render_mode=None,
        board_width=30,
        board_height=30,
        number_of_wargame_models=n_models,
        number_of_opponent_models=n_models if shooting else 0,
        max_groups=n_units,
        models=squads,
        opponent_models=list(squads) if shooting else None,
        number_of_objectives=2,
        number_of_battle_rounds=2,
        skip_phases=skip,
        opponent_policy=(
            OpponentPolicyConfig(type="scripted_advance_and_shoot")
            if shooting
            else None
        ),
    )


def _drive_episode(network: SetNetwork, env: PerModelEnv, seed: int) -> int:
    """A network-driven episode to termination; returns the model-step count."""
    agent = SetAgent(network)
    generator = torch.Generator().manual_seed(seed)
    observation, _ = env.reset(seed=seed)
    terminated = False
    guard = 0
    while not terminated:
        decision = agent.act(env, observation, generator=generator)
        observation, _r, terminated, _t, _ = env.step(decision.action)
        guard += 1
        assert guard < 2000, "episode did not terminate"
    return env.model_steps


def test_one_set_of_weights_serves_two_scenario_sizes() -> None:
    """Principle 2, end to end: the SAME instance plays 6-model and 10-model
    scenarios — different armies, units, objectives — with no reload."""
    torch.manual_seed(0)
    network = SetNetwork(SMALL_TRUNK, n_move_actions=96)
    small = PerModelEnv(_config(6, 2), build_info=False)
    large = PerModelEnv(_config(10, 5), build_info=False)
    # A `remain_stationary` declaration consumes its whole unit in one step,
    # so an episode costs between one step per unit and one per model, per
    # stepped phase.
    assert 2 * 2 <= _drive_episode(network, small, seed=1) <= 2 * 6
    assert 2 * 5 <= _drive_episode(network, large, seed=2) <= 2 * 10


def test_a_network_driven_shooting_episode_terminates() -> None:
    torch.manual_seed(0)
    env = PerModelEnv(_config(8, 2, shooting=True), build_info=False)
    network = SetNetwork.from_env(env, SMALL_TRUNK)
    steps = _drive_episode(network, env, seed=3)
    assert steps > 0


def test_padding_to_the_batch_maximum_changes_nothing() -> None:
    """A small observation batched beside a larger one reads identically to
    the same observation alone — padding is masked, never attended."""
    torch.manual_seed(0)
    network = SetNetwork(SMALL_TRUNK, n_move_actions=96)
    network.eval()
    small_env = PerModelEnv(_config(6, 2), build_info=False)
    large_env = PerModelEnv(_config(10, 5), build_info=False)
    small_obs, _ = small_env.reset(seed=5)
    large_obs, _ = large_env.reset(seed=6)
    small_tokens = build_token_observation(small_env, small_obs)
    large_tokens = build_token_observation(large_env, large_obs)

    alone = network(collate([small_tokens]))
    padded = network(collate([small_tokens, large_tokens]))

    n = small_tokens.player_tokens.shape[0]
    torch.testing.assert_close(
        alone.selector_logits[0], padded.selector_logits[0, :n], atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(alone.value[0], padded.value[0], atol=1e-5, rtol=1e-4)


def test_a_dead_models_features_reach_nobody() -> None:
    """Dead models are masked as keys: perturbing a corpse's token must not
    move any other model's latent, the selector or the value."""
    torch.manual_seed(0)
    env = PerModelEnv(_config(6, 2), build_info=False)
    observation, _ = env.reset(seed=7)
    victim = 3
    env.wargame_models[victim].take_damage(10**6)
    observation = env._observe()
    tokens = build_token_observation(env, observation)
    assert not tokens.player_alive[victim]

    network = SetNetwork(SMALL_TRUNK, n_move_actions=96)
    network.eval()
    with torch.no_grad():
        before = network(collate([tokens]))
        tokens.player_tokens[victim] += 100.0
        after = network(collate([tokens]))

    others = [i for i in range(6) if i != victim]
    torch.testing.assert_close(
        before.player_latents[0, others], after.player_latents[0, others]
    )
    torch.testing.assert_close(before.value, after.value)
    torch.testing.assert_close(
        before.selector_logits[0, others], after.selector_logits[0, others]
    )


def test_the_selector_offers_exactly_the_envs_own_mask() -> None:
    torch.manual_seed(0)
    env = PerModelEnv(_config(6, 2), build_info=False)
    observation, _ = env.reset(seed=9)
    network = SetNetwork(SMALL_TRUNK, n_move_actions=96)
    with torch.no_grad():
        output = network(collate([build_token_observation(env, observation)]))
    finite = torch.isfinite(output.selector_logits[0]).numpy()
    np.testing.assert_array_equal(finite, observation.selection_mask)


def test_the_target_pointer_masks_what_the_env_forbids() -> None:
    torch.manual_seed(0)
    env = PerModelEnv(_config(8, 2, shooting=True), build_info=False)
    observation, _ = env.reset(seed=11)
    # Walk to the shooting phase: submit STAY for everyone in movement.
    while observation.phase is not BattlePhase.shooting:
        assert observation.kind is StepKind.model_action
        index = int(np.flatnonzero(observation.selection_mask)[0])
        from wargame_rl.wargame.envs.per_model import PerModelAction

        observation, _r, terminated, _t, _ = env.step(PerModelAction(model_index=index))
        assert not terminated
    tokens = build_token_observation(env, observation)
    network = SetNetwork.from_env(env, SMALL_TRUNK)
    with torch.no_grad():
        output = network(collate([tokens]))
        index = int(np.flatnonzero(observation.selection_mask)[0])
        logits = network.action_logits(output, collate([tokens]), torch.tensor([index]))
    finite = torch.isfinite(logits.target[0]).numpy()
    np.testing.assert_array_equal(finite, tokens.target_mask[index])


def test_the_env_masks_agree_with_the_token_masks() -> None:
    """The token observation's displacement mask is sliced from the same
    function the whole-phase facade puts on its observation."""
    env = PerModelEnv(_config(6, 2), build_info=False)
    observation, _ = env.reset(seed=13)
    tokens = build_token_observation(env, observation)
    full = env.current_action_mask()
    handler = env.player_action_handler
    movement = handler.movement_slice
    np.testing.assert_array_equal(tokens.displacement_mask[:, 0], full[:, 0])
    np.testing.assert_array_equal(
        tokens.displacement_mask[:, 1:], full[:, movement.start : movement.end]
    )


@pytest.mark.parametrize("n_models,n_units", [(6, 2), (9, 3)])
def test_state_dict_is_size_independent(n_models: int, n_units: int) -> None:
    """Weights built against one scenario load into a network built against
    another — the load_state_dict failure Principle 2 exists to remove."""
    torch.manual_seed(0)
    env = PerModelEnv(_config(n_models, n_units), build_info=False)
    network = SetNetwork.from_env(env, SMALL_TRUNK)
    donor = SetNetwork(SMALL_TRUNK, n_move_actions=96)
    network.load_state_dict(donor.state_dict())
