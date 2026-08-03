from __future__ import annotations

import torch

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import (
    observation_to_tensor,
    observations_to_tensor_batch,
)
from wargame_rl.wargame.model.net import TransformerNetwork


def _shooting_env_config() -> WargameEnvConfig:
    return WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=2,
        board_width=30,
        board_height=30,
        models=[
            ModelConfig(x=5, y=5, weapons=[WeaponProfile(range=24)]),
            ModelConfig(x=10, y=10, weapons=[WeaponProfile(range=24)]),
        ],
        opponent_models=[
            ModelConfig(x=20, y=5),
            ModelConfig(x=25, y=10),
        ],
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        opponent_policy=OpponentPolicyConfig(type="random"),
        n_movement_angles=8,
        n_speed_bins=3,
    )


def _advance_to_shooting(env: WargameEnv) -> tuple:
    stay = WargameEnvAction(actions=[STAY_ACTION] * env.config.number_of_wargame_models)
    return env.step(stay)


def test_transformer_policy_dead_player_row_is_stay_only() -> None:
    env = WargameEnv(config=WargameEnvConfig())
    net = TransformerNetwork.policy_from_env(env)

    env.reset(seed=0)
    env.wargame_models[0].take_damage(env.wargame_models[0].stats["current_wounds"])
    stay = WargameEnvAction(actions=[STAY_ACTION] * env.config.number_of_wargame_models)
    obs, _, _, _, _ = env.step(stay)

    logits = net(observation_to_tensor(obs, net.device))
    assert torch.isfinite(logits[0, 0, STAY_ACTION])
    assert torch.isneginf(logits[0, 0, 1:]).all()


def test_transformer_policy_dead_opponent_shooting_column_is_neginf() -> None:
    env = WargameEnv(config=_shooting_env_config())
    net = TransformerNetwork.policy_from_env(env)

    env.reset(seed=42)
    env.opponent_models[0].take_damage(env.opponent_models[0].stats["current_wounds"])
    obs, _, _, _, _ = _advance_to_shooting(env)

    shooting_slice = env._action_handler.shooting_slice
    assert shooting_slice is not None

    tensors = observation_to_tensor(obs, net.device)
    logits = net(tensors)
    dead_target_col = shooting_slice.start
    assert not tensors[5][:, dead_target_col].any()
    assert torch.isneginf(logits[0, :, dead_target_col]).all()


def test_transformer_policy_without_shooting_keeps_shoot_head_disabled() -> None:
    env = WargameEnv(config=WargameEnvConfig(number_of_opponent_models=0))
    net = TransformerNetwork.policy_from_env(env)
    assert net.shoot_query_proj is None
    assert net.shoot_key_proj is None


def test_transformer_shooting_scores_land_in_correct_opponent_columns() -> None:
    """The bilinear head's score for (player i, opponent j) lands in env column
    ``shooting_slice.start + j`` for the matching player row."""
    env = WargameEnv(config=_shooting_env_config())
    net = TransformerNetwork.policy_from_env(env)
    net.eval()

    env.reset(seed=8)
    obs, _, _, _, _ = _advance_to_shooting(env)
    shooting_slice = env._action_handler.shooting_slice
    assert shooting_slice is not None

    tensors = observation_to_tensor(obs, net.device)
    mask = tensors[5]
    with torch.no_grad():
        state = net.encode_state(tensors)
        logits = net.policy_from_encoded(state)

        n_p = state.n_wargame_models
        n_o = state.n_opponents
        assert n_p > 0 and n_o > 0

        start = state.n_prefix
        player_latents = state.encoded[:, start : start + n_p, :]
        opp_latents = state.encoded[:, start + n_p : start + n_p + n_o, :]
        expected = net._shooting_scores(player_latents, opp_latents)  # (1, n_p, n_o)

    checked = 0
    for pi in range(n_p):
        for oj in range(n_o):
            col = shooting_slice.start + oj
            if not bool(mask[pi, col]):
                continue  # env masked this target; -inf is expected, skip
            assert torch.allclose(logits[0, pi, col], expected[0, pi, oj], atol=1e-5)
            checked += 1
    assert checked > 0  # the scenario must exercise at least one live target


def test_transformer_policy_batched_matches_single_obs() -> None:
    """Batched forward (variable alive counts) equals stacked single forwards."""
    env = WargameEnv(config=_shooting_env_config())
    net = TransformerNetwork.policy_from_env(env)
    net.eval()

    env.reset(seed=3)
    env.opponent_models[0].take_damage(env.opponent_models[0].stats["current_wounds"])
    obs_a, _, _, _, _ = _advance_to_shooting(env)  # one opponent dead

    env.reset(seed=4)
    obs_b, _, _, _, _ = _advance_to_shooting(env)  # all alive

    with torch.no_grad():
        single_a = net(observation_to_tensor(obs_a, net.device))[0]
        single_b = net(observation_to_tensor(obs_b, net.device))[0]
        batched = net(observations_to_tensor_batch([obs_a, obs_b], net.device))

    def _finite(t: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isneginf(t), torch.zeros_like(t), t)

    assert torch.equal(torch.isneginf(batched[0]), torch.isneginf(single_a))
    assert torch.equal(torch.isneginf(batched[1]), torch.isneginf(single_b))
    assert torch.allclose(_finite(batched[0]), _finite(single_a), atol=1e-5)
    assert torch.allclose(_finite(batched[1]), _finite(single_b), atol=1e-5)


def test_transformer_value_path_runs_with_masking() -> None:
    """The critic path still produces finite scalars through the shared encoding."""
    env = WargameEnv(config=_shooting_env_config())
    value_net = TransformerNetwork.value_from_env(env)
    value_net.eval()

    env.reset(seed=5)
    obs_a, _, _, _, _ = _advance_to_shooting(env)
    env.reset(seed=6)
    obs_b, _, _, _, _ = _advance_to_shooting(env)

    with torch.no_grad():
        single = value_net(observation_to_tensor(obs_a, value_net.device))
        batched = value_net(
            observations_to_tensor_batch([obs_a, obs_b], value_net.device)
        )

    assert single.shape == (1,)
    assert batched.shape == (2,)
    assert torch.isfinite(batched).all()


def test_self_attention_key_mask_ignores_masked_positions() -> None:
    """A key-padding mask makes masked-out tokens invisible to other queries."""
    from wargame_rl.wargame.model.common.config import TransformerConfig
    from wargame_rl.wargame.model.dqn.layers import Block

    config = TransformerConfig()
    block = Block(config).eval()

    torch.manual_seed(0)
    x = torch.randn(1, 5, config.embedding_size)
    key_mask = torch.ones(1, 1, 1, 5, dtype=torch.bool)
    key_mask[..., 4] = False  # last token masked out as a key

    with torch.no_grad():
        out = block(x, attn_mask=key_mask)
        x_changed = x.clone()
        x_changed[:, 4, :] = torch.randn(config.embedding_size)
        out_changed = block(x_changed, attn_mask=key_mask)

    # Queries 0..3 must not see the masked key, so their outputs are unchanged.
    assert torch.allclose(out[:, :4], out_changed[:, :4], atol=1e-5)


def test_dead_tokens_do_not_affect_alive_logits() -> None:
    """Mutating a dead opponent's features leaves alive players' logits unchanged.

    This is the core guarantee of the key-padding approach: dead rows are dropped
    as attention keys, so they cannot influence live tokens' encodings.
    """
    env = WargameEnv(config=_shooting_env_config())
    net = TransformerNetwork.policy_from_env(env)
    net.eval()

    env.reset(seed=13)
    env.opponent_models[0].take_damage(env.opponent_models[0].stats["current_wounds"])
    obs, _, _, _, _ = _advance_to_shooting(env)

    tensors = observation_to_tensor(obs, net.device)
    opp = tensors[3]
    feature_dim = int(opp.shape[-1])
    alive_idx = net._alive_feature_index(feature_dim, int(opp.shape[0]))
    dead_rows = (opp[:, alive_idx] <= 0.5).nonzero(as_tuple=True)[0]
    assert dead_rows.numel() > 0

    opp_mutated = opp.clone()
    torch.manual_seed(1)
    for row in dead_rows.tolist():
        noise = torch.randn(feature_dim, device=opp.device)
        noise[alive_idx] = opp[row, alive_idx]  # keep it dead
        opp_mutated[row] = noise
    mutated = [tensors[0], tensors[1], tensors[2], opp_mutated, tensors[4], tensors[5]]

    with torch.no_grad():
        before = net(tensors)
        after = net(mutated)

    assert torch.equal(torch.isneginf(before), torch.isneginf(after))
    finite = ~torch.isneginf(before)
    assert torch.allclose(before[finite], after[finite], atol=1e-5)


def test_transformer_policy_no_nan_with_dead_units() -> None:
    """Dead players/opponents never yield NaN logits and dead rows are stay-only."""
    env = WargameEnv(config=_shooting_env_config())
    net = TransformerNetwork.policy_from_env(env)
    net.eval()

    env.reset(seed=21)
    env.wargame_models[0].take_damage(env.wargame_models[0].stats["current_wounds"])
    env.opponent_models[0].take_damage(env.opponent_models[0].stats["current_wounds"])
    obs, _, _, _, _ = _advance_to_shooting(env)

    with torch.no_grad():
        logits = net(observation_to_tensor(obs, net.device))

    assert not torch.isnan(logits).any()
    finite = logits[~torch.isneginf(logits)]
    assert torch.isfinite(finite).all()
    # Dead player row collapses to stay-only.
    assert torch.isfinite(logits[0, 0, STAY_ACTION])
    assert torch.isneginf(logits[0, 0, 1:]).all()
