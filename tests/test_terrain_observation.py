"""Tests for Phase 2: Terrain in the Observation.

Covers TERR-08 (terrain tokens in observation) and TERR-09 (no mid-episode
shape change, no-terrain byte-identical).
"""

from __future__ import annotations

import torch

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import TerrainPieceConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import (
    TERRAIN_FEATURE_DIM,
    observation_to_tensor,
    observations_to_tensor_batch,
)
from wargame_rl.wargame.model.net import TransformerNetwork


def _make_terrain_config() -> WargameEnvConfig:
    return WargameEnvConfig(
        board_width=60,
        board_height=44,
        number_of_wargame_models=2,
        number_of_objectives=2,
        render_mode=None,
        terrain=[
            TerrainPieceConfig(footprint=(27, 8, 33, 16)),
            TerrainPieceConfig(footprint=(27, 28, 33, 36)),
        ],
    )


def _make_no_terrain_config() -> WargameEnvConfig:
    return WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_objectives=2,
        render_mode=None,
    )


# ---------------------------------------------------------------------------
# TERR-08: Terrain tokens in observation
# ---------------------------------------------------------------------------


class TestTerrainObservation:
    def test_terrain_obs_has_correct_token_count(self) -> None:
        """One terrain token per footprint."""
        env = WargameEnv(config=_make_terrain_config())
        obs, _ = env.reset(seed=42)
        assert len(obs.terrain) == 2

    def test_terrain_obs_carries_normalised_geometry(self) -> None:
        """Footprint corners normalised to [-1, 1]."""
        env = WargameEnv(config=_make_terrain_config())
        obs, _ = env.reset(seed=42)
        for t in obs.terrain:
            assert t.footprint.shape == (4,)
            assert t.footprint.min() >= -1.0
            assert t.footprint.max() <= 1.0

    def test_terrain_tensor_shape(self) -> None:
        """Terrain tensor has shape (n_terrain, TERRAIN_FEATURE_DIM)."""
        env = WargameEnv(config=_make_terrain_config())
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        terrain_t = tensors[4]
        assert terrain_t.shape == (2, TERRAIN_FEATURE_DIM)

    def test_terrain_tensor_batch_shape(self) -> None:
        """Batched terrain tensor has shape (batch, n_terrain, TERRAIN_FEATURE_DIM)."""
        env = WargameEnv(config=_make_terrain_config())
        obs1, _ = env.reset(seed=42)
        obs2, _, _, _, _ = env.step(
            WargameEnvAction(actions=list(env.action_space.sample()))
        )
        batch = observations_to_tensor_batch([obs1, obs2])
        terrain_t = batch[4]
        assert terrain_t.shape == (2, 2, TERRAIN_FEATURE_DIM)

    def test_terrain_is_static_across_steps(self) -> None:
        """Terrain tokens don't change mid-episode (footprints are static)."""
        env = WargameEnv(config=_make_terrain_config())
        obs1, _ = env.reset(seed=42)
        t1 = observation_to_tensor(obs1)[4]
        for _ in range(5):
            obs2, _, _, _, _ = env.step(
                WargameEnvAction(actions=list(env.action_space.sample()))
            )
        t2 = observation_to_tensor(obs2)[4]
        assert torch.equal(t1, t2)


# ---------------------------------------------------------------------------
# TERR-09: No-terrain → byte-identical, no mid-episode shape change
# ---------------------------------------------------------------------------


class TestNoTerrainBackwardCompat:
    def test_no_terrain_tensor_has_zero_rows(self) -> None:
        """With no terrain, terrain tensor is (0, TERRAIN_FEATURE_DIM)."""
        env = WargameEnv(config=_make_no_terrain_config())
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs)
        terrain_t = tensors[4]
        assert terrain_t.shape == (0, TERRAIN_FEATURE_DIM)

    def test_no_terrain_obs_is_empty(self) -> None:
        """With no terrain, observation.terrain is empty list."""
        env = WargameEnv(config=_make_no_terrain_config())
        obs, _ = env.reset(seed=42)
        assert obs.terrain == []
        assert obs.n_terrain == 0

    def test_no_terrain_observation_shape_unchanged(self) -> None:
        """Non-terrain tensors match between terrain and no-terrain configs
        (same board/model/objective counts), confirming no shape pollution."""
        cfg_with = WargameEnvConfig(
            board_width=50,
            board_height=50,
            number_of_wargame_models=2,
            number_of_objectives=2,
            render_mode=None,
            terrain=[TerrainPieceConfig(footprint=(10, 10, 20, 20))],
        )
        cfg_without = WargameEnvConfig(
            board_width=50,
            board_height=50,
            number_of_wargame_models=2,
            number_of_objectives=2,
            render_mode=None,
        )
        env_with = WargameEnv(config=cfg_with)
        env_without = WargameEnv(config=cfg_without)
        obs_with, _ = env_with.reset(seed=42)
        obs_without, _ = env_without.reset(seed=42)
        t_with = observation_to_tensor(obs_with)
        t_without = observation_to_tensor(obs_without)
        # game, obj, player, opponent shapes are the same
        for i in range(4):
            assert t_with[i].shape == t_without[i].shape
        # mask shape is the same
        assert t_with[5].shape == t_without[5].shape


# ---------------------------------------------------------------------------
# Network integration: terrain_embedding is None when terrain_size == 0
# ---------------------------------------------------------------------------


class TestNetworkTerrainIntegration:
    def test_transformer_no_terrain_has_no_embedding(self) -> None:
        """With no terrain, TransformerNetwork.terrain_embedding is None."""
        env = WargameEnv(config=_make_no_terrain_config())
        net = TransformerNetwork.from_env(env, is_policy=True)
        assert net.terrain_embedding is None
        assert net.terrain_size == 0

    def test_transformer_with_terrain_has_embedding(self) -> None:
        """With terrain, TransformerNetwork has a terrain embedding layer."""
        env = WargameEnv(config=_make_terrain_config())
        net = TransformerNetwork.from_env(env, is_policy=True)
        assert net.terrain_embedding is not None
        assert net.terrain_size == TERRAIN_FEATURE_DIM

    def test_transformer_forward_with_terrain(self) -> None:
        """Transformer forward pass works with terrain tokens."""
        env = WargameEnv(config=_make_terrain_config())
        net = TransformerNetwork.from_env(env, is_policy=True)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs, net.device)
        with torch.no_grad():
            logits = net(tensors)
        n_models = env.config.number_of_wargame_models
        assert logits.shape == (1, n_models, env._action_handler.n_actions)

    def test_transformer_value_with_terrain(self) -> None:
        """Transformer value network works with terrain tokens."""
        env = WargameEnv(config=_make_terrain_config())
        net = TransformerNetwork.from_env(env, is_policy=False)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs, net.device)
        with torch.no_grad():
            value = net(tensors)
        assert value.shape == (1, env.config.number_of_wargame_models)

    def test_transformer_no_terrain_forward_unchanged(self) -> None:
        """No-terrain config: transformer forward is unaffected by the pipeline change."""
        env = WargameEnv(config=_make_no_terrain_config())
        net = TransformerNetwork.from_env(env, is_policy=True)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs, net.device)
        with torch.no_grad():
            logits = net(tensors)
        n_models = env.config.number_of_wargame_models
        assert logits.shape == (1, n_models, env._action_handler.n_actions)

    def test_player_token_positions_unchanged_with_terrain(self) -> None:
        """Player tokens sit at the same positions (n_prefix..n_prefix+n_models)
        regardless of terrain, so per-model action heads are unaffected."""
        env = WargameEnv(config=_make_terrain_config())
        net = TransformerNetwork.from_env(env, is_policy=True)
        obs, _ = env.reset(seed=42)
        tensors = observation_to_tensor(obs, net.device)
        with torch.no_grad():
            state = net.encode_state(tensors)
        n_objectives = env.config.number_of_objectives
        expected_prefix = 1 + n_objectives  # game token + objective tokens
        assert state.n_prefix == expected_prefix
        assert state.n_wargame_models == env.config.number_of_wargame_models
