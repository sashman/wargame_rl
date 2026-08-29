"""The production trunk is 8 layers of 256 at 8 heads, and shrinking is opt-in.

`TransformerNetwork.from_spec` takes a `transformer_config` so a test can build
a small trunk -- the fixtures in `conftest.py` do, because the suite was paying
~12.7M parameters on a 2-model 20x20 board and it dominated both the runtime and
the 153 MB a checkpoint costs.

⚠ **That makes the production default worth pinning.** A default nobody asserts
is a default somebody eventually "optimises", and here it would silently orphan
every checkpoint in `checkpoints/` and void every score in `CLAUDE.md` -- the
weights would simply be the wrong shape, and only a `load_state_dict` far from
the change would say so.

The other half is that `None` must stay *exactly* the old path: every score on
file was measured through a call that passed no config at all, so `None` and an
explicit `TransformerConfig()` have to build the identical network.
"""

from __future__ import annotations

import pytest
import torch

from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.config import TransformerConfig
from wargame_rl.wargame.model.net import TransformerNetwork


@pytest.fixture
def small_env() -> WargameEnv:
    return WargameEnv(config=WargameEnvConfig(render_mode=None))


def _parameter_count(network: TransformerNetwork) -> int:
    return sum(p.numel() for p in network.parameters())


class TestTheProductionDefaultIsPinned:
    def test_the_shipped_trunk_is_still_eight_layers_of_256(self) -> None:
        """Every checkpoint in `checkpoints/` has this shape baked in."""
        defaults = TransformerConfig()

        assert (defaults.n_layers, defaults.n_heads, defaults.embedding_size) == (
            8,
            8,
            256,
        )

    def test_omitting_the_config_builds_the_production_network(
        self, small_env: WargameEnv
    ) -> None:
        """`None` is the path every recorded score was measured through.

        Asserted on the parameter count rather than on the config object,
        because what a checkpoint has to match is the *shape*.
        """
        torch.manual_seed(0)
        implicit = TransformerNetwork.policy_from_env(env=small_env)
        torch.manual_seed(0)
        explicit = TransformerNetwork.policy_from_env(
            env=small_env, transformer_config=TransformerConfig()
        )

        assert _parameter_count(implicit) == _parameter_count(explicit)
        assert implicit.config.model_dump() == TransformerConfig().model_dump()


class TestShrinkingIsRealAndOptIn:
    def test_a_small_trunk_is_dramatically_smaller(self, small_env: WargameEnv) -> None:
        """The whole point: the fixtures' trunk must actually be cheap.

        Two orders of magnitude, so a regression that silently restored the
        production size in `conftest.py` fails here rather than only showing up
        as a slower CI run nobody attributes.
        """
        production = TransformerNetwork.policy_from_env(env=small_env)
        small = TransformerNetwork.policy_from_env(
            env=small_env,
            transformer_config=TransformerConfig(n_layers=2, embedding_size=32),
        )

        assert _parameter_count(small) * 50 < _parameter_count(production)

    def test_the_fixtures_use_the_small_trunk(self, policy_net: object) -> None:
        """Guards the fixtures themselves, which is where the saving lives."""
        from tests.conftest import TEST_TRANSFORMER_CONFIG

        assert isinstance(policy_net, TransformerNetwork)
        assert policy_net.config.model_dump() == TEST_TRANSFORMER_CONFIG.model_dump()

    def test_a_shrunk_checkpoint_loads_back_at_its_own_shape(
        self, small_env: WargameEnv, tmp_path: object
    ) -> None:
        """A size that can be written must be a size that can be read.

        `from_state_dict` used to build the production trunk and load into it,
        so any non-default checkpoint failed with a wall of missing keys naming
        every layer and never naming the cause. That would have made
        `--n-layers` / `--embedding-size` a footgun for `simulate` and every
        `measure-*` recipe, which all load through this path.
        """
        import torch as _torch

        small = TransformerConfig(n_layers=2, embedding_size=32)
        written = TransformerNetwork.policy_from_env(
            env=small_env, transformer_config=small
        )

        loaded = TransformerNetwork.from_state_dict(small_env, written.state_dict())

        assert loaded.config.n_layers == 2
        assert loaded.config.embedding_size == 32
        for key, value in written.state_dict().items():
            assert _torch.equal(loaded.state_dict()[key].cpu(), value.cpu())

    def test_a_default_checkpoint_still_infers_nothing(
        self, small_env: WargameEnv
    ) -> None:
        """Weights already at the default must return None, not an equal object.

        None is the path every recorded score was loaded through.
        """
        from wargame_rl.wargame.model.net import trunk_config_from_state_dict

        production = TransformerNetwork.policy_from_env(env=small_env)

        assert trunk_config_from_state_dict(production.state_dict()) is None

    def test_a_width_the_default_heads_cannot_divide_is_refused(self) -> None:
        """Head count is not recorded, so a width it cannot divide is unloadable.

        Better to say so than to build a trunk that loads and computes wrongly.
        """
        from wargame_rl.wargame.model.net import trunk_config_from_state_dict

        state_dict = {
            "game_embedding.weight": torch.zeros(12, 6),
            "transformer.h.0.ln_1.weight": torch.zeros(12),
        }

        with pytest.raises(ValueError, match="heads do not divide"):
            trunk_config_from_state_dict(state_dict)

    def test_a_small_network_still_produces_usable_logits(
        self, small_env: WargameEnv
    ) -> None:
        """A trunk too small to work would make every fixture-based test vacuous."""
        network = TransformerNetwork.policy_from_env(
            env=small_env,
            transformer_config=TransformerConfig(n_layers=2, embedding_size=32),
        )
        observation, _info = small_env.reset(seed=7)
        from wargame_rl.wargame.model.common.observation import observation_to_tensor

        logits = network(observation_to_tensor(observation))

        assert torch.isfinite(logits).all()
