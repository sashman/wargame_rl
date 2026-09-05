"""The KL anchor: a frozen reference policy the run is held near.

It exists for the failure in `docs/melee-teaching-goal.md` 47 -- PPO from a
behaviour clone destroys it -- and for the diagnosis that followed: PPO
optimises the policy it rolls out (argmax, undecoded) while the policy is
scored through a joint decode, so it spends decode headroom buying unaided
skill. The tests here pin the two properties that make the anchor usable: it
is a **provable** no-op when off, and it survives the masked `-inf` logits
this network emits.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import TEST_TRANSFORMER_CONFIG
from train import _validate_kl_anchor
from wargame_rl.wargame.envs.types import WargameEnvObservation
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.ppo.config import PPOConfig
from wargame_rl.wargame.model.ppo.lightning import PPOLightning, masked_categorical_kl
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer


def _module(env: WargameEnv, kl_ref_coef: float) -> PPOLightning:
    net = PPO_Transformer.from_env(env=env, transformer_config=TEST_TRANSFORMER_CONFIG)
    return PPOLightning(
        env=env,
        ppo_model=net.to("cpu"),
        log=False,
        kl_ref_coef=kl_ref_coef,
        show_inner_progress=False,
    )


class TestOffIsANoOp:
    """At coefficient 0 nothing is built, so a control run cannot differ."""

    def test_no_reference_is_attached_at_zero(self, env: WargameEnv) -> None:
        module = _module(env, 0.0)
        module.attach_kl_reference()
        assert module._kl_reference is None

    def test_a_reference_is_attached_above_zero(self, env: WargameEnv) -> None:
        module = _module(env, 0.1)
        module.attach_kl_reference()
        assert module._kl_reference is not None

    def test_the_reference_does_not_carry_gradients(self, env: WargameEnv) -> None:
        module = _module(env, 0.1)
        module.attach_kl_reference()
        assert module._kl_reference is not None
        assert not any(p.requires_grad for p in module._kl_reference.parameters())

    def test_the_reference_is_a_copy_not_the_live_model(self, env: WargameEnv) -> None:
        """Training must not drag the anchor along behind it."""
        module = _module(env, 0.1)
        module.attach_kl_reference()
        assert module._kl_reference is not None
        assert module._kl_reference is not module.ppo_model
        live = next(module.ppo_model.parameters())
        with torch.no_grad():
            live.add_(1.0)
        frozen = next(module._kl_reference.parameters())
        assert not torch.allclose(live, frozen)


class TestTheDivergenceItself:
    """The masked `-inf` trap is the whole reason this has its own helper."""

    def test_divergence_to_itself_is_zero(self, env: WargameEnv) -> None:
        module = _module(env, 0.1)
        module.attach_kl_reference()
        observation, _ = env.reset(seed=0)
        state = _state_tensors(observation, module)
        logits, _ = module.ppo_model(state)
        divergence = module._kl_divergence_to_reference(logits, state)
        assert float(divergence.detach()) == pytest.approx(0.0, abs=1e-6)

    def test_a_masked_entry_contributes_exactly_nothing(self) -> None:
        """`0 * nan` is `nan`; an illegal action must not poison the loss.

        Synthetic rather than sampled from the network: the small test env
        emits no masked logits at all, so a test taken from it would pass
        vacuously. The reference here is the SAME row with the masked entries
        present, so the expected value is the divergence of the legal part.
        """
        legal = torch.tensor([[0.5, -0.2, 1.3]])
        with_mask = torch.tensor([[0.5, -0.2, 1.3, float("-inf")]])
        reference = torch.tensor([[0.1, 0.4, -0.7, float("-inf")]])
        reference_legal = reference[:, :3]
        masked = masked_categorical_kl(with_mask, reference)
        unmasked = masked_categorical_kl(legal, reference_legal)
        assert torch.isfinite(masked)
        assert float(masked) == pytest.approx(float(unmasked), abs=1e-6)

    def test_two_masked_rows_are_not_nan(self) -> None:
        both = torch.tensor([[1.0, float("-inf"), 0.0, float("-inf")]])
        other = torch.tensor([[0.2, float("-inf"), 0.9, float("-inf")]])
        assert torch.isfinite(masked_categorical_kl(both, other))

    def test_it_is_zero_for_identical_rows_and_positive_otherwise(self) -> None:
        row = torch.tensor([[0.3, -1.0, 2.0, float("-inf")]])
        assert float(masked_categorical_kl(row, row)) == pytest.approx(0.0, abs=1e-7)
        other = torch.tensor([[1.1, 0.2, -0.4, float("-inf")]])
        assert float(masked_categorical_kl(row, other)) > 0.0

    def test_divergence_grows_as_the_policy_moves_away(self, env: WargameEnv) -> None:
        module = _module(env, 0.1)
        module.attach_kl_reference()
        observation, _ = env.reset(seed=0)
        state = _state_tensors(observation, module)
        divergences = []
        for _ in range(2):
            with torch.no_grad():
                for parameter in module.ppo_model.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.05)
            logits, _ = module.ppo_model(state)
            divergences.append(float(module._kl_divergence_to_reference(logits, state)))
        assert divergences[1] > divergences[0]

    def test_the_divergence_pulls_the_policy_back(self, env: WargameEnv) -> None:
        """Descending the KL alone must reduce it -- the sign is load-bearing."""
        module = _module(env, 1.0)
        module.attach_kl_reference()
        observation, _ = env.reset(seed=0)
        state = _state_tensors(observation, module)
        with torch.no_grad():
            for parameter in module.ppo_model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.05)
        optimizer = torch.optim.SGD(module.ppo_model.parameters(), lr=1e-2)
        logits, _ = module.ppo_model(state)
        before = module._kl_divergence_to_reference(logits, state)
        optimizer.zero_grad()
        before.backward()
        optimizer.step()
        logits, _ = module.ppo_model(state)
        after = module._kl_divergence_to_reference(logits, state)
        assert float(after) < float(before)


class TestTheAdaptiveController:
    """The coefficient nobody can pick a priori, replaced by a target."""

    @pytest.mark.parametrize(
        ("target", "drift", "expected"),
        [
            # Drifting further than asked -- pull harder.
            (0.3, 1.0, 0.2),
            # Staying closer than asked -- let go.
            (0.3, 0.05, 0.05),
            # Inside the band -- leave it alone.
            (0.3, 0.3, 0.1),
            (0.3, 0.35, 0.1),
        ],
    )
    def test_it_moves_the_coefficient_the_right_way(
        self, env: WargameEnv, target: float, drift: float, expected: float
    ) -> None:
        module = _module(env, 0.1)
        module.kl_ref_target = target
        module._adapt_kl_coefficient(drift)
        assert module.kl_ref_coef == pytest.approx(expected)

    def test_a_zero_target_leaves_the_coefficient_fixed(self, env: WargameEnv) -> None:
        module = _module(env, 0.1)
        module.kl_ref_target = 0.0
        for drift in (0.0, 5.0, 100.0):
            module._adapt_kl_coefficient(drift)
        assert module.kl_ref_coef == pytest.approx(0.1)

    def test_it_converges_onto_the_target_band(self, env: WargameEnv) -> None:
        """A run whose drift responds to the weight must settle, not oscillate.

        Stands in for the controller's only real job. `drift = 1.0 / coef` is a
        crude response model, but any monotone decreasing one exercises the
        same fixed point.
        """
        module = _module(env, 1e-3)
        module.kl_ref_target = 0.3
        for _ in range(40):
            module._adapt_kl_coefficient(1.0 / module.kl_ref_coef)
        settled = 1.0 / module.kl_ref_coef
        assert 0.3 / 1.5 <= settled <= 0.3 * 1.5

    def test_the_coefficient_cannot_run_away_in_either_direction(
        self, env: WargameEnv
    ) -> None:
        module = _module(env, 1.0)
        module.kl_ref_target = 0.3
        for _ in range(200):
            module._adapt_kl_coefficient(99.0)
        assert module.kl_ref_coef <= 1e4
        for _ in range(400):
            module._adapt_kl_coefficient(0.0)
        assert module.kl_ref_coef >= 1e-4


class TestTheConfigReachesTheModule:
    """`PPOLightning` takes `**kwargs`, so a config key it ignores is silent.

    That trap fired once during this feature's own development: a patch that
    added the field to `PPOConfig` but not to the module's signature left the
    option accepted, logged into wandb, and doing nothing.
    """

    @pytest.mark.parametrize("field", ["kl_ref_coef", "kl_ref_target"])
    def test_the_module_consumes_the_field(self, env: WargameEnv, field: str) -> None:
        config = PPOConfig()
        setattr(config, field, 0.25)
        net = PPO_Transformer.from_env(
            env=env, transformer_config=TEST_TRANSFORMER_CONFIG
        )
        # `log` comes from the config too, so it is not passed separately.
        module = PPOLightning(env=env, ppo_model=net.to("cpu"), **config.model_dump())
        assert getattr(module, field) == pytest.approx(0.25)


class TestTheStartupGuard:
    def test_an_anchor_without_a_warm_start_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requires warm_start_ckpt_path"):
            _validate_kl_anchor(0.1, None)

    @pytest.mark.parametrize(
        ("coefficient", "warm_start"),
        [(None, None), (0.0, None), (0.1, "some.ckpt")],
    )
    def test_every_other_combination_is_allowed(
        self, coefficient: float | None, warm_start: str | None
    ) -> None:
        _validate_kl_anchor(coefficient, warm_start)


def _state_tensors(
    observation: WargameEnvObservation, module: PPOLightning
) -> list[torch.Tensor]:
    from wargame_rl.wargame.model.common.observation import observation_to_tensor

    encoded = observation_to_tensor(observation, torch.device("cpu"))
    return list(encoded)


class TestResumingAnAnchoredRun:
    """The reference is a submodule, so it lands in the checkpoint.

    `attach_kl_reference` only runs on the warm-start path, and warm start and
    resume are mutually exclusive -- so without a slot rebuilt at load time the
    restore fails with `Unexpected key(s) in state_dict: "_kl_reference...."`
    and every anchored run is un-resumable. Caught by trying to resume one.
    """

    def test_the_reference_is_saved_with_the_state_dict(self, env: WargameEnv) -> None:
        module = _module(env, 0.1)
        module.attach_kl_reference()
        keys = module.state_dict().keys()
        assert any(k.startswith("_kl_reference.") for k in keys)

    def test_a_checkpoint_carrying_a_reference_rebuilds_the_slot(
        self, env: WargameEnv
    ) -> None:
        saved = _module(env, 0.1)
        saved.attach_kl_reference()
        checkpoint = {"state_dict": saved.state_dict()}
        fresh = _module(env, 0.1)
        assert fresh._kl_reference is None
        fresh.on_load_checkpoint(checkpoint)
        assert fresh._kl_reference is not None
        fresh.load_state_dict(checkpoint["state_dict"])

    def test_a_checkpoint_without_one_is_left_alone(self, env: WargameEnv) -> None:
        """A run that never had an anchor must not grow one on resume."""
        plain = _module(env, 0.0)
        checkpoint = {"state_dict": plain.state_dict()}
        fresh = _module(env, 0.0)
        fresh.on_load_checkpoint(checkpoint)
        assert fresh._kl_reference is None
        fresh.load_state_dict(checkpoint["state_dict"])
