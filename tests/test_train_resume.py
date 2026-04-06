from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pytorch_lightning import LightningModule

import train as train_module
from train import (
    _apply_warm_start_weights,
    _fit_with_optional_resume,
    _validate_checkpoint_mode,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.dqn.lightning import DQNLightning
from wargame_rl.wargame.model.net import MLPNetwork
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer


def _make_env() -> WargameEnv:
    return WargameEnv(
        config=WargameEnvConfig(
            board_width=12,
            board_height=12,
            number_of_wargame_models=2,
            number_of_objectives=1,
            objective_radius_size=1,
            number_of_battle_rounds=2,
            render_mode=None,
        )
    )


def _set_all_params(module: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.fill_(value)


def test_validate_checkpoint_mode_rejects_conflicting_flags() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_checkpoint_mode("a.ckpt", "b.ckpt")


def test_fit_with_optional_resume_passes_ckpt_path() -> None:
    calls: list[str | None] = []

    class DummyTrainer:
        def fit(
            self, model: LightningModule, ckpt_path: str | None = None
        ) -> None:  # pragma: no cover - trivial shim
            del model
            calls.append(ckpt_path)

    class DummyModule(LightningModule):
        pass

    trainer = DummyTrainer()
    model = DummyModule()
    _fit_with_optional_resume(trainer, model, None)  # type: ignore[arg-type]
    _fit_with_optional_resume(trainer, model, "resume.ckpt")  # type: ignore[arg-type]
    assert calls == [None, "resume.ckpt"]


def test_warm_start_loads_ppo_weights_only(tmp_path: Path) -> None:
    env = _make_env()
    source = PPOLightning(env=env, ppo_model=PPO_Transformer.from_env(env))
    target = PPOLightning(env=env, ppo_model=PPO_Transformer.from_env(env))
    _set_all_params(source.ppo_model, 0.1234)

    ckpt_path = tmp_path / "ppo.ckpt"
    torch.save({"state_dict": source.state_dict()}, ckpt_path)
    _apply_warm_start_weights(target, str(ckpt_path))

    source_param = next(source.ppo_model.parameters()).detach().clone()
    target_param = next(target.ppo_model.parameters()).detach().clone()
    assert torch.allclose(target_param, source_param)
    env.close()


def test_warm_start_loads_dqn_weights_and_syncs_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DQNLightning, "populate", lambda self: None)
    env = _make_env()
    source_net = MLPNetwork.policy_from_env(env)
    target_net = MLPNetwork.policy_from_env(env)
    source = DQNLightning(env=env, policy_net=source_net)
    target = DQNLightning(env=env, policy_net=target_net)
    _set_all_params(source.policy_net, 0.5678)

    ckpt_path = tmp_path / "dqn.ckpt"
    torch.save({"state_dict": source.state_dict()}, ckpt_path)
    _apply_warm_start_weights(target, str(ckpt_path))

    policy_param = next(target.policy_net.parameters()).detach().clone()
    source_param = next(source.policy_net.parameters()).detach().clone()
    target_param = next(target.target_net.parameters()).detach().clone()
    assert torch.allclose(policy_param, source_param)
    assert torch.allclose(target_param, policy_param)
    env.close()


def test_train_forwards_resume_ckpt_to_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_ckpt = tmp_path / "resume.ckpt"
    torch.save({"state_dict": {}}, resume_ckpt)

    fit_calls: list[str | None] = []

    class DummyTrainer:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fit(self, model: LightningModule, ckpt_path: str | None = None) -> None:
            del model
            fit_calls.append(ckpt_path)

    class DummyModel(LightningModule):
        pass

    monkeypatch.setattr(train_module, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        train_module, "create_environment", lambda env_config: _make_env()
    )
    monkeypatch.setattr(
        train_module,
        "PPO_Transformer",
        type("X", (), {"from_env": staticmethod(lambda env: object())}),
    )
    monkeypatch.setattr(train_module, "PPOLightning", lambda **kwargs: DummyModel())

    train_module.train(
        env_config_path=None,
        algorithm=train_module.AlgorithmType.PPO,
        network_type=train_module.NetworkType.TRANSFORMER,
        no_wandb=True,
        record_during_training=False,
        record_after_epoch=10,
        record_every_n_epochs=20,
        no_inner_progress=False,
        resume_ckpt_path=str(resume_ckpt),
        max_epochs=1,
        n_steps=8,
        n_eval_episodes=1,
    )

    assert fit_calls == [str(resume_ckpt)]
