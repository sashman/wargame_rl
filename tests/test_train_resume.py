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
        train_module,
        "create_environment",
        lambda env_config, **kwargs: _make_env(),
    )
    monkeypatch.setattr(
        train_module,
        "PPO_Transformer",
        # Takes the PPO config too: it decides whether the model is built with a
        # shooting slice, and therefore whether it decodes targets autoregressively.
        type("X", (), {"from_env": staticmethod(lambda env, config=None: object())}),
    )
    monkeypatch.setattr(train_module, "PPOLightning", lambda **kwargs: DummyModel())

    train_module.train(
        env_config_path=None,
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


def test_resume_can_load_a_checkpoint_that_pickles_the_env() -> None:
    """⚠ Regression: `--resume-ckpt-path` was broken for EVERY checkpoint here.

    PyTorch 2.6 flipped `torch.load`'s `weights_only` default to True, and these
    checkpoints pickle the whole `WargameEnv` as a Lightning hparam, so
    Lightning's internal restore raised `UnpicklingError: Unsupported global ...
    WargameEnv`. The run died in seconds and the launcher still exited 0, which
    is the silent-failure shape this project has been bitten by before.
    """
    import torch

    from train import _trusted_checkpoint_load

    class _OnlyWithWeightsOnlyFalse:
        pass

    original = torch.load
    seen: dict[str, object] = {}

    def fake_load(*args: object, **kwargs: object) -> str:
        seen["weights_only"] = kwargs.get("weights_only")
        return "loaded"

    torch.load = fake_load  # type: ignore[assignment]
    try:
        with _trusted_checkpoint_load():
            assert torch.load("anything.ckpt") == "loaded"
        assert seen["weights_only"] is False, (
            "the resume path must not use torch 2.6's weights_only default"
        )
        # ...and the patch must be undone, or every later load is unguarded.
        assert torch.load is fake_load
    finally:
        torch.load = original  # type: ignore[assignment]
