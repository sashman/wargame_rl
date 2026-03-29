"""End-to-end test: run the training entrypoint (PPO) for one epoch.

Collected last so that faster unit/integration tests run first (file name sorts after test_*).
"""

import os
import sys


def test_training_smoke() -> None:
    """Smoke test: call train() for one epoch with no wandb; assert it completes without raising."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from train import AlgorithmType, train
    from wargame_rl.wargame.model.dqn.config import NetworkType

    env_config_path = os.path.join(
        project_root, "examples", "env_config", "4_models_2_objectives_fixed.yaml"
    )

    train(
        render_mode=None,
        env_config_path=env_config_path,
        algorithm=AlgorithmType.PPO,
        network_type=NetworkType.TRANSFORMER,
        record_during_training=False,
        record_after_epoch=10,
        record_every_n_epochs=20,
        max_epochs=1,
        no_wandb=True,
    )
