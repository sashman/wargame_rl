"""The per-model behaviour clone (#331): the inverse column map round-trips
through the agent's decoder on every recorded decision, the fit raises the
held-out match, and the saved `.pt` loads and plays.

Driven on the small two-units-a-side scenario with `squad_march_take` as the
teacher, on the CPU: the point is the plumbing, not the fidelity.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.tokens import Head
from wargame_rl.wargame.model.per_model.checkpoint import load_checkpoint
from wargame_rl.wargame.model.per_model.clone import (
    fit_clone,
    match_report,
    record_demonstrations,
    save_clone,
)
from wargame_rl.wargame.model.per_model.net import SetNetwork, SetNetworkConfig
from wargame_rl.wargame.selectors import build_per_model_chooser


def _small_network(env: PerModelEnv) -> SetNetwork:
    return SetNetwork.from_env(env, SetNetworkConfig(n_layers=1, embedding_size=32))


def test_recorded_decisions_round_trip_and_cover_every_head() -> None:
    # Arrange: a shooting scenario, the take script on the player seat.
    env = PerModelEnv(small_config(rounds=2))
    chooser = build_per_model_chooser("squad_march_take", [env], seed=0)
    # Act: the recorder decodes every recorded pair back and compares.
    demos = record_demonstrations(env, chooser.choose, 2)
    # Assert: something was recorded, per episode, over both heads a
    # movement-and-shooting game uses, and no closing step slipped in.
    assert demos and {t.env_index for t in demos} == {0, 1}
    assert {t.head for t in demos} >= {Head.declaration, Head.displacement}
    assert all(t.column >= 0 and t.model >= 0 for t in demos)


def test_the_fit_raises_the_held_out_match_and_the_clone_plays(
    tmp_path: Path,
) -> None:
    # Arrange
    torch.manual_seed(0)
    env = PerModelEnv(small_config(rounds=2))
    chooser = build_per_model_chooser("squad_march_take", [env], seed=0)
    demos = record_demonstrations(env, chooser.choose, 4)
    train = [t for t in demos if t.env_index < 3]
    test = [t for t in demos if t.env_index == 3]
    network = _small_network(env)
    before = match_report(network, test)
    # Act
    losses = fit_clone(network, train, epochs=6, seed=0, batch_size=32, lr=3e-3)
    after = match_report(network, test)
    # Assert: the loss fell and the joint held-out match rose.
    assert losses[-1] < losses[0]
    assert after["joint"] > before["joint"]
    # Assert: the saved checkpoint loads with the clone's weights and plays.
    out = tmp_path / "clone.pt"
    save_clone(
        out,
        network,
        env_config=env.config.model_dump(mode="json"),
        seed=0,
        revision="test",
        teacher="squad_march_take",
        n_episodes=4,
        epochs=6,
        match=after,
    )
    loaded = load_checkpoint(out)
    assert loaded.rounds == 0
    assert (out.with_suffix(".clone.json")).exists()
    player = build_per_model_chooser(str(out), [PerModelEnv(env.config)], seed=0)
    assert player.kind == "checkpoint"
