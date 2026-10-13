"""The per-model behaviour clone (#331): the inverse column map round-trips
through the agent's decoder on every recorded decision, the fit raises the
held-out match, and the saved `.pt` loads and plays.

Driven on the small two-units-a-side scenario with `squad_march_take` as the
teacher, on the CPU: the point is the plumbing, not the fidelity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.tokens import Head, TokenScenario, build_tokens
from wargame_rl.wargame.model.per_model.checkpoint import load_checkpoint
from wargame_rl.wargame.model.per_model.clone import (
    explained_variance,
    fit_clone,
    fit_critic,
    match_report,
    record_demonstrations,
    save_clone,
    value_targets,
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
    policy = [t for t in demos if t.has_policy]
    closing = [t for t in demos if not t.has_policy]
    assert policy and all(t.model >= 0 for t in policy)
    assert closing and all(t.is_close for t in closing)
    assert sum(t.done for t in demos) == 2


def test_the_critic_fit_moves_the_value_head_and_nothing_else() -> None:
    # Arrange: recorded games with their returns, a small network.
    torch.manual_seed(1)
    env = PerModelEnv(small_config(rounds=2))
    chooser = build_per_model_chooser("squad_march_take", [env], seed=0)
    demos = record_demonstrations(env, chooser.choose, 3)
    targets = value_targets(demos, gamma=0.9)
    network = _small_network(env)
    before = {k: v.clone() for k, v in network.state_dict().items()}
    ev_before = explained_variance(network, demos, targets)
    # Act
    losses = fit_critic(network, demos, targets, epochs=8, seed=0, lr=3e-3)
    ev_after = explained_variance(network, demos, targets)
    # Assert: the loss fell, the fit explains more, and only the value head moved.
    assert losses[-1] < losses[0]
    assert ev_after > ev_before
    for key, value in network.state_dict().items():
        if key.startswith("value_head"):
            continue
        assert torch.equal(value, before[key]), key


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


def test_compact_tokens_halve_the_float_arrays_and_collate_back_to_float32() -> None:
    """The recorded demonstration is stored at half precision (a 24-body army's
    relation block is ~37 KB per decision in float32) and read back widened."""
    from wargame_rl.wargame.model.per_model.batch import collate
    from wargame_rl.wargame.model.per_model.clone import compact_tokens

    env = PerModelEnv(small_config())
    observation, _ = env.reset(seed=3)
    scenario = TokenScenario.for_episode(env, env.player_seat)
    tokens = build_tokens(env, env.player_seat, observation, scenario)
    compact = compact_tokens(tokens)

    assert compact.self_relations.dtype == np.float16
    assert compact.self_relations.nbytes * 2 == tokens.self_relations.nbytes
    assert compact.selector_mask is tokens.selector_mask
    wide = collate([tokens], device=torch.device("cpu"))
    narrow = collate([compact], device=torch.device("cpu"))
    assert narrow.self_relations.dtype == torch.float32
    torch.testing.assert_close(
        narrow.self_relations, wide.self_relations, rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(narrow.players, wide.players, rtol=2e-3, atol=2e-3)
