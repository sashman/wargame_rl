"""The set network: size-independent by construction, and the masks are the env's.

The receipts for Principle 2, each on the network alone: one instance plays
two differently sized scenarios with no reload; its state dict has the same
shapes whichever scenario it was built against; padding an observation into
a larger batch changes nothing it outputs; a dead model's features reach
nobody. Then the contract with the facade: the selector offers exactly the
env's selector mask and each head masks exactly what the env forbids. And
the size pin, so a fixture cannot shrink the production default.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.per_model_seats import random_legal_action, small_config
from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.envs.per_model.tokens import (
    Head,
    TokenObservation,
    TokenScenario,
    build_tokens,
)
from wargame_rl.wargame.model.per_model import SetNetwork, SetNetworkConfig, collate

SMALL_TRUNK = SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)


def _tokens(env: PerModelEnv, seed: int = 1, steps: int = 0) -> TokenObservation:
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        observation, *_rest = env.step(random_legal_action(observation.decision, rng))
    seat = env.player_seat
    return build_tokens(env, seat, observation, TokenScenario.for_episode(env, seat))


def _network(env: PerModelEnv, seed: int = 0) -> SetNetwork:
    torch.manual_seed(seed)
    network = SetNetwork.from_env(env, SMALL_TRUNK)
    network.eval()
    return network


def _large_env() -> PerModelEnv:
    return PerModelEnv(
        small_config(n_models=10, group_ids=[i // 2 for i in range(10)], max_groups=5)
    )


def test_one_set_of_weights_serves_two_scenario_sizes() -> None:
    small_env = PerModelEnv(small_config())
    network = _network(small_env)
    for env in (small_env, _large_env()):
        tokens = _tokens(env)
        batch = collate([tokens])
        output = network(batch)
        assert output.player_latents.shape[1] == tokens.n_players
        assert output.value.shape == (1,)
        model = int(torch.argmax(output.selector_logits[0]))
        heads = network.heads(output, batch, torch.tensor([model]))
        assert heads.unit.shape == (1, 1 + tokens.n_units)
        assert torch.isfinite(
            heads.declaration[0][tokens.declaration_mask[model]]
        ).all()


@pytest.mark.parametrize(
    "n_models, max_groups",
    [(6, 2), (10, 5), (3, 1)],
)
def test_state_dict_shapes_do_not_depend_on_the_scenario(
    n_models: int, max_groups: int
) -> None:
    reference = {
        key: tuple(value.shape)
        for key, value in _network(PerModelEnv(small_config())).state_dict().items()
    }
    env = PerModelEnv(
        small_config(
            n_models=n_models,
            group_ids=[i % max_groups for i in range(n_models)],
            max_groups=max_groups,
        )
    )
    shapes = {
        key: tuple(value.shape) for key, value in _network(env).state_dict().items()
    }
    assert shapes == reference


def test_padding_into_a_larger_batch_changes_nothing() -> None:
    """A six-model observation alone, and beside a ten-model one, agrees."""
    small_env = PerModelEnv(small_config())
    network = _network(small_env)
    small = _tokens(small_env, steps=2)
    large = _tokens(_large_env(), steps=3)
    alone = network(collate([small]))
    padded = network(collate([small, large]))
    p = small.n_players
    assert torch.allclose(
        alone.player_latents[0], padded.player_latents[0, :p], atol=1e-5
    )
    assert torch.allclose(alone.selector_logits[0], padded.selector_logits[0, :p])
    assert torch.allclose(alone.value, padded.value[:1], atol=1e-5)
    model = int(torch.argmax(alone.selector_logits[0]))
    heads_alone = network.heads(alone, collate([small]), torch.tensor([model]))
    heads_padded = network.heads(
        padded, collate([small, large]), torch.tensor([model, 0])
    )
    u = small.n_units
    assert torch.allclose(heads_alone.unit[0], heads_padded.unit[0, : 1 + u], atol=1e-5)
    assert torch.allclose(
        heads_alone.declaration[0], heads_padded.declaration[0], atol=1e-5
    )
    assert torch.allclose(
        heads_alone.displacement[0], heads_padded.displacement[0], atol=1e-5
    )
    assert not torch.isfinite(heads_padded.unit[0, 1 + u :]).any(), (
        "pad columns are masked"
    )


def test_a_dead_models_features_reach_nobody() -> None:
    env = PerModelEnv(small_config())
    network = _network(env)
    observation, _ = env.reset(seed=1)
    env.wargame_models[2].stats["current_wounds"] = 0
    seat = env.player_seat
    built = build_tokens(env, seat, observation, TokenScenario.for_episode(env, seat))
    # The point predates the corpse, so its selector still names model 2; the
    # env would never do that, and the network's key mask is what is on trial.
    selector = built.selector_mask.copy()
    selector[2] = False
    tokens = TokenObservation(**{**built.__dict__, "selector_mask": selector})
    perturbed = TokenObservation(
        **{
            **tokens.__dict__,
            "players": _perturb_row(tokens.players, 2),
            "self_relations": _perturb_row(tokens.self_relations, 2),
        }
    )
    a = network(collate([tokens]))
    b = network(collate([perturbed]))
    alive = torch.from_numpy(tokens.player_alive)
    assert torch.allclose(
        a.player_latents[0][alive], b.player_latents[0][alive], atol=1e-6
    )
    assert torch.allclose(a.selector_logits[0][alive], b.selector_logits[0][alive])
    assert torch.allclose(a.value, b.value, atol=1e-6)
    assert not torch.isfinite(a.selector_logits[0, 2]), "a corpse is never selectable"


def _perturb_row(array: np.ndarray, row: int) -> np.ndarray:
    out = array.copy()
    out[row] = out[row] + np.float32(3.0)
    return out


def test_the_selector_offers_exactly_the_envs_mask() -> None:
    env = PerModelEnv(small_config(melee=True, opponent_x=22, rounds=2))
    network = _network(env)
    observation, _ = env.reset(seed=3)
    rng = np.random.default_rng(3)
    scenario = TokenScenario.for_episode(env, env.player_seat)
    done = False
    checked = 0
    while not done:
        tokens = build_tokens(env, env.player_seat, observation, scenario)
        output = network(collate([tokens]))
        finite = torch.isfinite(output.selector_logits[0]).numpy()
        assert np.array_equal(finite, observation.decision.selector_mask)
        checked += 1
        observation, _r, done, _t, _i = env.step(
            random_legal_action(observation.decision, rng)
        )
    assert checked > 10


def test_every_head_masks_what_the_env_forbids() -> None:
    env = PerModelEnv(small_config(melee=True, opponent_x=22, rounds=2))
    network = _network(env)
    observation, _ = env.reset(seed=6)
    rng = np.random.default_rng(6)
    scenario = TokenScenario.for_episode(env, env.player_seat)
    heads_seen: set[Head] = set()
    done = False
    while not done:
        tokens = build_tokens(env, env.player_seat, observation, scenario)
        batch = collate([tokens])
        output = network(batch)
        for model in np.flatnonzero(tokens.selector_mask):
            heads = network.heads(output, batch, torch.tensor([int(model)]))
            assert np.array_equal(
                torch.isfinite(heads.declaration[0]).numpy(),
                tokens.declaration_mask[model],
            )
            assert np.array_equal(
                torch.isfinite(heads.displacement[0]).numpy(),
                tokens.displacement_mask[model],
            )
            assert np.array_equal(
                torch.isfinite(heads.unit[0]).numpy(), tokens.unit_mask[model]
            )
        heads_seen.add(tokens.head)
        observation, _r, done, _t, _i = env.step(
            random_legal_action(observation.decision, rng)
        )
    assert {Head.declaration, Head.displacement, Head.unit_pointer} <= heads_seen


def test_a_row_with_no_living_player_still_gives_finite_outputs() -> None:
    env = PerModelEnv(small_config())
    network = _network(env)
    observation, _ = env.reset(seed=1)
    for model in env.wargame_models:
        model.stats["current_wounds"] = 0
    seat = env.player_seat
    tokens = build_tokens(env, seat, observation, TokenScenario.for_episode(env, seat))
    output = network(collate([tokens]))
    assert torch.isfinite(output.player_latents).all()
    assert torch.isfinite(output.value).all()


# ----------------------------------------------------------------- the size


def test_the_default_trunk_is_four_layers_of_128_with_eight_heads() -> None:
    config = SetNetworkConfig()
    assert (config.n_layers, config.embedding_size, config.n_heads) == (4, 128, 8)
    env = PerModelEnv(small_config())
    default = SetNetwork.from_env(env)
    explicit = SetNetwork.from_env(env, SetNetworkConfig())
    assert _count(default) == _count(explicit)
    assert default.config == SetNetworkConfig()


def test_the_small_trunk_is_much_smaller_and_still_produces_finite_logits() -> None:
    env = PerModelEnv(small_config())
    small = _network(env)
    production = SetNetwork.from_env(env)
    assert _count(small) * 20 < _count(production)
    tokens = _tokens(env)
    output = small(collate([tokens]))
    assert torch.isfinite(output.selector_logits[0][tokens.selector_mask]).all()
    assert torch.isfinite(output.value).all()


def test_a_width_the_heads_cannot_divide_is_refused() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SetNetworkConfig(embedding_size=100, n_heads=8)


def _count(network: torch.nn.Module) -> int:
    return sum(p.numel() for p in network.parameters())
