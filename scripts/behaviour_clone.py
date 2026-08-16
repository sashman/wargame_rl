"""Clone a scripted baseline into the policy network, so PPO starts in its basin.

**Why this exists.** On the real tables the trained agent trails
`squad_march_shoot` by ~30 vp_margin, and three config-level levers have been
measured null (see
`reports/2026-08-16-the-cap-makes-it-a-denial-game.md`). The reason is a
coordination gap, not a mispricing, and it shows up as two facts that are only
consistent if the agent sits in a *local* optimum:

- A **unilateral** squad advance is punished: a properly paired teleport audit
  measured the moved squad losing **29.4 of its own income** and 1.7 of 5
  models. PPO trains on the per-model vector, so every gradient step toward
  advancing is downhill.
- The **joint** advancing policy is better *by the agent's own training reward*:
  `squad_march_shoot` earns **30.29** an episode against the agent's **24.77**,
  ahead on every calculator.

Gradient descent cannot cross that: the path is downhill and the destination is
a discrete distance away. Behaviour cloning crosses it in one step, and because
the destination scores *higher* on the training reward, PPO warm-started there
has no incentive to drift back.

Usage: just behaviour-clone <policy> <env_config> [n_episodes] [epochs] [out]

The output is a checkpoint `train.py --warm-start-ckpt-path` accepts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from pydantic_yaml import parse_yaml_raw_as
from torch import nn

from wargame_rl.wargame.envs.baseline.evaluate import selector_for
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.device import auto_device
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.model.net import TransformerNetwork

# Disjoint from every other seed band in the repo: evaluation uses 700000+,
# in-run eval 500000+, the logged baselines 10000+. A clone trained on the
# evaluation layouts would make the held-out score meaningless.
CLONE_SEED_BASE = 800_000

# The prefix `PPOLightning.load_state_dict` expects. `_apply_warm_start_weights`
# loads with `strict=False`, so a wrong prefix here loads **nothing at all** and
# trains a random network while the run claims a warm start -- which is why
# `main` verifies the key overlap instead of trusting it.
POLICY_PREFIX = "ppo_model.policy_network."


def collect(
    policy_name: str, config: WargameEnvConfig, n_episodes: int
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Play `policy_name` and record what it saw and what it did.

    Returns the five state tensors stacked over steps, the action mask, and the
    action the scripted policy chose for each model.

    Every step is kept, movement and shooting alike: the phase is part of the
    game-feature vector, so one network learns both and the clone reproduces the
    whole policy rather than half of it.
    """
    env = create_environment(env_config=config)
    select = selector_for(build_baseline_policy(policy_name))

    states: list[list[torch.Tensor]] = []
    masks: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []

    for index in range(n_episodes):
        observation, _ = env.reset(seed=CLONE_SEED_BASE + index)
        terminated = truncated = False
        while not (terminated or truncated):
            tensors = observation_to_tensor(observation)
            action = select(observation, env)
            states.append([t.detach().clone() for t in tensors[:5]])
            masks.append(tensors[5].detach().clone())
            actions.append(torch.tensor(action.actions, dtype=torch.long))
            observation, _r, terminated, truncated, _i = env.step(action)
        if (index + 1) % 25 == 0:
            print(f"  collected {index + 1}/{n_episodes} episodes", flush=True)

    env.close()
    stacked = [torch.stack([s[i] for s in states]) for i in range(5)]
    return stacked, torch.stack(masks), torch.stack(actions)


def train(
    net: TransformerNetwork,
    states: list[torch.Tensor],
    masks: torch.Tensor,
    actions: torch.Tensor,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Fit the network to the scripted actions by masked cross-entropy.

    Only rows the mask leaves a real choice on contribute. A destroyed model has
    exactly one legal action (`stay`), so including it would spend most of the
    loss teaching the network to agree about corpses -- and on a 25-model board
    late in an episode that is the majority of rows.
    """
    net.to(device).train()
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    n_steps = actions.shape[0]

    for epoch in range(epochs):
        order = torch.randperm(n_steps)
        total, batches, correct, counted = 0.0, 0, 0, 0
        for start in range(0, n_steps, batch_size):
            index = order[start : start + batch_size]
            batch_states = [s[index].to(device) for s in states]
            batch_mask = masks[index].to(device)
            batch_actions = actions[index].to(device)

            logits = net(batch_states)
            logits = logits.masked_fill(~batch_mask, float("-inf"))

            # A row with one legal action teaches nothing and would dominate:
            # a destroyed model has only `stay`, and late in a 25-model episode
            # those are most of the rows.
            choosing = batch_mask.sum(dim=-1) > 1
            # And the target itself must be legal. A masked-out target sits at
            # -inf, which makes the cross-entropy infinite and the step NaN --
            # the scripted policies honour the mask, so this should never fire,
            # but it fails silently and catastrophically if one ever stops.
            chosen_legal = batch_mask.gather(-1, batch_actions.unsqueeze(-1)).squeeze(
                -1
            )
            choosing = choosing & chosen_legal
            if not bool(choosing.any()):
                continue
            flat_logits = logits[choosing]
            flat_actions = batch_actions[choosing]
            loss = loss_fn(flat_logits, flat_actions)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()

            total += float(loss.detach())
            batches += 1
            correct += int((flat_logits.argmax(dim=-1) == flat_actions).sum())
            counted += int(flat_actions.numel())

        accuracy = correct / counted if counted else float("nan")
        print(
            f"  epoch {epoch + 1}/{epochs}  loss {total / max(batches, 1):.4f}"
            f"  action-match {accuracy:.3f}",
            flush=True,
        )


def main() -> None:
    """Collect scripted play, clone it, and write a warm-start checkpoint."""
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy_name = sys.argv[1]
    config_path = sys.argv[2]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    out_path = (
        Path(sys.argv[5]) if len(sys.argv) > 5 else Path("checkpoints/clone.ckpt")
    )

    config = parse_yaml_raw_as(WargameEnvConfig, Path(config_path).read_text())
    config.render_mode = None
    device = auto_device()

    print(f"collecting {n_episodes} episodes of '{policy_name}' on {config_path}")
    states, masks, actions = collect(policy_name, config, n_episodes)
    print(f"  {actions.shape[0]} steps, {actions.shape[1]} models per step")

    env = create_environment(env_config=config)
    net = TransformerNetwork.policy_from_env(env)
    env.close()

    print(f"cloning on {device}")
    train(net, states, masks, actions, epochs, batch_size=32, device=device)

    state_dict = {
        POLICY_PREFIX + key: value.cpu() for key, value in net.state_dict().items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, out_path)
    print(f"wrote {out_path} ({len(state_dict)} tensors)")

    # `_apply_warm_start_weights` loads with strict=False, so a prefix mistake is
    # silent. Prove the keys land on a real module before anyone trains on this.
    reference = TransformerNetwork.policy_from_env(
        create_environment(env_config=config)
    )
    expected = {POLICY_PREFIX + k for k in reference.state_dict()}
    overlap = len(expected & set(state_dict))
    print(f"key check: {overlap}/{len(expected)} policy tensors will be applied")
    if overlap != len(expected):
        raise SystemExit("checkpoint keys do not match the policy network")


if __name__ == "__main__":
    main()
