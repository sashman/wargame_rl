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

Usage: just behaviour-clone <policy|ckpt> <env_config> [n_episodes] [epochs] [out]
       [seed] [decode_topk]

**The teacher may be a checkpoint, and it may be decoded.** Passing a
checkpoint path with `decode_topk` > 1 distils *joint constrained decoding*
into the weights: the demonstrations are the most probable coherency-legal
combination the teacher's own distribution allows, which on three seeds is
worth +40.5 vp_margin and coherency 0.639 -> 0.936 at play time. Cloning it
asks whether the network can carry that improvement itself.

⚠ **A per-model fit does not inherit a joint property.** A 98.3%-action-match
clone of `squad_march_take` holds 0.40 unit coherency against its teacher's
0.95, because matching each marginal says nothing about whether independent
samples from those marginals are jointly legal. Expect the student to need the
decoder too; the question is whether it needs it *less*, and whether the two
compound.

**Clone twice before quoting a clone.** Two clones from identical data and
identical settings, differing only in weight initialisation, measured **115.8**
and **111.1** on the held-out tables — a 4.7 vp spread across the bar itself.
The demonstration data is deterministic (`CLONE_SEED_BASE + episode`), so that
variance is entirely the fit, and it is large enough to decide whether a clone
appears to beat the bar or merely tie it. `seed` makes a clone reproducible;
running several with different seeds is how you find out which you have.

The output is a checkpoint `train.py --warm-start-ckpt-path` accepts.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from pydantic_yaml import parse_yaml_raw_as
from torch import nn

from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.common.device import auto_device
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.model.net import TransformerNetwork
from wargame_rl.wargame.model.ppo.config import PPOConfig
from wargame_rl.wargame.selectors import build_action_selector

# Disjoint from every other seed band in the repo: evaluation uses 700000+,
# in-run eval 500000+, the logged baselines 10000+. A clone trained on the
# evaluation layouts would make the held-out score meaningless.
CLONE_SEED_BASE = 800_000

# The prefix `PPOLightning.load_state_dict` expects. `_apply_warm_start_weights`
# loads with `strict=False`, so a wrong prefix here loads **nothing at all** and
# trains a random network while the run claims a warm start -- which is why
# `main` verifies the key overlap instead of trusting it.
POLICY_PREFIX = "ppo_model.policy_network."

# The critic's prefix. Cloning the policy ALONE leaves PPO with a randomly
# initialised value function, so its first updates are driven by noise -- and
# measured, that destroys a good clone: 115.8 -> 98.2..106.3 held-out, damage
# proportional to learning rate and indifferent to gamma. Supplying a fitted
# critic is the direct test of that.
VALUE_PREFIX = "ppo_model.value_network."


def unit_match_counts(
    predicted: torch.Tensor,
    actions: torch.Tensor,
    choosing: torch.Tensor,
    group_ids: torch.Tensor,
) -> tuple[int, int]:
    """Count unit-steps where **every** deciding member matched the demonstrator.

    Args:
        predicted: ``(batch, n_models)`` the network's chosen action per model.
        actions: ``(batch, n_models)`` the action the demonstrator chose.
        choosing: ``(batch, n_models)`` which models had a real choice to make.
        group_ids: ``(n_models,)`` unit membership, this project's name for the
            rules' *unit*.

    Returns:
        ``(matched_units, counted_units)``. A unit-step with no deciding member
        is not counted at all, mirroring `domain/coherency.py`, which iterates
        living models so a wiped unit never registers as a failure.

    **Why per-model action match is the wrong number for a joint property.**
    A unit is coherent only if *all* of its models are placed correctly, so
    fidelity has to be scored the same way. Measured: a clone at **98.3%**
    action match holds unit coherency **0.580** against its demonstrator's
    **0.884** -- because `0.983 ** 5` is only ~92% of unit-steps clean, and the
    positional error is cumulative, since nothing re-forms a model that drifts
    out. Per-model match reports 98% while the property the rules care about is
    gone.
    """
    matched = (predicted == actions) & choosing
    matched_units = counted_units = 0
    for group_id in torch.unique(group_ids):
        members = group_ids == group_id
        deciding = (choosing & members).sum(dim=-1)
        agreeing = (matched & members).sum(dim=-1)
        valid = deciding > 0
        counted_units += int(valid.sum())
        matched_units += int(((agreeing == deciding) & valid).sum())
    return matched_units, counted_units


def collect(
    select: ActionSelector, config: WargameEnvConfig, n_episodes: int, gamma: float
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Play the demonstrator and record what it saw, what it did, and what it earned.

    Returns the five state tensors stacked over steps, the action mask, the
    action the demonstrator chose for each model, and the **discounted per-model
    return** from each step to the end of its episode.

    The return is per model, not per episode, because that is what PPO's critic
    predicts -- `value_from_encoded` emits one value per player token.

    Every step is kept, movement and shooting alike: the phase is part of the
    game-feature vector, so one network learns both and the clone reproduces the
    whole policy rather than half of it.

    The phase index is returned **separately** as well, because the fit needs to
    balance the target classes *within* a phase and cannot recover the phase from
    the mask: a charge reuses the movement slice, so the two phases have the same
    legal-action shape and differ only in which rungs the 2D6 left open.

    The selector is passed in rather than built here, because the demonstrator
    is no longer necessarily a scripted policy: a checkpoint played under joint
    constrained decoding is one too, and it is the interesting one.
    """
    env = create_environment(env_config=config)

    states: list[list[torch.Tensor]] = []
    masks: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    phases: list[int] = []

    for index in range(n_episodes):
        observation, _ = env.reset(seed=CLONE_SEED_BASE + index)
        terminated = truncated = False
        episode_rewards: list[torch.Tensor] = []
        while not (terminated or truncated):
            tensors = observation_to_tensor(observation)
            phases.append(
                list(BattlePhase).index(
                    env.game_clock_state.phase or BattlePhase.movement
                )
            )
            action = select(observation, env)
            states.append([t.detach().clone() for t in tensors[:5]])
            masks.append(tensors[5].detach().clone())
            actions.append(torch.tensor(action.actions, dtype=torch.long))
            observation, _r, terminated, truncated, _i = env.step(action)
            episode_rewards.append(
                torch.tensor(env.last_per_model_reward, dtype=torch.float32)
            )
        # Discount backwards within the episode; the bootstrap is zero because
        # the episode has actually ended rather than been truncated mid-return.
        running = torch.zeros_like(episode_rewards[0])
        episode_returns: list[torch.Tensor] = []
        for reward in reversed(episode_rewards):
            running = reward + gamma * running
            episode_returns.append(running.clone())
        returns.extend(reversed(episode_returns))
        if (index + 1) % 25 == 0:
            print(f"  collected {index + 1}/{n_episodes} episodes", flush=True)

    env.close()
    stacked = [torch.stack([s[i] for s in states]) for i in range(5)]
    return (
        stacked,
        torch.stack(masks),
        torch.stack(actions),
        torch.stack(returns),
        torch.tensor(phases, dtype=torch.long),
    )


# Above this a single rare row would dominate its batch and the step goes
# unstable. At the measured 3.7% charge-order rate the uncapped weight is ~26,
# so this binds -- deliberately: the aim is to make the rare action *visible*,
# not to make it the only thing the fit sees.
MAX_CLASS_WEIGHT = 12.0


def phase_balanced_weights(
    actions: torch.Tensor, masks: torch.Tensor, phases: torch.Tensor
) -> torch.Tensor:
    """Per-row loss weights that balance STAY against everything else, per phase.

    ⚠ **Per PHASE, not globally, and the distinction is the whole point.** STAY
    is the *rare* class in the movement phase (the script always moves) and the
    *dominant* one in the charge phase (only an eligible unit in reach declares).
    One global balance would therefore push in opposite directions in the two
    phases and cancel. Measured on the charging teacher: STAY is 96.3% of
    deciding charge-phase rows and a few per cent of movement-phase rows.

    Counted over rows the mask leaves a real choice on, matching what the fit
    actually trains on -- a destroyed model has only STAY and is excluded there,
    so counting it here would inflate the STAY share and under-weight the rare
    class exactly where it matters most.

    Weights are normalised to mean 1 over the deciding rows, so this changes the
    BALANCE of the loss and not its scale, and the learning rate carries over.
    """
    # ⚠ Normalised to one device first. The collected tensors do not all live on
    # the same one -- `observation_to_tensor` can return CUDA tensors while the
    # phase index is built on the CPU here -- and a comparison across the two
    # raises rather than broadcasting. This is a one-off precompute, so the CPU
    # is the cheap and safe place to do it.
    home = actions.device
    actions = actions.cpu()
    masks = masks.cpu()
    phases = phases.cpu()
    deciding = masks.sum(dim=-1) > 1
    is_stay = actions == STAY_ACTION
    weights = torch.ones_like(actions, dtype=torch.float32)
    for phase in phases.unique():
        rows = (phases == phase).unsqueeze(-1) & deciding
        if not bool(rows.any()):
            continue
        stay = int((rows & is_stay).sum())
        other = int((rows & ~is_stay).sum())
        if stay == 0 or other == 0:
            continue
        # Balance the two classes against each other, then cap.
        stay_w = min((stay + other) / (2.0 * stay), MAX_CLASS_WEIGHT)
        other_w = min((stay + other) / (2.0 * other), MAX_CLASS_WEIGHT)
        weights[rows & is_stay] = stay_w
        weights[rows & ~is_stay] = other_w
    mean = weights[deciding].mean() if bool(deciding.any()) else torch.tensor(1.0)
    return (weights / mean.clamp(min=1e-8)).to(home)


def train(
    net: TransformerNetwork,
    states: list[torch.Tensor],
    masks: torch.Tensor,
    actions: torch.Tensor,
    group_ids: torch.Tensor,
    phases: torch.Tensor,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Fit the network to the scripted actions by masked cross-entropy.

    Only rows the mask leaves a real choice on contribute. A destroyed model has
    exactly one legal action (`stay`), so including it would spend most of the
    loss teaching the network to agree about corpses -- and on a 25-model board
    late in an episode that is the majority of rows.

    Reports **two** fidelity numbers per epoch. `action-match` is the per-model
    rate; `unit-match` is the fraction of unit-steps on which every deciding
    member agreed. Read the second one when the clone has to satisfy anything
    joint -- see `unit_match_counts` for why the first is blind to it. The loss
    is still per model: this is a measurement, not an objective.

    ⚠ **The rows are WEIGHTED so a rare target is not drowned by a common one.**
    Unweighted, this fit learned the charge phase as "always STAY": a charge
    order is ~3.7% of the deciding rows in that phase, so predicting STAY scores
    94% and the loss has almost nothing to gain from the other 6%. Measured, the
    resulting clones echoed **0.8-2.4%** of the teacher's charge orders while
    matching its shooting at 0.99 -- they had not failed to coordinate, they had
    failed to *declare*. `phase_balanced_weights` is the fix.
    """
    net.to(device).train()
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    n_steps = actions.shape[0]
    weights = phase_balanced_weights(actions, masks, phases)

    group_ids = group_ids.to(device)
    for epoch in range(epochs):
        order = torch.randperm(n_steps)
        total, batches, correct, counted = 0.0, 0, 0, 0
        units_matched, units_counted = 0, 0
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
            flat_weights = weights[index].to(device)[choosing]
            per_row = loss_fn(flat_logits, flat_actions)
            loss = (per_row * flat_weights).sum() / flat_weights.sum().clamp(min=1e-8)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()

            total += float(loss.detach())
            batches += 1
            correct += int((flat_logits.argmax(dim=-1) == flat_actions).sum())
            counted += int(flat_actions.numel())
            # Scored on the whole batch rather than the flattened selection,
            # because a unit's members have to be compared side by side and
            # flattening has already thrown their grouping away.
            matched, total_units = unit_match_counts(
                logits.argmax(dim=-1), batch_actions, choosing, group_ids
            )
            units_matched += matched
            units_counted += total_units

        accuracy = correct / counted if counted else float("nan")
        unit_accuracy = units_matched / units_counted if units_counted else float("nan")
        print(
            f"  epoch {epoch + 1}/{epochs}  loss {total / max(batches, 1):.4f}"
            f"  action-match {accuracy:.3f}  unit-match {unit_accuracy:.3f}",
            flush=True,
        )


def train_value(
    net: TransformerNetwork,
    states: list[torch.Tensor],
    returns: torch.Tensor,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Fit the critic to the demonstrator's own discounted returns.

    Plain MSE over every model, dead ones included: a destroyed model's return
    is a real number the critic should predict, and masking them would leave the
    value of "this model is gone" untrained exactly when it starts to matter.

    Reported as explained variance rather than raw loss, because the loss scale
    means nothing on its own — 1 - Var(residual)/Var(target) says whether the
    critic has learned anything, and it is the number PPO's advantages depend on.
    """
    net.to(device).train()
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    n_steps = returns.shape[0]

    for epoch in range(epochs):
        order = torch.randperm(n_steps)
        residual, total = 0.0, 0.0
        target_mean = float(returns.mean())
        for start in range(0, n_steps, batch_size):
            index = order[start : start + batch_size]
            batch_states = [s[index].to(device) for s in states]
            batch_returns = returns[index].to(device)

            predicted = net(batch_states)
            loss = torch.nn.functional.mse_loss(predicted, batch_returns)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()

            residual += float(((predicted - batch_returns) ** 2).sum().detach())
            total += float(((batch_returns - target_mean) ** 2).sum().detach())

        explained = 1.0 - residual / total if total > 0 else float("nan")
        print(
            f"  critic epoch {epoch + 1}/{epochs}  explained variance {explained:.3f}",
            flush=True,
        )


def main() -> None:
    """Collect scripted play, clone it, and write a warm-start checkpoint."""
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    teacher = sys.argv[1]
    config_path = sys.argv[2]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    out_path = (
        Path(sys.argv[5]) if len(sys.argv) > 5 else Path("checkpoints/clone.ckpt")
    )
    # Decoding the TEACHER, not the student. At K>1 the demonstrations are the
    # most probable coherency-legal action *combination* the teacher's own
    # distribution allows, so the student is fitted to legal joint play rather
    # than to the independent argmax that produced 33% cancelled unit-moves.
    decode_topk = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    # Seeds the weight init and the batch shuffling -- the ONLY things that
    # differ between two clones of the same demonstrations, and worth 4.7 vp of
    # held-out score. Unseeded, a clone is not reproducible and a single one is
    # not quotable.
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    torch.manual_seed(seed)

    config = parse_yaml_raw_as(WargameEnvConfig, Path(config_path).read_text())
    config.render_mode = None
    device = auto_device()

    # The critic's target must be discounted the way the run that consumes it
    # will discount, so this mirrors PPOConfig rather than inventing a value.
    gamma = PPOConfig().gamma

    # Collection is deterministic in (policy, config, n_episodes, gamma), and it
    # is the slow half -- 1200 episodes is ~30 minutes against ~20 for the fit.
    # Caching it is what makes "clone several times and look at the spread"
    # affordable, which the variance above makes mandatory rather than nice.
    # A checkpoint path is not a filename, and two teachers differing only in
    # `decode_topk` produce different demonstrations -- both have to reach the
    # cache key or a decoded run silently reuses an argmax collection.
    resolved = build_action_selector(
        teacher, create_environment(env_config=config), decode_topk
    )
    select, teacher_label = resolved.select, resolved.label
    # ⚠ **The key carries a FINGERPRINT OF THE CONFIG, not just its filename.**
    # It used to name the file stem, so editing a config in place produced a
    # cache hit on demonstrations collected under the old one -- and when the
    # charge declaration landed, that meant 102-action, 60-turn demonstrations
    # being fitted to a 104-action, 80-turn network, with every recorded action
    # indexing a different action space. Silent, and catastrophic.
    #
    # `v2` in the name is separate and still needed: the payload gained
    # `phases`, and a v1 payload would KeyError at best and fit unweighted at
    # worst.
    fingerprint = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:12]
    cache = Path("checkpoints/clone_data") / (
        f"{teacher_label}-k{decode_topk}-{Path(config_path).stem}"
        f"-{n_episodes}-g{gamma}-{fingerprint}-v2.pt"
    )
    if cache.exists():
        print(f"reusing collected demonstrations from {cache}")
        payload = torch.load(cache, weights_only=False)
        states, masks, actions, returns, phases = (
            payload["states"],
            payload["masks"],
            payload["actions"],
            payload["returns"],
            payload["phases"],
        )
    else:
        print(
            f"collecting {n_episodes} episodes of '{teacher_label}' "
            f"(decode_topk={decode_topk}) on {config_path}"
        )
        states, masks, actions, returns, phases = collect(
            select, config, n_episodes, gamma
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "states": states,
                "masks": masks,
                "actions": actions,
                "returns": returns,
                "phases": phases,
            },
            cache,
        )
        print(f"cached demonstrations to {cache}")
    print(
        f"  {actions.shape[0]} steps, {actions.shape[1]} models per step, "
        f"returns discounted at gamma={gamma}"
    )

    env = create_environment(env_config=config)
    net = TransformerNetwork.policy_from_env(env)
    # Read off the env rather than stored with the demonstrations: unit
    # membership is fixed by the config, so it needs no re-collection and the
    # cached demonstration sets (~780 MB each) stay valid.
    group_ids = torch.tensor(
        [model.group_id for model in env.wargame_models], dtype=torch.long
    )
    env.close()

    print(f"cloning on {device} (seed {seed})")
    train(
        net,
        states,
        masks,
        actions,
        group_ids,
        phases,
        epochs,
        batch_size=32,
        device=device,
    )

    print("fitting the critic on the demonstrator's own returns")
    value_env = create_environment(env_config=config)
    value_net = TransformerNetwork.value_from_env(value_env)
    value_env.close()
    train_value(value_net, states, returns, epochs, batch_size=32, device=device)

    state_dict = {
        POLICY_PREFIX + key: value.cpu() for key, value in net.state_dict().items()
    }
    state_dict.update(
        {
            VALUE_PREFIX + key: value.cpu()
            for key, value in value_net.state_dict().items()
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, out_path)
    print(f"wrote {out_path} ({len(state_dict)} tensors)")

    # `_apply_warm_start_weights` loads with strict=False, so a prefix mistake is
    # silent. Prove the keys land on a real module before anyone trains on this.
    check_env = create_environment(env_config=config)
    expected = {
        POLICY_PREFIX + key
        for key in TransformerNetwork.policy_from_env(check_env).state_dict()
    } | {
        VALUE_PREFIX + key
        for key in TransformerNetwork.value_from_env(check_env).state_dict()
    }
    check_env.close()
    overlap = len(expected & set(state_dict))
    print(
        f"key check: {overlap}/{len(expected)} tensors (policy + critic) will be applied"
    )
    if overlap != len(expected):
        raise SystemExit("checkpoint keys do not match the policy/value networks")


if __name__ == "__main__":
    main()
