"""The clone's checkpoint keys must land on the real policy network.

`train.py::_apply_warm_start_weights` loads a warm start with
``load_state_dict(strict=False)``. That is deliberate — a warm start is allowed
to supply the policy without the critic — but it means a **wrong key prefix
loads nothing at all, silently**, and the run trains a random network while
reporting that it warm-started. The repo has been bitten by this family before:
`torch.compile` is not wired precisely because it prefixes every key with
``_orig_mod.``.

So the prefix is pinned here rather than trusted. If `PPOLightning`'s attribute
names ever change, this fails loudly instead of producing a run whose "warm
start" was a no-op.
"""

from __future__ import annotations

import pytest
import torch

from scripts.behaviour_clone import (
    POLICY_PREFIX,
    STAY_ACTION,
    VALUE_PREFIX,
    collect,
    phase_balanced_weights,
    unit_match_counts,
)
from wargame_rl.wargame.envs.types import (
    WargameEnvAction,
    WargameEnvConfig,
    WargameEnvObservation,
)
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.net import TransformerNetwork
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer


def _make_module() -> tuple[PPOLightning, WargameEnv]:
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=3,
            number_of_objectives=2,
            number_of_battle_rounds=4,
        )
    )
    module = PPOLightning(
        env=env,
        ppo_model=PPO_Transformer.from_env(env),
        log=False,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        num_rollout_envs=2,
    )
    return module, env


def test_clone_prefix_matches_every_policy_tensor() -> None:
    """Every cloned tensor must have a home in the Lightning module."""
    module, env = _make_module()
    policy = TransformerNetwork.policy_from_env(env)

    cloned = {POLICY_PREFIX + key for key in policy.state_dict()}
    target = set(module.state_dict())

    missing = cloned - target
    assert not missing, (
        f"{len(missing)} cloned keys have no home, e.g. {sorted(missing)[:3]}"
    )
    assert cloned, "no policy tensors produced"


def test_the_critic_prefix_lands_too() -> None:
    """The clone ships a fitted critic, and its keys must apply as well.

    Cloning the policy *alone* leaves PPO with a randomly initialised value
    function, and measured, that destroys a good clone: 115.8 -> 98.2..106.3
    held-out, damage proportional to learning rate and indifferent to gamma. The
    critic is only worth shipping if it actually loads, and `strict=False` will
    never say otherwise.
    """
    module, env = _make_module()
    critic = TransformerNetwork.value_from_env(env)
    policy = TransformerNetwork.policy_from_env(env)

    cloned = {VALUE_PREFIX + key for key in critic.state_dict()}
    missing = cloned - set(module.state_dict())

    assert not missing, (
        f"{len(missing)} critic keys have no home: {sorted(missing)[:3]}"
    )
    assert cloned, "no critic tensors produced"
    policy_keys = {POLICY_PREFIX + key for key in policy.state_dict()}
    assert not (cloned & policy_keys), "policy and critic key sets must be disjoint"


def test_loading_a_clone_actually_changes_the_weights() -> None:
    """The end the prefix serves: `strict=False` must not silently no-op.

    Asserting the key names alone would still pass if `load_state_dict` ignored
    them, so this loads a clone whose weights are deliberately distinct and
    checks the module's policy tensors actually moved.
    """
    module, env = _make_module()
    policy = TransformerNetwork.policy_from_env(env)

    with torch.no_grad():
        for tensor in policy.state_dict().values():
            if tensor.is_floating_point():
                tensor.fill_(0.5)
    state_dict = {POLICY_PREFIX + k: v for k, v in policy.state_dict().items()}

    before = module.state_dict()[POLICY_PREFIX + "game_embedding.weight"].clone()
    module.load_state_dict(state_dict, strict=False)
    after = module.state_dict()[POLICY_PREFIX + "game_embedding.weight"]

    assert not torch.equal(before, after), "warm start was a silent no-op"
    assert torch.allclose(after, torch.full_like(after, 0.5))


def test_unit_match_is_all_or_nothing_per_unit() -> None:
    """A unit-step counts only when *every* deciding member agreed.

    This is the number `action-match` cannot give: coherency is a joint
    property, so one wrong model spoils its whole unit. Measured, the gap is
    not academic — a clone at 98.3% action match held unit coherency 0.580
    against its demonstrator's 0.884.
    """
    group_ids = torch.tensor([0, 0, 1, 1])
    actions = torch.tensor([[5, 5, 7, 7], [5, 5, 7, 7]])
    choosing = torch.ones(2, 4, dtype=torch.bool)
    # Step 0: unit 0 has one member wrong, unit 1 is perfect.
    # Step 1: both units perfect.
    predicted = torch.tensor([[5, 9, 7, 7], [5, 5, 7, 7]])

    matched, counted = unit_match_counts(predicted, actions, choosing, group_ids)

    assert counted == 4, "two units over two steps"
    assert matched == 3, "only unit 0 at step 0 should fail"
    # The per-model rate is 7/8 = 0.875 on the same data, which is the point:
    # it reports a far healthier number than the joint property warrants.


def test_a_unit_with_nothing_to_decide_is_not_counted() -> None:
    """A wiped or fully-masked unit is skipped, not scored as a failure.

    Mirrors `domain/coherency.py`, which iterates *living* models so that a
    casualty never registers as a breach. Counting dead units as matches would
    inflate the rate as an army dies — the same confound `coherency_rate`
    already carries.
    """
    group_ids = torch.tensor([0, 0, 1, 1])
    actions = torch.tensor([[5, 5, 7, 7]])
    predicted = torch.tensor([[5, 5, 0, 0]])
    # Unit 1 has no deciding member: its models are destroyed.
    choosing = torch.tensor([[True, True, False, False]])

    matched, counted = unit_match_counts(predicted, actions, choosing, group_ids)

    assert counted == 1, "only unit 0 had a decision to make"
    assert matched == 1


def test_collect_takes_any_selector_and_records_its_actions() -> None:
    """`collect` must accept a decoded checkpoint teacher, not only a script.

    Distilling joint constrained decoding is the whole reason the demonstrator
    became a parameter: the teacher is a checkpoint played at `decode_topk` > 1,
    and what is recorded has to be the action the teacher *actually played*. A
    plumbing slip that recorded the undecoded argmax instead would train the
    student on the wrong target and look completely normal.
    """
    config = WargameEnvConfig(
        render_mode=None,
        number_of_wargame_models=3,
        number_of_objectives=2,
        number_of_battle_rounds=2,
    )
    played: list[list[int]] = []

    def select(observation: WargameEnvObservation, env: WargameEnv) -> WargameEnvAction:
        # The HIGHEST legal action per model, not a constant: the recorded
        # actions must differ across steps and models, or a `collect` that
        # wrote zeros -- or the teacher's undecoded argmax -- would pass here.
        assert observation.action_mask is not None
        chosen = [int(row.nonzero()[0][-1]) for row in observation.action_mask]
        played.append(chosen)
        return WargameEnvAction(actions=chosen)

    states, masks, actions, returns, phases = collect(
        select, config, n_episodes=1, gamma=0.9
    )

    # The phase per step, recorded so the fit can balance target classes WITHIN
    # a phase: STAY is the rare class in movement and the dominant one in the
    # charge, and the mask cannot tell those two apart because a charge reuses
    # the movement slice.
    assert phases.shape[0] == actions.shape[0]
    assert actions.shape[0] == len(played)
    assert actions.shape[1] == config.number_of_wargame_models
    assert torch.equal(actions, torch.tensor(played, dtype=torch.long))
    assert masks.shape[0] == actions.shape[0]
    assert returns.shape == actions.shape
    assert all(tensor.shape[0] == actions.shape[0] for tensor in states)


class TestPhaseBalancedWeights:
    """A rare target must not be drowned by a common one — per PHASE.

    Unweighted, the clone fit learned the charge phase as "always STAY": a charge
    order is ~3.7% of that phase's deciding rows, so predicting STAY scores 94%
    and the loss has almost nothing to gain from the rest. Measured, the clones
    echoed 0.8–2.4% of their teacher's charge orders while matching its shooting
    at 0.99 — they had not failed to coordinate, they had failed to declare.
    """

    @staticmethod
    def _rows(stay_count: int, other_count: int, phase: int):  # type: ignore[no-untyped-def]
        n = stay_count + other_count
        actions = torch.tensor(
            [[STAY_ACTION]] * stay_count + [[7]] * other_count, dtype=torch.long
        )
        masks = torch.ones((n, 1, 12), dtype=torch.bool)
        phases = torch.full((n,), phase, dtype=torch.long)
        return actions, masks, phases

    def test_the_rare_target_is_weighted_up(self) -> None:
        """96 STAY against 4 charges is the measured charge-phase balance."""
        # Arrange
        actions, masks, phases = self._rows(96, 4, phase=3)

        # Act
        weights = phase_balanced_weights(actions, masks, phases)

        # Assert
        assert weights[actions == 7].mean() > weights[actions == STAY_ACTION].mean()

    def test_the_balance_is_PER_PHASE_and_not_global(self) -> None:
        """⚠ STAY is rare in movement and dominant in the charge.

        A single global balance would push in opposite directions in the two
        phases and cancel, which is why the phase is recorded at collection
        rather than inferred from the mask — a charge reuses the movement slice,
        so the two are indistinguishable there.
        """
        # Arrange: charge phase 96/4 STAY-heavy, movement phase 4/96 STAY-light.
        charge = self._rows(96, 4, phase=3)
        movement = self._rows(4, 96, phase=1)
        actions = torch.cat([charge[0], movement[0]])
        masks = torch.cat([charge[1], movement[1]])
        phases = torch.cat([charge[2], movement[2]])

        # Act
        weights = phase_balanced_weights(actions, masks, phases)

        # Assert: the up-weighted class is the rare one in EACH phase.
        in_charge = phases == 3
        in_movement = phases == 1
        charge_stay = weights[in_charge & (actions == STAY_ACTION).squeeze(-1)].mean()
        charge_other = weights[in_charge & (actions == 7).squeeze(-1)].mean()
        move_stay = weights[in_movement & (actions == STAY_ACTION).squeeze(-1)].mean()
        move_other = weights[in_movement & (actions == 7).squeeze(-1)].mean()
        assert charge_other > charge_stay
        assert move_stay > move_other

    def test_a_row_with_one_legal_action_is_not_counted(self) -> None:
        """A destroyed model has only STAY and is excluded from the fit.

        Counting it here would inflate the STAY share and under-weight the rare
        class exactly where it matters most.
        """
        # Arrange
        actions, masks, phases = self._rows(4, 4, phase=3)
        corpses = torch.zeros((96, 1, 12), dtype=torch.bool)
        corpses[:, :, STAY_ACTION] = True
        masks = torch.cat([masks, corpses])
        actions = torch.cat([actions, torch.full((96, 1), STAY_ACTION)])
        phases = torch.cat([phases, torch.full((96,), 3)])

        # Act
        weights = phase_balanced_weights(actions, masks, phases)

        # Assert: 4 v 4 among DECIDING rows, so the two classes weigh the same.
        deciding = masks.sum(dim=-1) > 1
        stay = weights[deciding & (actions == STAY_ACTION)].mean()
        other = weights[deciding & (actions == 7)].mean()
        assert stay == pytest.approx(float(other), rel=1e-6)

    def test_the_mean_weight_is_one_so_the_learning_rate_carries_over(self) -> None:
        """This changes the BALANCE of the loss, never its scale."""
        # Arrange
        actions, masks, phases = self._rows(96, 4, phase=3)

        # Act
        weights = phase_balanced_weights(actions, masks, phases)

        # Assert
        assert float(weights[masks.sum(dim=-1) > 1].mean()) == pytest.approx(1.0)
