import numpy as np
import torch

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.agent_base import BaseAgent
from wargame_rl.wargame.model.common.observation import (
    apply_action_mask,
    observation_to_tensor,
)
from wargame_rl.wargame.model.net import RL_Network


class ArgmaxAgent(BaseAgent):
    """Agent that plays the highest-scoring valid action of any `RL_Network`.

    Takes a bare policy network rather than a PPO actor-critic, which is what
    `simulate.py` and the measurement scripts have: they load weights straight
    into a `TransformerNetwork`. Greedy argmax over the logits is the same
    action `PPOModel.get_action(deterministic=True)` would pick, so a checkpoint
    scores the same through either path.
    """

    def __init__(self, env: WargameEnv) -> None:
        """Args:
        env: environment to interact with.
        """
        super().__init__(env)
        self.reset()

    def get_action(
        self,
        policy_net: RL_Network,
        observation: WargameEnvObservation,
        epsilon: float,
    ) -> WargameEnvAction:
        """Using the given network, decide what action to carry out.

        Uses an epsilon-greedy policy with action masking — only valid
        actions (according to ``observation.action_mask``) are considered.
        """
        mask = observation.action_mask  # (n_models, n_actions) or None

        if np.random.random() < epsilon:
            if mask is not None:
                action = WargameEnvAction.random(mask)
            else:
                action = WargameEnvAction(self.env.action_space.sample())
        else:
            with torch.no_grad():
                tensors = observation_to_tensor(observation, policy_net.device)
                mask_tensor = tensors[5]  # (n_models, n_actions)
                state = tensors[:5]
                logits = policy_net(state)
                assert logits.shape[0] == 1
                assert len(logits.shape) == 3
                logits = apply_action_mask(logits, mask_tensor.unsqueeze(0))
                _, action_indexes = logits.max(dim=-1)
                action = WargameEnvAction(actions=action_indexes.flatten().tolist())

        self._last_log_prob = None
        return action
