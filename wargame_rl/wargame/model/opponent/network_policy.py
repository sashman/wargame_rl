"""Opponent policy: play a trained checkpoint on the opponent side.

Fills the `model` registry key `docs/opponent-policies.md` has reserved since
the opponent system was written:

```yaml
opponent_policy:
  type: model
  params:
    checkpoint: checkpoints/<run>/last.ckpt
    decode_topk: 3
```

**Switching a config's opponent invalidates every number measured on it.**

Lives under `model/` rather than `envs/opponent/` because it needs a network,
and `envs` importing `model` would be a dependency inversion *and* a real
import cycle — `model/net.py` imports `envs.wargame`. The env-layer half is
`envs/opponent/selector_policy.py`, which this subclasses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from wargame_rl.wargame.envs.domain.battle_factory import unit_count
from wargame_rl.wargame.envs.opponent.registry import register_policy
from wargame_rl.wargame.envs.opponent.selector_policy import SelectorOpponentPolicy
from wargame_rl.wargame.model.common.decoding import decode_joint_coherent
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.model.net import (
    TransformerNetwork,
    convert_state_dict,
    spec_from_observation,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.types import (
        WargameEnvAction,
        WargameEnvConfig,
        WargameEnvObservation,
    )
    from wargame_rl.wargame.envs.wargame import WargameEnv


class AsymmetricArmiesError(ValueError):
    """The observation encoding cannot describe these two armies at once."""


def require_equal_armies(config: WargameEnvConfig) -> None:
    """Refuse a config whose two armies differ in size or unit count.

    Not a nicety, and not about fairness. `TransformerNetwork._alive_feature_index`
    locates the `alive` column by counting **backwards** from the last column,
    assuming the trailing expected-damage block is exactly `n_opponents` wide;
    `_alive_from_features` then falls back to treating **every row as alive**
    when that index lands out of range. So on unequal armies a network seated on
    the opponent side reads casualties as live models and never raises — it
    simply plays a board that does not exist.

    This is the restriction the size-agnostic policy work exists to remove.
    """
    players = config.number_of_wargame_models
    opponents = config.number_of_opponent_models
    if players != opponents:
        raise AsymmetricArmiesError(
            f"a network opponent needs equal armies: {players} player models "
            f"against {opponents} opponent models. The observation encoding "
            "locates the alive column by counting back from the trailing "
            "expected-damage block, and degrades silently when the two widths "
            "disagree."
        )
    player_units = unit_count(players, config.max_groups, config.models)
    opponent_units = unit_count(opponents, config.max_groups, config.opponent_models)
    if player_units != opponent_units:
        raise AsymmetricArmiesError(
            f"a network opponent needs equal unit counts: {player_units} player "
            f"units against {opponent_units}. The shooting action slice is one "
            "entry per enemy unit, so the two seats would need different action "
            "spaces."
        )


class NetworkOpponentPolicy(SelectorOpponentPolicy):
    """Drive the opponent army with a policy loaded from a checkpoint."""

    def __init__(self, env: WargameEnv, **kwargs: object) -> None:
        checkpoint = kwargs.pop("checkpoint", None)
        if not isinstance(checkpoint, str):
            raise ValueError(
                "the `model` opponent policy requires a `checkpoint` param "
                "naming a .ckpt, e.g. params: {checkpoint: checkpoints/run/last.ckpt}"
            )
        require_equal_armies(env.config)

        self._checkpoint = checkpoint
        decode_topk = kwargs.pop("decode_topk", 1)
        self._decode_topk = int(decode_topk) if isinstance(decode_topk, int) else 1
        self._decode_stay = bool(kwargs.pop("decode_stay", False))
        # Loaded now so a missing or malformed path fails at env construction,
        # loudly, rather than part-way through a scoring run. The *network* is
        # built lazily -- see `_network`.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw = payload["state_dict"] if "state_dict" in payload else payload
        # Raises on an unrecognised prefix rather than loading nothing. The
        # warm-start path in `train.py` uses `strict=False` with no prefix
        # rewriting, so a wrong prefix there trains a random network while
        # reporting success; this must not copy that shape.
        self._state_dict = convert_state_dict(raw)
        self._net: TransformerNetwork | None = None

        super().__init__(
            env,
            select=self._select_action_for,
            label=checkpoint,
            # A network can emit any unmasked action, so it shoots whenever the
            # opponent's action space has a shooting slice at all.
            shoots=None,
        )

    @property
    def checkpoint_path(self) -> str:
        """The checkpoint this opponent is playing."""
        return self._checkpoint

    def _network(self, observation: WargameEnvObservation) -> TransformerNetwork:
        """Build the network on first use, sized from the opponent's own seat.

        **Not built in `__init__`, and not via `from_env`.** `from_env` calls
        `env.reset()`, and this policy is constructed *inside*
        `WargameEnv.__init__` — the statement that binds `_opponent_policy` —
        so a reset there would re-enter a half-built env, and would consume the
        layout RNG and shift every seeded episode besides. Sizing from the first
        observation instead is safe because `reset` has run by then.

        The handler is the **mirror's**, i.e. the env's opponent-side one.
        Reading `env._action_handler` through the mirror's `__getattr__` would
        fall through to the player's and size this network with the wrong action
        count — silently, on a symmetric config.
        """
        if self._net is None:
            spec = spec_from_observation(
                observation,
                self.mirror.player_action_handler,
                self.mirror.config.objective_budget,
            )
            net = TransformerNetwork.from_spec(spec, is_policy=True)
            net.load_state_dict(self._state_dict)
            net.eval()
            self._net = net
        return self._net

    def _select_action_for(
        self, observation: WargameEnvObservation, env: WargameEnv
    ) -> WargameEnvAction:
        """Greedy over the network's logits, optionally decoded jointly.

        `decode_joint_coherent` reads only `config`, `game_clock_state`,
        `player_models`, `opponent_models`, `player_action_handler` and
        `rules_quantities` — every one of which the mirror either swaps or
        genuinely shares — so the joint constrained decode works on this seat
        unchanged.
        """
        from wargame_rl.wargame.envs.types import WargameEnvAction

        net = self._network(observation)
        with torch.no_grad():
            logits = net(observation_to_tensor(observation, net.device))
            actions = [int(a) for a in logits.argmax(dim=-1).flatten().tolist()]
            if self._decode_topk > 1:
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                actions = decode_joint_coherent(
                    log_probs,
                    actions,
                    env,
                    self._decode_topk,
                    include_stay=self._decode_stay,
                )
        return WargameEnvAction(actions=actions)


register_policy("model", NetworkOpponentPolicy)
