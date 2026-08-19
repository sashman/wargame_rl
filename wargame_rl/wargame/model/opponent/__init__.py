"""Opponent policies that need a network.

Importing this package registers the `model` opponent key. Registration flows
**downward** -- the upper layer registers into the lower layer's registry --
which is what keeps `model -> envs` one-way while still letting a config name a
checkpoint as its opponent.
"""

from wargame_rl.wargame.model.opponent.network_policy import NetworkOpponentPolicy

__all__ = ["NetworkOpponentPolicy"]
