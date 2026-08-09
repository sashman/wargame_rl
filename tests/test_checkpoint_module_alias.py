"""Checkpoints written before DQN was removed must still load.

Lightning pickles the whole `PPO_Transformer` into every checkpoint's
`hyper_parameters`, so a checkpoint records the *import path* of `Block` and
`LayerNorm` as of the day it was written. Those layers moved from
`model/dqn/layers.py` to `model/common/layers.py` on 2026-08-09, which made
every existing checkpoint raise `ModuleNotFoundError` on `torch.load` — caught
only by running `measure-checkpoint` against a real trained run, because
nothing in the suite loaded a pre-move checkpoint.

`checkpoints/` is the only copy of any trained weights (nothing is uploaded to
wandb), so this is not a cosmetic compatibility concern.
"""

from __future__ import annotations

import io
import pickle

# Imported for its side effect as well as its symbols: importing the module is
# what installs the `sys.modules` alias under test.
from wargame_rl.wargame.model.common.config import TransformerConfig
from wargame_rl.wargame.model.common.layers import Block, LayerNorm

OLD_MODULE = b"wargame_rl.wargame.model.dqn.layers"
NEW_MODULE = b"wargame_rl.wargame.model.common.layers"


def test_old_layer_module_path_still_resolves() -> None:
    """`find_class` on the pre-move path returns the moved classes.

    This is the exact call `torch.load` makes and the exact one that raised
    `ModuleNotFoundError`, so it is tested directly rather than only through a
    checkpoint.
    """
    unpickler = pickle.Unpickler(io.BytesIO(b"\x80\x05N."))
    assert unpickler.find_class(OLD_MODULE.decode(), "Block") is Block
    assert unpickler.find_class(OLD_MODULE.decode(), "LayerNorm") is LayerNorm


def test_a_layer_pickled_under_the_old_path_unpickles() -> None:
    """A real pickled layer naming the old module loads as the moved class.

    Protocol 0 writes module paths as newline-terminated text rather than
    length-prefixed, so the old path can be substituted at any length — which
    is what makes a pre-move checkpoint reproducible without a binary fixture.
    """
    payload = pickle.dumps(Block(TransformerConfig()), protocol=0)
    assert NEW_MODULE in payload, "the layer no longer records its module path"

    restored = pickle.loads(payload.replace(NEW_MODULE, OLD_MODULE))

    assert isinstance(restored, Block)
    assert isinstance(restored.ln_1, LayerNorm)
