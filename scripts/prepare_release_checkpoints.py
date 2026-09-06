"""Strip a training checkpoint down to the weights a release should carry.

**Why this exists.** A Lightning checkpoint here is ~198 MB, and only a quarter
of that is the policy. The rest is optimiser moments, LR-scheduler and loop
state, callback state, and -- on an anchored run -- a full second copy of the
network held as the KL reference. None of it is usable without the training
loop, and the reference copy is *the warm start*, not the trained policy, so
shipping it invites someone to score the wrong weights.

**What it guarantees.** Every retained tensor is compared to the original with
`torch.equal`, so a released file is bit-identical to the run that produced it
or the script raises. That check is the point: a release whose weights merely
*look* right is how a random network gets published as a trained one.

Usage: just prepare-release <out_dir> <label> <ckpt>...
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

# Kept because the network is rebuilt from them at load time. Everything else in
# a Lightning checkpoint describes how training was progressing, not what was
# learned.
_KEEP = ("state_dict", "hyper_parameters", "hparams_name", "epoch", "global_step")
_REFERENCE_PREFIX = "_kl_reference"


def strip(checkpoint: dict) -> tuple[dict, int, int]:
    """Return the release payload, and how many tensors were kept and dropped."""
    state = checkpoint["state_dict"]
    policy = {k: v for k, v in state.items() if not k.startswith(_REFERENCE_PREFIX)}
    dropped = len(state) - len(policy)
    payload = {k: checkpoint[k] for k in _KEEP if k in checkpoint}
    payload["state_dict"] = policy
    return payload, len(policy), dropped


def verify(original: dict, payload: dict) -> None:
    """Raise unless every retained tensor is bit-identical to the original."""
    source = original["state_dict"]
    for key, tensor in payload["state_dict"].items():
        if not torch.equal(tensor, source[key]):
            raise SystemExit(f"tensor differs after stripping: {key}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(2)
    out_dir, label = Path(sys.argv[1]), sys.argv[2]
    out_dir.mkdir(parents=True, exist_ok=True)
    sums: list[str] = []
    for index, raw in enumerate(sys.argv[3:], start=1):
        source = Path(raw)
        original = torch.load(source, map_location="cpu", weights_only=False)
        payload, kept, dropped = strip(original)
        verify(original, payload)
        epoch = payload.get("epoch", "na")
        target = out_dir / f"{label}-s{index}-epoch{epoch}.ckpt"
        torch.save(payload, target)
        before = source.stat().st_size / 1048576
        after = target.stat().st_size / 1048576
        print(
            f"{target.name}: {kept} tensors kept, {dropped} reference tensors "
            f"dropped, {before:.1f} MB -> {after:.1f} MB, all bit-identical"
        )
        sums.append(f"{sha256(target)}  {target.name}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    print(f"wrote {out_dir / 'SHA256SUMS.txt'}")


if __name__ == "__main__":
    main()
