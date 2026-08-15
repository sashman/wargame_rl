"""`_cuda_is_usable` must match on the major architecture, not the exact string.

CUDA is binary-compatible *within* a major version: a cubin built for `sm_86`
runs on any `sm_8x` device with `x >= 6`. An exact-string guard therefore
rejects working hardware -- measured on an RTX 4090 (`sm_89`) against a build
listing `sm_86`, where Lightning selected the GPU for training all day while
every script reaching this helper silently ran on CPU. One evaluation took
36 seconds on the GPU and over 10 minutes on the CPU.

A false negative here is much quieter than the crash the guard exists to
prevent, which is why it survived. `test_an_older_major_is_still_refused` keeps
the original protection: the guard was written for a real `sm_61` card that
accepted `.to("cuda")` and then died at the first forward.
"""

from __future__ import annotations

import pytest

from wargame_rl.wargame.model.common import device as device_module


@pytest.mark.parametrize(
    ("capability", "arches", "usable", "case"),
    [
        (
            (8, 9),
            ["sm_70", "sm_80", "sm_86", "sm_90"],
            True,
            "sm_86 cubin runs on sm_89",
        ),
        ((8, 6), ["sm_86"], True, "exact match still works"),
        ((8, 0), ["sm_86", "sm_90"], False, "sm_86 does NOT run on the older sm_80"),
        ((6, 1), ["sm_70", "sm_75", "sm_80"], False, "the original sm_61 failure"),
        ((9, 0), ["sm_86"], False, "a different major is never compatible"),
        ((8, 9), [], True, "no arch list: trust availability rather than refuse"),
    ],
)
def test_binary_compatibility_is_within_a_major_version(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    arches: list[str],
    usable: bool,
    case: str,
) -> None:
    # Arrange
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device_module.torch.cuda, "get_arch_list", lambda: arches)
    monkeypatch.setattr(
        device_module.torch.cuda, "get_device_capability", lambda: capability
    )

    # Act / Assert
    assert device_module._cuda_is_usable() is usable, case


def test_no_cuda_device_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange / Act / Assert: nothing below this point should consult the
    # arch list at all.
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)

    assert device_module._cuda_is_usable() is False
