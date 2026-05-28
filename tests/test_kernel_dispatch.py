"""Kernel dispatch tests: backend selection logic + stub behaviour."""

from __future__ import annotations

import pytest

from tessera.config import TesseraConfig
from tessera.kernel_dispatch import KernelBackend, select_backend_kind


def test_explicit_backend_overrides_auto() -> None:
    assert select_backend_kind(KernelBackend.TRITON) == KernelBackend.TRITON
    assert select_backend_kind("triton") == KernelBackend.TRITON


def test_auto_returns_a_valid_variant() -> None:
    kind = select_backend_kind("auto")
    assert isinstance(kind, KernelBackend)


def test_flash_attn4_stub_raises_not_implemented(deepseek_v3_toml) -> None:
    cfg = TesseraConfig.from_toml(deepseek_v3_toml)
    from tessera.backends.flash_attn4 import FlashAttn4Backend

    be = FlashAttn4Backend(cfg)
    assert not be.is_available()
    with pytest.raises(NotImplementedError, match="Sprint 0"):
        be.forward()  # type: ignore[call-arg]


def test_unknown_backend_string_raises() -> None:
    with pytest.raises(ValueError):
        select_backend_kind("does-not-exist")
