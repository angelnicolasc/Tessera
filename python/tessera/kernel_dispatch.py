"""Kernel dispatch.

The ``KernelBackend`` enum has one entry per upstream attention kernel Tessera can mount on.
``get_mla_backend`` picks the best available backend at runtime based on the current GPU's
compute capability and the configured override.

Sprint 0 ships full wrappers for ``flash_mla``, ``flash_infer`` and the Triton fallback (the
latter just delegates to vLLM's built-in MLA backend at call time). ``flash_attn4`` is a
documented stub: the dispatcher will return its wrapper if explicitly requested, and the
wrapper raises ``NotImplementedError`` with a pointer to the tracking issue. This shape lets
FA4 land later as a 50-line PR without changing call sites.
"""

from __future__ import annotations

import enum
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tessera.backends.base import MlaAttentionBackend
    from tessera.config import TesseraConfig


class KernelBackend(enum.StrEnum):
    """Selectable attention-kernel backends."""

    FLASH_MLA = "flash_mla"
    FLASH_INFER = "flash_infer"
    FLASH_ATTN4 = "flash_attn4"
    TRITON = "triton"


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _cuda_capability() -> tuple[int, int] | None:
    """Return ``(major, minor)`` or ``None`` if torch / CUDA is not available."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def select_backend_kind(
    requested: KernelBackend | str | None = None,
) -> KernelBackend:
    """Decide which backend to instantiate. ``requested == "auto"`` (the default) follows the
    playbook's priority order: FlashMLA on SM ≥ 9.0, FlashInfer otherwise if available,
    Triton fallback last.
    """
    if isinstance(requested, str) and requested != "auto":
        requested = KernelBackend(requested)
    if isinstance(requested, KernelBackend):
        return requested
    cap = _cuda_capability()
    if cap is not None and cap[0] >= 9 and _has_module("flash_mla"):
        return KernelBackend.FLASH_MLA
    if _has_module("flashinfer"):
        return KernelBackend.FLASH_INFER
    return KernelBackend.TRITON


def get_mla_backend(config: TesseraConfig) -> MlaAttentionBackend:
    """Construct the configured attention backend wrapper."""
    kind = select_backend_kind(config.kernel.backend)
    if kind is KernelBackend.FLASH_MLA:
        from tessera.backends.flash_mla import FlashMLABackend

        return FlashMLABackend(config)
    if kind is KernelBackend.FLASH_INFER:
        from tessera.backends.flash_infer import FlashInferMLABackend

        return FlashInferMLABackend(config)
    if kind is KernelBackend.FLASH_ATTN4:
        from tessera.backends.flash_attn4 import FlashAttn4Backend

        return FlashAttn4Backend(config)
    from tessera.backends.triton_fallback import TritonMLABackend

    return TritonMLABackend(config)
