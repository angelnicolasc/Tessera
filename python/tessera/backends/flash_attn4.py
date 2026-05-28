"""FlashAttention-4 backend — **Sprint 0 stub**.

FA4 is the FlashAttention-4 warp-specialised pipeline for Blackwell (see
`reversion-tessera02.md`). The dispatcher knows about it so users can request it explicitly;
this wrapper raises ``NotImplementedError`` so misuse is loud rather than silently slow.

When FA4 lands, this file is the only place that needs to change: ``kernel_dispatch`` already
routes the enum variant here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tessera.backends.base import MlaAttentionBackend

if TYPE_CHECKING:
    import torch


_TRACKING_URL = "https://github.com/angelnicolasc/tessera/issues?q=label%3Afa4"


class FlashAttn4Backend(MlaAttentionBackend):
    """Placeholder for the FlashAttention-4 backend."""

    backend_name = "flash_attn4"

    def is_available(self) -> bool:
        return False

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:  # noqa: ARG002
        msg = (
            "FlashAttention-4 backend is not implemented in Sprint 0. "
            f"Track progress at {_TRACKING_URL}."
        )
        raise NotImplementedError(msg)
