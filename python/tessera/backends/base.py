"""Backend-agnostic interface for MLA attention kernels.

The interface is intentionally narrow: every backend is invoked the same way by the
Tessera-aware vLLM plugin, which hands it the active block page table and the absorbed query.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tessera.config import TesseraConfig


class MlaAttentionBackend(ABC):
    """Common surface for FlashMLA / FlashInfer / FA4 / Triton."""

    backend_name: str = "abstract"

    def __init__(self, config: TesseraConfig) -> None:
        self._config = config

    @property
    def config(self) -> TesseraConfig:
        return self._config

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the underlying upstream kernel is importable on this machine."""

    @abstractmethod
    def forward(
        self,
        q_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        c_kv_block_table: torch.Tensor,
        k_rope_block_table: torch.Tensor,
        seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Run absorbed MLA decode attention.

        Shapes (decode step):

        * ``q_absorbed``: ``[batch, num_heads, d_c]`` — query with ``W_UK`` absorbed.
        * ``q_rope``: ``[batch, num_heads, d_r]`` — query RoPE component.
        * ``c_kv_block_table`` / ``k_rope_block_table``: ``[batch, max_blocks]`` of Tessera
          block ids; the backend will resolve them to device pointers via the block manager.
        * ``seqlens``: ``[batch]`` int32.

        Returns ``[batch, num_heads, d_h]`` (`d_h` is the post-W_UV head dimension).
        """
