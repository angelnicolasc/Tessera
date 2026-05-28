"""FlashMLA backend wrapper. Imports ``flash_mla`` lazily so the rest of the package works on
machines where the kernel isn't installed.

WS9: ``forward`` is now fully wired:
- Shape assertions guard against silent mismatches before the kernel call.
- FP8 per-layer scales are forwarded when ``ckv_dtype == fp8_e4m3`` and ``fp8_scales``
  is provided.
- Page-table construction from Tessera block IDs is handled by ``build_page_tables``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tessera.backends.base import MlaAttentionBackend

if TYPE_CHECKING:
    import torch

    from tessera.config import TesseraConfig


def build_page_tables(
    block_ids: list[list[int]],
    *,
    device: Any,
) -> torch.Tensor:
    """Convert a ragged list of Tessera block IDs into the page-table tensor FlashMLA expects.

    Args:
        block_ids: outer list = batch; inner list = block IDs in sequence order. Ragged
            rows are padded to the maximum row length with -1 (unused entries).
        device: torch device (passed through from the active GPU).

    Returns:
        ``[batch, max_blocks]`` int32 tensor on ``device``.
    """
    import torch  # noqa: PLC0415

    max_blocks = max((len(row) for row in block_ids), default=0)
    batch = len(block_ids)
    table = torch.full((batch, max_blocks), fill_value=-1, dtype=torch.int32, device=device)
    for b, row in enumerate(block_ids):
        if row:
            table[b, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return table


class FlashMLABackend(MlaAttentionBackend):
    """Production attention backend on Hopper/Blackwell (SM ≥ 9.0).

    Tessera does not own the kernel: it provides FlashMLA the paged page tables it expects
    (64-token blocks, ``c_kv`` + ``k_rope`` pointers) and forwards arguments straight through.

    FP8 path (``ckv_dtype == "fp8_e4m3"``): pass ``fp8_scales`` as a ``[num_layers]`` float
    tensor. The kernel applies per-layer scaling inside the attention softmax; calibration is
    required — see ``ADR-0007`` and ``tessera.fp8_calibrate``.
    """

    backend_name = "flash_mla"

    def __init__(self, config: TesseraConfig) -> None:
        super().__init__(config)
        self._flash_mla: Any | None = None

    def _load(self) -> Any:
        if self._flash_mla is not None:
            return self._flash_mla
        try:
            import flash_mla  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment-dependent
            msg = (
                "flash_mla is not installed. Install with `pip install flash-mla` or build "
                "from https://github.com/deepseek-ai/FlashMLA"
            )
            raise RuntimeError(msg) from exc
        self._flash_mla = flash_mla
        return flash_mla

    def is_available(self) -> bool:
        try:
            self._load()
        except RuntimeError:
            return False
        return True

    def forward(
        self,
        q_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        c_kv_block_table: torch.Tensor,
        k_rope_block_table: torch.Tensor,
        seqlens: torch.Tensor,
        *,
        fp8_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run absorbed MLA decode attention via FlashMLA.

        Args:
            q_absorbed: ``[batch, num_heads, d_c]`` — query with ``W_UK`` absorbed.
            q_rope: ``[batch, num_heads, d_r]`` — RoPE component.
            c_kv_block_table: ``[batch, max_blocks]`` int32 page table for ``c_kv``.
            k_rope_block_table: ``[batch, max_blocks]`` int32 page table for ``k_rope``.
            seqlens: ``[batch]`` int32 per-sequence KV lengths.
            fp8_scales: ``[num_layers]`` float32 per-layer scale factors (FP8 path only).
                If ``None`` and ``ckv_dtype == "fp8_e4m3"``, the kernel defaults to 1.0 per
                layer — strongly discouraged; calibrate first (ADR-0007).

        Returns:
            ``[batch, num_heads, d_h]`` BF16 output.
        """
        kernel = self._load()
        cfg    = self._config

        # ── Shape assertions (catch mismatches before they become CUDA errors) ────────
        batch      = q_absorbed.shape[0]
        num_heads  = cfg.model.num_heads
        d_c        = cfg.model.latent_dim
        d_r        = cfg.model.rope_key_dim

        if q_absorbed.shape != (batch, num_heads, d_c):
            msg = (
                f"q_absorbed shape mismatch: expected ({batch}, {num_heads}, {d_c}), "
                f"got {tuple(q_absorbed.shape)}"
            )
            raise ValueError(msg)
        if q_rope.shape != (batch, num_heads, d_r):
            msg = (
                f"q_rope shape mismatch: expected ({batch}, {num_heads}, {d_r}), "
                f"got {tuple(q_rope.shape)}"
            )
            raise ValueError(msg)
        if c_kv_block_table.shape[0] != batch or k_rope_block_table.shape[0] != batch:
            msg = (
                f"block table batch dim mismatch: q batch={batch}, "
                f"c_kv_block_table={c_kv_block_table.shape[0]}, "
                f"k_rope_block_table={k_rope_block_table.shape[0]}"
            )
            raise ValueError(msg)
        if seqlens.shape[0] != batch:
            msg = f"seqlens batch dim mismatch: expected {batch}, got {seqlens.shape[0]}"
            raise ValueError(msg)

        # ── Scale ────────────────────────────────────────────────────────────────────
        scale = cfg.kernel.softmax_scale
        if scale is None:
            scale = 1.0 / (d_c ** 0.5)

        # ── FP8 scale forwarding ──────────────────────────────────────────────────────
        kwargs: dict[str, Any] = {}
        if cfg.block.ckv_dtype == "fp8_e4m3":
            if fp8_scales is not None:
                kwargs["fp8_scales"] = fp8_scales
            # If fp8_scales is None with FP8 dtype we pass nothing: FlashMLA defaults to 1.0.
            # This is correct for testing; production must calibrate (ADR-0007).

        return kernel.mla_decode(  # type: ignore[no-any-return]
            q_absorbed=q_absorbed,
            q_rope=q_rope,
            c_kv_pages=c_kv_block_table,
            k_rope_pages=k_rope_block_table,
            seqlens=seqlens,
            block_size=cfg.block.block_size_tokens,
            softmax_scale=scale,
            **kwargs,
        )

    def forward_from_config(
        self,
        q_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        c_kv_block_table: torch.Tensor,
        k_rope_block_table: torch.Tensor,
        seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper: reads ``fp8_scales`` from the active config automatically.

        Equivalent to calling :meth:`forward` with the ``fp8_scales`` tensor constructed
        from ``self._config.fp8_scales``. Use this when you don't want to manage scale
        tensors manually.
        """
        import torch as _torch  # noqa: PLC0415

        fp8_scales: torch.Tensor | None = None
        if self._config.block.ckv_dtype == "fp8_e4m3" and self._config.fp8_scales is not None:
            raw = self._config.fp8_scales
            num_layers = self._config.model.num_layers
            fp8_scales = _torch.tensor(
                [raw.get(i, 1.0) for i in range(num_layers)],
                dtype=_torch.float32,
            )
        return self.forward(
            q_absorbed=q_absorbed,
            q_rope=q_rope,
            c_kv_block_table=c_kv_block_table,
            k_rope_block_table=k_rope_block_table,
            seqlens=seqlens,
            fp8_scales=fp8_scales,
        )
