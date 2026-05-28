"""FlashInfer MLA backend wrapper. Used on Ampere+ when FlashMLA is unavailable.

WS9: wrapper is now memoized per ``(num_qo_heads, d_c, d_r, page_size)`` config key.
TD-006 resolved: workspace allocation happens once; plan() is called on every forward pass
since seqlens change per batch, but the heavy allocation is amortised across requests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tessera.backends.base import MlaAttentionBackend

if TYPE_CHECKING:
    import torch

    from tessera.config import TesseraConfig

# Cache key: all config params that drive wrapper construction.
_WrapperKey = tuple[int, int, int, int]  # (num_qo_heads, d_c, d_r, page_size)


class FlashInferMLABackend(MlaAttentionBackend):
    """Wraps ``flashinfer.BatchMLAPagedAttentionWrapper`` (FlashInfer ≥ 0.2).

    The wrapper holds GPU workspace buffers whose size depends on the model config.
    Constructing one per forward call (the Sprint 0 behaviour) caused repeated HBM
    allocations on the hot path. The wrapper is now memoized per config; plan() is still
    called before every run() because sequence lengths change between requests.
    """

    backend_name = "flash_infer"

    def __init__(self, config: TesseraConfig) -> None:
        super().__init__(config)
        self._wrapper_cls: Any | None = None
        # Memoised wrapper instances keyed by (num_qo_heads, d_c, d_r, page_size).
        self._wrapper_cache: dict[_WrapperKey, Any] = {}

    def _load(self) -> Any:
        """Return the ``BatchMLAPagedAttentionWrapper`` class (import once)."""
        if self._wrapper_cls is not None:
            return self._wrapper_cls
        try:
            import flashinfer  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment-dependent
            msg = "flashinfer is not installed. `pip install flashinfer>=0.2`"
            raise RuntimeError(msg) from exc
        if not hasattr(flashinfer, "BatchMLAPagedAttentionWrapper"):
            msg = "installed flashinfer is too old; need >= 0.2 with BatchMLAPagedAttentionWrapper"
            raise RuntimeError(msg)
        self._wrapper_cls = flashinfer.BatchMLAPagedAttentionWrapper
        return self._wrapper_cls

    def is_available(self) -> bool:
        try:
            self._load()
        except RuntimeError:
            return False
        return True

    def _get_wrapper(self) -> Any:
        """Return a memoised wrapper for the current config, constructing it on first use."""
        cfg = self._config
        key: _WrapperKey = (
            cfg.model.num_heads,
            cfg.model.latent_dim,
            cfg.model.rope_key_dim,
            cfg.block.block_size_tokens,
        )
        if key not in self._wrapper_cache:
            wrapper_cls = self._load()
            # workspace_buffer=None lets FlashInfer allocate its own; production callers
            # may pass a pre-allocated buffer to avoid internal HBM fragmentation.
            self._wrapper_cache[key] = wrapper_cls(workspace_buffer=None)  # type: ignore[call-arg]
        return self._wrapper_cache[key]

    def forward(
        self,
        q_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        c_kv_block_table: torch.Tensor,
        k_rope_block_table: torch.Tensor,
        seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Run absorbed MLA decode attention via FlashInfer.

        The wrapper is memoised across calls. plan() is invoked before every run()
        because ``qo_indptr`` / ``kv_indptr`` encode per-batch sequence lengths.
        """
        wrapper = self._get_wrapper()
        cfg = self._config
        scale = cfg.kernel.softmax_scale
        if scale is None:
            scale = 1.0 / (cfg.model.latent_dim ** 0.5)

        wrapper.plan(
            qo_indptr=seqlens,
            kv_indptr=seqlens,
            kv_indices=c_kv_block_table,
            num_qo_heads=cfg.model.num_heads,
            d_c=cfg.model.latent_dim,
            d_r=cfg.model.rope_key_dim,
            page_size=cfg.block.block_size_tokens,
            sm_scale=scale,
        )
        return wrapper.run(q_absorbed, q_rope, c_kv_block_table, k_rope_block_table)  # type: ignore[no-any-return]
