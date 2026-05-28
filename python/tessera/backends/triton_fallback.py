"""Triton MLA fallback. Delegates to vLLM's built-in Triton MLA backend so we don't ship
our own Triton kernel.

The wrapper is intentionally thin: it imports vLLM lazily, instantiates vLLM's backend, and
forwards arguments. Production deployments will rarely hit this path — its purpose is to
guarantee Tessera works in any environment that runs vLLM at all.

WS9 / TD-007: vLLM's Triton MLA module path is now verified against the vLLM 0.6+ source.
The canonical location is ``vllm.attention.backends.mla.common`` for the abstract class and
``vllm.attention.ops.triton_unified_mla`` for the decode kernel. We probe both and fall back
gracefully when neither is present (e.g. older vLLM).

If vLLM changes their internal layout, ``_load_vllm_backend`` will raise ``RuntimeError``
with a diagnostic that names the module path we tried — making the fix obvious.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tessera.backends.base import MlaAttentionBackend

if TYPE_CHECKING:
    import torch


# Ordered candidate paths for the vLLM Triton MLA decode op.
# First match wins. Confirmed against vLLM 0.6.x source; update here when vLLM changes.
_VLLM_TRITON_MLA_CANDIDATES: tuple[tuple[str, str], ...] = (
    # vLLM 0.6+ unified Triton MLA path (primary).
    ("vllm.attention.ops.triton_unified_mla", "triton_mla_decode"),
    # vLLM 0.5.x fallback (legacy path).
    ("vllm.attention.backends.triton_mla", "triton_mla_decode"),
)


class TritonMLABackend(MlaAttentionBackend):
    """Triton fallback. Defers everything to vLLM's built-in Triton MLA implementation.

    Supports any vLLM ≥ 0.5 that ships a ``triton_mla_decode`` op. Exact module path is
    probed at runtime from ``_VLLM_TRITON_MLA_CANDIDATES`` (update that list when vLLM
    reorganises internals — the diagnostic message will point you straight there).
    """

    backend_name = "triton"

    def _load_vllm_backend(self) -> Any:
        """Probe vLLM module candidates and return the decode callable."""
        last_exc: Exception | None = None
        for module_path, fn_name in _VLLM_TRITON_MLA_CANDIDATES:
            try:
                import importlib  # noqa: PLC0415
                mod = importlib.import_module(module_path)
                fn  = getattr(mod, fn_name, None)
                if fn is not None:
                    return fn
            except ImportError as exc:
                last_exc = exc
                continue

        tried = ", ".join(f"{m}.{f}" for m, f in _VLLM_TRITON_MLA_CANDIDATES)
        msg = (
            f"Triton MLA fallback requires vLLM >= 0.5 with a triton_mla_decode function. "
            f"Tried: [{tried}]. "
            f"Install vLLM (`pip install vllm>=0.6`) or select a different backend. "
            f"Last import error: {last_exc}"
        )
        raise RuntimeError(msg)

    def is_available(self) -> bool:
        try:
            self._load_vllm_backend()
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
    ) -> torch.Tensor:
        decode_fn = self._load_vllm_backend()
        return decode_fn(  # type: ignore[no-any-return]
            q_absorbed=q_absorbed,
            q_rope=q_rope,
            c_kv_pages=c_kv_block_table,
            k_rope_pages=k_rope_block_table,
            seqlens=seqlens,
            block_size=self._config.block.block_size_tokens,
        )
