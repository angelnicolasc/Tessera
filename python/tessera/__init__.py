"""Tessera — MLA-aware KV block manager for multi-agent inference.

Public surface:

* ``TesseraConfig`` — typed configuration loaded from ``tessera.toml``.
* ``BlockManager`` / ``ShareTable`` / ``UsearchIndex`` — native Rust types re-exported
  from ``tessera._native`` (built via maturin).
* ``SegmentIndex`` — two-layer xxhash3 + HNSW segment index with an explicit latency budget.
* ``KernelBackend`` / ``get_mla_backend`` — kernel dispatch (FlashMLA / FlashInfer / Triton).
* ``TesseraBlockAllocator`` — vLLM V1 ``BlockAllocator`` plugin implementation.

The native module is imported lazily so that ``import tessera`` succeeds even when the
extension has not been built yet (e.g. during ``ruff`` / ``pyright`` runs in CI).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from tessera.config import TesseraConfig
from tessera.kernel_dispatch import KernelBackend, get_mla_backend
from tessera.segment_index import SegmentIndex, SegmentMatch

if TYPE_CHECKING:
    from tessera import _native as _native_module  # noqa: F401  (re-exported for type checkers)
    from tessera.vllm_plugin import TesseraBlockAllocator


__all__ = [
    "REQUIRED_BLOCK_SIZE_TOKENS",
    "BlockManager",
    "CkvDtype",
    "CompressionScheme",
    "DistributedSegmentIndex",
    "KernelBackend",
    "MlaBlockConfig",
    "MockTransport",
    "MultiRankCoordinator",
    "RankId",
    "SegmentIndex",
    "SegmentMatch",
    "ShareTable",
    "TesseraBlockAllocator",
    "TesseraConfig",
    "UsearchIndex",
    "World",
    "get_mla_backend",
    "metrics_snapshot_text",
    "spawn_multirank_world",
]

__version__ = "0.4.0"


def __getattr__(name: str) -> Any:
    """Lazy re-export of native and multi-rank types so import order doesn't matter."""
    native_attrs = {
        "BlockManager",
        "ShareTable",
        "UsearchIndex",
        "MlaBlockConfig",
        "CompressionScheme",
        "CkvDtype",
        "REQUIRED_BLOCK_SIZE_TOKENS",
        "metrics_snapshot_text",
        # Sprint 3 multi-rank surface (native).
        "RankId",
        "World",
        "MockTransport",
        "DistributedSegmentIndex",
    }
    if name in native_attrs:
        native = import_module("tessera._native")
        return getattr(native, name)
    if name == "TesseraBlockAllocator":
        return import_module("tessera.vllm_plugin").TesseraBlockAllocator
    if name in {"MultiRankCoordinator", "spawn_multirank_world"}:
        mod = import_module("tessera.multi_rank")
        return getattr(mod, name)
    raise AttributeError(f"module 'tessera' has no attribute {name!r}")
