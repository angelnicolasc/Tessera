"""Thin Python wrapper around the native ``ShareTable``. Provided so call sites import a
stable ``tessera.cross_agent`` API instead of reaching into ``tessera._native``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tessera._native import ShareTable as _NativeShareTable


class CrossAgentShareTable:
    """Cross-agent share table. Backed by the native Rust implementation."""

    def __init__(self) -> None:
        from tessera import _native

        self._inner: _NativeShareTable = _native.ShareTable()

    def add_share(self, req_id: int, block_id: int) -> None:
        """Register ``req_id`` as an owner of ``block_id``."""
        self._inner.add_share(req_id, block_id)

    def release_request(self, req_id: int) -> list[int]:
        """Release every shared block held by ``req_id``. The returned ids must be freed on
        the block manager."""
        return self._inner.release_request(req_id)

    def owners(self, block_id: int) -> list[int] | None:
        """List of owner request ids for a block, or ``None`` if not shared."""
        return self._inner.owners(block_id)

    def shared_block_count(self) -> int:
        """Distinct shared blocks tracked."""
        return self._inner.shared_block_count()

    def sharing_rate(self) -> float:
        """Fraction of allocated block references that are shared."""
        return self._inner.sharing_rate()
