"""vLLM V1 ``BlockAllocator`` plugin.

Registered via the ``vllm.block_allocator = tessera.vllm_plugin:TesseraBlockAllocator`` entry
point in ``pyproject.toml``. vLLM discovers it through ``importlib.metadata.entry_points``.

This module imports vLLM lazily inside the methods that need it. That has two consequences:

1. Importing ``tessera.vllm_plugin`` is safe on a machine without vLLM installed (so tests
   can verify the protocol shape via attribute introspection).
2. ``ruff`` / ``pyright`` in CI don't need vLLM as a dependency.

WS1 change: ``free(block)`` is rewired to call ``manager.release_request(req_id)`` which
leverages the new per-request lifecycle index. Shared blocks are still released via
``CrossAgentShareTable.release_request`` first.

WS7 change: ``find_shared_prefix`` now fires an async Layer 2 HNSW lookup for each exact
miss (off hot path via ``asyncio.gather``). Results are delivered to future requests; the
current prefill proceeds immediately without waiting.
"""

from __future__ import annotations

import asyncio
import ctypes
from typing import TYPE_CHECKING, Any

from tessera.config import TesseraConfig
from tessera.cross_agent import CrossAgentShareTable
from tessera.segment_index import SegmentIndex, hash_ckv_bytes

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    _Scales = dict[int, float]


#: Protocol surface vLLM V1 requires from a block allocator. Listed explicitly so that the
#: introspection-based protocol test in ``tests/test_vllm_plugin_protocol.py`` doesn't need
#: vLLM installed to verify we match.
REQUIRED_V1_METHODS: tuple[str, ...] = (
    "allocate_mutable_block",
    "allocate_immutable_blocks",
    "free",
    "get_num_free_blocks",
    "get_num_total_blocks",
)


class TesseraBlockAllocator:
    """Tessera-backed vLLM V1 ``BlockAllocator``.

    Bridges three components:

    * ``tessera._native.BlockManager`` — the Rust block manager (allocation, dedup, CoW,
      per-request lifecycle via ``release_request``).
    * ``tessera.segment_index.SegmentIndex`` — two-layer content-addressed sharing.
    * ``tessera.cross_agent.CrossAgentShareTable`` — ref-counted multi-owner bookkeeping.
    """

    def __init__(
        self,
        config: TesseraConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
        transport: Any | None = None,
    ) -> None:
        """Construct a rank-aware allocator.

        Args:
            config: Tessera configuration.
            rank: This process's rank within the world. Default 0 (singleton retrocompat).
            world_size: Total world size. Default 1.
            transport: Optional :class:`tessera._native.MockTransport`-shaped handle for
                cross-rank coordination. When ``world_size == 1`` or ``transport is None``,
                multi-rank coordination is disabled and the allocator behaves identically to
                Sprints 0-2. Production deployments pass a ``P2pCudaTransport`` here
                (once Sprint-3 cloud-burst lands the runtime impl — Sprint 3 ships the
                ``MockTransport``-driven path end-to-end).
        """
        from tessera import _native

        self._config = config
        self._native = _native
        self._rank = int(rank)
        self._world_size = int(world_size)

        native_cfg = config.to_native_config()
        if self._world_size > 1:
            world = _native.World.single_node(local=self._rank, size=self._world_size)
            self._manager = _native.BlockManager.with_world(
                native_cfg, config.runtime.gpu_memory_bytes, rank=self._rank, world=world
            )
        else:
            self._manager = _native.BlockManager(native_cfg, config.runtime.gpu_memory_bytes)

        self._transport = transport  # may be None for singleton

        latent_dim = (
            config.model.latent_dim
            if config.is_mla
            else config.model.num_heads * config.model.head_dim
        )
        self._segment_index = SegmentIndex(
            latent_dim=latent_dim,
            num_layers=config.model.num_layers,
            hnsw_m=config.segment_index.hnsw_m,
            hnsw_ef_construction=config.segment_index.hnsw_ef_construction,
            hnsw_ef_search=config.segment_index.hnsw_ef_search,
            similarity_threshold=config.segment_index.similarity_threshold,
            latency_budget_us=config.segment_index.hnsw_latency_budget_us,
        )

        # Distributed segment index — only meaningful with peers + transport.
        self._distributed_index: Any | None = None
        if self._world_size > 1 and self._transport is not None:
            world = _native.World.single_node(local=self._rank, size=self._world_size)
            self._distributed_index = _native.DistributedSegmentIndex(
                latent_dim,
                world,
                self._transport,
                int(config.segment_index.hnsw_latency_budget_us) * 2,  # double for fan-out RTT
            )

        self._share_table = CrossAgentShareTable()
        self._fp8_scales: dict[int, float] | None = config.fp8_scales
        from tessera import observability as _obs

        _obs.init_tracing(config.observability.tracing_endpoint)

    @property
    def rank(self) -> int:
        """Rank id of this allocator (0 for singleton)."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Total world size."""
        return self._world_size

    def transfer_request_to_rank(self, req_id: int, target: int) -> int:
        """Migrate every private block owned by ``req_id`` to ``target`` (PD-disaggregation,
        ADR-0016). Returns the number of blocks transferred. Raises ``RuntimeError`` when
        called without a configured transport.
        """
        if self._transport is None:
            msg = "transfer_request_to_rank requires a configured cross-rank transport"
            raise RuntimeError(msg)
        return self._manager.transfer_request_to_rank(req_id, target, self._transport)

    # ─────────────── vLLM V1 BlockAllocator protocol ───────────────────────

    def allocate_mutable_block(self, prev_block: Any | None, device: Any) -> Any:
        """Allocate a fresh writable block.

        ``device`` is the torch device handed in by vLLM; ignored by Tessera since the block
        manager owns its own memory budget.
        """
        from tessera import observability as _obs

        del device
        req_id = int(getattr(prev_block, "req_id", 0) or 0)
        token_start = int(getattr(prev_block, "token_end", 0) or 0)
        block_size = self._config.block.block_size_tokens
        with _obs.span("tessera.allocate"):
            block_id = self._manager.allocate(req_id, token_start, token_start + block_size)
        return _TesseraBlock(
            allocator=self,
            block_id=block_id,
            req_id=req_id,
            token_start=token_start,
            token_end=token_start + block_size,
            is_full=False,
        )

    def allocate_immutable_blocks(self, prev_block: Any | None, block_ids: list[int]) -> list[Any]:
        """Acquire references to existing immutable (sealed) blocks. Used when vLLM resolves
        a prefix-cache hit.
        """
        del prev_block
        out: list[Any] = []
        for bid in block_ids:
            self._manager.increment_ref(int(bid))
            out.append(
                _TesseraBlock(
                    allocator=self,
                    block_id=int(bid),
                    req_id=0,
                    token_start=0,
                    token_end=self._config.block.block_size_tokens,
                    is_full=True,
                )
            )
        return out

    def free(self, block: Any) -> None:
        """Release all blocks associated with ``block.req_id``.

        WS1 rewiring: instead of freeing a single block_id, we delegate to
        ``manager.release_request(req_id)`` which atomically frees all private blocks
        tracked for this request. Shared blocks (via share table) are handled first.
        """
        from tessera import observability as _obs

        req_id = int(getattr(block, "req_id", 0) or 0)

        with _obs.span("tessera.release_request"):
            # Release shared-block references via the cross-agent share table.
            if getattr(block, "is_shared", False):
                to_release = self._share_table.release_request(req_id)
                for bid in to_release:
                    self._manager.free(bid)

            # Release all private blocks owned by this request (WS1 lifecycle tracking).
            self._manager.release_request(req_id)

    def get_num_free_blocks(self) -> int:
        """Number of blocks currently free."""
        return self._manager.total_blocks - self._manager.used_blocks

    def get_num_total_blocks(self) -> int:
        """Total blocks managed."""
        return self._manager.total_blocks

    # ─────────────── Tessera-specific hooks ────────────────────────────────

    def post_prefill_seal(self, block_id: int, c_kv: NDArray[np.floating]) -> int:
        """Called by Tessera-aware wrappers after prefill writes ``c_kv`` for a block.

        Returns the canonical block id (may differ from ``block_id`` if the seal hit an
        existing duplicate). The HNSW add is scheduled off the hot path.
        """
        from tessera import observability as _obs

        with _obs.span("tessera.seal"):
            canonical_id, content_hash, was_dedup = self._manager.seal(block_id)
        self._write_fp8_scales(canonical_id)
        if not was_dedup:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(  # noqa: RUF006  fire-and-forget HNSW indexing
                    self._async_hnsw_add(canonical_id, content_hash, c_kv)
                )
            except RuntimeError:
                # No running loop (synchronous context, e.g. unit test) — index immediately.
                self._segment_index.add(canonical_id, content_hash, c_kv)
        return canonical_id

    def _write_fp8_scales(self, block_id: int) -> None:
        """Write per-layer FP8 scale factors into the block's scale region.

        On the CPU mock backend, ``fp8_scales_ptr`` always returns ``None`` so this is a no-op.
        On a CUDA backend the ptr is a device memory address; we memcpy the scale array in.
        """
        if self._fp8_scales is None:
            return
        ptr = self._manager.fp8_scales_ptr(block_id)
        if ptr is None:
            return
        import numpy as np

        num_layers = self._config.model.num_layers
        scales = np.array(
            [self._fp8_scales.get(i, 1.0) for i in range(num_layers)],
            dtype=np.float32,
        )
        ctypes.memmove(ptr, scales.ctypes.data, scales.nbytes)

    async def _async_hnsw_add(
        self, block_id: int, content_hash: int, c_kv: NDArray[np.floating]
    ) -> None:
        self._segment_index.add(block_id, content_hash, c_kv)

    def find_shared_prefix(
        self,
        req_id: int,
        c_kv_blocks: list[NDArray[np.floating]],
    ) -> list[int | tuple[int, int] | None]:
        """For each block in an incoming prefill, check for a matching ``c_kv`` block.

        Lookup ladder:

        1. **Layer 1 (local exact xxh3)** — sync, hot path. Hit returns a local ``block_id``
           and registers the share locally.
        2. **Layer 1 distributed** — if multi-rank, fan out to peers via
           :class:`DistributedSegmentIndex.lookup_hash`. Hit returns ``(rank, block_id)``;
           caller routes the request to the owning rank (Sprint 3 returns the tuple to the
           caller; the integration code is expected to call ``transfer_request_to_rank`` to
           pull, or to short-circuit decode on that rank). Sprint 4 will add transparent
           reverse-pull here.
        3. **Layer 2 (HNSW)** — fired asynchronously for misses (WS7); results benefit
           future requests.

        Returns one entry per input block: an ``int`` for a local hit, a ``(rank, block)``
        tuple for a remote hit, or ``None`` for misses.
        """
        from tessera import observability as _obs

        results: list[int | tuple[int, int] | None] = []
        hnsw_miss_tasks: list[Any] = []

        with _obs.span("tessera.lookup_approximate"):
            for c_kv in c_kv_blocks:
                content_hash = hash_ckv_bytes(c_kv)
                local_match = self._segment_index.lookup_exact(content_hash)
                if local_match is not None:
                    self._manager.increment_ref(local_match)
                    self._share_table.add_share(req_id, local_match)
                    results.append(local_match)
                    if self._distributed_index is not None:
                        self._distributed_index.record_local_hit()
                    continue

                # Distributed Layer 1 — only consult when we have peers and a transport.
                remote_hit: tuple[int, int] | None = None
                if self._distributed_index is not None:
                    remote_hit = self._distributed_index.lookup_hash(content_hash)

                if remote_hit is not None:
                    # Remote rank holds the block. Caller decides how to consume it.
                    results.append(remote_hit)
                else:
                    results.append(None)
                    # Local Layer 2: schedule async HNSW lookup for future requests.
                    try:
                        loop = asyncio.get_running_loop()
                        task = loop.create_task(self._async_hnsw_lookup(c_kv))
                        hnsw_miss_tasks.append(task)
                    except RuntimeError:
                        # No running event loop — synchronous context (e.g. tests).
                        pass

        # Fire-and-forget: HNSW results are not awaited here.
        _ = hnsw_miss_tasks
        return results

    async def _async_hnsw_lookup(self, c_kv: NDArray[np.floating]) -> None:
        """Background HNSW lookup. Result is discarded if below threshold; otherwise
        the calling future requests benefit from a populated approximate index.
        """
        await self._segment_index.lookup_approximate(c_kv)


class _TesseraBlock:
    """Lightweight Block descriptor exposed to vLLM."""

    __slots__ = (
        "_allocator",
        "block_id",
        "is_full",
        "is_shared",
        "req_id",
        "token_end",
        "token_start",
    )

    def __init__(
        self,
        *,
        allocator: TesseraBlockAllocator,
        block_id: int,
        req_id: int,
        token_start: int,
        token_end: int,
        is_full: bool,
        is_shared: bool = False,
    ) -> None:
        self._allocator = allocator
        self.block_id = block_id
        self.req_id = req_id
        self.token_start = token_start
        self.token_end = token_end
        self.is_full = is_full
        self.is_shared = is_shared

    def __repr__(self) -> str:
        return (
            f"_TesseraBlock(id={self.block_id}, req={self.req_id}, "
            f"range=[{self.token_start},{self.token_end}), "
            f"full={self.is_full}, shared={self.is_shared})"
        )
