"""Two-layer segment index orchestration (WS7 polish).

* **Layer 1** — exact ``xxhash3`` lookup. Synchronous, sub-microsecond, on the prefill hot
  path.
* **Layer 2** — HNSW approximate match via the native ``UsearchIndex``. Wrapped in
  ``asyncio.wait_for`` with an explicit latency budget. Returning ``None`` on timeout is a
  correctness-preserving miss: the request just computes its own ``c_kv`` instead of reusing
  a sibling's.

WS7 changes vs Sprint 0:
- Per-instance ``ThreadPoolExecutor`` replaced by a process-global singleton
  (``_GLOBAL_EXECUTOR``) bounded to ``cpu_count() // 2`` workers.
- ``asyncio.Semaphore`` (default 16) caps concurrent HNSW queries so the threadpool
  never becomes a runaway resource.
- ``segment_index_queue_depth`` gauge updated on every ``lookup_approximate`` entry/exit
  via ``observability.set_segment_index_queue_depth``.
- ``close()`` no longer shuts down the global executor; it releases per-instance state.

Only ``c_kv`` is indexed. ``k_rope`` is position-dependent and cannot be shared (ADR-0001 /
ADR-0008 / docs/src/segment_index.md).
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xxhash
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tessera._native import UsearchIndex as _NativeUsearchIndex

# ─── Process-global executor for HNSW queries (WS7) ──────────────────────────
# Bounded to cpu_count() // 2 to avoid thrashing on HNSW-heavy workloads while
# leaving cores available for the main inference thread and Tokio runtime.
_MAX_HNSW_WORKERS: int = max(1, (os.cpu_count() or 2) // 2)
_GLOBAL_EXECUTOR: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Return the process-global HNSW executor, creating it on first call."""
    global _GLOBAL_EXECUTOR  # noqa: PLW0603
    if _GLOBAL_EXECUTOR is None:
        _GLOBAL_EXECUTOR = ThreadPoolExecutor(
            max_workers=_MAX_HNSW_WORKERS,
            thread_name_prefix="tessera-hnsw",
        )
    return _GLOBAL_EXECUTOR


# ─── Concurrent query cap (backpressure) ─────────────────────────────────────
# Default 16: enough headroom for burst multi-agent prefills without letting
# all threads pile into HNSW simultaneously.
_DEFAULT_HNSW_CONCURRENCY: int = 16


@dataclass(frozen=True, slots=True)
class SegmentMatch:
    """One match returned by the segment index."""

    block_id: int
    similarity: float
    was_exact: bool


def hash_ckv_bytes(c_kv: NDArray[np.floating]) -> int:
    """Stable content hash over a ``c_kv`` tensor's bytes. Position-independent by construction."""
    if not c_kv.flags["C_CONTIGUOUS"]:
        c_kv = np.ascontiguousarray(c_kv)
    return xxhash.xxh3_64(c_kv.tobytes()).intdigest()


def descriptor_from_ckv(c_kv: NDArray[np.floating]) -> NDArray[np.float32]:
    """Compute the per-block ANN descriptor: mean over (layer, token) reduces ``c_kv`` to
    a ``[d_c]`` vector. This is the only place MLA's storage shape leaks into the index
    pipeline; DSA will replace this function, not the trait.
    """
    if c_kv.ndim != 3:  # [num_layers, block_size_tokens, d_c]
        msg = f"expected c_kv with shape [L, T, d_c]; got ndim={c_kv.ndim}"
        raise ValueError(msg)
    return np.asarray(c_kv, dtype=np.float32).mean(axis=(0, 1))


class SegmentIndex:
    """Two-layer content-addressed segment index over MLA ``c_kv`` blocks."""

    DEFAULT_LATENCY_BUDGET_US: int = 50_000
    """Default async HNSW lookup budget (50 ms). Miss-on-timeout is safe; the previous
    500 μs default was below the floor of `asyncio.timeout` + thread-executor dispatch
    + usearch search even on idle machines, so it always timed out into a false miss."""

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        *,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        similarity_threshold: float = 0.97,
        latency_budget_us: int = DEFAULT_LATENCY_BUDGET_US,
        max_concurrent_queries: int = _DEFAULT_HNSW_CONCURRENCY,
        backend: _NativeUsearchIndex | None = None,
    ) -> None:
        """Construct a segment index.

        Args:
            latent_dim: ``d_c`` — descriptor dimensionality. Must match the model.
            num_layers: ``L`` — retained for diagnostics and future multi-layer hashing.
            hnsw_m / hnsw_ef_construction / hnsw_ef_search: HNSW tuning (see ADR-0008).
            similarity_threshold: minimum cosine similarity for a match to be returned.
            latency_budget_us: maximum wall-clock budget for an async HNSW lookup. On
                timeout the lookup returns ``None`` — a safe miss.
            max_concurrent_queries: cap on simultaneous HNSW queries via Semaphore.
                Provides backpressure when many agents prefill concurrently.
            backend: optional pre-constructed native index (lets tests inject a mock).
        """
        self._latent_dim = latent_dim
        self._num_layers = num_layers
        self._threshold = similarity_threshold
        self._budget_s = latency_budget_us / 1_000_000.0

        self._exact: dict[int, int] = {}

        # Semaphore for concurrent-query backpressure (WS7).
        self._sem = asyncio.Semaphore(max_concurrent_queries)

        if backend is None:
            from tessera import _native

            backend = _native.UsearchIndex(
                latent_dim,
                hnsw_m,
                hnsw_ef_construction,
                hnsw_ef_search,
            )
        self._backend = backend

    # ─────────────────── Layer 1: exact hash (sync) ───────────────────────────

    def lookup_exact(self, content_hash: int) -> int | None:
        """O(1) exact lookup. Synchronous; safe on the prefill hot path."""
        return self._exact.get(content_hash)

    # ─────────────────── Layer 2: HNSW approximate (async) ───────────────────

    async def lookup_approximate(self, c_kv: NDArray[np.floating]) -> SegmentMatch | None:
        """HNSW lookup with the configured latency budget and backpressure cap.

        Returns ``None`` if no match clears ``similarity_threshold``, the budget is
        exceeded, or the semaphore cannot be acquired within the budget window.
        Either case is a safe miss — the caller falls back to computing fresh.
        """
        from tessera import observability

        descriptor = descriptor_from_ckv(c_kv)
        executor = _get_executor()
        loop = asyncio.get_running_loop()

        observability.set_segment_index_queue_depth(
            self._sem._value  # type: ignore[attr-defined]  # asyncio internals
        )

        try:
            async with asyncio.timeout(self._budget_s):
                async with self._sem:
                    with observability.span("tessera.hnsw_query"):
                        matches: list[tuple[int, float]] = await loop.run_in_executor(
                            executor, self._backend.query, descriptor, 1
                        )
        except TimeoutError:
            try:
                from tessera import observability as _obs

                _obs.hnsw_budget_exceeded()
            except Exception:  # noqa: S110  metric path is intentionally best-effort
                pass  # pragma: no cover
            return None
        finally:
            # Always update queue depth after the query completes/times out.
            observability.set_segment_index_queue_depth(
                self._sem._value  # type: ignore[attr-defined]
            )

        if not matches:
            return None
        block_id, similarity = matches[0]
        if similarity < self._threshold:
            return None
        return SegmentMatch(block_id=block_id, similarity=similarity, was_exact=False)

    # ─────────────────── mutators ─────────────────────────────────────────────

    def add(self, block_id: int, content_hash: int, c_kv: NDArray[np.floating]) -> None:
        """Register a newly sealed block in both layers."""
        self._exact[content_hash] = block_id
        descriptor = descriptor_from_ckv(c_kv)
        self._backend.add(block_id, descriptor)

    def remove(self, block_id: int, content_hash: int) -> None:
        """Drop a block from both layers (called on eviction)."""
        self._exact.pop(content_hash, None)
        self._backend.remove(block_id)

    def __len__(self) -> int:
        return len(self._exact)

    def close(self) -> None:
        """Release per-instance state. Does NOT shut down the global executor."""
        # Clear local state; the global executor belongs to the process lifetime.
        self._exact.clear()

    def __enter__(self) -> SegmentIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
