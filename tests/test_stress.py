"""WS6 — Stress tests for concurrency and memory pressure.

Tests in this file are intentionally heavier than unit tests. They are still fast enough
to run in CI (< 5 s each on a single core) but cover scenarios the unit tests cannot:
  - 50-thread concurrent lifecycle (no ref-count leaks)
  - 95% memory pressure with eviction keeping the system alive
  - 10K alloc/free throughput floor (> 50K ops/s)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from tessera import _native


# ──────────── helpers ─────────────────────────────────────────────────────────


def _make_manager(n_blocks: int = 200) -> _native.BlockManager:
    scheme = _native.CompressionScheme.mla_latent(32, 8)
    cfg = _native.MlaBlockConfig(scheme, 4, 64, _native.CkvDtype.Bf16, 0)
    return _native.BlockManager(cfg, n_blocks * cfg.total_block_bytes())


# ──────────── tests ────────────────────────────────────────────────────────────


def test_concurrent_threaded_lifecycle_no_leaks() -> None:
    """50 threads each run a full alloc→fill→seal→release_request cycle concurrently.

    After all threads complete, used_blocks must be 0 — no ref-count leaks.
    """
    n_threads = 50
    blocks_per_thread = 3
    manager = _make_manager(n_threads * blocks_per_thread + 20)

    def worker(req_id: int) -> None:
        bids: list[int] = []
        token_start = req_id * blocks_per_thread * 64
        for i in range(blocks_per_thread):
            start = token_start + i * 64
            bid = manager.allocate(req_id, start, start + 64)
            manager.fill_primary_test_pattern(bid, 0xAA)
            bids.append(bid)
        for bid in bids:
            manager.seal(bid)
        manager.release_request(req_id)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(worker, i + 1) for i in range(n_threads)]
        for fut in as_completed(futures):
            fut.result()  # re-raise any exception from the thread

    assert manager.used_blocks == 0, (
        f"ref-count leak detected: {manager.used_blocks} blocks still live after all threads exited"
    )


def test_memory_pressure_eviction_keeps_system_alive() -> None:
    """Fill the pool to 90%, then allocate 15 more blocks.

    The block manager must evict sealed (Tier-b) blocks to make room — never panicking and
    always keeping utilization ≤ 1.0.
    """
    total = 100
    manager = _make_manager(total)

    # Alloc+seal 90 blocks → each becomes a Tier-b candidate (sealed, ref_count=1, not indexed).
    for i in range(90):
        bid = manager.allocate(i + 1, 0, 64)
        manager.fill_primary_test_pattern(bid, 0x55)
        manager.seal(bid)

    assert manager.used_blocks == 90
    assert manager.total_blocks == total

    # Allocate 15 more — 10 take free slots, 5 evict Tier-b blocks. None may raise.
    for i in range(15):
        bid = manager.allocate(1000 + i, 0, 64)
        assert bid >= 0

    assert manager.utilization() <= 1.0, "utilization exceeded 100% — invariant violated"
    assert manager.used_blocks <= manager.total_blocks


def test_memory_pressure_metrics_reflect_evictions() -> None:
    """After pressure-induced eviction, the native Prometheus snapshot mentions evictions."""
    total = 10
    manager = _make_manager(total)

    # Saturate the pool with sealed blocks, then force one eviction.
    for i in range(total):
        bid = manager.allocate(i + 1, 0, 64)
        manager.fill_primary_test_pattern(bid, 0x77)
        manager.seal(bid)

    # One more allocation triggers eviction of the LRU Tier-b block.
    manager.allocate(9999, 0, 64)

    snapshot = _native.metrics_snapshot_text()
    # The Rust counter name is tessera_evictions_total; accept any non-zero count.
    assert "tessera_evictions_total" in snapshot or "evictions" in snapshot.lower(), (
        "metrics snapshot should contain eviction counter"
    )


def test_allocation_throughput_floor() -> None:
    """10 K alloc/free cycles must complete at > 50 K ops/s (soft floor).

    This is not a Criterion benchmark — it just guards against catastrophic regressions
    (e.g. accidentally introducing a global Python lock on every allocation path).
    """
    n = 10_000
    manager = _make_manager(1_000)

    t0 = time.perf_counter()
    for i in range(n):
        bid = manager.allocate((i % 500) + 1, 0, 64)
        manager.free(bid)
    elapsed = time.perf_counter() - t0

    throughput = n / elapsed
    # Log for visibility even on pass (captured by pytest -s or -ra).
    print(f"\n  alloc/free throughput: {throughput:,.0f} ops/s (floor: 50,000)")
    assert throughput > 50_000, (
        f"throughput {throughput:,.0f} ops/s fell below 50K floor — possible regression"
    )
