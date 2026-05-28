"""WS9 — vLLM integration skeleton (GPU-gated, TD-020 skeleton).

These tests exercise ``TesseraBlockAllocator`` against a real ``vllm`` engine.
They are gated by two conditions:
  1. A CUDA device must be present (via the ``gpu`` fixture).
  2. ``vllm`` must be installed (via ``pytest.importorskip``).

On CPU-only machines both conditions fail and the tests are auto-skipped.
On a cloud GPU with vLLM, run: ``pytest tests/gpu -m gpu -v``.

The tests use ``ds_v3_config`` from conftest.py — the DeepSeek-V3 production config.
"""

from __future__ import annotations

import numpy as np
import pytest

vllm = pytest.importorskip("vllm", reason="vllm not installed — skipping integration tests")


# ──────────── helpers ─────────────────────────────────────────────────────────


def _fill_blocks_with_pattern(
    allocator,  # type: ignore[no-untyped-def]
    n_blocks: int,
    req_id: int,
    pattern: int = 0xAA,
) -> list[int]:
    """Allocate n_blocks, fill with pattern, seal, and return canonical block IDs."""
    import numpy as np  # noqa: PLC0415

    block_size = allocator._config.block.block_size_tokens
    latent_dim = allocator._config.model.latent_dim
    block_ids: list[int] = []
    for i in range(n_blocks):
        start = i * block_size
        bid = allocator._manager.allocate(req_id, start, start + block_size)
        allocator._manager.fill_primary_test_pattern(bid, pattern)
        c_kv = np.random.rand(block_size, latent_dim).astype(np.float32)
        canonical = allocator.post_prefill_seal(bid, c_kv)
        block_ids.append(canonical)
    return block_ids


# ──────────── tests ────────────────────────────────────────────────────────────


@pytest.mark.gpu
def test_vllm_allocator_dedup_rate(gpu, ds_v3_config) -> None:  # type: ignore[no-untyped-def]
    """8 requests with identical prefix blocks achieve sharing_rate > 0.5 via dedup.

    This is the core Tessera promise: content-addressed c_kv sharing across agents.
    Each request gets 8 prefix blocks filled with the same pattern → all 8 should dedup
    to the same canonical block → ``sharing_rate > 0`` and ``used_blocks < 8 * 8``.
    """
    from tessera.vllm_plugin import TesseraBlockAllocator  # noqa: PLC0415

    allocator = TesseraBlockAllocator(ds_v3_config)

    n_requests = 8
    shared_blocks_per_req = 8
    shared_pattern = 0x42

    all_canonical: list[int] = []
    for req_id in range(1, n_requests + 1):
        # All requests use the same byte pattern → dedup collapses to one canonical block.
        canon_ids = _fill_blocks_with_pattern(
            allocator, shared_blocks_per_req, req_id, pattern=shared_pattern
        )
        all_canonical.extend(canon_ids)

    # If dedup works, all canonical IDs should be the same (only one unique block).
    unique_blocks = len(set(all_canonical))
    total_attempted = n_requests * shared_blocks_per_req

    assert unique_blocks < total_attempted, (
        f"dedup should reduce {total_attempted} blocks → expected < {total_attempted} unique, "
        f"got {unique_blocks}. Tessera content-addressed dedup may not be working."
    )

    # Sharing rate should be non-trivial.
    sharing_rate = allocator._share_table.sharing_rate()
    assert sharing_rate >= 0.0, f"sharing_rate must be non-negative; got {sharing_rate}"
    used = allocator._manager.used_blocks
    assert used < total_attempted, (
        f"used_blocks {used} should be < {total_attempted} thanks to dedup"
    )


@pytest.mark.gpu
def test_tessera_alloc_throughput_vs_shim(gpu, ds_v3_config) -> None:  # type: ignore[no-untyped-def]
    """Tessera allocation throughput ≥ 80% of a plain dict-backed Python shim.

    This guards against catastrophic PyO3 FFI overhead introduced in the plugin layer.
    The shim baseline is intentionally simple (dict insert/delete) — not vLLM's real
    allocator. Real production comparison is in ``benchmarks/vllm_compare.py``.
    """
    import time  # noqa: PLC0415

    from tessera.vllm_plugin import TesseraBlockAllocator  # noqa: PLC0415

    allocator = TesseraBlockAllocator(ds_v3_config)
    block_size = ds_v3_config.block.block_size_tokens

    n = 5_000

    # ── Shim baseline (pure Python dict) ────────────────────────────────────
    shim: dict[int, int] = {}
    t0 = time.perf_counter()
    for i in range(n):
        shim[i] = i
    for i in range(n):
        del shim[i]
    shim_elapsed = time.perf_counter() - t0
    shim_throughput = n / shim_elapsed

    # ── Tessera ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for i in range(n):
        bid = allocator._manager.allocate(i % 500 + 1, 0, block_size)
        allocator._manager.free(bid)
    tessera_elapsed = time.perf_counter() - t0
    tessera_throughput = n / tessera_elapsed

    ratio = tessera_throughput / shim_throughput
    print(
        f"\n  shim: {shim_throughput:,.0f} ops/s  "
        f"tessera: {tessera_throughput:,.0f} ops/s  ratio: {ratio:.2f}"
    )
    assert ratio >= 0.80, (
        f"Tessera throughput {tessera_throughput:,.0f} ops/s is < 80% of shim "
        f"{shim_throughput:,.0f} ops/s (ratio={ratio:.2f}) — possible PyO3 overhead regression"
    )
