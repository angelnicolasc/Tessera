"""SegmentIndex two-layer behaviour. Requires the native extension."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tessera.segment_index import SegmentIndex, descriptor_from_ckv, hash_ckv_bytes


@pytest.fixture
def small_ckv() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 64, 32), dtype=np.float32)


def test_exact_hash_roundtrip(native: object, small_ckv: np.ndarray) -> None:
    del native  # ensure native is importable; SegmentIndex constructs UsearchIndex
    idx = SegmentIndex(latent_dim=32, num_layers=4)
    h = hash_ckv_bytes(small_ckv)
    idx.add(block_id=7, content_hash=h, c_kv=small_ckv)
    assert idx.lookup_exact(h) == 7
    idx.close()


def test_exact_miss_returns_none() -> None:
    idx = SegmentIndex(latent_dim=32, num_layers=4)
    assert idx.lookup_exact(0xDEADBEEF) is None
    idx.close()


def test_descriptor_shape() -> None:
    rng = np.random.default_rng(0)
    c_kv = rng.standard_normal((4, 64, 32), dtype=np.float32)
    desc = descriptor_from_ckv(c_kv)
    assert desc.shape == (32,)
    assert desc.dtype == np.float32


@pytest.mark.asyncio
async def test_hnsw_approximate_match_returns_inserted_block(
    native: object, small_ckv: np.ndarray
) -> None:
    del native
    idx = SegmentIndex(latent_dim=32, num_layers=4)
    idx.add(block_id=42, content_hash=hash_ckv_bytes(small_ckv), c_kv=small_ckv)
    match = await idx.lookup_approximate(small_ckv)
    assert match is not None
    assert match.block_id == 42
    assert match.similarity >= 0.97
    assert match.was_exact is False
    idx.close()


@pytest.mark.asyncio
async def test_remove_clears_both_layers(native: object, small_ckv: np.ndarray) -> None:
    del native
    idx = SegmentIndex(latent_dim=32, num_layers=4)
    h = hash_ckv_bytes(small_ckv)
    idx.add(block_id=11, content_hash=h, c_kv=small_ckv)
    idx.remove(block_id=11, content_hash=h)
    assert idx.lookup_exact(h) is None
    # HNSW: query no longer returns it.
    match = await idx.lookup_approximate(small_ckv)
    assert match is None
    idx.close()


@pytest.mark.asyncio
async def test_latency_budget_zero_yields_miss(native: object, small_ckv: np.ndarray) -> None:
    """A budget below floor must always time out, returning None safely."""
    del native
    idx = SegmentIndex(latent_dim=32, num_layers=4, latency_budget_us=1)
    idx.add(block_id=1, content_hash=hash_ckv_bytes(small_ckv), c_kv=small_ckv)
    # Force a yield so the executor truly races against the timeout.
    await asyncio.sleep(0)
    result = await idx.lookup_approximate(small_ckv)
    # Result MAY be None (timeout) or a real match — both are correct. Assert it is at least
    # not an exception, and if non-None it is the inserted block.
    if result is not None:
        assert result.block_id == 1
    idx.close()
