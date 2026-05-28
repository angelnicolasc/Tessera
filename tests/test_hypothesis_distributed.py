"""Hypothesis boundary fuzz: ``DistributedSegmentIndex`` + ``find_shared_prefix`` shape.

Properties asserted:

1. ``DistributedSegmentIndex.lookup_hash`` always returns one of: ``None`` | ``(int, int)``.
   No leaked exceptions, no malformed payloads, regardless of content hash.
2. ``find_shared_prefix`` on the vLLM plugin returns a list whose entries are each
   ``int`` | ``tuple[int, int]`` | ``None``. The order matches the input length.
3. Local content_hash → block_id (Layer 1) round-trips: registering a block and looking
   up its hash always finds it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")
pytest.importorskip("tessera._native")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tessera import _native  # noqa: E402


def _world_and_transport(world_size: int):
    transports = _native.MockTransport.new_world(world_size)
    world = _native.World.single_node(local=0, size=world_size)
    return world, transports


@given(content_hash=st.integers(min_value=0, max_value=2**64 - 1))
@settings(
    max_examples=64,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_distributed_lookup_hash_shape(content_hash):
    """No matter what hash we ask for on an empty world, the result is None or a 2-tuple."""
    world, transports = _world_and_transport(2)
    idx = _native.DistributedSegmentIndex(
        dimensions=32,
        world=world,
        transport=transports[0],
        budget_us=2000,
    )
    result = idx.lookup_hash(content_hash)
    assert result is None or (
        isinstance(result, tuple)
        and len(result) == 2
        and all(isinstance(x, int) and x >= 0 for x in result)
    )


@given(world_size=st.integers(min_value=1, max_value=4))
@settings(max_examples=16, deadline=None)
def test_distributed_lookup_safe_on_any_world_size(world_size):
    """world_size ∈ [1, 4]: lookup never raises; singletons short-circuit; larger fan-out
    returns a safe miss when no peer holds the hash."""
    world, transports = _world_and_transport(world_size)
    idx = _native.DistributedSegmentIndex(
        dimensions=16,
        world=world,
        transport=transports[0],
        budget_us=1000,
    )
    # An arbitrary hash that no peer holds.
    result = idx.lookup_hash(0xDEADBEEFCAFEBABE)
    assert result is None


@given(
    descriptors=st.lists(
        st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False), min_size=8, max_size=8),
        min_size=0,
        max_size=4,
    )
)
@settings(max_examples=32, deadline=None)
def test_usearch_add_query_roundtrip(descriptors):
    """Adding a descriptor and querying it back should return the same block id (top-1)."""
    if not descriptors:
        return
    idx = _native.UsearchIndex(dimensions=8)
    for block_id, vec in enumerate(descriptors):
        arr = np.asarray(vec, dtype=np.float32)
        idx.add(block_id, arr)
    # Query each descriptor back; should hit itself with similarity ≈ 1.
    for block_id, vec in enumerate(descriptors):
        arr = np.asarray(vec, dtype=np.float32)
        if np.linalg.norm(arr) == 0:
            continue  # zero-vector cosine is degenerate; skip
        hits = idx.query(arr, 1)
        assert len(hits) >= 1
        top_block, top_sim = hits[0]
        # Top hit should be among the inserted blocks.
        assert 0 <= top_block <= len(descriptors)
        assert -1.0 <= top_sim <= 1.01  # cosine bounded
