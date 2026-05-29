"""End-to-end multi-rank tests over a simulated TP=4 world. CPU-only.

Each test exercises one invariant of the rank-aware block manager / transport / distributed
segment index stack. The fixture (`tp4_world`) provides 4 interconnected block managers
sharing a `MockTransport` world; tests assert behaviour against that snapshot.
"""

from __future__ import annotations

import pytest

from tessera.multi_rank import MultiRankCoordinator


@pytest.mark.integration
def test_distinct_ranks_isolate_private_blocks(tp4_world: MultiRankCoordinator) -> None:
    """Allocations on rank 0 must not appear on other ranks."""
    rank0, rank1 = tp4_world.managers[0], tp4_world.managers[1]
    bid0 = rank0.allocate(req_id=1, token_start=0, token_end=64)
    assert rank0.used_blocks == 1
    assert rank1.used_blocks == 0
    # Global ids encode the owning rank.
    g0 = rank0.global_id(bid0)
    assert g0 == (0, bid0)


@pytest.mark.integration
def test_cross_rank_payload_roundtrip(tp4_world: MultiRankCoordinator) -> None:
    """Pushing a block from rank-0 to rank-2 via transfer_request_to_rank moves used_blocks
    accounting from source to destination."""
    src = tp4_world.managers[0]
    dst = tp4_world.managers[2]

    # Source rank: allocate 3 blocks for req 7 with distinct patterns.
    for i in range(3):
        bid = src.allocate(req_id=7, token_start=i * 64, token_end=(i + 1) * 64)
        src.fill_primary_test_pattern(bid, 0x10 + i)
    assert src.used_blocks == 3
    assert dst.used_blocks == 0

    moved = src.transfer_request_to_rank(req_id=7, target=2, transport=tp4_world.transports[0])
    assert moved == 3
    assert src.used_blocks == 0
    assert dst.used_blocks == 3


@pytest.mark.integration
def test_release_request_is_rank_local(tp4_world: MultiRankCoordinator) -> None:
    """release_request on rank 0 must not affect rank 1's blocks for the same req_id."""
    rank0, rank1 = tp4_world.managers[0], tp4_world.managers[1]
    rank0.allocate(req_id=42, token_start=0, token_end=64)
    rank0.allocate(req_id=42, token_start=64, token_end=128)
    rank1.allocate(req_id=42, token_start=0, token_end=64)

    freed_on_rank0 = rank0.release_request(req_id=42)
    assert freed_on_rank0 == 2
    assert rank0.used_blocks == 0
    # Rank 1's allocation for the same req_id is untouched.
    assert rank1.used_blocks == 1


@pytest.mark.integration
def test_transport_event_log_records_each_cross_rank_op(
    tp4_world: MultiRankCoordinator,
) -> None:
    """Every push/fetch hit must appear in the transport's event log exactly once."""
    src = tp4_world.managers[0]
    bid = src.allocate(req_id=99, token_start=0, token_end=64)
    src.fill_primary_test_pattern(bid, 0xAB)

    handle = tp4_world.transports[0]
    handle.clear_events()
    src.transfer_request_to_rank(req_id=99, target=3, transport=handle)

    events = handle.events()
    push_events = [e for e in events if e.startswith("Push")]
    assert len(push_events) == 1, f"expected 1 push event, got {events}"


@pytest.mark.integration
def test_distributed_segment_index_remote_hit(
    small_mla_config, tp2_world: MultiRankCoordinator
) -> None:
    """Sprint 3 hash-only path: a content_hash held by a remote rank's hash table is found
    via DistributedSegmentIndex. We don't wire BlockManagerPeerAdapter.lookup_hash to return
    real hits (it returns None in the adapter), so this test verifies the safe-miss path:
    no remote hit yields None gracefully."""
    from tessera import _native

    world = _native.World.single_node(local=0, size=2)
    idx = _native.DistributedSegmentIndex(
        dimensions=small_mla_config.model.latent_dim,
        world=world,
        transport=tp2_world.transports[0],
        budget_us=10_000,
    )
    hit = idx.lookup_hash(0xDEADBEEF)
    assert hit is None


@pytest.mark.integration
def test_singleton_world_retrocompat(small_mla_config) -> None:
    """world_size=1 must still work end-to-end via spawn_multirank_world (covers ADR-0014
    fallback path: no peers, transport.singleton)."""
    from tessera.multi_rank import spawn_multirank_world

    coord = spawn_multirank_world(small_mla_config, world_size=1)
    assert coord.world_size == 1
    assert len(coord.managers) == 1
    assert len(coord.transports) == 1
    mgr = coord.managers[0]
    bid = mgr.allocate(req_id=1, token_start=0, token_end=64)
    assert mgr.used_blocks == 1
    mgr.free(bid)
    assert mgr.used_blocks == 0
