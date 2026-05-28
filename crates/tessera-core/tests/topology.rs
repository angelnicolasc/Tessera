//! Multi-node topology semantics (Sprint 4 / WS2).
//!
//! Verifies the new `Topology::node_of` / `is_same_node` / `World::peer_tier` helpers + the
//! `MockTransport::with_topology` builder.

use std::sync::Arc;
use std::time::Duration;

use tessera_core::block::BlockId;
use tessera_core::rank::{LatencyTier, NodeId, RankId, Topology, World};
use tessera_core::transport::{
    mock::{MockPeer, MockTransport},
    BlockPayload, LatencyProfile, RankTransport, ReservationToken,
};

struct EchoPeer;
impl MockPeer for EchoPeer {
    fn provide_block(&self, _: BlockId) -> anyhow::Result<BlockPayload> {
        Ok(BlockPayload { c_kv: vec![], k_rope: vec![], fp8_scales: None })
    }
    fn accept_pushed(&self, _: BlockPayload) -> anyhow::Result<BlockId> {
        Ok(BlockId(0))
    }
    fn lookup_hash(&self, _: u64) -> anyhow::Result<Option<BlockId>> {
        Ok(None)
    }
    fn reserve(&self, _: u64, _: u32) -> anyhow::Result<ReservationToken> {
        Ok(ReservationToken(0))
    }
}

#[test]
fn single_node_topology_classifies_every_peer_as_intra_node() {
    let topology = Topology::SingleNode;
    assert_eq!(topology.node_of(RankId(0)), Some(NodeId(0)));
    assert_eq!(topology.node_of(RankId(99)), Some(NodeId(0)));
    assert!(topology.is_same_node(RankId(0), RankId(1)));

    let world = World::new(RankId(0), 4, Topology::SingleNode).unwrap();
    assert_eq!(world.peer_tier(RankId(3)), LatencyTier::IntraNode);
}

#[test]
fn multi_node_topology_classifies_correctly() {
    // 4 ranks: 0,1 on node-0; 2,3 on node-1.
    let topology = Topology::MultiNode {
        node_of: vec![NodeId(0), NodeId(0), NodeId(1), NodeId(1)],
    };
    assert_eq!(topology.node_of(RankId(0)), Some(NodeId(0)));
    assert_eq!(topology.node_of(RankId(2)), Some(NodeId(1)));
    assert_eq!(topology.node_of(RankId(99)), None, "out-of-range rank returns None");

    assert!(topology.is_same_node(RankId(0), RankId(1)));
    assert!(!topology.is_same_node(RankId(0), RankId(2)));
    assert!(!topology.is_same_node(RankId(0), RankId(99)));

    let world = World::new(RankId(0), 4, topology).unwrap();
    assert_eq!(world.peer_tier(RankId(1)), LatencyTier::IntraNode);
    assert_eq!(world.peer_tier(RankId(2)), LatencyTier::IntraRack);
    assert_eq!(world.peer_tier(RankId(3)), LatencyTier::IntraRack);
}

#[test]
fn latency_tier_as_str_is_stable() {
    assert_eq!(LatencyTier::IntraNode.as_str(), "intra_node");
    assert_eq!(LatencyTier::IntraRack.as_str(), "intra_rack");
    assert_eq!(LatencyTier::CrossRack.as_str(), "cross_rack");
}

#[tokio::test(start_paused = true)]
async fn with_topology_builder_wraps_handles_with_latency_injector() {
    let topology = Topology::MultiNode {
        node_of: vec![NodeId(0), NodeId(1)],
    };
    let profile = LatencyProfile {
        intra_node_us: 1,
        intra_rack_us: 200,
        cross_rack_us: 0,
        jitter_us: 0,
        drop_rate: 0.0,
    };
    let transports = MockTransport::with_topology(2, topology, profile, 0xCAFE);
    assert_eq!(transports.len(), 2);
    assert_eq!(transports[0].name(), "latency_injector");

    // Register a peer on the inner mock — we need to reach it via the original handle,
    // which means walking around the injector. For this test we just verify that calls
    // through the wrapped transport experience the cross-node latency.
    let start = tokio::time::Instant::now();
    // The peer registry is shared across all bases produced by `new_world` inside
    // `with_topology`. Tests that need to inject a real peer should construct the
    // injector explicitly (see `latency_injection.rs`). Here we accept the structural
    // assertion: calling fetch on an unregistered peer fails (no peer registered) but
    // the injector still sleeps first.
    let result = transports[0].fetch_block(RankId(1), BlockId(0)).await;
    let elapsed = start.elapsed();
    // Should have slept ≥ intra_rack_us before failing.
    assert!(
        elapsed >= Duration::from_micros(200),
        "with_topology must inject cross-node latency; observed {elapsed:?}"
    );
    assert!(result.is_err(), "unregistered peer: expect peer-lookup error");
}

// The DistributedSegmentIndex tier-budget scaling test lives in
// `crates/tessera-index/tests/distributed.rs` to avoid a `tessera-core` ↔ `tessera-index`
// dev-dep cycle.
