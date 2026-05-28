//! Tests for `DistributedSegmentIndex`. Uses `MockTransport` with a `CapturePeer` per rank.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tessera_core::block::BlockId;
use tessera_core::rank::{NodeId, RankId, Topology, World};
use tessera_core::transport::{
    mock::{MockPeer, MockTransport},
    BlockPayload, RankTransport, ReservationToken,
};
use tessera_index::{DistributedSegmentIndex, TierBudget, UsearchIndex};

/// Test peer with a content_hash → block_id table.
struct HashPeer {
    hashes: Mutex<std::collections::HashMap<u64, BlockId>>,
    /// Optional artificial delay before responding to query_hash.
    delay: Option<Duration>,
}

impl HashPeer {
    fn new(entries: Vec<(u64, BlockId)>, delay: Option<Duration>) -> Arc<Self> {
        Arc::new(Self {
            hashes: Mutex::new(entries.into_iter().collect()),
            delay,
        })
    }
}

impl MockPeer for HashPeer {
    fn provide_block(&self, _block_id: BlockId) -> anyhow::Result<BlockPayload> {
        Ok(BlockPayload { c_kv: vec![], k_rope: vec![], fp8_scales: None })
    }
    fn accept_pushed(&self, _payload: BlockPayload) -> anyhow::Result<BlockId> {
        Ok(BlockId(0))
    }
    fn lookup_hash(&self, content_hash: u64) -> anyhow::Result<Option<BlockId>> {
        if let Some(d) = self.delay {
            std::thread::sleep(d);
        }
        Ok(self.hashes.lock().get(&content_hash).copied())
    }
    fn reserve(&self, _: u64, _: u32) -> anyhow::Result<ReservationToken> {
        Ok(ReservationToken(0))
    }
}

fn build_index(
    world_size: u32,
    local_rank: RankId,
    transport: MockTransport,
    budget: Duration,
) -> DistributedSegmentIndex {
    let local = Arc::new(UsearchIndex::new(tessera_index::UsearchConfig::default_for_dim(32))
        .expect("usearch construction must succeed"));
    let world = Arc::new(World::new(local_rank, world_size, Topology::SingleNode).unwrap());
    let transport: Arc<dyn RankTransport> = Arc::new(transport);
    DistributedSegmentIndex::new(local, world, transport, budget)
}

#[tokio::test]
async fn remote_hit_returns_global_id_pointing_at_owner_rank() {
    let handles = MockTransport::new_world(3);
    let peer0 = HashPeer::new(vec![], None);
    let peer1 = HashPeer::new(vec![], None);
    let peer2 = HashPeer::new(vec![(0xABCD, BlockId(42))], None);
    handles[0].register_peer(RankId(0), Arc::clone(&peer0) as Arc<dyn MockPeer>);
    handles[0].register_peer(RankId(1), Arc::clone(&peer1) as Arc<dyn MockPeer>);
    handles[0].register_peer(RankId(2), Arc::clone(&peer2) as Arc<dyn MockPeer>);

    let idx = build_index(3, RankId(0), handles[0].clone(), Duration::from_millis(50));
    let hit = idx.lookup_hash(0xABCD).await.unwrap();
    assert!(hit.is_some());
    let hit = hit.unwrap();
    assert_eq!(hit.global.rank, RankId(2));
    assert_eq!(hit.global.block, BlockId(42));
    assert!(!hit.local);
}

#[tokio::test]
async fn all_peers_miss_returns_none() {
    let handles = MockTransport::new_world(3);
    let peer1 = HashPeer::new(vec![(0xCAFE, BlockId(7))], None);
    let peer2 = HashPeer::new(vec![(0xBEEF, BlockId(9))], None);
    handles[0].register_peer(RankId(1), Arc::clone(&peer1) as Arc<dyn MockPeer>);
    handles[0].register_peer(RankId(2), Arc::clone(&peer2) as Arc<dyn MockPeer>);

    let idx = build_index(3, RankId(0), handles[0].clone(), Duration::from_millis(50));
    let hit = idx.lookup_hash(0xDEAD).await.unwrap();
    assert!(hit.is_none());
}

#[tokio::test]
async fn singleton_world_short_circuits_without_fanout() {
    let handles = MockTransport::new_world(1);
    let idx = build_index(1, RankId(0), handles[0].clone(), Duration::from_millis(50));
    let hit = idx.lookup_hash(0xFFFF).await.unwrap();
    assert!(hit.is_none());
    // No query_hash events in the log because there are no peers.
    let events = handles[0].events();
    assert!(events
        .iter()
        .all(|e| !matches!(e, tessera_core::transport::mock::MockEvent::QueryHash { .. })));
}

#[test]
fn budget_scales_per_tier_multi_node() {
    let base = Duration::from_micros(100);
    let local: Arc<dyn tessera_index::IndexBackend> =
        Arc::new(UsearchIndex::new(tessera_index::UsearchConfig::default_for_dim(16)).unwrap());
    let topology = Topology::MultiNode {
        node_of: vec![NodeId(0), NodeId(0), NodeId(1)],
    };
    let world = Arc::new(World::new(RankId(0), 3, topology).unwrap());
    let handles = MockTransport::new_world(3);
    let transport: Arc<dyn RankTransport> = Arc::new(handles[0].clone());

    let idx = DistributedSegmentIndex::new(local, world, transport, base)
        .with_tier_multipliers(TierBudget {
            intra_node: 1.0,
            intra_rack: 8.0,
            cross_rack: 80.0,
        });

    assert_eq!(idx.effective_budget_for(RankId(1)), base, "same-node tier");
    assert_eq!(idx.effective_budget_for(RankId(2)), base.mul_f32(8.0), "cross-node tier");
}

#[tokio::test]
async fn budget_exhausted_returns_safe_miss() {
    let handles = MockTransport::new_world(2);
    // Peer 1 holds the hash but with a delay larger than the budget.
    let peer1 = HashPeer::new(
        vec![(0xCAFE, BlockId(100))],
        Some(Duration::from_millis(80)),
    );
    handles[0].register_peer(RankId(1), Arc::clone(&peer1) as Arc<dyn MockPeer>);

    let idx = build_index(2, RankId(0), handles[0].clone(), Duration::from_millis(10));
    let start = std::time::Instant::now();
    let hit = idx.lookup_hash(0xCAFE).await.unwrap();
    let elapsed = start.elapsed();
    // Budget honoured (within reasonable scheduler slack).
    assert!(elapsed < Duration::from_millis(60), "lookup took {elapsed:?}");
    assert!(hit.is_none(), "budget exceeded must return None safely");
}
