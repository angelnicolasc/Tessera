//! Tests for the `MockTransport` in-process implementation. Sprint 3 / WS2.

use std::sync::Arc;

use parking_lot::Mutex;
use tessera_core::block::BlockId;
use tessera_core::rank::{RankId, Topology};
use tessera_core::transport::{
    mock::{MockEvent, MockPeer, MockTransport},
    BlockPayload, RankTransport,
};

/// Test-only peer that records all incoming calls plus a content-hash → block id table.
struct CapturePeer {
    /// Local payload store: block_id → payload bytes.
    payloads: Mutex<std::collections::HashMap<BlockId, BlockPayload>>,
    /// Content hash → block id mapping for `lookup_hash`.
    hashes: Mutex<std::collections::HashMap<u64, BlockId>>,
    /// Push acceptance log.
    accepted: Mutex<Vec<usize>>,
    /// On-seal-announce log.
    seals_received: Mutex<Vec<(RankId, BlockId, u64)>>,
}

impl CapturePeer {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            payloads: Mutex::new(std::collections::HashMap::new()),
            hashes: Mutex::new(std::collections::HashMap::new()),
            accepted: Mutex::new(Vec::new()),
            seals_received: Mutex::new(Vec::new()),
        })
    }

    fn put(&self, block_id: BlockId, payload: BlockPayload, content_hash: u64) {
        self.payloads.lock().insert(block_id, payload);
        self.hashes.lock().insert(content_hash, block_id);
    }
}

impl MockPeer for CapturePeer {
    fn provide_block(&self, block_id: BlockId) -> anyhow::Result<BlockPayload> {
        self.payloads
            .lock()
            .get(&block_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no payload for {block_id:?}"))
    }

    fn accept_pushed(&self, payload: BlockPayload) -> anyhow::Result<BlockId> {
        let mut acc = self.accepted.lock();
        let id = BlockId(acc.len() as u32 + 1000);
        acc.push(payload.byte_len());
        self.payloads.lock().insert(id, payload);
        Ok(id)
    }

    fn lookup_hash(&self, content_hash: u64) -> anyhow::Result<Option<BlockId>> {
        Ok(self.hashes.lock().get(&content_hash).copied())
    }

    fn on_seal_announce(
        &self,
        src: RankId,
        block_id: BlockId,
        content_hash: u64,
        _descriptor: &[f32],
    ) -> anyhow::Result<()> {
        self.seals_received
            .lock()
            .push((src, block_id, content_hash));
        Ok(())
    }
}

#[tokio::test]
async fn broadcast_seal_reaches_all_peers_except_source() {
    let handles = MockTransport::new_world(4);
    let peers: Vec<_> = (0..4).map(|_| CapturePeer::new()).collect();
    for (i, p) in peers.iter().enumerate() {
        handles[0].register_peer(RankId(i as u32), Arc::clone(p) as Arc<dyn MockPeer>);
    }

    handles[0]
        .broadcast_seal(RankId(0), BlockId(7), 0xCAFE, vec![0.1, 0.2, 0.3])
        .await
        .unwrap();

    // Source did NOT receive its own broadcast.
    assert!(peers[0].seals_received.lock().is_empty());
    // Every other peer got it exactly once.
    for (i, peer) in peers.iter().enumerate().take(4).skip(1) {
        let received = peer.seals_received.lock();
        assert_eq!(received.len(), 1, "rank {i} missed the broadcast");
        assert_eq!(received[0], (RankId(0), BlockId(7), 0xCAFE));
    }

    // Event log records exactly one BroadcastSeal.
    let events = handles[0].events();
    let broadcasts: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, MockEvent::BroadcastSeal { .. }))
        .collect();
    assert_eq!(broadcasts.len(), 1);
}

#[tokio::test]
async fn fetch_block_returns_payload_from_owner_peer() {
    let handles = MockTransport::new_world(2);
    let peer1 = CapturePeer::new();
    let payload = BlockPayload {
        c_kv: vec![1, 2, 3, 4],
        k_rope: vec![5, 6],
        fp8_scales: None,
    };
    peer1.put(BlockId(42), payload.clone(), 0xDEAD);
    handles[0].register_peer(RankId(1), Arc::clone(&peer1) as Arc<dyn MockPeer>);

    let fetched = handles[0]
        .fetch_block(RankId(1), BlockId(42))
        .await
        .unwrap();
    assert_eq!(fetched.c_kv, payload.c_kv);
    assert_eq!(fetched.k_rope, payload.k_rope);
    assert!(fetched.fp8_scales.is_none());
    assert_eq!(fetched.byte_len(), 6);
}

#[tokio::test]
async fn push_block_assigns_destination_block_id() {
    let handles = MockTransport::new_world(2);
    let peer1 = CapturePeer::new();
    handles[0].register_peer(RankId(1), Arc::clone(&peer1) as Arc<dyn MockPeer>);

    let payload = BlockPayload {
        c_kv: vec![0xAA; 64],
        k_rope: vec![0xBB; 16],
        fp8_scales: None,
    };
    let assigned = handles[0].push_block(RankId(1), payload).await.unwrap();
    // CapturePeer assigns starting at 1000.
    assert_eq!(assigned, BlockId(1000));
    assert_eq!(peer1.accepted.lock().len(), 1);
    assert_eq!(peer1.accepted.lock()[0], 80);
}

#[tokio::test]
async fn query_hash_returns_local_block_id_when_present() {
    let handles = MockTransport::new_world(2);
    let peer1 = CapturePeer::new();
    peer1.put(
        BlockId(11),
        BlockPayload {
            c_kv: vec![],
            k_rope: vec![],
            fp8_scales: None,
        },
        0xFEED,
    );
    handles[0].register_peer(RankId(1), Arc::clone(&peer1) as Arc<dyn MockPeer>);

    let hit = handles[0].query_hash(RankId(1), 0xFEED).await.unwrap();
    assert_eq!(hit, Some(BlockId(11)));
    let miss = handles[0].query_hash(RankId(1), 0xBAD).await.unwrap();
    assert_eq!(miss, None);
}

#[tokio::test]
async fn release_announce_propagates_to_peers() {
    let handles = MockTransport::new_world(3);
    // No payloads needed; just check that the event log records propagation.
    handles[0]
        .announce_release(RankId(0), BlockId(99))
        .await
        .unwrap();
    let events = handles[0].events();
    let releases: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            MockEvent::Release { src, block_id } => Some((*src, *block_id)),
            _ => None,
        })
        .collect();
    assert_eq!(releases.len(), 1);
    assert_eq!(releases[0], (RankId(0), BlockId(99)));
}

#[tokio::test]
async fn topology_is_single_node_for_mock() {
    let handles = MockTransport::new_world(2);
    assert!(matches!(handles[0].topology(), Topology::SingleNode));
    assert_eq!(handles[0].name(), "mock");
}
