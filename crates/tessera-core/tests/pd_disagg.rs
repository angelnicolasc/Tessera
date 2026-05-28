//! PD-disaggregation hook tests. Validates `transfer_request_to_rank` end-to-end on a
//! 2-rank simulated world using `MockTransport` + `CpuMockBackend`.

use std::sync::Arc;

use parking_lot::Mutex;
use tessera_core::{
    block::BlockId,
    rank::{RankId, Topology, World},
    transport::{
        mock::{MockPeer, MockTransport},
        BlockPayload, RankTransport,
    },
    CkvDtype, CompressionScheme, MlaBlockConfig, TesseraBlockManager, TokenRange,
};

fn small_cfg() -> MlaBlockConfig {
    MlaBlockConfig::new(
        CompressionScheme::MlaLatent { latent_dim: 32, rope_key_dim: 8 },
        4,
        64,
        CkvDtype::Bf16,
        0,
    )
    .unwrap()
}

/// Peer whose `accept_pushed` forwards into a [`TesseraBlockManager`] on the destination
/// side. This is the canonical wiring for the PD-disagg push path.
struct BlockManagerPeer {
    manager: Arc<TesseraBlockManager>,
    /// Tracks the synthetic req_id used by the destination side for incoming blocks.
    accept_req_id: u64,
    /// Records the assigned destination block ids in order.
    received: Mutex<Vec<BlockId>>,
}

impl BlockManagerPeer {
    fn new(manager: Arc<TesseraBlockManager>, accept_req_id: u64) -> Arc<Self> {
        Arc::new(Self {
            manager,
            accept_req_id,
            received: Mutex::new(Vec::new()),
        })
    }
}

impl MockPeer for BlockManagerPeer {
    fn provide_block(&self, block_id: BlockId) -> anyhow::Result<BlockPayload> {
        self.manager
            .export_payload(block_id)
            .map_err(|e| anyhow::anyhow!("export_payload: {e}"))
    }

    fn accept_pushed(&self, payload: BlockPayload) -> anyhow::Result<BlockId> {
        let bid = self
            .manager
            .import_payload(self.accept_req_id, TokenRange::new(0, 64), &payload)
            .map_err(|e| anyhow::anyhow!("import_payload: {e}"))?;
        self.received.lock().push(bid);
        Ok(bid)
    }

    fn lookup_hash(&self, _content_hash: u64) -> anyhow::Result<Option<BlockId>> {
        Ok(None)
    }
}

#[tokio::test]
async fn transfer_request_moves_all_owned_blocks() {
    // Source rank 0; target rank 1. Each has its own block manager.
    let handles = MockTransport::new_world(2);
    let world0 = Arc::new(World::new(RankId(0), 2, Topology::SingleNode).unwrap());
    let world1 = Arc::new(World::new(RankId(1), 2, Topology::SingleNode).unwrap());
    let src = Arc::new(
        TesseraBlockManager::new_with_world(small_cfg(), 16 * 1024 * 1024, RankId(0), world0)
            .unwrap(),
    );
    let dst = Arc::new(
        TesseraBlockManager::new_with_world(small_cfg(), 16 * 1024 * 1024, RankId(1), world1)
            .unwrap(),
    );

    // Source side: allocate 3 blocks for req 7 with distinct patterns.
    let req_id = 7u64;
    let mut src_blocks = Vec::new();
    for i in 0..3 {
        let bid = src.allocate(req_id, TokenRange::new(0, 64)).unwrap();
        src.fill_primary_test_pattern(bid, 0x10 + i as u8).unwrap();
        src_blocks.push(bid);
    }
    assert_eq!(src.used_blocks(), 3);

    // Register the destination block manager as rank-1's peer.
    let dst_peer = BlockManagerPeer::new(Arc::clone(&dst), 99);
    handles[0].register_peer(RankId(1), Arc::clone(&dst_peer) as Arc<dyn MockPeer>);

    let transport: Arc<dyn RankTransport> = Arc::new(handles[0].clone());
    let moved = src
        .transfer_request_to_rank(req_id, RankId(1), &transport)
        .await
        .unwrap();
    assert_eq!(moved, 3);
    assert_eq!(src.used_blocks(), 0, "source must release after transfer");
    assert_eq!(dst.used_blocks(), 3, "destination must hold the imported blocks");
    assert_eq!(dst_peer.received.lock().len(), 3);
}

#[tokio::test]
async fn transfer_of_unknown_request_returns_zero() {
    let handles = MockTransport::new_world(2);
    let world0 = Arc::new(World::new(RankId(0), 2, Topology::SingleNode).unwrap());
    let src = Arc::new(
        TesseraBlockManager::new_with_world(small_cfg(), 16 * 1024 * 1024, RankId(0), world0)
            .unwrap(),
    );
    let transport: Arc<dyn RankTransport> = Arc::new(handles[0].clone());
    let moved = src
        .transfer_request_to_rank(424242, RankId(1), &transport)
        .await
        .unwrap();
    assert_eq!(moved, 0);
}

#[tokio::test]
async fn payload_roundtrip_preserves_content_hash() {
    // export -> import round-trip on the same content should produce the same byte stream.
    let world = Arc::new(World::new(RankId(0), 1, Topology::SingleNode).unwrap());
    let mgr = Arc::new(
        TesseraBlockManager::new_with_world(small_cfg(), 16 * 1024 * 1024, RankId(0), world)
            .unwrap(),
    );
    let bid = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    mgr.fill_primary_test_pattern(bid, 0xC3).unwrap();
    let payload = mgr.export_payload(bid).unwrap();
    let bid2 = mgr.import_payload(2, TokenRange::new(0, 64), &payload).unwrap();
    let payload2 = mgr.export_payload(bid2).unwrap();
    assert_eq!(payload.c_kv, payload2.c_kv);
    assert_eq!(payload.k_rope, payload2.k_rope);
}
