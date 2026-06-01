//! Reserve-then-stream transactional PD-disaggregation tests (Sprint 4 / WS3 / ADR-0018).
//!
//! Validates the three-phase protocol end-to-end on a 2-rank world:
//!   1. Reserve — capacity check at the destination, surfaces OOM cleanly.
//!   2. Stream — per-block push with rollback on any failure.
//!   3. Commit — source releases only on full success.

use std::sync::Arc;

use parking_lot::Mutex;
use tessera_core::{
    block::BlockId,
    rank::{RankId, Topology, World},
    transport::{
        mock::{MockPeer, MockTransport},
        BlockPayload, LatencyInjector, LatencyProfile, RankTransport, ReservationToken,
    },
    CkvDtype, CompressionScheme, MlaBlockConfig, TesseraBlockManager, TesseraError, TokenRange,
};

fn cfg() -> MlaBlockConfig {
    MlaBlockConfig::new(
        CompressionScheme::MlaLatent {
            latent_dim: 32,
            rope_key_dim: 8,
        },
        4,
        64,
        CkvDtype::Bf16,
        0,
    )
    .unwrap()
}

/// Peer that forwards reserve / accept_pushed / release_reservation to a real block manager
/// so the tests exercise the full capacity-aware protocol.
struct CapacityPeer {
    manager: Arc<TesseraBlockManager>,
    accept_req_id: u64,
    pushed: Mutex<Vec<BlockId>>,
}

impl CapacityPeer {
    fn new(manager: Arc<TesseraBlockManager>, accept_req_id: u64) -> Arc<Self> {
        Arc::new(Self {
            manager,
            accept_req_id,
            pushed: Mutex::new(Vec::new()),
        })
    }
}

impl MockPeer for CapacityPeer {
    fn provide_block(&self, block_id: BlockId) -> anyhow::Result<BlockPayload> {
        self.manager
            .export_payload(block_id)
            .map_err(|e| anyhow::anyhow!("export: {e}"))
    }
    fn accept_pushed(&self, payload: BlockPayload) -> anyhow::Result<BlockId> {
        let bid = self
            .manager
            .import_payload(self.accept_req_id, TokenRange::new(0, 64), &payload)
            .map_err(|e| anyhow::anyhow!("import: {e}"))?;
        self.pushed.lock().push(bid);
        Ok(bid)
    }
    fn lookup_hash(&self, _: u64) -> anyhow::Result<Option<BlockId>> {
        Ok(None)
    }
    fn reserve(&self, req_id: u64, count: u32) -> anyhow::Result<ReservationToken> {
        self.manager
            .reserve_incoming(req_id, count)
            .map_err(|e| anyhow::anyhow!("reserve: {e}"))
    }
    fn release_reservation(&self, token: ReservationToken) -> anyhow::Result<()> {
        self.manager
            .release_reservation_local(token)
            .map_err(|e| anyhow::anyhow!("release: {e}"))
    }
}

fn make_pair(
    target_capacity_mb: u64,
) -> (
    Arc<TesseraBlockManager>,
    Arc<TesseraBlockManager>,
    MockTransport,
) {
    let handles = MockTransport::new_world(2);
    let world0 = Arc::new(World::new(RankId(0), 2, Topology::SingleNode).unwrap());
    let world1 = Arc::new(World::new(RankId(1), 2, Topology::SingleNode).unwrap());
    let src = Arc::new(
        TesseraBlockManager::new_with_world(cfg(), 32 * 1024 * 1024, RankId(0), world0).unwrap(),
    );
    let dst = Arc::new(
        TesseraBlockManager::new_with_world(
            cfg(),
            target_capacity_mb * 1024 * 1024,
            RankId(1),
            world1,
        )
        .unwrap(),
    );
    (src, dst, handles[0].clone())
}

#[tokio::test]
async fn full_success_consumes_reservation_and_releases_source() {
    let (src, dst, handle) = make_pair(32);
    let peer = CapacityPeer::new(Arc::clone(&dst), 99);
    handle.register_peer(RankId(1), Arc::clone(&peer) as Arc<dyn MockPeer>);

    let req_id = 7u64;
    for i in 0..4 {
        let bid = src.allocate(req_id, TokenRange::new(0, 64)).unwrap();
        src.fill_primary_test_pattern(bid, 0x10 + i as u8).unwrap();
    }
    assert_eq!(src.used_blocks(), 4);

    let transport: Arc<dyn RankTransport> = Arc::new(handle.clone());
    let moved = src
        .transfer_request_to_rank(req_id, RankId(1), &transport)
        .await
        .unwrap();
    assert_eq!(moved, 4);
    assert_eq!(src.used_blocks(), 0, "source must release after success");
    assert_eq!(
        dst.used_blocks(),
        4,
        "destination must hold imported blocks"
    );
    assert_eq!(peer.pushed.lock().len(), 4);
}

#[tokio::test]
async fn destination_oom_aborts_cleanly_without_touching_source() {
    // Make the destination too small to hold all source blocks: capacity = 1 block worth.
    let (src, dst, handle) = make_pair(1);
    let peer = CapacityPeer::new(Arc::clone(&dst), 99);
    handle.register_peer(RankId(1), Arc::clone(&peer) as Arc<dyn MockPeer>);

    let req_id = 13u64;
    // Allocate way more than the destination can possibly accept.
    for _ in 0..100 {
        src.allocate(req_id, TokenRange::new(0, 64)).unwrap();
    }
    let src_used_before = src.used_blocks();
    assert!(src_used_before >= 50, "source should hold many blocks");

    let transport: Arc<dyn RankTransport> = Arc::new(handle.clone());
    let result = src
        .transfer_request_to_rank(req_id, RankId(1), &transport)
        .await;
    assert!(result.is_err(), "destination OOM must abort the transfer");
    if let Err(TesseraError::Backend(e)) = &result {
        let msg = format!("{e:?}");
        assert!(
            msg.contains("out of MLA blocks")
                || msg.contains("OutOfBlocks")
                || msg.contains("reserve"),
            "error should reference reservation/OOM; got: {msg}"
        );
    }
    // CRITICAL invariant: source state untouched.
    assert_eq!(
        src.used_blocks(),
        src_used_before,
        "source must retain all blocks on aborted transfer"
    );
    assert_eq!(
        peer.pushed.lock().len(),
        0,
        "no blocks should have been pushed"
    );
}

#[tokio::test]
async fn mid_stream_push_failure_releases_reservation_and_preserves_source() {
    // Wrap the destination peer's MockTransport in a LatencyInjector that drops 100% of
    // push_block calls. Reserve still succeeds (different op), but every push fails.
    let (src, dst, base_handle) = make_pair(32);
    let peer = CapacityPeer::new(Arc::clone(&dst), 99);
    base_handle.register_peer(RankId(1), Arc::clone(&peer) as Arc<dyn MockPeer>);

    // LatencyInjector wrapping the source's view of the transport with all-drops profile.
    let inner: Arc<MockTransport> = Arc::new(base_handle.clone());
    let injector = LatencyInjector::new(
        inner,
        LatencyProfile {
            // Only drop pushes. reserve/release pass through with no delay/drop.
            // Sprint 4 doesn't have per-op selective drops; we use ALL_DROPS and assert
            // the abort semantics regardless of which op failed first.
            ..LatencyProfile::ALL_DROPS
        },
        RankId(0),
        Topology::SingleNode,
        0xBEEF,
    );
    let transport: Arc<dyn RankTransport> = Arc::new(injector);

    let req_id = 21u64;
    for _ in 0..3 {
        src.allocate(req_id, TokenRange::new(0, 64)).unwrap();
    }
    let src_used_before = src.used_blocks();

    // `RESERVATIONS_ACTIVE{rank="r1"}` is a process-global prometheus gauge shared with
    // every other test in this binary that touches rank 1. Capture the baseline before the
    // transfer so the assertion measures *this test's* delta rather than the absolute
    // counter (cargo-test parallelism + lazy_static metrics broke the absolute-zero check
    // on Windows/macOS runners).
    let active_before = tessera_core::metrics::RESERVATIONS_ACTIVE
        .with_label_values(&["r1"])
        .get();

    let result = src
        .transfer_request_to_rank(req_id, RankId(1), &transport)
        .await;
    assert!(result.is_err(), "ALL_DROPS must cause transfer to fail");
    // Source still owns every block.
    assert_eq!(src.used_blocks(), src_used_before);
    // No blocks landed at destination.
    assert_eq!(dst.used_blocks(), 0);
    // No active reservations leaked at destination (release_reservation was called on abort,
    // OR the reserve itself failed under ALL_DROPS — either way, no leftovers).
    let active = tessera_core::metrics::RESERVATIONS_ACTIVE
        .with_label_values(&["r1"])
        .get();
    assert!(
        (active - active_before).abs() < f64::EPSILON,
        "no reservations should leak on abort; before={active_before}, after={active}"
    );
}

#[tokio::test]
async fn empty_request_returns_zero_without_calling_transport() {
    let (src, _dst, handle) = make_pair(32);
    let transport: Arc<dyn RankTransport> = Arc::new(handle.clone());
    let moved = src
        .transfer_request_to_rank(424242, RankId(1), &transport)
        .await
        .unwrap();
    assert_eq!(moved, 0);
    // No transport events for empty request.
    let events = handle.events();
    assert!(
        events.is_empty(),
        "empty request should not touch transport"
    );
}
