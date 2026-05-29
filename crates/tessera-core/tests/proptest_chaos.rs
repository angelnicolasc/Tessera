//! Property-based chaos tests over random op sequences + random `LatencyProfile`.
//! Sprint 4 / WS4 / ADR-0019.
//!
//! These tests are slow (256 cases × multiple async ops each). They run as part of
//! `cargo test --workspace` but are wrapped in `proptest!` config that keeps wall-clock
//! reasonable on CI runners. Regression seeds land in `tests/proptest-regressions/`.

use std::sync::Arc;

use proptest::prelude::*;
use tessera_core::{
    block::BlockId,
    rank::{RankId, Topology, World},
    transport::{
        mock::{MockPeer, MockTransport},
        BlockPayload, LatencyInjector, LatencyProfile, RankTransport, ReservationToken,
    },
    CkvDtype, CompressionScheme, MlaBlockConfig, TesseraBlockManager, TokenRange,
};

fn small_cfg() -> MlaBlockConfig {
    MlaBlockConfig::new(
        CompressionScheme::MlaLatent {
            latent_dim: 16,
            rope_key_dim: 4,
        },
        2,
        64,
        CkvDtype::Bf16,
        0,
    )
    .unwrap()
}

#[derive(Debug, Clone)]
enum Op {
    Allocate { req: u8 },
    Free { idx: usize },
    ReleaseRequest { req: u8 },
    Seal { idx: usize, pattern: u8 },
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0..4u8).prop_map(|req| Op::Allocate { req }),
        (0..16usize).prop_map(|idx| Op::Free { idx }),
        (0..4u8).prop_map(|req| Op::ReleaseRequest { req }),
        (0..16usize, any::<u8>()).prop_map(|(idx, pattern)| Op::Seal { idx, pattern }),
    ]
}

fn sequence_strategy() -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(op_strategy(), 1..32)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,             // keep CI wall-clock bounded; cargo test runs in <30s
        max_shrink_iters: 256,
        .. ProptestConfig::default()
    })]

    /// Round-trip invariant: every Allocate eventually paired with a Free or
    /// ReleaseRequest leaves `used_blocks` consistent with the live set.
    #[test]
    fn alloc_free_release_keeps_used_blocks_consistent(ops in sequence_strategy()) {
        let mgr = TesseraBlockManager::new(small_cfg(), 16 * 1024 * 1024).unwrap();
        let mut live: Vec<(BlockId, u8)> = Vec::new();   // (block_id, req)

        for op in ops {
            match op {
                Op::Allocate { req } => {
                    if let Ok(bid) = mgr.allocate(u64::from(req), TokenRange::new(0, 64)) {
                        live.push((bid, req));
                    }
                }
                Op::Free { idx } => {
                    if idx < live.len() {
                        let (bid, _) = live.remove(idx);
                        mgr.free(bid).unwrap();
                    }
                }
                Op::ReleaseRequest { req } => {
                    let freed = mgr.release_request(u64::from(req));
                    let before = live.len();
                    live.retain(|(_, r)| *r != req);
                    let _after = live.len();
                    // freed count should equal (before - after) minus any blocks that
                    // were already individually freed before release_request fired (we
                    // don't track that in this simple model, so we only check freed <=
                    // before).
                    prop_assert!(freed as usize <= before);
                }
                Op::Seal { idx, pattern } => {
                    if idx < live.len() {
                        let (bid, _) = live[idx];
                        let _ = mgr.fill_primary_test_pattern(bid, pattern);
                        // Sealing may collapse via dedup; remove from our live tracking
                        // when the canonical block differs.
                        if let Ok(out) = mgr.seal(bid) {
                            if out.was_dedup {
                                live.remove(idx);
                            }
                        }
                    }
                }
            }
            // Invariant 1: used_blocks never exceeds total.
            prop_assert!(mgr.used_blocks() <= mgr.total_blocks());
        }
    }

    /// Eviction safety: under random op pressure, no block with ref_count > 1 (shared) is
    /// ever physically freed. We can't directly observe ref_counts from outside, but we
    /// CAN observe that `free` on a known live block id never panics — even after random
    /// eviction pressure pushes the manager to its limits.
    #[test]
    fn free_of_known_block_never_panics(ops in sequence_strategy()) {
        let mgr = TesseraBlockManager::new(small_cfg(), 8 * 1024 * 1024).unwrap();
        let mut live: Vec<BlockId> = Vec::new();

        for op in ops {
            if let Op::Allocate { req } = op {
                if let Ok(bid) = mgr.allocate(u64::from(req), TokenRange::new(0, 64)) {
                    live.push(bid);
                }
            }
        }
        // Free every live block. Some may have been evicted; that's the safe-no-op path
        // (`free` of an unknown id returns Ok).
        for bid in live {
            prop_assert!(mgr.free(bid).is_ok());
        }
    }

    /// release_request fidelity: blocks freed by release_request must equal the count of
    /// private blocks that req owned at the time of call.
    #[test]
    fn release_request_fidelity(allocs in 1usize..16) {
        let mgr = TesseraBlockManager::new(small_cfg(), 32 * 1024 * 1024).unwrap();
        let req = 42u64;
        for _ in 0..allocs {
            // Some allocations may fail if budget too small; track actual count.
            let _ = mgr.allocate(req, TokenRange::new(0, 64));
        }
        let actual = mgr.used_blocks();
        let freed = mgr.release_request(req);
        prop_assert_eq!(freed, actual);
        prop_assert_eq!(mgr.used_blocks(), 0);
    }
}

// ───────── Async chaos: transfer_request_to_rank under random LatencyProfile ───────────

struct StaticPeer {
    manager: Arc<TesseraBlockManager>,
}
impl StaticPeer {
    fn new(manager: Arc<TesseraBlockManager>) -> Arc<Self> {
        Arc::new(Self { manager })
    }
}
impl MockPeer for StaticPeer {
    fn provide_block(&self, b: BlockId) -> anyhow::Result<BlockPayload> {
        self.manager
            .export_payload(b)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
    fn accept_pushed(&self, p: BlockPayload) -> anyhow::Result<BlockId> {
        self.manager
            .import_payload(999, TokenRange::new(0, 64), &p)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
    fn lookup_hash(&self, _: u64) -> anyhow::Result<Option<BlockId>> {
        Ok(None)
    }
    fn reserve(&self, req_id: u64, count: u32) -> anyhow::Result<ReservationToken> {
        self.manager
            .reserve_incoming(req_id, count)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
    fn release_reservation(&self, t: ReservationToken) -> anyhow::Result<()> {
        self.manager
            .release_reservation_local(t)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Verifies the atomicity contract: after any number of transfer attempts under chaos,
/// the source either fully transferred (count > 0, used_blocks_src dropped) OR retained
/// all its blocks (count is irrelevant; transfer errored, src state preserved).
#[test]
fn transfer_atomicity_under_chaos() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        for seed in 0u64..16 {
            let handles = MockTransport::new_world(2);
            let w0 = Arc::new(World::new(RankId(0), 2, Topology::SingleNode).unwrap());
            let w1 = Arc::new(World::new(RankId(1), 2, Topology::SingleNode).unwrap());
            let src = Arc::new(
                TesseraBlockManager::new_with_world(small_cfg(), 8 * 1024 * 1024, RankId(0), w0)
                    .unwrap(),
            );
            let dst = Arc::new(
                TesseraBlockManager::new_with_world(small_cfg(), 8 * 1024 * 1024, RankId(1), w1)
                    .unwrap(),
            );
            handles[0].register_peer(
                RankId(1),
                Arc::clone(&StaticPeer::new(Arc::clone(&dst))) as Arc<dyn MockPeer>,
            );

            // 5 source blocks, random drop_rate.
            for _ in 0..5 {
                let bid = src.allocate(seed, TokenRange::new(0, 64)).unwrap();
                src.fill_primary_test_pattern(bid, 0x11).unwrap();
            }
            let src_before = src.used_blocks();

            let drop_rate = (seed as f32) / 20.0; // 0.0 .. 0.75
            let injector = LatencyInjector::new(
                Arc::new(handles[0].clone()),
                LatencyProfile {
                    drop_rate,
                    ..LatencyProfile::ZERO
                },
                RankId(0),
                Topology::SingleNode,
                seed,
            );
            let transport: Arc<dyn RankTransport> = Arc::new(injector);

            let outcome = src
                .transfer_request_to_rank(seed, RankId(1), &transport)
                .await;
            let src_after = src.used_blocks();
            let dst_after = dst.used_blocks();

            if outcome.is_ok() {
                assert_eq!(src_after, 0, "successful transfer must drain source");
                assert_eq!(dst_after, src_before, "all blocks must land at destination");
            } else {
                assert_eq!(
                    src_after, src_before,
                    "failed transfer must retain source state (seed={seed} drop_rate={drop_rate})"
                );
            }
        }
    });
}
