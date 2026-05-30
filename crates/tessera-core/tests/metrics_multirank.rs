//! Sprint 3 multi-rank metric wiring.
//!
//! These tests don't snapshot exact counter deltas (Prometheus globals accumulate across
//! the entire test binary), but they assert that the new families *can* be incremented
//! without panic and that they appear in the text snapshot.

use std::sync::Arc;

use tessera_core::{
    block::BlockId,
    metrics,
    rank::{RankId, Topology, World},
    transport::{
        mock::{MockPeer, MockTransport},
        BlockPayload, RankTransport,
    },
};

struct StaticPeer;
impl MockPeer for StaticPeer {
    fn provide_block(&self, _: BlockId) -> anyhow::Result<BlockPayload> {
        Ok(BlockPayload {
            c_kv: vec![1, 2, 3],
            k_rope: vec![4],
            fp8_scales: None,
        })
    }
    fn accept_pushed(&self, _: BlockPayload) -> anyhow::Result<BlockId> {
        Ok(BlockId(0))
    }
    fn lookup_hash(&self, _: u64) -> anyhow::Result<Option<BlockId>> {
        Ok(None)
    }
}

#[tokio::test]
async fn cross_rank_transfer_metric_is_recorded_on_fetch() {
    let handles = MockTransport::new_world(2);
    handles[0].register_peer(RankId(1), Arc::new(StaticPeer));

    // Capture before/after.
    let before = metrics::CROSS_RANK_TRANSFERS_TOTAL
        .with_label_values(&["r1", "r0", "fetch"])
        .get();
    let _ = handles[0].fetch_block(RankId(1), BlockId(7)).await.unwrap();
    let after = metrics::CROSS_RANK_TRANSFERS_TOTAL
        .with_label_values(&["r1", "r0", "fetch"])
        .get();
    assert!(
        after > before,
        "fetch should increment the cross-rank transfer counter"
    );
}

#[tokio::test]
async fn broadcast_seal_increments_per_peer() {
    let handles = MockTransport::new_world(4);
    for r in 1..4 {
        handles[0].register_peer(RankId(r), Arc::new(StaticPeer));
    }
    let mut before = [0.0; 3];
    for (i, r) in (1..4).enumerate() {
        before[i] = metrics::CROSS_RANK_TRANSFERS_TOTAL
            .with_label_values(&["r0", &format!("r{r}"), "broadcast_seal"])
            .get();
    }
    handles[0]
        .broadcast_seal(RankId(0), BlockId(11), 0xCAFE, vec![0.5; 8])
        .await
        .unwrap();
    for (i, r) in (1..4).enumerate() {
        let after = metrics::CROSS_RANK_TRANSFERS_TOTAL
            .with_label_values(&["r0", &format!("r{r}"), "broadcast_seal"])
            .get();
        assert!(
            after > before[i],
            "broadcast must increment counter for peer r{r}"
        );
    }
}

#[test]
fn metric_families_appear_in_snapshot_text() {
    // Touch each new family at least once so it materialises in the text encoder output.
    metrics::BLOCKS_PER_RANK.with_label_values(&["r0"]).set(1.0);
    metrics::PD_DISAGG_TRANSFERS_TOTAL
        .with_label_values(&["r0", "r1"])
        .inc();
    metrics::DISTRIBUTED_INDEX_LOCAL_HITS_TOTAL.inc();
    metrics::DISTRIBUTED_INDEX_MISSES_TOTAL.inc();
    metrics::DISTRIBUTED_INDEX_FANOUT_LATENCY_SECONDS.observe(0.0001);
    // The remaining two families are exercised by the async tests above, but those
    // share process-global prometheus state and may not have run yet when this test
    // executes (cargo-test runs in parallel). Touch them here so the assertion
    // doesn't depend on test ordering.
    metrics::CROSS_RANK_TRANSFERS_TOTAL
        .with_label_values(&["r0", "r1", "fetch"])
        .inc();
    metrics::DISTRIBUTED_INDEX_REMOTE_HITS_TOTAL
        .with_label_values(&["r1"])
        .inc();

    let text = metrics::snapshot_text();
    for family in [
        "tessera_blocks_per_rank",
        "tessera_cross_rank_transfers_total",
        "tessera_pd_disagg_transfers_total",
        "tessera_distributed_index_local_hits_total",
        "tessera_distributed_index_remote_hits_total",
        "tessera_distributed_index_misses_total",
        "tessera_distributed_index_fanout_latency_seconds",
    ] {
        assert!(
            text.contains(family),
            "snapshot text missing metric family {family}"
        );
    }
    // World correctness: world ctor invariants are unaffected by metrics.
    assert!(World::new(RankId(0), 1, Topology::SingleNode).is_some());
}
