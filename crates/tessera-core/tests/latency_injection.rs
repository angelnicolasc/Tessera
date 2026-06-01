//! Tests for `LatencyInjector` (Sprint 4 / WS1).
//!
//! Determinism: every test seeds the injector explicitly. Where wall-clock matters we use
//! `tokio::time::pause` + `tokio::time::advance` so the test suite remains fast and
//! independent of host scheduler jitter.

use std::sync::Arc;
use std::time::Duration;

use tessera_core::block::BlockId;
use tessera_core::rank::{NodeId, RankId, Topology};
use tessera_core::transport::{
    mock::{MockPeer, MockTransport},
    BlockPayload, LatencyInjector, LatencyProfile, RankTransport, ReservationToken,
};

/// Minimal peer that always satisfies queries with empty payloads — keeps tests focused on
/// the injection layer.
struct EchoPeer;
impl MockPeer for EchoPeer {
    fn provide_block(&self, _: BlockId) -> anyhow::Result<BlockPayload> {
        Ok(BlockPayload {
            c_kv: vec![],
            k_rope: vec![],
            fp8_scales: None,
        })
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

fn wrap_mock(
    profile: LatencyProfile,
    topology: Topology,
    seed: u64,
) -> LatencyInjector<MockTransport> {
    let handles = MockTransport::new_world(2);
    handles[0].register_peer(RankId(1), Arc::new(EchoPeer));
    let inner = Arc::new(handles[0].clone());
    LatencyInjector::new(inner, profile, RankId(0), topology, seed)
}

#[tokio::test(start_paused = true)]
async fn intra_node_tier_latency_is_honoured() {
    let injector = wrap_mock(
        LatencyProfile {
            intra_node_us: 100,
            intra_rack_us: 0,
            cross_rack_us: 0,
            jitter_us: 0,
            drop_rate: 0.0,
        },
        Topology::SingleNode,
        42,
    );

    let start = tokio::time::Instant::now();
    let _ = injector.fetch_block(RankId(1), BlockId(7)).await.unwrap();
    let elapsed = start.elapsed();
    // start_paused = true means tokio::time::sleep advances virtual time. The sleep was
    // 100 µs base + 0 jitter; tokio::time auto-advances on sleep.
    assert!(
        elapsed >= Duration::from_micros(100),
        "fetch_block should sleep ≥ intra_node_us; observed {elapsed:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn multi_node_topology_picks_cross_node_tier() {
    let topo = Topology::MultiNode {
        node_of: vec![NodeId(0), NodeId(1)],
    };
    let injector = wrap_mock(
        LatencyProfile {
            intra_node_us: 1,
            intra_rack_us: 500,
            cross_rack_us: 500,
            jitter_us: 0,
            drop_rate: 0.0,
        },
        topo,
        7,
    );

    let start = tokio::time::Instant::now();
    let _ = injector.fetch_block(RankId(1), BlockId(0)).await.unwrap();
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_micros(500),
        "peer on different node should pay intra_rack_us; observed {elapsed:?}"
    );
}

#[tokio::test]
async fn drop_rate_one_point_zero_always_fails() {
    let injector = wrap_mock(LatencyProfile::ALL_DROPS, Topology::SingleNode, 1);
    let err = injector
        .fetch_block(RankId(1), BlockId(0))
        .await
        .unwrap_err();
    assert!(
        format!("{err}").contains("simulated drop"),
        "expected simulated drop error, got: {err}"
    );
}

#[tokio::test]
async fn drop_rate_zero_never_fails() {
    let injector = wrap_mock(LatencyProfile::ZERO, Topology::SingleNode, 9);
    for _ in 0..32 {
        injector.fetch_block(RankId(1), BlockId(0)).await.unwrap();
    }
}

#[tokio::test]
async fn deterministic_with_explicit_seed() {
    let p = LatencyProfile {
        intra_node_us: 10,
        intra_rack_us: 0,
        cross_rack_us: 0,
        jitter_us: 100,
        drop_rate: 0.5,
    };
    // Two injectors with the same seed should make the same drop decisions across the
    // same call sequence.
    let inj_a = wrap_mock(p, Topology::SingleNode, 0xC0FFEE);
    let inj_b = wrap_mock(p, Topology::SingleNode, 0xC0FFEE);
    let mut seq_a = Vec::new();
    let mut seq_b = Vec::new();
    for _ in 0..16 {
        seq_a.push(inj_a.query_hash(RankId(1), 0).await.is_err());
        seq_b.push(inj_b.query_hash(RankId(1), 0).await.is_err());
    }
    assert_eq!(
        seq_a, seq_b,
        "same seed must produce identical drop sequences"
    );
}

#[tokio::test]
async fn metric_increments_on_drop() {
    let before = tessera_core::metrics::LATENCY_INJECTED_DROPS_TOTAL
        .with_label_values(&["fetch_block"])
        .get();
    let injector = wrap_mock(LatencyProfile::ALL_DROPS, Topology::SingleNode, 5);
    let _ = injector.fetch_block(RankId(1), BlockId(0)).await;
    let after = tessera_core::metrics::LATENCY_INJECTED_DROPS_TOTAL
        .with_label_values(&["fetch_block"])
        .get();
    assert!(after > before, "drop counter should increment");
}
