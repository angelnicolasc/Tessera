//! Prometheus metrics exposed by the block manager.
//!
//! These are global statics so they can be referenced from any layer (the block manager
//! itself, the share table, the Python plugin) without threading a `Registry` through the
//! constructor signatures. Initialisation is idempotent via `lazy_static`.
//!
//! Names and help-text follow the playbook §12 spec.

use lazy_static::lazy_static;
use prometheus::{
    register_counter, register_counter_vec, register_gauge, register_gauge_vec,
    register_histogram, Counter, CounterVec, Gauge, GaugeVec, Histogram,
};

lazy_static! {
    /// Fraction of MLA blocks currently allocated (0.0 — 1.0).
    pub static ref BLOCK_UTILIZATION: Gauge = register_gauge!(
        "tessera_block_utilization",
        "Fraction of MLA blocks currently in use (0.0–1.0)"
    )
    .expect("failed to register tessera_block_utilization");

    /// Fraction of allocated block references that are shared (rather than privately held).
    pub static ref SHARING_RATE: Gauge = register_gauge!(
        "tessera_sharing_rate",
        "Fraction of allocated block references that are shared (vs. private)"
    )
    .expect("failed to register tessera_sharing_rate");

    /// Total number of blocks collapsed via xxhash3 exact match in seal().
    pub static ref EXACT_DEDUP_HITS: Counter = register_counter!(
        "tessera_exact_dedup_hits_total",
        "Blocks deduplicated via xxhash3 exact-match in seal()"
    )
    .expect("failed to register tessera_exact_dedup_hits_total");

    /// Total number of blocks matched via HNSW approximate lookup (Python side).
    pub static ref HNSW_MATCH_HITS: Counter = register_counter!(
        "tessera_hnsw_match_hits_total",
        "Blocks reused via HNSW approximate match"
    )
    .expect("failed to register tessera_hnsw_match_hits_total");

    /// Number of HNSW lookups that exceeded their latency budget and returned None.
    pub static ref HNSW_LATENCY_EXCEEDED: Counter = register_counter!(
        "tessera_hnsw_budget_exceeded_total",
        "HNSW lookups that exceeded their latency budget"
    )
    .expect("failed to register tessera_hnsw_budget_exceeded_total");

    /// Number of copy-on-write forks performed by the block manager.
    pub static ref COW_FORKS: Counter = register_counter!(
        "tessera_cow_forks_total",
        "Copy-on-write block forks triggered"
    )
    .expect("failed to register tessera_cow_forks_total");

    /// Effective compression ratio for the configured block layout relative to MHA BF16.
    pub static ref COMPRESSION_RATIO: Gauge = register_gauge!(
        "tessera_compression_ratio_vs_mha",
        "Effective compression ratio relative to MHA BF16 block storage"
    )
    .expect("failed to register tessera_compression_ratio_vs_mha");

    /// Total blocks freed by `release_request`. Each unit is one block freed, not one
    /// request released — the count tracks overall lifecycle throughput.
    pub static ref REQUEST_RELEASES_TOTAL: Counter = register_counter!(
        "tessera_request_releases_total",
        "Total MLA blocks freed via release_request()"
    )
    .expect("failed to register tessera_request_releases_total");

    /// Blocks evicted by tier. `tier=a`: orphaned (ref_count==0); `tier=b`: unindexed
    /// inactive (ref_count==1, not in segment index); `tier=c`: indexed inactive
    /// (ref_count==1, in segment index — highest reuse value, evicted last).
    pub static ref EVICTIONS_TOTAL: CounterVec = register_counter_vec!(
        "tessera_evictions_total",
        "Blocks evicted by tier (a=orphaned, b=unindexed-inactive, c=indexed-inactive)",
        &["tier"]
    )
    .expect("failed to register tessera_evictions_total");

    // ───────── Sprint 3 — Multi-rank / TP metrics ─────────────────────────────

    /// Allocated blocks per rank. Labeled by `rank` (e.g. `r0`, `r1`).
    pub static ref BLOCKS_PER_RANK: GaugeVec = register_gauge_vec!(
        "tessera_blocks_per_rank",
        "Allocated blocks per rank",
        &["rank"]
    )
    .expect("failed to register tessera_blocks_per_rank");

    /// Cross-rank block transfers (broadcasts, fetches). Labels: `src`, `dst`, `kind`
    /// where `kind` ∈ {`broadcast_seal`, `fetch`, `push`, `release_announce`}.
    pub static ref CROSS_RANK_TRANSFERS_TOTAL: CounterVec = register_counter_vec!(
        "tessera_cross_rank_transfers_total",
        "Cross-rank block transfers initiated by this rank",
        &["src", "dst", "kind"]
    )
    .expect("failed to register tessera_cross_rank_transfers_total");

    /// PD-disaggregation transfers (whole-request migrations). Labels: `src`, `dst`.
    pub static ref PD_DISAGG_TRANSFERS_TOTAL: CounterVec = register_counter_vec!(
        "tessera_pd_disagg_transfers_total",
        "Whole-request KV transfers between ranks (PD-disaggregation)",
        &["src", "dst"]
    )
    .expect("failed to register tessera_pd_disagg_transfers_total");

    /// Distributed segment index — local hits (resolved without fan-out).
    pub static ref DISTRIBUTED_INDEX_LOCAL_HITS_TOTAL: Counter = register_counter!(
        "tessera_distributed_index_local_hits_total",
        "Distributed segment index lookups resolved by the local rank"
    )
    .expect("failed to register tessera_distributed_index_local_hits_total");

    /// Distributed segment index — remote hits, labeled by source rank.
    pub static ref DISTRIBUTED_INDEX_REMOTE_HITS_TOTAL: CounterVec = register_counter_vec!(
        "tessera_distributed_index_remote_hits_total",
        "Distributed segment index lookups resolved by a remote rank",
        &["src_rank"]
    )
    .expect("failed to register tessera_distributed_index_remote_hits_total");

    /// Distributed segment index — full misses (no rank had the block).
    pub static ref DISTRIBUTED_INDEX_MISSES_TOTAL: Counter = register_counter!(
        "tessera_distributed_index_misses_total",
        "Distributed segment index lookups that produced no hit on any rank"
    )
    .expect("failed to register tessera_distributed_index_misses_total");

    /// Distributed segment index — fan-out wall-clock latency. Bounded by the configured
    /// per-call budget; samples exceeding the budget are still recorded for diagnostics.
    pub static ref DISTRIBUTED_INDEX_FANOUT_LATENCY_SECONDS: Histogram = register_histogram!(
        "tessera_distributed_index_fanout_latency_seconds",
        "Wall-clock latency of distributed segment index fan-out lookups",
        vec![1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2]
    )
    .expect("failed to register tessera_distributed_index_fanout_latency_seconds");

    // ───────── Sprint 4 — chaos injection + transactional reservations ─────

    /// Simulated drops induced by `LatencyInjector`. Labels: `op` (broadcast_seal, fetch_block,
    /// push_block, etc). Production deployments should see this at zero unless a staging
    /// chaos rig is intentionally configured.
    pub static ref LATENCY_INJECTED_DROPS_TOTAL: CounterVec = register_counter_vec!(
        "tessera_latency_injected_drops_total",
        "Cross-rank transport calls dropped by the LatencyInjector (chaos / staging only)",
        &["op"]
    )
    .expect("failed to register tessera_latency_injected_drops_total");

    /// Active block-slot reservations across all ranks. Labels: `rank` (the destination
    /// holding the reservation).
    pub static ref RESERVATIONS_ACTIVE: GaugeVec = register_gauge_vec!(
        "tessera_reservations_active",
        "Active reservation tokens on this rank (pending PD-disagg transfers)",
        &["rank"]
    )
    .expect("failed to register tessera_reservations_active");

    /// PD-disagg transfers that aborted (e.g. target reservation failed, mid-stream push
    /// dropped). Labeled by `reason` so dashboards can distinguish capacity failures from
    /// transport failures.
    pub static ref TRANSFER_ABORTS_TOTAL: CounterVec = register_counter_vec!(
        "tessera_transfer_aborts_total",
        "PD-disaggregation transfers that aborted before full success",
        &["reason"]
    )
    .expect("failed to register tessera_transfer_aborts_total");
}

/// Snapshot all metrics as Prometheus text format. Convenient for unit tests and for
/// embedding a `/metrics` endpoint without pulling in a full HTTP server.
pub fn snapshot_text() -> String {
    use prometheus::Encoder;
    let metric_families = prometheus::gather();
    let encoder = prometheus::TextEncoder::new();
    let mut buf = Vec::new();
    encoder
        .encode(&metric_families, &mut buf)
        .expect("encoding prometheus metrics never fails for in-memory write");
    String::from_utf8(buf).expect("prometheus text encoder is always valid UTF-8")
}
