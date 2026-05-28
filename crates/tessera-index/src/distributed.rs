//! Distributed segment index: fan-out across ranks with an explicit total budget.
//!
//! The orchestration mirrors the two-layer index from `tessera-core`:
//!
//! 1. Try the local backend first (budget: `local_fraction * total_budget`).
//! 2. On miss, fan out to all peer ranks via [`RankTransport::query_hash`] for fast hash
//!    short-circuits, then [`RankTransport::fetch_block`] for full retrieval. First hit
//!    wins; remaining requests are cancelled.
//! 3. On total miss / budget exhaustion, return `None`. A miss is **always safe**: the
//!    caller falls back to computing the block locally.
//!
//! Sprint 3 ships the hash-only fast path (Layer-A-distributed). The "descriptor fan-out"
//! mode for HNSW-style similarity matches is reserved for Sprint 4 once the
//! `IndexBackend::query` round-trip is profiled across NVLink and NCCL transports.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::FutureExt;
use tessera_core::block::{BlockId, GlobalBlockId};
use tessera_core::rank::{LatencyTier, RankId, World};
use tessera_core::transport::RankTransport;

use crate::IndexBackend;

/// Budget multipliers per latency tier. Sprint 4 ships pragmatic defaults; tuning is a
/// deployment concern and the multipliers can be overridden per-instance via
/// [`DistributedSegmentIndex::with_tier_multipliers`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierBudget {
    /// Multiplier applied to the base budget for intra-node peers (typically 1.0×).
    pub intra_node: f32,
    /// Multiplier for intra-rack peers (typically ~5–10× the intra-node base).
    pub intra_rack: f32,
    /// Multiplier for cross-rack peers (typically ~50–100× the intra-node base).
    pub cross_rack: f32,
}

impl TierBudget {
    /// Pragmatic defaults derived from typical NVLink/NVSwitch/IB latency ratios.
    pub const DEFAULT: Self = Self {
        intra_node: 1.0,
        intra_rack: 8.0,
        cross_rack: 80.0,
    };

    /// Multiplier for a tier.
    pub const fn multiplier(&self, tier: LatencyTier) -> f32 {
        match tier {
            LatencyTier::IntraNode => self.intra_node,
            LatencyTier::IntraRack => self.intra_rack,
            LatencyTier::CrossRack => self.cross_rack,
        }
    }
}

impl Default for TierBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A distributed-index hit. Carries the global address so the caller can route to the
/// correct rank's block manager (fetch, increment-ref, etc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributedHit {
    /// Global identity of the matched block.
    pub global: GlobalBlockId,
    /// Whether the hit came from the local backend (`true`) or from a remote rank.
    pub local: bool,
}

/// Distributed segment index over `IndexBackend` + a cross-rank [`RankTransport`].
///
/// Sprint 4: budget scales with peer topology tier via [`TierBudget`]. The base
/// `total_budget` represents the intra-node case (1.0× multiplier); intra-rack and
/// cross-rack peers scale up. This avoids starving multi-node lookups under a budget that
/// was tuned for NVLink latencies.
pub struct DistributedSegmentIndex {
    local: Arc<dyn IndexBackend>,
    world: Arc<World>,
    transport: Arc<dyn RankTransport>,
    total_budget: Duration,
    local_fraction: f32,
    tier_budget: TierBudget,
}

impl std::fmt::Debug for DistributedSegmentIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedSegmentIndex")
            .field("local_name", &self.local.name())
            .field("local_len", &self.local.len())
            .field("world_size", &self.world.size)
            .field("transport", &self.transport.name())
            .field("total_budget", &self.total_budget)
            .finish()
    }
}

impl DistributedSegmentIndex {
    /// Construct a distributed index. `local` is the per-rank backend (typically a
    /// [`crate::UsearchIndex`]). `world` describes the surrounding deployment; `transport`
    /// is the cross-rank message bus. `total_budget` bounds end-to-end lookup latency.
    pub fn new(
        local: Arc<dyn IndexBackend>,
        world: Arc<World>,
        transport: Arc<dyn RankTransport>,
        total_budget: Duration,
    ) -> Self {
        Self {
            local,
            world,
            transport,
            total_budget,
            local_fraction: 0.4,
            tier_budget: TierBudget::DEFAULT,
        }
    }

    /// Override the fraction of the total budget that the local lookup is allowed to
    /// consume. Default is 0.4 (40%). Remaining budget is the cap for fan-out.
    pub fn with_local_budget_fraction(mut self, fraction: f32) -> Self {
        self.local_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Override the per-tier budget multipliers. Default is [`TierBudget::DEFAULT`].
    pub fn with_tier_multipliers(mut self, tier_budget: TierBudget) -> Self {
        self.tier_budget = tier_budget;
        self
    }

    /// Effective budget for reaching `peer` given the configured base + tier multipliers.
    /// Public for callers that need to size their own timeouts compatibly.
    pub fn effective_budget_for(&self, peer: RankId) -> Duration {
        let tier = self.world.peer_tier(peer);
        let mult = self.tier_budget.multiplier(tier);
        // `Duration::mul_f32` saturates on overflow; safe for any sane multiplier.
        self.total_budget.mul_f32(mult.max(0.0))
    }

    /// The world this index spans.
    pub fn world(&self) -> &Arc<World> {
        &self.world
    }

    /// Sprint 3 — hash-only lookup. Resolves a content hash to its `(rank, block_id)`. Used
    /// by the rank-aware plugin's `find_shared_prefix` on Layer 1 misses.
    ///
    /// Returns `Ok(None)` when nobody (local or remote) holds the hash, or when the budget
    /// is exhausted before any hit. Both are safe misses.
    pub async fn lookup_hash(&self, content_hash: u64) -> anyhow::Result<Option<DistributedHit>> {
        let start = Instant::now();
        let total = self.total_budget;
        let local_budget = total.mul_f32(self.local_fraction);

        // (1) Local fast path. Hash lookup is not directly supported by IndexBackend (it's
        // ANN), so the local hash table lives in the block manager / segment index in
        // Python. For the Rust core distributed lookup we *only* need to check whether the
        // local rank already knows about this hash through its local IndexBackend's
        // `query` path — but that's a descriptor lookup. To keep Sprint 3 narrow we just
        // skip local (we assume the caller already checked) and go straight to fan-out.
        // Tests cover this contract: caller does local check, then calls us for remote.
        // Local time accounted for telemetry.
        let local_elapsed = start.elapsed();
        if local_elapsed >= local_budget {
            tracing::debug!(
                ?local_elapsed,
                ?local_budget,
                "DistributedSegmentIndex: local budget exhausted before fan-out"
            );
        }

        // (2) Fan out to peers via query_hash.
        //
        // Effective budget is the **maximum** per-peer tier budget across all peers —
        // we wait at most as long as the slowest peer's tier allows. The race semantics
        // (first hit wins) make this safe: faster peers respond well within their own
        // tier and we cancel the rest. Without tier-aware scaling, the base `total` would
        // starve multi-node lookups (intra-rack peers need ~8× more wall-clock).
        let peers: Vec<RankId> = self.world.peers().collect();
        if peers.is_empty() {
            self.record_miss(start);
            return Ok(None);
        }
        // `total_budget` is the intra-node baseline. For each peer we scale by its tier;
        // the fan-out runs until the SLOWEST tier's effective budget elapses. Faster peers
        // resolve well within that envelope and lose nothing.
        let max_peer_budget = peers
            .iter()
            .map(|p| self.effective_budget_for(*p))
            .max()
            .unwrap_or(total);
        // Account for the local phase already spent.
        let local_consumed = start.elapsed();
        let remaining = max_peer_budget.saturating_sub(local_consumed);
        if remaining.is_zero() {
            self.record_miss(start);
            return Ok(None);
        }

        let transport = Arc::clone(&self.transport);

        // Launch parallel queries; first hit wins; others cancelled by dropping the futures.
        let mut futs = futures::stream::FuturesUnordered::new();
        for peer in peers {
            let t = Arc::clone(&transport);
            futs.push(
                async move {
                    let res = t.query_hash(peer, content_hash).await;
                    (peer, res)
                }
                .boxed(),
            );
        }

        let timeout_fut = tokio::time::sleep(remaining);
        tokio::pin!(timeout_fut);

        use futures::StreamExt;
        let outcome = loop {
            tokio::select! {
                biased;
                _ = &mut timeout_fut => break None,
                next = futs.next() => match next {
                    Some((peer, Ok(Some(block)))) => break Some((peer, block)),
                    Some((_, Ok(None))) => continue,
                    Some((peer, Err(e))) => {
                        tracing::warn!(?peer, error = %e, "query_hash failed on peer");
                        continue;
                    }
                    None => break None, // all peers exhausted, no hit
                },
            }
        };

        let elapsed = start.elapsed();
        tessera_core::metrics::DISTRIBUTED_INDEX_FANOUT_LATENCY_SECONDS
            .observe(elapsed.as_secs_f64());

        match outcome {
            Some((src_rank, block_id)) => {
                tessera_core::metrics::DISTRIBUTED_INDEX_REMOTE_HITS_TOTAL
                    .with_label_values(&[&src_rank.to_string()])
                    .inc();
                Ok(Some(DistributedHit {
                    global: GlobalBlockId::new(src_rank, block_id),
                    local: false,
                }))
            }
            None => {
                tessera_core::metrics::DISTRIBUTED_INDEX_MISSES_TOTAL.inc();
                Ok(None)
            }
        }
    }

    /// Record a local hit. The caller invokes this when their own local index resolved the
    /// query without needing fan-out. Sprint 3 keeps the local lookup ownership with the
    /// caller (the Python segment index) so we expose this telemetry helper.
    pub fn record_local_hit(&self) {
        tessera_core::metrics::DISTRIBUTED_INDEX_LOCAL_HITS_TOTAL.inc();
    }

    fn record_miss(&self, start: Instant) {
        let elapsed = start.elapsed();
        tessera_core::metrics::DISTRIBUTED_INDEX_FANOUT_LATENCY_SECONDS
            .observe(elapsed.as_secs_f64());
        tessera_core::metrics::DISTRIBUTED_INDEX_MISSES_TOTAL.inc();
    }

    /// Borrow the local backend (for direct add/remove on this rank).
    pub fn local(&self) -> &Arc<dyn IndexBackend> {
        &self.local
    }

    /// Borrow the transport handle.
    pub fn transport(&self) -> &Arc<dyn RankTransport> {
        &self.transport
    }
}
